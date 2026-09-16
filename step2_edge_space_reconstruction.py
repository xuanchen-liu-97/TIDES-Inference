"""TIDES Step 2: reconstruct the observational edge-space field family.

The three-layer TIDES architecture uses the stage-wise lifted coefficients

    B^(r) in R^(M x L),  r = 1,...,K,

as the canonical Step-2 object.  For preprocessed node-space vector-field
observations Y, a candidate edge set D, a local interaction library Psi, and
stage labels, Step 2 solves the linear observation model

    y_r = A_r vec(B^(r)) + eta_r

independently by stage and returns the *entire* uncertainty-floor feasible
family

    B_epsilon = { B_(1:K) : ||y - A b|| / ||y|| <= epsilon }.

No physical topology/dynamics hypothesis enters this module.  The temporal
change tensors

    Delta B^(r) = B^(r+1) - B^(r)

are exposed as a derived view of the same family.

Implementation notes
--------------------
The global design is block diagonal across temporal stages.  The dense
reference backend therefore factorizes each stage separately.  This is both
cheaper and more informative than forming one giant SVD.  In scaled
coordinates c_r = S_r b_r, a dense stage family has

    c_r = c0_r + V_r^T a_r + z_r,

where z_r lies in the numerical nullspace and

    sum_r ||Sigma_r a_r||^2 <= delta^2 - ||r0||^2.

For large designs, a LinearOperator/LSQR backend keeps the family implicit.
The feasibility oracle remains exact with respect to the supplied numerical
operator; explicit identified/null projectors and random family sampling are
available when dense SVD geometry is present.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

try:  # package form
    from .solvers_linear_regression import solve_least_squares
except ImportError:  # flat-file form
    from solvers_linear_regression import solve_least_squares

try:
    from scipy.sparse.linalg import LinearOperator
except Exception:  # pragma: no cover - scipy only needed for operator mode
    LinearOperator = None


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


# -----------------------------------------------------------------------------
# Input handling
# -----------------------------------------------------------------------------


def _validate_standard_pairwise_incidence(D: FloatArray) -> None:
    D = np.asarray(D, dtype=float)
    if D.ndim != 2:
        raise ValueError("D must have shape (n_nodes, n_candidate_edges).")

    nz = np.abs(D) > 1.0e-12
    if not np.all(np.sum(nz, axis=0) == 2):
        raise ValueError(
            "Each candidate pair must have exactly two nonzero incidence entries."
        )
    for m in range(D.shape[1]):
        vals = np.sort(D[nz[:, m], m])
        if not np.allclose(vals, np.array([-1.0, 1.0]), atol=1.0e-12, rtol=0.0):
            raise ValueError(
                "Endpoint-output libraries require standard oriented incidence "
                "columns with values {-1,+1}."
            )


def _pair_endpoints_from_incidence(D: FloatArray) -> tuple[IntArray, IntArray]:
    _validate_standard_pairwise_incidence(D)
    M = D.shape[1]
    tail = np.empty(M, dtype=np.int64)
    head = np.empty(M, dtype=np.int64)
    for m in range(M):
        tail[m] = int(np.flatnonzero(D[:, m] < -0.5)[0])
        head[m] = int(np.flatnonzero(D[:, m] > 0.5)[0])
    return tail, head


def _coerce_library(
    library: Any,
    *,
    n_samples: int,
    n_edges: int,
) -> tuple[FloatArray, tuple[str, ...], str, Mapping[str, Any]]:
    """Return the evaluated edge-local feature tensor and lightweight metadata.

    Preferred input is ``PairwiseLocalInteractionLibrary`` from
    ``pairwise_local_interaction_library.py``.  A raw array is also accepted:

        (T,M,L,2)  generalized endpoint-output atoms;
        (T,M,L)    scalar oriented-edge atoms (assembled through D).
    """

    if hasattr(library, "endpoint_features"):
        features = np.asarray(library.endpoint_features, dtype=float)
        labels = tuple(str(x) for x in getattr(library, "feature_labels", ()))
        representation = str(
            getattr(library, "feature_representation", "endpoint_pairwise")
        )
        metadata = dict(getattr(library, "metadata", {}))
    else:
        features = np.asarray(library, dtype=float)
        labels = tuple()
        representation = "raw_endpoint_pairwise" if features.ndim == 4 else "raw_scalar"
        metadata = {}

    if features.ndim not in {3, 4}:
        raise ValueError(
            "library features must have shape (T,M,L) or (T,M,L,2)."
        )
    if features.shape[:2] != (n_samples, n_edges):
        raise ValueError(
            "library feature shape must begin with "
            f"({n_samples}, {n_edges}); got {features.shape}."
        )
    if features.ndim == 4 and features.shape[3] != 2:
        raise ValueError("Endpoint-output features require final dimension 2.")
    if features.shape[2] < 1:
        raise ValueError("The local interaction library must contain at least one atom.")
    if not np.all(np.isfinite(features)):
        raise ValueError("Local interaction library contains non-finite values.")

    L = int(features.shape[2])
    if labels and len(labels) != L:
        raise ValueError("feature_labels must contain one label per library atom.")
    if not labels:
        labels = tuple(f"psi_{ell + 1}" for ell in range(L))

    return features, labels, representation, metadata


def _validate_stage_labels(stage_of_sample: Sequence[int], n_samples: int) -> IntArray:
    stage = np.asarray(stage_of_sample, dtype=np.int64).reshape(-1)
    if stage.size != n_samples:
        raise ValueError("stage_of_sample must contain one label per observation.")
    if np.any(stage < 0):
        raise ValueError("stage_of_sample must contain non-negative labels.")

    unique = np.unique(stage)
    if unique.size == 0 or unique[0] != 0:
        raise ValueError("stage_of_sample must start at stage 0.")
    expected = np.arange(int(unique[-1]) + 1, dtype=np.int64)
    if not np.array_equal(unique, expected):
        raise ValueError("stage labels must be contiguous 0,1,...,K-1.")
    return stage


# -----------------------------------------------------------------------------
# Stage-wise design construction
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class StageEdgeSpaceDesign:
    """Linear observation design for one temporal stage."""

    stage_index: int
    sample_indices: IntArray
    y: FloatArray
    matrix: object
    column_scales: FloatArray

    n_stage_samples: int
    n_nodes: int
    n_edges: int
    n_library_atoms: int
    n_parameters: int
    n_scalar_observations: int

    feature_representation: str
    design_mode: str

    # Compact local data retained for Step 3; does not change Step-2 solves.
    endpoint_features: Optional[FloatArray] = None
    edge_endpoints: Optional[IntArray] = None


def _stage_column_scales(
    D: FloatArray,
    features: FloatArray,
    ids: IntArray,
) -> FloatArray:
    """Exact RMS scales for one stage in canonical B coordinates."""

    T_r = int(ids.size)
    N = int(D.shape[0])
    n_scalar = T_r * N

    if features.ndim == 4:
        ss = np.sum(features[ids] ** 2, axis=(0, 3))  # (M,L)
    else:
        dnorm2 = np.sum(D * D, axis=0)  # (M,)
        ss = dnorm2[:, None] * np.sum(features[ids] ** 2, axis=0)

    scales = np.sqrt(np.maximum(ss.ravel() / max(n_scalar, 1), 0.0))
    return np.where(np.isfinite(scales) & (scales > 1.0e-14), scales, 1.0)


def _build_dense_stage_matrix(
    D: FloatArray,
    features: FloatArray,
    ids: IntArray,
) -> FloatArray:
    """Explicit A_r mapping vec(B^(r)) to vec(Y_r)."""

    T_r = int(ids.size)
    N, M = D.shape
    L = int(features.shape[2])

    if features.ndim == 4:
        tail, head = _pair_endpoints_from_incidence(D)
        block = np.zeros((T_r, N, M, L), dtype=float)
        local = features[ids]
        for m in range(M):
            block[:, tail[m], m, :] = local[:, m, :, 0]
            block[:, head[m], m, :] = local[:, m, :, 1]
        return block.reshape(T_r * N, M * L)

    # Legacy/scalar oriented-edge response.  Each scalar edge response is
    # distributed to node space through the oriented incidence column D[:,m].
    return np.einsum(
        "tml,nm->tnml",
        features[ids],
        D,
        optimize=True,
    ).reshape(T_r * N, M * L)


def _build_stage_operator(
    Y: FloatArray,
    D: FloatArray,
    features: FloatArray,
    ids: IntArray,
):
    if LinearOperator is None:
        raise ImportError("scipy is required for Step-2 operator / LSQR mode.")

    T_r = int(ids.size)
    N, M = D.shape
    L = int(features.shape[2])
    p = M * L

    if features.ndim == 4:
        tail, head = _pair_endpoints_from_incidence(D)
        S_tail = np.zeros((N, M), dtype=float)
        S_head = np.zeros((N, M), dtype=float)
        S_tail[tail, np.arange(M)] = 1.0
        S_head[head, np.arange(M)] = 1.0
        local = features[ids]

        def matvec(beta):
            B = np.asarray(beta, dtype=float).reshape(M, L)
            tail_response = np.einsum(
                "ml,tml->tm", B, local[:, :, :, 0], optimize=True
            )
            head_response = np.einsum(
                "ml,tml->tm", B, local[:, :, :, 1], optimize=True
            )
            pred = tail_response @ S_tail.T + head_response @ S_head.T
            return pred.ravel()

        def rmatvec(v):
            R = np.asarray(v, dtype=float).reshape(T_r, N)
            rt = R[:, tail]
            rh = R[:, head]
            G = (
                np.einsum("tm,tml->ml", rt, local[:, :, :, 0], optimize=True)
                + np.einsum("tm,tml->ml", rh, local[:, :, :, 1], optimize=True)
            )
            return G.ravel()

    else:
        local = features[ids]

        def matvec(beta):
            B = np.asarray(beta, dtype=float).reshape(M, L)
            edge_response = np.einsum("ml,tml->tm", B, local, optimize=True)
            return (edge_response @ D.T).ravel()

        def rmatvec(v):
            R = np.asarray(v, dtype=float).reshape(T_r, N)
            edge_residual = R @ D
            G = np.einsum("tm,tml->ml", edge_residual, local, optimize=True)
            return G.ravel()

    return LinearOperator(
        shape=(T_r * N, p),
        matvec=matvec,
        rmatvec=rmatvec,
        dtype=float,
    )


def _resolve_stage_design_mode(
    *,
    n_entries: int,
    design_mode: str,
    solver_method: str,
    direct_max_entries: int,
) -> str:
    design_mode = str(design_mode).lower()
    solver_method = str(solver_method).lower()

    if design_mode not in {"auto", "dense", "operator"}:
        raise ValueError("design_mode must be 'auto', 'dense', or 'operator'.")
    if solver_method not in {"auto", "dense_lstsq", "lsqr"}:
        raise ValueError("solver_method must be 'auto', 'dense_lstsq', or 'lsqr'.")

    if solver_method == "dense_lstsq":
        return "dense"
    if design_mode == "dense":
        return "dense"
    if design_mode == "operator":
        return "operator"
    if solver_method == "lsqr":
        return "operator"
    return "dense" if n_entries <= int(direct_max_entries) else "operator"


def build_edge_space_stage_designs(
    Y: ArrayLike,
    D: ArrayLike,
    library: Any,
    stage_of_sample: Sequence[int],
    *,
    design_mode: str = "auto",
    solver_method: str = "auto",
    direct_max_entries: int = 20_000_000,
) -> tuple[tuple[StageEdgeSpaceDesign, ...], FloatArray, tuple[str, ...], str, Mapping[str, Any], IntArray]:
    """Build the absolute stage-wise Step-2 designs.

    The parameterization is directly ``B^(1),...,B^(K)``.  No baseline-plus-
    change coordinates and no support restriction are used.
    """

    Y_arr = np.asarray(Y, dtype=float)
    D_arr = np.asarray(D, dtype=float)
    if Y_arr.ndim != 2:
        raise ValueError("Y must have shape (n_observations, n_nodes).")
    if D_arr.ndim != 2:
        raise ValueError("D must have shape (n_nodes, n_candidate_edges).")
    if D_arr.shape[0] != Y_arr.shape[1]:
        raise ValueError("D.shape[0] must equal Y.shape[1].")
    if not np.all(np.isfinite(Y_arr)) or not np.all(np.isfinite(D_arr)):
        raise ValueError("Y and D must contain only finite values.")

    T, N = Y_arr.shape
    M = int(D_arr.shape[1])
    if T < 1 or N < 1 or M < 1:
        raise ValueError("Y and D must be non-empty.")

    stage = _validate_stage_labels(stage_of_sample, T)
    features, labels, representation, library_metadata = _coerce_library(
        library,
        n_samples=T,
        n_edges=M,
    )
    if features.ndim == 4:
        _validate_standard_pairwise_incidence(D_arr)

    L = int(features.shape[2])
    K = int(np.max(stage)) + 1
    designs: list[StageEdgeSpaceDesign] = []

    for r in range(K):
        ids = np.flatnonzero(stage == r).astype(np.int64)
        if ids.size == 0:
            raise ValueError(f"Stage {r} has no observations.")

        n_scalar = int(ids.size) * N
        p = M * L
        resolved_mode = _resolve_stage_design_mode(
            n_entries=n_scalar * p,
            design_mode=design_mode,
            solver_method=solver_method,
            direct_max_entries=direct_max_entries,
        )

        if resolved_mode == "dense":
            matrix = _build_dense_stage_matrix(D_arr, features, ids)
        else:
            matrix = _build_stage_operator(Y_arr, D_arr, features, ids)

        scales = _stage_column_scales(D_arr, features, ids)
        designs.append(
            StageEdgeSpaceDesign(
                stage_index=int(r),
                sample_indices=ids,
                y=Y_arr[ids].ravel(),
                matrix=matrix,
                column_scales=scales,
                n_stage_samples=int(ids.size),
                n_nodes=int(N),
                n_edges=int(M),
                n_library_atoms=int(L),
                n_parameters=int(p),
                n_scalar_observations=int(n_scalar),
                feature_representation=str(representation),
                design_mode=str(resolved_mode),
                endpoint_features=features[ids] if features.ndim == 4 else None,
                edge_endpoints=np.column_stack(_pair_endpoints_from_incidence(D_arr)) if features.ndim == 4 else None,
            )
        )

    return (
        tuple(designs),
        features,
        labels,
        representation,
        library_metadata,
        stage,
    )


# -----------------------------------------------------------------------------
# Stage geometry and the full observational family
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class StageSolutionGeometry:
    """Reference solution and identified/null geometry for one stage."""

    stage_index: int
    reference_b: FloatArray
    reference_B: FloatArray
    fitted_y: FloatArray
    residual: FloatArray
    residual_norm: float

    column_scales: FloatArray
    rank: Optional[int]
    nullity: Optional[int]
    singular_values_scaled: Optional[FloatArray]
    identified_basis_scaled: Optional[FloatArray]  # rows of V_q^T, shape (q,p)
    rank_cutoff_scaled: Optional[float]

    solver_method: str
    design_mode: str


@dataclass(frozen=True)
class EdgeSpaceFieldFamily:
    """Step-2 output: the full uncertainty-floor feasible B_(1:K) family."""

    designs: tuple[StageEdgeSpaceDesign, ...]
    stage_geometry: tuple[StageSolutionGeometry, ...]
    stage_of_sample: IntArray

    uncertainty_floor: float
    target_norm: float
    residual_threshold: float
    minimum_residual_norm: float
    minimum_relative_residual: float
    slack_squared: float

    n_nodes: int
    n_edges: int
    n_library_atoms: int
    n_stages: int
    parameter_count: int

    feature_labels: tuple[str, ...]
    feature_representation: str
    library_metadata: Mapping[str, Any]
    metadata: Mapping[str, Any]

    @property
    def is_empty(self) -> bool:
        return bool(self.slack_squared < 0.0)

    @property
    def rank(self) -> Optional[int]:
        ranks = [g.rank for g in self.stage_geometry]
        return None if any(q is None for q in ranks) else int(sum(int(q) for q in ranks))

    @property
    def nullity(self) -> Optional[int]:
        nullities = [g.nullity for g in self.stage_geometry]
        return (
            None
            if any(q is None for q in nullities)
            else int(sum(int(q) for q in nullities))
        )

    @property
    def reference_B_stages(self) -> FloatArray:
        return np.stack([g.reference_B for g in self.stage_geometry], axis=0)

    @property
    def reference_delta_B(self) -> FloatArray:
        return np.diff(self.reference_B_stages, axis=0)

    def difference_view(self, B_stages: ArrayLike) -> FloatArray:
        B = self._coerce_B_stages(B_stages)
        return np.diff(B, axis=0)

    def _coerce_B_stages(self, B_stages: ArrayLike) -> FloatArray:
        B = np.asarray(B_stages, dtype=float)
        expected = (self.n_stages, self.n_edges, self.n_library_atoms)
        if B.shape != expected:
            raise ValueError(f"B_stages must have shape {expected}; got {B.shape}.")
        if not np.all(np.isfinite(B)):
            raise ValueError("B_stages must contain only finite values.")
        return B

    def residual_by_stage(self, B_stages: ArrayLike) -> tuple[FloatArray, ...]:
        B = self._coerce_B_stages(B_stages)
        out: list[FloatArray] = []
        for r, design in enumerate(self.designs):
            beta = B[r].ravel()
            pred = np.asarray(design.matrix @ beta, dtype=float).reshape(-1)
            out.append(np.asarray(design.y - pred, dtype=float))
        return tuple(out)

    def residual_norm(self, B_stages: ArrayLike) -> float:
        residuals = self.residual_by_stage(B_stages)
        return float(np.sqrt(sum(float(np.dot(r, r)) for r in residuals)))

    def relative_residual(self, B_stages: ArrayLike) -> float:
        denom = max(float(self.target_norm), np.finfo(float).tiny)
        return float(self.residual_norm(B_stages) / denom)

    def is_feasible(
        self,
        B_stages: ArrayLike,
        *,
        rtol: float = 1.0e-10,
        atol: float = 0.0,
    ) -> bool:
        rho = self.relative_residual(B_stages)
        return bool(rho <= self.uncertainty_floor * (1.0 + rtol) + atol)

    def identified_component(self, stage_index: int, delta_B: ArrayLike) -> FloatArray:
        """Project a coefficient perturbation onto the data-identified stage subspace.

        Projection is orthogonal in the internally column-scaled coordinates,
        then mapped back to canonical B coordinates.
        """

        r = int(stage_index)
        g = self.stage_geometry[r]
        Vq = g.identified_basis_scaled
        if Vq is None:
            raise RuntimeError("Identified projector requires dense SVD geometry.")
        d = np.asarray(delta_B, dtype=float).reshape(-1)
        if d.size != self.n_edges * self.n_library_atoms:
            raise ValueError("delta_B has incompatible size.")
        c = g.column_scales * d
        c_id = Vq.T @ (Vq @ c)
        return (c_id / g.column_scales).reshape(self.n_edges, self.n_library_atoms)

    def null_component(self, stage_index: int, delta_B: ArrayLike) -> FloatArray:
        """Project a perturbation onto the stage observational nullspace."""

        d = np.asarray(delta_B, dtype=float).reshape(
            self.n_edges, self.n_library_atoms
        )
        return d - self.identified_component(stage_index, d)

    def sample_feasible(
        self,
        *,
        rng: Optional[np.random.Generator] = None,
        null_scale: float = 1.0,
        boundary_fraction: float = 0.9,
    ) -> FloatArray:
        """Sample one feasible B_(1:K) member using dense stage geometries.

        ``null_scale`` controls the initial canonical coefficient norm of a random
        null perturbation relative to the reference-field norm.  The final
        perturbation is automatically shrunk when numerical near-null leakage
        would otherwise exceed the uncertainty floor.
        """

        if self.is_empty:
            raise RuntimeError("Cannot sample an empty observational family.")
        if not (0.0 <= boundary_fraction <= 1.0):
            raise ValueError("boundary_fraction must lie in [0,1].")
        if null_scale < 0.0 or not np.isfinite(null_scale):
            raise ValueError("null_scale must be finite and non-negative.")
        if any(g.identified_basis_scaled is None for g in self.stage_geometry):
            raise RuntimeError("Family sampling requires dense SVD geometry in every stage.")

        rng = np.random.default_rng() if rng is None else rng
        B0 = self.reference_B_stages
        perturb = np.zeros_like(B0)

        # Random null perturbation in canonical B coordinates.
        for r, g in enumerate(self.stage_geometry):
            if g.nullity is None or g.nullity <= 0 or null_scale == 0.0:
                continue
            raw = rng.normal(size=(self.n_edges, self.n_library_atoms))
            dn = self.null_component(r, raw)
            nrm = float(np.linalg.norm(dn))
            if nrm <= 1.0e-14:
                continue
            reference_scale = max(float(np.linalg.norm(B0[r])), 1.0)
            perturb[r] += dn * (null_scale * reference_scale / nrm)

        # Add an identified perturbation that uses the requested fraction of the
        # available residual budget in the ideal SVD geometry.
        q_total = sum(int(g.rank or 0) for g in self.stage_geometry)
        budget = np.sqrt(max(float(self.slack_squared), 0.0))
        target_response = float(boundary_fraction) * budget

        if q_total > 0 and target_response > 0.0:
            h = rng.normal(size=q_total)
            hnorm = float(np.linalg.norm(h))
            if hnorm > 0.0:
                h *= target_response / hnorm
                cursor = 0
                for r, g in enumerate(self.stage_geometry):
                    q = int(g.rank or 0)
                    if q == 0:
                        continue
                    hr = h[cursor : cursor + q]
                    cursor += q
                    s = np.asarray(g.singular_values_scaled[:q], dtype=float)
                    Vq = np.asarray(g.identified_basis_scaled, dtype=float)
                    dc = Vq.T @ (hr / s)
                    perturb[r] += (dc / g.column_scales).reshape(
                        self.n_edges, self.n_library_atoms
                    )

        # Numerical rank thresholds can turn extremely small singular directions
        # into approximate rather than exact nulls.  Enforce feasibility using the
        # actual observation operator, shrinking only the perturbation if needed.
        candidate = B0 + perturb
        rho = self.relative_residual(candidate)
        if rho <= self.uncertainty_floor * (1.0 + 1.0e-10):
            return candidate

        response_sq = max(
            self.residual_norm(candidate) ** 2 - self.minimum_residual_norm ** 2,
            0.0,
        )
        if response_sq <= 0.0:
            return B0.copy()
        max_response = np.sqrt(max(float(self.slack_squared), 0.0))
        shrink = min(1.0, max_response / np.sqrt(response_sq))
        shrink *= 1.0 - 1.0e-10
        candidate = B0 + shrink * perturb
        if not self.is_feasible(candidate, rtol=1.0e-8):
            # Roundoff-safe fallback.
            return B0.copy()
        return candidate


# -----------------------------------------------------------------------------
# Stage solvers
# -----------------------------------------------------------------------------


def _dense_stage_geometry(
    design: StageEdgeSpaceDesign,
    *,
    rcond: Optional[float],
) -> StageSolutionGeometry:
    A = np.asarray(design.matrix, dtype=float)
    y = np.asarray(design.y, dtype=float)
    scales = np.asarray(design.column_scales, dtype=float)

    As = A / scales[None, :]
    U, s, Vh = np.linalg.svd(As, full_matrices=False)

    if s.size == 0:
        rank = 0
        cutoff = 0.0
    else:
        cutoff = (
            np.finfo(float).eps * max(As.shape) * float(s[0])
            if rcond is None
            else float(rcond) * float(s[0])
        )
        rank = int(np.count_nonzero(s > cutoff))

    if rank > 0:
        Uq = U[:, :rank]
        sq = s[:rank]
        Vq = Vh[:rank, :]
        c0 = Vq.T @ ((Uq.T @ y) / sq)
    else:
        Vq = np.zeros((0, As.shape[1]), dtype=float)
        c0 = np.zeros(As.shape[1], dtype=float)

    b0 = c0 / scales
    fitted = A @ b0
    residual = y - fitted

    M = design.n_edges
    L = design.n_library_atoms
    return StageSolutionGeometry(
        stage_index=int(design.stage_index),
        reference_b=np.asarray(b0, dtype=float),
        reference_B=np.asarray(b0, dtype=float).reshape(M, L),
        fitted_y=np.asarray(fitted, dtype=float),
        residual=np.asarray(residual, dtype=float),
        residual_norm=float(np.linalg.norm(residual)),
        column_scales=scales.copy(),
        rank=int(rank),
        nullity=int(design.n_parameters - rank),
        singular_values_scaled=np.asarray(s, dtype=float),
        identified_basis_scaled=np.asarray(Vq, dtype=float),
        rank_cutoff_scaled=float(cutoff),
        solver_method="dense_svd",
        design_mode="dense",
    )


def _operator_stage_geometry(
    design: StageEdgeSpaceDesign,
    *,
    solver_kwargs: Mapping[str, Any],
) -> StageSolutionGeometry:
    forbidden = {"method", "column_scales"} & set(solver_kwargs)
    if forbidden:
        raise ValueError(
            "Operator Step 2 controls method/column_scales internally; remove: "
            + ", ".join(sorted(forbidden))
        )

    result = solve_least_squares(
        design.matrix,
        design.y,
        method="lsqr",
        column_scaling=True,
        column_scales=design.column_scales,
        **dict(solver_kwargs),
    )
    b0 = np.asarray(result.coefficients, dtype=float).reshape(-1)
    fitted = np.asarray(result.fitted_values, dtype=float).reshape(-1)
    residual = np.asarray(result.residual, dtype=float).reshape(-1)

    return StageSolutionGeometry(
        stage_index=int(design.stage_index),
        reference_b=b0,
        reference_B=b0.reshape(design.n_edges, design.n_library_atoms),
        fitted_y=fitted,
        residual=residual,
        residual_norm=float(np.linalg.norm(residual)),
        column_scales=np.asarray(design.column_scales, dtype=float).copy(),
        rank=None,
        nullity=None,
        singular_values_scaled=None,
        identified_basis_scaled=None,
        rank_cutoff_scaled=None,
        solver_method="lsqr",
        design_mode="operator",
    )


# -----------------------------------------------------------------------------
# Public Step-2 inference
# -----------------------------------------------------------------------------


def reconstruct_edge_space_family_from_observations(
    Y: ArrayLike,
    D: ArrayLike,
    library: Any,
    stage_of_sample: Sequence[int],
    *,
    uncertainty_floor: float,
    design_mode: str = "auto",
    solver_method: str = "auto",
    direct_max_entries: int = 20_000_000,
    rcond: Optional[float] = None,
    solver_kwargs: Optional[Mapping[str, Any]] = None,
) -> EdgeSpaceFieldFamily:
    """Reconstruct the complete observational edge-space field family.

    Parameters
    ----------
    Y
        Preprocessed node-space vector-field observations, shape ``(T,N)``.
        Known node-local/intrinsic terms should already be removed if they are
        outside the chosen pairwise edge-space library.
    D
        Oriented candidate-edge incidence matrix, shape ``(N,M)``.
    library
        Preferred: ``PairwiseLocalInteractionLibrary`` evaluated on the same T
        observation states.  Raw arrays of shape ``(T,M,L,2)`` (endpoint-output)
        or ``(T,M,L)`` (scalar oriented-edge) are also accepted.
    stage_of_sample
        Contiguous zero-based stage label for every observation.
    uncertainty_floor
        Relative observational tolerance epsilon defining ``B_epsilon``.
    design_mode
        ``'auto'``, ``'dense'``, or ``'operator'``.  Resolution is performed per
        stage because the absolute-B design is block diagonal in time.
    solver_method
        ``'auto'``, ``'dense_lstsq'``, or ``'lsqr'``.  In dense mode Step 2 uses
        an SVD directly so that the identified/null geometry is retained.
    direct_max_entries
        Per-stage dense/operator switch threshold in auto mode.
    rcond
        Relative SVD rank cutoff for dense stages.  ``None`` uses the standard
        machine-precision cutoff ``eps * max(shape) * s_max``.
    solver_kwargs
        Extra LSQR controls for operator stages.

    Returns
    -------
    EdgeSpaceFieldFamily
        The Step-2 family object.  ``reference_B_stages`` is only one convenient
        minimum-residual representative; the inference result is the family
        defined by ``uncertainty_floor`` and the stored observation operators.
    """

    epsilon = float(uncertainty_floor)
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("uncertainty_floor must be finite and strictly positive.")

    (
        designs,
        features,
        feature_labels,
        representation,
        library_metadata,
        stage,
    ) = build_edge_space_stage_designs(
        Y,
        D,
        library,
        stage_of_sample,
        design_mode=design_mode,
        solver_method=solver_method,
        direct_max_entries=direct_max_entries,
    )

    solver_method = str(solver_method).lower()
    kwargs = {} if solver_kwargs is None else dict(solver_kwargs)
    geometries: list[StageSolutionGeometry] = []

    for design in designs:
        if design.design_mode == "dense":
            if solver_method == "lsqr":
                # Caller explicitly asked for LSQR even though the dense matrix is
                # available.  Keep the requested numerical backend; geometry is
                # consequently implicit.
                op_design = StageEdgeSpaceDesign(
                    **{**design.__dict__, "design_mode": "operator"}
                )
                geometries.append(
                    _operator_stage_geometry(op_design, solver_kwargs=kwargs)
                )
            else:
                geometries.append(_dense_stage_geometry(design, rcond=rcond))
        else:
            geometries.append(
                _operator_stage_geometry(design, solver_kwargs=kwargs)
            )

    Y_arr = np.asarray(Y, dtype=float)
    target_norm = float(np.linalg.norm(Y_arr.ravel()))
    minimum_residual_sq = float(
        sum(g.residual_norm * g.residual_norm for g in geometries)
    )
    minimum_residual_norm = float(np.sqrt(max(minimum_residual_sq, 0.0)))
    denom = max(target_norm, np.finfo(float).tiny)
    minimum_relative = float(minimum_residual_norm / denom)
    threshold = float(epsilon * target_norm)
    slack_sq = float(threshold * threshold - minimum_residual_sq)

    M = designs[0].n_edges
    L = designs[0].n_library_atoms
    N = designs[0].n_nodes
    K = len(designs)

    rank_values = [g.rank for g in geometries]
    null_values = [g.nullity for g in geometries]
    total_rank = None if any(q is None for q in rank_values) else int(sum(rank_values))
    total_nullity = (
        None if any(q is None for q in null_values) else int(sum(null_values))
    )

    metadata = {
        "backend": "absolute-stage-edge-space-solution-family",
        "canonical_parameterization": "B^(1),...,B^(K)",
        "feature_representation": str(representation),
        "n_preprocessed_observations": int(Y_arr.shape[0]),
        "n_scalar_observations": int(Y_arr.size),
        "n_nodes": int(N),
        "n_candidate_edges": int(M),
        "n_library_atoms": int(L),
        "n_stages": int(K),
        "parameter_count": int(K * M * L),
        "rank": total_rank,
        "nullity": total_nullity,
        "minimum_relative_residual": minimum_relative,
        "uncertainty_floor": epsilon,
        "family_empty": bool(slack_sq < 0.0),
        "stage_design_modes": tuple(d.design_mode for d in designs),
        "stage_solver_methods": tuple(g.solver_method for g in geometries),
        "stage_ranks": tuple(g.rank for g in geometries),
        "stage_nullities": tuple(g.nullity for g in geometries),
        "stage_sample_counts": tuple(d.n_stage_samples for d in designs),
    }

    return EdgeSpaceFieldFamily(
        designs=tuple(designs),
        stage_geometry=tuple(geometries),
        stage_of_sample=stage.copy(),
        uncertainty_floor=epsilon,
        target_norm=target_norm,
        residual_threshold=threshold,
        minimum_residual_norm=minimum_residual_norm,
        minimum_relative_residual=minimum_relative,
        slack_squared=slack_sq,
        n_nodes=int(N),
        n_edges=int(M),
        n_library_atoms=int(L),
        n_stages=int(K),
        parameter_count=int(K * M * L),
        feature_labels=tuple(feature_labels),
        feature_representation=str(representation),
        library_metadata=dict(library_metadata),
        metadata=metadata,
    )


# Short public alias for the three-layer pipeline.
reconstruct_edge_space_family = reconstruct_edge_space_family_from_observations


__all__ = [
    "StageEdgeSpaceDesign",
    "StageSolutionGeometry",
    "EdgeSpaceFieldFamily",
    "build_edge_space_stage_designs",
    "reconstruct_edge_space_family_from_observations",
    "reconstruct_edge_space_family",
]
