"""TIDES Step 3: reconstruct stage-wise edge-space vector fields.

Step 3 receives the change supports selected by Step 2 and estimates the
corresponding *unpenalized* vector-field coefficients in the original,
unprojected observation model.

For stage r,

    B^(r) = B^(1) + sum_{k<r} Delta B^(k),

where ``B^(1)`` is unrestricted over all candidate edge rows and only the rows
admitted by Step 2 are free in each ``Delta B^(k)`` block.

For preprocessed observations ``Y`` and edge-feature responses ``Psi``, Step 3
solves one global linear regression.  ``Psi`` may be the historical scalar edge
feature tensor ``(T,M,L)`` or the generalized endpoint-output tensor
``(T,M,L,2)``; in both cases the unknown coefficient tensor remains ``(M,L)``.

    vec(Y) = A_selected beta + residual,

with no sparsity penalty.  Numerical least-squares machinery lives in
``solvers_linear_regression.py``; this module owns the TIDES-specific design,
parameter unpacking, and reconstruction of ``B^(r)``.

Important separation from Step 2
--------------------------------
Step-2 projected coefficients are never reused as final amplitudes.  Step 3
reconstructs ``B^(1)`` and every selected ``Delta B`` jointly from the full
unprojected model.

Current formal entry point
--------------------------
``reconstruct_vector_field_from_observations`` accepts the same preprocessed
``Y``, incidence/aggregation matrix ``D``, edge features, and stage labels used
by Step 2, together with either the Step-2 result itself, its constraints, or
plain support arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

try:  # package form
    from .solvers_linear_regression import (
        LinearRegressionResult,
        solve_least_squares,
    )
except ImportError:  # flat-file form
    from solvers_linear_regression import (
        LinearRegressionResult,
        solve_least_squares,
    )

try:
    from scipy.sparse.linalg import LinearOperator
except Exception:  # pragma: no cover - scipy only needed for large/operator mode
    LinearOperator = None


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

# Compatibility type alias retained so the current tides.py can still import
# this module while the end-to-end wrapper is migrated to the observation-based API.
LibraryFunction = Callable[[FloatArray], FloatArray]


def _resolve_feature_output_modes(
    n_features: int,
    feature_output_modes: Optional[Sequence[str]],
) -> tuple[str, ...]:
    """Resolve feature-wise assembly modes with legacy compatibility."""

    L = int(n_features)
    if feature_output_modes is None:
        return tuple("legacy_difference" for _ in range(L))
    modes = tuple(str(x).lower() for x in feature_output_modes)
    if len(modes) != L:
        raise ValueError(
            "feature_output_modes must contain exactly one entry per edge feature."
        )
    bad = sorted(set(modes) - {"common", "difference", "legacy_difference"})
    if bad:
        raise ValueError(
            "feature_output_modes entries must be 'common' or 'difference' "
            f"(legacy internal mode also accepted); got {bad!r}."
        )
    return modes


def _validate_standard_pairwise_incidence(D: FloatArray) -> None:
    D_arr = np.asarray(D, dtype=float)
    nz = np.abs(D_arr) > 1.0e-12
    if D_arr.ndim != 2 or not np.all(np.sum(nz, axis=0) == 2):
        raise ValueError(
            "General common/difference output modes require two nonzero incidence "
            "entries per candidate edge."
        )
    for m in range(D_arr.shape[1]):
        vals = np.sort(D_arr[nz[:, m], m])
        if not np.allclose(vals, np.array([-1.0, 1.0]), atol=1.0e-12, rtol=0.0):
            raise ValueError(
                "General common/difference output modes require standard oriented "
                "incidence columns with values {-1,+1}."
            )


def _feature_output_directions(
    D: FloatArray,
    modes: Sequence[str],
) -> FloatArray:
    """Node-space output directions, shape ``(N,M,L)``."""

    D_arr = np.asarray(D, dtype=float)
    modes = tuple(modes)
    if any(mode != "legacy_difference" for mode in modes):
        _validate_standard_pairwise_incidence(D_arr)
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    out = np.empty((D_arr.shape[0], D_arr.shape[1], len(modes)), dtype=float)
    for ell, mode in enumerate(modes):
        if mode == "legacy_difference":
            out[:, :, ell] = D_arr
        elif mode == "difference":
            out[:, :, ell] = inv_sqrt2 * D_arr
        elif mode == "common":
            out[:, :, ell] = inv_sqrt2 * np.abs(D_arr)
        else:
            raise RuntimeError(f"Unknown feature output mode {mode!r}.")
    return out



def _pair_endpoints_from_incidence(D: FloatArray) -> tuple[IntArray, IntArray]:
    D_arr = np.asarray(D, dtype=float)
    _validate_standard_pairwise_incidence(D_arr)
    M = D_arr.shape[1]
    tail = np.empty(M, dtype=np.int64)
    head = np.empty(M, dtype=np.int64)
    for m in range(M):
        tail[m] = int(np.flatnonzero(D_arr[:, m] < -0.5)[0])
        head[m] = int(np.flatnonzero(D_arr[:, m] > 0.5)[0])
    return tail, head


def _modes_from_support_source(
    source,
    n_features: int,
    explicit: Optional[Sequence[str]],
) -> tuple[str, ...]:
    """Use explicit modes, else inherit Step-2 metadata, else legacy output."""

    if explicit is not None:
        return _resolve_feature_output_modes(n_features, explicit)
    metadata = getattr(source, "metadata", None)
    if isinstance(metadata, dict) and metadata.get("feature_output_modes") is not None:
        return _resolve_feature_output_modes(
            n_features, metadata["feature_output_modes"]
        )
    return _resolve_feature_output_modes(n_features, None)


@dataclass(frozen=True)
class VectorFieldReconstructionDesign:
    """Selected-support global Step-3 regression design."""

    y: FloatArray
    matrix: object
    column_scales: FloatArray

    supports: tuple[IntArray, ...]
    anchor_slice: slice
    delta_slices: tuple[slice, ...]
    nuisance_slice: slice

    n_samples: int
    n_nodes: int
    n_edges: int
    n_edge_features: int
    feature_output_modes: tuple[str, ...]
    feature_representation: str
    n_stages: int
    n_parameters: int
    n_scalar_observations: int
    n_nuisance_parameters: int
    design_mode: str


@dataclass(frozen=True)
class VectorFieldReconstructionResult:
    """Output of TIDES Step 3."""

    B_anchor: FloatArray
    delta_B: tuple[FloatArray, ...]
    B_stages: FloatArray

    fitted_vector_field: FloatArray
    residual_vector_field: FloatArray
    residual_norm: float
    relative_residual: float

    parameter_count: int
    n_observations: int
    n_preprocessed_observations: int
    stage_of_sample: IntArray

    solver_result: LinearRegressionResult
    stationary_nuisance_coefficients: Optional[FloatArray]
    metadata: dict
    design: Optional[VectorFieldReconstructionDesign] = None

    @property
    def rank(self) -> Optional[int]:
        return self.solver_result.rank

    @property
    def identifiable(self) -> Optional[bool]:
        return self.solver_result.identifiable

    @property
    def condition_number_scaled(self) -> float:
        return self.solver_result.condition_number_scaled

    @property
    def condition_number_raw(self) -> float:
        return self.solver_result.condition_number_raw

    @property
    def lsqr_info(self) -> dict:
        """Compatibility diagnostic view for earlier notebook code."""
        return {
            "solver": self.solver_result.method,
            "rank": self.solver_result.rank,
            "n_columns": self.solver_result.n_features,
            "condition_number_scaled": self.solver_result.condition_number_scaled,
            "condition_number_raw": self.solver_result.condition_number_raw,
            "converged": self.solver_result.converged,
            "iterations": self.solver_result.iterations,
            "solver_status": self.solver_result.solver_status,
            "normal_equation_relative_residual": (
                self.solver_result.normal_equation_relative_residual
            ),
        }


# -----------------------------------------------------------------------------
# Input / support handling
# -----------------------------------------------------------------------------


def _extract_supports(change_constraints_or_supports) -> tuple[IntArray, ...]:
    """Accept a Step-2 result, constraints, or plain support arrays."""

    source = change_constraints_or_supports

    if hasattr(source, "constraints"):
        source = source.constraints
    elif hasattr(source, "supports") and not isinstance(source, (list, tuple)):
        source = source.supports

    supports: list[IntArray] = []
    for item in source:
        if hasattr(item, "support"):
            item = item.support

        s = np.asarray(item, dtype=np.int64).reshape(-1)
        if s.size:
            if np.unique(s).size != s.size:
                raise ValueError("A Step-3 change support contains duplicate edge indices.")
            s = np.sort(s)
        supports.append(s)

    return tuple(supports)


def _validate_observations(
    Y: ArrayLike,
    D: ArrayLike,
    edge_features: ArrayLike,
    stage_of_sample: Sequence[int],
    change_constraints_or_supports,
):
    Y = np.asarray(Y, dtype=float)
    D = np.asarray(D, dtype=float)
    Psi = np.asarray(edge_features, dtype=float)
    stage = np.asarray(stage_of_sample, dtype=np.int64).reshape(-1)
    supports = _extract_supports(change_constraints_or_supports)

    if Y.ndim != 2:
        raise ValueError("Y must have shape (n_observations, n_nodes).")
    if D.ndim != 2:
        raise ValueError("D must have shape (n_nodes, n_candidate_edges).")
    if Psi.ndim not in {3,4}:
        raise ValueError(
            "edge_features must have shape (T,M,L) or generalized endpoint "
            "shape (T,M,L,2)."
        )
    if Psi.ndim == 4 and Psi.shape[3] != 2:
        raise ValueError("endpoint-output features require final dimension 2.")

    T, N = Y.shape
    if D.shape[0] != N:
        raise ValueError("D.shape[0] must equal Y.shape[1].")
    M = D.shape[1]
    if Psi.shape[:2] != (T, M):
        raise ValueError(
            "edge_features.shape[:2] must equal (Y.shape[0], D.shape[1])."
        )
    L = Psi.shape[2]
    if T == 0 or N == 0 or M == 0 or L == 0:
        raise ValueError("Y, D, and edge_features must be non-empty.")
    if Psi.ndim == 4:
        _validate_standard_pairwise_incidence(D)

    if stage.size != T:
        raise ValueError("stage_of_sample must have one entry per observation.")
    if np.any(stage < 0):
        raise ValueError("stage_of_sample must contain non-negative labels.")
    unique = np.unique(stage)
    if unique.size == 0 or unique[0] != 0:
        raise ValueError("stage_of_sample must start at stage 0.")
    expected = np.arange(int(unique[-1]) + 1, dtype=np.int64)
    if not np.array_equal(unique, expected):
        raise ValueError("stage labels must be contiguous 0,1,...,R-1.")

    R = int(unique[-1]) + 1
    if len(supports) != R - 1:
        raise ValueError(
            "There must be exactly one change support per detected transition."
        )
    for supp in supports:
        if supp.size and (int(np.min(supp)) < 0 or int(np.max(supp)) >= M):
            raise ValueError("A Step-3 change support contains an invalid edge index.")
    for name, arr in (("Y",Y),("D",D),("edge_features",Psi)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must be finite.")
    return Y, D, Psi, stage, supports


def _coerce_stationary_nuisance(
    stationary_nuisance: Optional[ArrayLike],
    *,
    n_samples: int,
    n_nodes: int,
):
    if stationary_nuisance is None:
        return None

    A = np.asarray(stationary_nuisance, dtype=float)

    if A.ndim == 3 and A.shape[:2] == (n_samples, n_nodes):
        out = A
    elif A.ndim == 2 and A.shape[0] == n_samples * n_nodes:
        out = A.reshape(n_samples, n_nodes, A.shape[1])
    else:
        raise ValueError(
            "stationary_nuisance must have shape (T,N,Q) or (T*N,Q)."
        )

    if not np.all(np.isfinite(out)):
        raise ValueError("stationary_nuisance must be finite.")

    return out


# -----------------------------------------------------------------------------
# Parameterization / design
# -----------------------------------------------------------------------------


def _parameter_layout(
    *,
    n_edges: int,
    n_edge_features: int,
    supports: tuple[IntArray, ...],
    n_nuisance: int,
):
    M = int(n_edges)
    L = int(n_edge_features)

    anchor_slice = slice(0, M * L)
    p = M * L

    delta_slices: list[slice] = []
    for support in supports:
        q = p + int(support.size) * L
        delta_slices.append(slice(p, q))
        p = q

    nuisance_slice = slice(p, p + int(n_nuisance))
    p += int(n_nuisance)

    return anchor_slice, tuple(delta_slices), nuisance_slice, int(p)


def _unpack_coefficients(
    beta: FloatArray,
    *,
    n_edges: int,
    n_edge_features: int,
    supports: tuple[IntArray, ...],
    anchor_slice: slice,
    delta_slices: tuple[slice, ...],
    nuisance_slice: slice,
    n_nuisance: int,
):
    beta = np.asarray(beta, dtype=float).reshape(-1)
    M = int(n_edges)
    L = int(n_edge_features)

    B0 = beta[anchor_slice].reshape(M, L)

    deltas: list[FloatArray] = []
    for support, sl in zip(supports, delta_slices):
        dB = np.zeros((M, L), dtype=float)
        if support.size:
            dB[support, :] = beta[sl].reshape(support.size, L)
        deltas.append(dB)

    gamma = (
        beta[nuisance_slice].copy()
        if n_nuisance > 0
        else None
    )

    return B0, tuple(deltas), gamma


def _stage_matrices(B0: FloatArray, deltas: Sequence[FloatArray]) -> FloatArray:
    R = len(deltas) + 1
    out = np.empty((R,) + B0.shape, dtype=float)
    out[0] = B0

    for r in range(1, R):
        out[r] = out[r - 1] + deltas[r - 1]

    return out


def _analytic_column_scales(
    D: FloatArray,
    Psi: FloatArray,
    stage: IntArray,
    supports: tuple[IntArray, ...],
    nuisance: Optional[FloatArray],
    feature_output_modes: Sequence[str],
    *,
    anchor_slice: slice,
    delta_slices: tuple[slice, ...],
    nuisance_slice: slice,
    n_parameters: int,
) -> FloatArray:
    """Exact RMS scales of the selected generalized design columns."""

    T, M, L = Psi.shape[:3]
    N = D.shape[0]
    n_scalar = T * N
    scales = np.ones(n_parameters, dtype=float)

    if Psi.ndim == 4:
        anchor_ss = np.sum(Psi * Psi, axis=(0, 3))  # (M,L)
    else:
        directions = _feature_output_directions(D, feature_output_modes)
        direction_norm2 = np.sum(directions * directions, axis=0)
        anchor_ss = direction_norm2 * np.sum(Psi * Psi, axis=0)
    scales[anchor_slice] = np.sqrt(
        np.maximum(anchor_ss.ravel() / n_scalar, 0.0)
    )

    for k, (support, sl) in enumerate(zip(supports, delta_slices)):
        if not support.size:
            continue
        active = stage >= (k + 1)
        if Psi.ndim == 4:
            ss = np.sum(Psi[active][:, support, :, :] ** 2, axis=(0, 3))
        else:
            directions = _feature_output_directions(D, feature_output_modes)
            direction_norm2 = np.sum(directions * directions, axis=0)
            ss = (
                direction_norm2[support, :]
                * np.sum(Psi[active][:, support, :] ** 2, axis=0)
            )
        scales[sl] = np.sqrt(np.maximum(ss.ravel() / n_scalar, 0.0))

    if nuisance is not None:
        ss = np.sum(nuisance * nuisance, axis=(0, 1))
        scales[nuisance_slice] = np.sqrt(np.maximum(ss / n_scalar, 0.0))

    scales = np.where(
        np.isfinite(scales) & (scales > 1e-14),
        scales,
        1.0,
    )
    return scales


def _build_dense_matrix(
    D: FloatArray,
    Psi: FloatArray,
    stage: IntArray,
    supports: tuple[IntArray, ...],
    nuisance: Optional[FloatArray],
    feature_output_modes: Sequence[str],
) -> FloatArray:
    """Explicit selected-support generalized design; correctness path."""

    T, M, L = Psi.shape[:3]
    N = D.shape[0]
    n_scalar = T * N

    if Psi.ndim == 4:
        tail, head = _pair_endpoints_from_incidence(D)
        anchor4 = np.zeros((T, N, M, L), dtype=float)
        for m in range(M):
            anchor4[:, tail[m], m, :] = Psi[:, m, :, 0]
            anchor4[:, head[m], m, :] = Psi[:, m, :, 1]
        anchor = anchor4.reshape(n_scalar, M * L)
    else:
        directions = _feature_output_directions(D, feature_output_modes)
        anchor = np.einsum(
            "tml,nml->tnml",
            Psi,
            directions,
            optimize=True,
        ).reshape(n_scalar, M * L)

    blocks = [anchor]
    for k, support in enumerate(supports):
        if not support.size:
            continue
        active = (stage >= (k + 1)).astype(float)
        if Psi.ndim == 4:
            tail, head = _pair_endpoints_from_incidence(D)
            block4 = np.zeros((T, N, support.size, L), dtype=float)
            for q, m in enumerate(support):
                block4[:, tail[m], q, :] = (
                    active[:, None] * Psi[:, m, :, 0]
                )
                block4[:, head[m], q, :] = (
                    active[:, None] * Psi[:, m, :, 1]
                )
            block = block4.reshape(n_scalar, support.size * L)
        else:
            directions = _feature_output_directions(D, feature_output_modes)
            block = np.einsum(
                "t,tsl,nsl->tnsl",
                active,
                Psi[:, support, :],
                directions[:, support, :],
                optimize=True,
            ).reshape(n_scalar, support.size * L)
        blocks.append(block)

    if nuisance is not None:
        blocks.append(nuisance.reshape(n_scalar, nuisance.shape[2]))
    return np.column_stack(blocks)


def _build_linear_operator(
    Y: FloatArray,
    D: FloatArray,
    Psi: FloatArray,
    stage: IntArray,
    supports: tuple[IntArray, ...],
    nuisance: Optional[FloatArray],
    feature_output_modes: Sequence[str],
    *,
    anchor_slice: slice,
    delta_slices: tuple[slice, ...],
    nuisance_slice: slice,
    n_parameters: int,
):
    if LinearOperator is None:
        raise ImportError(
            "scipy is required for Step-3 LinearOperator / LSQR mode."
        )

    T, N = Y.shape
    M = D.shape[1]
    L = Psi.shape[2]
    R = int(np.max(stage)) + 1
    Q = 0 if nuisance is None else nuisance.shape[2]
    endpoint_mode = Psi.ndim == 4

    if endpoint_mode:
        tail, head = _pair_endpoints_from_incidence(D)
        # Sparse selector matrices as dense N x M incidence masks; D is already
        # dense in the current TIDES observation API.
        S = np.zeros_like(D)
        Rmat = np.zeros_like(D)
        S[tail, np.arange(M)] = 1.0
        Rmat[head, np.arange(M)] = 1.0
        modes = tuple("endpoint_pair" for _ in range(L))
    else:
        modes = tuple(feature_output_modes)
        inv_sqrt2 = 1.0 / np.sqrt(2.0)
        mode_matrices: dict[str, FloatArray] = {}
        for mode in sorted(set(modes)):
            if mode == "legacy_difference":
                mode_matrices[mode] = D
            elif mode == "difference":
                mode_matrices[mode] = inv_sqrt2 * D
            elif mode == "common":
                mode_matrices[mode] = inv_sqrt2 * np.abs(D)
            else:
                raise RuntimeError(f"Unknown output mode {mode!r}.")
        mode_indices = {
            mode: np.asarray(
                [ell for ell, value in enumerate(modes) if value == mode],
                dtype=np.int64,
            )
            for mode in mode_matrices
        }

    stage_indices = tuple(np.flatnonzero(stage == r) for r in range(R))

    def unpack(beta):
        return _unpack_coefficients(
            beta,
            n_edges=M,
            n_edge_features=L,
            supports=supports,
            anchor_slice=anchor_slice,
            delta_slices=delta_slices,
            nuisance_slice=nuisance_slice,
            n_nuisance=Q,
        )

    def matvec(beta):
        B0, deltas, gamma = unpack(beta)
        B_stages = _stage_matrices(B0, deltas)
        pred = np.zeros_like(Y)
        for r, ids in enumerate(stage_indices):
            if not ids.size:
                continue
            if endpoint_mode:
                tail_response = np.einsum(
                    "ml,tml->tm",
                    B_stages[r],
                    Psi[ids, :, :, 0],
                    optimize=True,
                )
                head_response = np.einsum(
                    "ml,tml->tm",
                    B_stages[r],
                    Psi[ids, :, :, 1],
                    optimize=True,
                )
                pred[ids] += tail_response @ S.T + head_response @ Rmat.T
            else:
                for mode, H_mode in mode_matrices.items():
                    ell = mode_indices[mode]
                    if not ell.size:
                        continue
                    edge_response = np.einsum(
                        "ml,tml->tm",
                        B_stages[r][:, ell],
                        Psi[ids][:, :, ell],
                        optimize=True,
                    )
                    pred[ids] += edge_response @ H_mode.T
        if nuisance is not None:
            pred += np.einsum("tnq,q->tn", nuisance, gamma, optimize=True)
        return pred.ravel()

    def rmatvec(v):
        Rnode = np.asarray(v, dtype=float).reshape(T, N)
        stage_grad: list[FloatArray] = []
        for r, ids in enumerate(stage_indices):
            G = np.zeros((M, L), dtype=float)
            if ids.size:
                if endpoint_mode:
                    rt = Rnode[ids][:, tail]
                    rh = Rnode[ids][:, head]
                    G = (
                        np.einsum(
                            "tm,tml->ml", rt, Psi[ids, :, :, 0], optimize=True
                        )
                        + np.einsum(
                            "tm,tml->ml", rh, Psi[ids, :, :, 1], optimize=True
                        )
                    )
                else:
                    for mode, H_mode in mode_matrices.items():
                        ell = mode_indices[mode]
                        if not ell.size:
                            continue
                        edge_residual = Rnode[ids] @ H_mode
                        tmp = np.einsum(
                            "tm,tml->ml",
                            edge_residual,
                            Psi[ids],
                            optimize=True,
                        )
                        G[:, ell] = tmp[:, ell]
            stage_grad.append(G)

        grad = np.zeros(n_parameters, dtype=float)
        grad[anchor_slice] = np.sum(stage_grad, axis=0).ravel()
        suffix = np.zeros((M, L), dtype=float)
        suffix_after = [np.zeros((M, L), dtype=float) for _ in supports]
        for r in range(R - 1, 0, -1):
            suffix += stage_grad[r]
            suffix_after[r - 1] = suffix.copy()
        for k, (support, sl) in enumerate(zip(supports, delta_slices)):
            if support.size:
                grad[sl] = suffix_after[k][support, :].ravel()
        if nuisance is not None:
            grad[nuisance_slice] = np.einsum(
                "tnq,tn->q", nuisance, Rnode, optimize=True
            )
        return grad

    return LinearOperator(
        shape=(T * N, n_parameters),
        matvec=matvec,
        rmatvec=rmatvec,
        dtype=float,
    )


def build_vector_field_reconstruction_design(
    Y: ArrayLike,
    D: ArrayLike,
    edge_features: ArrayLike,
    stage_of_sample: Sequence[int],
    change_constraints_or_supports,
    *,
    feature_output_modes: Optional[Sequence[str]] = None,
    stationary_nuisance: Optional[ArrayLike] = None,
    design_mode: str = "auto",
    direct_max_entries: int = 20_000_000,
    solver_method: str = "auto",
) -> VectorFieldReconstructionDesign:
    """Build the selected-support full Step-3 regression design.

    ``design_mode='auto'`` uses an explicit dense matrix for modest problems
    and a LinearOperator for larger ones.  A requested dense least-squares
    backend forces the dense design; a requested LSQR backend uses the operator
    path unless ``design_mode='dense'`` is explicitly requested.
    """

    Y, D, Psi, stage, supports = _validate_observations(
        Y,
        D,
        edge_features,
        stage_of_sample,
        change_constraints_or_supports,
    )

    T, N = Y.shape
    M = D.shape[1]
    L = Psi.shape[2]
    R = int(np.max(stage)) + 1
    if Psi.ndim == 4:
        if feature_output_modes is not None:
            raise ValueError(
                "feature_output_modes must be omitted for endpoint-output features."
            )
        output_modes = tuple("endpoint_pair" for _ in range(L))
        feature_representation = "endpoint_pairwise"
    else:
        output_modes = _modes_from_support_source(
            change_constraints_or_supports, L, feature_output_modes
        )
        feature_representation = "scalar_mode_resolved"

    nuisance = _coerce_stationary_nuisance(
        stationary_nuisance,
        n_samples=T,
        n_nodes=N,
    )
    Q = 0 if nuisance is None else int(nuisance.shape[2])

    anchor_slice, delta_slices, nuisance_slice, n_parameters = _parameter_layout(
        n_edges=M,
        n_edge_features=L,
        supports=supports,
        n_nuisance=Q,
    )

    n_scalar = T * N
    estimated_entries = n_scalar * n_parameters

    design_mode = str(design_mode).lower()
    solver_method = str(solver_method).lower()

    if design_mode not in {"auto", "dense", "operator"}:
        raise ValueError("design_mode must be 'auto', 'dense', or 'operator'.")
    if solver_method not in {"auto", "dense_lstsq", "lsqr"}:
        raise ValueError(
            "solver_method must be 'auto', 'dense_lstsq', or 'lsqr'."
        )

    if solver_method == "dense_lstsq":
        resolved_mode = "dense"
    elif design_mode == "auto":
        if solver_method == "lsqr":
            resolved_mode = "operator"
        else:
            resolved_mode = (
                "dense"
                if estimated_entries <= int(direct_max_entries)
                else "operator"
            )
    else:
        resolved_mode = design_mode

    if resolved_mode == "operator" and solver_method == "dense_lstsq":
        raise ValueError(
            "dense_lstsq requires an explicit dense Step-3 design."
        )

    scales = _analytic_column_scales(
        D,
        Psi,
        stage,
        supports,
        nuisance,
        output_modes,
        anchor_slice=anchor_slice,
        delta_slices=delta_slices,
        nuisance_slice=nuisance_slice,
        n_parameters=n_parameters,
    )

    if resolved_mode == "dense":
        matrix = _build_dense_matrix(D, Psi, stage, supports, nuisance, output_modes)
    else:
        matrix = _build_linear_operator(
            Y,
            D,
            Psi,
            stage,
            supports,
            nuisance,
            output_modes,
            anchor_slice=anchor_slice,
            delta_slices=delta_slices,
            nuisance_slice=nuisance_slice,
            n_parameters=n_parameters,
        )

    return VectorFieldReconstructionDesign(
        y=Y.ravel(),
        matrix=matrix,
        column_scales=scales,
        supports=supports,
        anchor_slice=anchor_slice,
        delta_slices=delta_slices,
        nuisance_slice=nuisance_slice,
        n_samples=int(T),
        n_nodes=int(N),
        n_edges=int(M),
        n_edge_features=int(L),
        feature_output_modes=tuple(output_modes),
        feature_representation=str(feature_representation),
        n_stages=int(R),
        n_parameters=int(n_parameters),
        n_scalar_observations=int(n_scalar),
        n_nuisance_parameters=int(Q),
        design_mode=resolved_mode,
    )


# -----------------------------------------------------------------------------
# Formal Step-3 inference
# -----------------------------------------------------------------------------


def reconstruct_vector_field_from_observations(
    Y: ArrayLike,
    D: ArrayLike,
    edge_features: ArrayLike,
    stage_of_sample: Sequence[int],
    change_constraints_or_supports,
    *,
    feature_output_modes: Optional[Sequence[str]] = None,
    stationary_nuisance: Optional[ArrayLike] = None,
    solver_method: str = "auto",
    solver_kwargs: Optional[dict] = None,
    design_mode: str = "auto",
    direct_max_entries: int = 20_000_000,
    return_design: bool = False,
) -> VectorFieldReconstructionResult:
    """Reconstruct ``B^(1)`` and selected ``Delta B`` blocks jointly.

    Parameters
    ----------
    Y
        Preprocessed node-space vector-field observations, shape ``(T,N)``.
        Any known intrinsic term must already have been removed.
    D
        Edge-to-node aggregation/incidence matrix, shape ``(N,M)``.
    edge_features
        Edge-function responses, shape ``(T,M,L)``.
    stage_of_sample
        Detected stage label for each preprocessed observation.
    change_constraints_or_supports
        Step-2 result, Step-2 constraints, or one zero-based edge support array
        per transition.
    stationary_nuisance
        Optional stationary node-space nuisance design, shape ``(T,N,Q)`` or
        ``(T*N,Q)``.  Coefficients are fitted jointly and shared across stages.
    solver_method
        ``'auto'``, ``'dense_lstsq'``, or ``'lsqr'``.  N=8 correctness tests
        should normally use the dense SVD backend selected by ``'auto'``.
    solver_kwargs
        Additional keyword arguments forwarded to ``solve_least_squares``.
    design_mode
        ``'auto'``, ``'dense'``, or ``'operator'``.
    direct_max_entries
        Dense/operator switch threshold when both solver and design are auto.
    return_design
        Attach the constructed Step-3 design to the result for diagnostics.
    """

    Y_arr, D_arr, Psi, stage, supports = _validate_observations(
        Y,
        D,
        edge_features,
        stage_of_sample,
        change_constraints_or_supports,
    )

    nuisance = _coerce_stationary_nuisance(
        stationary_nuisance,
        n_samples=Y_arr.shape[0],
        n_nodes=Y_arr.shape[1],
    )

    design = build_vector_field_reconstruction_design(
        Y_arr,
        D_arr,
        Psi,
        stage,
        change_constraints_or_supports,
        feature_output_modes=feature_output_modes,
        stationary_nuisance=nuisance,
        design_mode=design_mode,
        direct_max_entries=direct_max_entries,
        solver_method=solver_method,
    )

    kwargs = {} if solver_kwargs is None else dict(solver_kwargs)

    forbidden = {"method", "column_scales"} & set(kwargs)
    if forbidden:
        raise ValueError(
            "Pass solver_method through the Step-3 argument and let Step 3 "
            "supply its own exact column scales; remove from solver_kwargs: "
            + ", ".join(sorted(forbidden))
        )

    linear_result = solve_least_squares(
        design.matrix,
        design.y,
        method=solver_method,
        column_scaling=True,
        column_scales=design.column_scales,
        **kwargs,
    )

    beta = np.asarray(linear_result.coefficients, dtype=float).reshape(-1)

    B0, deltas, gamma = _unpack_coefficients(
        beta,
        n_edges=design.n_edges,
        n_edge_features=design.n_edge_features,
        supports=design.supports,
        anchor_slice=design.anchor_slice,
        delta_slices=design.delta_slices,
        nuisance_slice=design.nuisance_slice,
        n_nuisance=design.n_nuisance_parameters,
    )

    B_stages = _stage_matrices(B0, deltas)

    fitted = np.asarray(linear_result.fitted_values, dtype=float).reshape(
        design.n_samples,
        design.n_nodes,
    )
    residual = np.asarray(linear_result.residual, dtype=float).reshape(
        design.n_samples,
        design.n_nodes,
    )

    metadata = {
        "backend": "global-selected-support-unpenalized-linear-regression",
        "support_source": "caller-provided Step-2 constraints/supports",
        "design_mode": design.design_mode,
        "solver_method": linear_result.method,
        "n_preprocessed_observations": int(design.n_samples),
        "n_scalar_observations": int(design.n_scalar_observations),
        "n_nodes": int(design.n_nodes),
        "n_candidate_edges": int(design.n_edges),
        "n_edge_features": int(design.n_edge_features),
        "feature_output_modes": tuple(design.feature_output_modes),
        "feature_representation": str(design.feature_representation),
        "output_generalization_active": bool(
            design.feature_representation != "scalar_mode_resolved"
            or any(mode != "legacy_difference" for mode in design.feature_output_modes)
        ),
        "n_stages": int(design.n_stages),
        "n_transitions": int(design.n_stages - 1),
        "anchor_parameters": int(design.n_edges * design.n_edge_features),
        "selected_change_groups": int(sum(s.size for s in design.supports)),
        "selected_change_parameters": int(
            sum(s.size for s in design.supports) * design.n_edge_features
        ),
        "stationary_nuisance_parameters": int(design.n_nuisance_parameters),
        "parameter_count": int(design.n_parameters),
        "rank": None if linear_result.rank is None else int(linear_result.rank),
        "identifiable": (
            None
            if linear_result.identifiable is None
            else bool(linear_result.identifiable)
        ),
        "condition_number_scaled": float(
            linear_result.condition_number_scaled
        ),
        "condition_number_raw": float(linear_result.condition_number_raw),
        "relative_residual": float(linear_result.relative_residual),
        "normal_equation_relative_residual": float(
            linear_result.normal_equation_relative_residual
        ),
        "solver_converged": bool(linear_result.converged),
    }

    return VectorFieldReconstructionResult(
        B_anchor=B0,
        delta_B=deltas,
        B_stages=B_stages,
        fitted_vector_field=fitted,
        residual_vector_field=residual,
        residual_norm=float(linear_result.residual_norm),
        relative_residual=float(linear_result.relative_residual),
        parameter_count=int(design.n_parameters),
        n_observations=int(design.n_scalar_observations),
        n_preprocessed_observations=int(design.n_samples),
        stage_of_sample=stage.copy(),
        solver_result=linear_result,
        stationary_nuisance_coefficients=gamma,
        metadata=metadata,
        design=design if return_design else None,
    )


# Short public name for the formal observation-based Step-3 API.
# This keeps ``tides.py`` imports stable while the pipeline wrapper is updated.
reconstruct_vector_field = reconstruct_vector_field_from_observations


__all__ = [
    "LibraryFunction",
    "VectorFieldReconstructionDesign",
    "VectorFieldReconstructionResult",
    "build_vector_field_reconstruction_design",
    "reconstruct_vector_field_from_observations",
    "reconstruct_vector_field",
]
