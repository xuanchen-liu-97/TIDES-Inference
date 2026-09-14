"""TIDES Step 2: infer supports of piecewise-sparse vector-field changes.

For the varying-structure branch, observations obey the cumulative model

    y = X_base vec(B^(1)) + X_change beta_delta,

with

    B^(r) = B^(1) + sum_{k<r} Delta B^(k).

Only the temporal changes are assumed group-sparse.  The stationary anchor is
an unrestricted nuisance and the shared interaction-law factorisation used in
Step 4 is deliberately *not* imposed here.

Two computational backends implement the same Step-2 statistical principle.

``dense``
    Correctness/reference backend.  It explicitly builds the cumulative change
    matrix, projects out the stationary anchor with a dense SVD basis, solves
    grouped BPDN, ranks groups by convex coefficient norm, and certifies the
    first floor-feasible ranked prefix.

``scalable``
    Memory-scalable backend.  It never materialises the full cumulative change
    matrix.  The stationary anchor is stored sparsely, projected with high-
    accuracy LSQR, and projected temporal group blocks are generated lazily.
    The default scalable solver builds a floor-feasible working set in batches,
    solves restricted grouped BPDN with the dense-SVD reference kernel, and then
    globally audits every omitted group with the BPDN dual/KKT conditions.
    Violating groups are reactivated until the full convex problem is certified.
    Final support size is still chosen only by the independent ranked-prefix
    uncertainty-floor certificate.  The older conditional-screening backend is
    retained as an explicit regression baseline.

General pairwise input/output representation
--------------------------------------------
The inference core accepts either the historical scalar/mode-resolved edge
features ``(T,M,L)`` or generalized endpoint-output features ``(T,M,L,2)``.
The latter gather the complete sender/receiver state and directly specify the
two endpoint contributions of each basis function, so no equal-and-opposite
``D q`` output assumption is required.

``build_pairwise_polynomial_features`` retains the orthonormal common/difference
coordinate construction for scalar/mode-resolved experiments.  The preferred
fully generalized builder, ``build_pairwise_endpoint_polynomial_features``,
uses sender/receiver states and a swap-equivariant polynomial basis.  To avoid
a structural edge-self gauge, own-state-only monomials are assigned to the
one-body sector; the returned pairwise library contains all cross-only and
genuine joint monomials up to the requested degree.  Thus topology, pairwise
state gathering, and unknown local dynamics remain separated without making
individual edge coefficients non-identifiable by construction.

Noise awareness
---------------
Both backends require only an externally calibrated uncertainty floor.  The
clean benchmark uses a preprocessing-resolution floor; noisy/weak-form
observation layers may supply a larger effective uncertainty budget without
changing the sparse-selection principle.

Legacy Adaptive Group-LASSO and forward/backward search remain available on the
dense backend as explicit regression baselines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Literal, Mapping, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

import scipy.sparse as sp
from scipy.sparse.linalg import lsqr, splu

try:  # package form
    from .solvers_sparse_structure import (
        AdaptiveGroupLassoResult,
        ForwardBackwardGroupSearchResult,
        GroupBasisPursuitResult,
        ScreenedGroupBasisPursuitResult,
        WorkingSetGroupBasisPursuitResult,
        solve_adaptive_group_lasso,
        solve_forward_backward_group_search,
        solve_group_basis_pursuit_denoising,
        solve_screened_group_basis_pursuit_denoising,
        solve_working_set_group_basis_pursuit_denoising,
    )
    from .solvers_linear_regression import (
        LinearRegressionResult,
        solve_least_squares,
    )
except ImportError:  # flat-file form
    from solvers_sparse_structure import (
        AdaptiveGroupLassoResult,
        ForwardBackwardGroupSearchResult,
        GroupBasisPursuitResult,
        ScreenedGroupBasisPursuitResult,
        WorkingSetGroupBasisPursuitResult,
        solve_adaptive_group_lasso,
        solve_forward_backward_group_search,
        solve_group_basis_pursuit_denoising,
        solve_screened_group_basis_pursuit_denoising,
        solve_working_set_group_basis_pursuit_denoising,
    )
    from solvers_linear_regression import (
        LinearRegressionResult,
        solve_least_squares,
    )


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
GroupLabel = tuple[int, int]  # (transition ordinal, candidate-edge index)


@dataclass(frozen=True)
class PairwisePolynomialFeatures:
    """Dynamics-agnostic pairwise local-state polynomial features.

    For the oriented incidence convention ``D[:,m] = -e_i + e_j`` we first
    gather the complete endpoint state in the orthonormal common/difference
    coordinates

        c_m = (x_i + x_j) / sqrt(2),
        d_m = (x_j - x_i) / sqrt(2).

    This is an invertible orthogonal change of coordinates from ``(x_i,x_j)``.
    For an undirected interaction, reversing the arbitrary bookkeeping
    orientation leaves ``c`` unchanged and sends ``d -> -d``.  Orientation-
    equivariant endpoint dynamics therefore decompose into a common output
    channel that is even in ``d`` and a difference output channel that is odd
    in ``d``.  ``output_modes`` records that topological assembly channel for
    every polynomial feature without doubling the feature dimension.
    """

    local_states: FloatArray              # (T, M, 2), orthonormal physical (c,d)
    normalized_local_states: FloatArray   # (T, M, 2), coordinates used by basis
    edge_features: FloatArray             # (T, M, L)
    exponents: tuple[tuple[int, int], ...] # (power of c, power of d)
    feature_labels: tuple[str, ...]
    output_modes: tuple[str, ...]          # 'common' for even d, 'difference' for odd d
    component_labels: tuple[str, ...]      # mode-resolved labels for Step 4
    coordinate_scale: FloatArray           # shape (2,), scale-only normalization
    degree: int
    include_constant: bool
    normalization: str
    coordinate_convention: str


def _validate_standard_pairwise_incidence(D: FloatArray) -> None:
    """Require one ``-1`` tail and one ``+1`` head per candidate pair."""

    D_arr = np.asarray(D, dtype=float)
    nz = np.abs(D_arr) > 1.0e-12
    if D_arr.ndim != 2 or not np.all(np.sum(nz, axis=0) == 2):
        raise ValueError(
            "General pairwise common/difference coordinates require exactly two "
            "nonzero incidence entries per candidate edge."
        )
    for m in range(D_arr.shape[1]):
        vals = np.sort(D_arr[nz[:, m], m])
        if not np.allclose(vals, np.array([-1.0, 1.0]), atol=1.0e-12, rtol=0.0):
            raise ValueError(
                "General pairwise common/difference coordinates require standard "
                "oriented incidence columns with values {-1,+1}; interaction "
                "strengths belong in inferred coefficients, not in D."
            )


def _resolve_feature_output_modes(
    n_features: int,
    feature_output_modes: Optional[Sequence[str]],
) -> tuple[str, ...]:
    """Resolve feature-wise node-space assembly directions.

    ``None`` preserves the historical TIDES convention exactly: every feature
    uses the unnormalised incidence direction ``D[:,m]``.  Explicit
    ``'common'`` / ``'difference'`` modes use the orthonormal endpoint-output
    basis ``|D|/sqrt(2)`` / ``D/sqrt(2)``.
    """

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
            f"(legacy internal mode also accepted); got invalid values {bad!r}."
        )
    return modes


def _feature_output_directions(
    D: FloatArray,
    modes: Sequence[str],
) -> FloatArray:
    """Return node-space output directions with shape ``(N,M,L)``."""

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
        else:  # defensive: modes are validated above
            raise RuntimeError(f"Unknown output mode {mode!r}.")
    return out


def build_pairwise_polynomial_features(
    node_states: ArrayLike,
    D: ArrayLike,
    *,
    degree: int = 2,
    include_constant: bool = False,
    normalization: Literal["none", "rms", "maxabs"] = "rms",
    coordinate_scale: Optional[Sequence[float]] = None,
) -> PairwisePolynomialFeatures:
    """Construct the orientation-equivariant generic pairwise feature library.

    No target dynamical law is supplied.  Each standard incidence edge first
    receives the complete endpoint state through the orthonormal coordinates

        c_m = (x_i + x_j) / sqrt(2),
        d_m = (x_j - x_i) / sqrt(2).

    We then evaluate every monomial ``c**a * d**b`` up to the requested total
    degree.  Orientation equivariance fixes only the *output assembly channel*:
    even powers of ``d`` contribute through the common endpoint direction
    ``|D|/sqrt(2)``, while odd powers contribute through the difference direction
    ``D/sqrt(2)``.  This is a topological symmetry constraint, not information
    about the unknown target law.

    ``normalization`` rescales ``c`` and ``d`` but does not centre them, so parity
    under edge reversal and the physical zero-relative-state origin are preserved.
    """

    X = np.asarray(node_states, dtype=float)
    D_arr = np.asarray(D, dtype=float)
    if X.ndim != 2:
        raise ValueError("node_states must have shape (n_observations, n_nodes).")
    if D_arr.ndim != 2:
        raise ValueError("D must have shape (n_nodes, n_candidate_edges).")
    if X.shape[1] != D_arr.shape[0]:
        raise ValueError("node_states.shape[1] must equal D.shape[0].")
    if X.shape[0] == 0 or X.shape[1] == 0 or D_arr.shape[1] == 0:
        raise ValueError("node_states and D must be non-empty.")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(D_arr)):
        raise ValueError("node_states and D must be finite.")

    degree = int(degree)
    if degree < 1:
        raise ValueError("degree must be >= 1.")

    _validate_standard_pairwise_incidence(D_arr)

    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    d = inv_sqrt2 * (X @ D_arr)
    c = inv_sqrt2 * (X @ np.abs(D_arr))
    local = np.stack([c, d], axis=2)

    normalization = str(normalization).lower()
    if normalization not in {"none", "rms", "maxabs"}:
        raise ValueError("normalization must be 'none', 'rms', or 'maxabs'.")

    if coordinate_scale is not None:
        scale = np.asarray(coordinate_scale, dtype=float).reshape(-1)
        if scale.shape != (2,) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError("coordinate_scale must contain two finite positive values.")
    elif normalization == "none":
        scale = np.ones(2, dtype=float)
    elif normalization == "rms":
        scale = np.sqrt(np.mean(local * local, axis=(0, 1)))
        scale = np.where(scale > 1.0e-14, scale, 1.0)
    else:
        scale = np.max(np.abs(local), axis=(0, 1))
        scale = np.where(scale > 1.0e-14, scale, 1.0)

    local_scaled = local / scale[None, None, :]
    cs = local_scaled[:, :, 0]
    ds = local_scaled[:, :, 1]

    exponents: list[tuple[int, int]] = []
    start_degree = 0 if include_constant else 1
    for total in range(start_degree, degree + 1):
        for a in range(total, -1, -1):
            b = total - a
            exponents.append((int(a), int(b)))

    features = np.empty((X.shape[0], D_arr.shape[1], len(exponents)), dtype=float)
    labels: list[str] = []
    modes: list[str] = []
    component_labels: list[str] = []
    for ell, (a, b) in enumerate(exponents):
        term = np.ones_like(cs)
        if a:
            term = term * np.power(cs, a)
        if b:
            term = term * np.power(ds, b)
        features[:, :, ell] = term
        if a == 0 and b == 0:
            label = "1"
        else:
            parts = []
            if a:
                parts.append("c" if a == 1 else f"c^{a}")
            if b:
                parts.append("d" if b == 1 else f"d^{b}")
            label = "*".join(parts)
        mode = "common" if (b % 2 == 0) else "difference"
        labels.append(label)
        modes.append(mode)
        component_labels.append(f"{mode}:{label}")

    return PairwisePolynomialFeatures(
        local_states=np.asarray(local, dtype=float),
        normalized_local_states=np.asarray(local_scaled, dtype=float),
        edge_features=np.asarray(features, dtype=float),
        exponents=tuple(exponents),
        feature_labels=tuple(labels),
        output_modes=tuple(modes),
        component_labels=tuple(component_labels),
        coordinate_scale=np.asarray(scale, dtype=float),
        degree=int(degree),
        include_constant=bool(include_constant),
        normalization=str(normalization),
        coordinate_convention="orthonormal-common-difference",
    )


@dataclass(frozen=True)
class PairwiseEndpointPolynomialFeatures:
    """Gauge-fixed, orientation-equivariant pairwise endpoint library.

    A general swap-equivariant endpoint law can be written as

        g_i = F(x_i, x_j),
        g_j = F(x_j, x_i).

    Monomials depending only on the *receiving node itself* (``x_i**n`` in
    ``g_i`` and ``x_j**n`` in ``g_j``) form a one-body sector and cannot be
    uniquely assigned to individual incident edges without extra microscopic
    assumptions.  This builder therefore returns the canonical pairwise sector
    modulo that one-body gauge: for every total degree ``n`` it keeps

        x_i**a x_j**b  in g_i,
        x_j**a x_i**b  in g_j,

    with ``a+b=n`` and ``b>=1``.  Cross-only terms (``a=0``) and genuine joint
    terms (``a,b>0``) are both retained.  The omitted own-only sector should be
    modelled separately as node-local dynamics when required.
    """

    local_states: FloatArray          # (T,M,2), orthonormal (c,d)
    endpoint_states: FloatArray       # (T,M,2), oriented (tail,head)
    normalized_endpoint_states: FloatArray
    endpoint_features: FloatArray     # (T,M,L,2): contributions at (tail,head)
    powers: tuple[tuple[int, int], ...]  # (self power a, neighbour power b>=1)
    feature_labels: tuple[str, ...]
    component_labels: tuple[str, ...]
    state_scale: float
    degree: int
    normalization: str
    feature_representation: str


def _pair_endpoints_from_incidence(D: FloatArray) -> tuple[IntArray, IntArray]:
    """Return tail/head node indices from standard oriented incidence columns."""

    D_arr = np.asarray(D, dtype=float)
    _validate_standard_pairwise_incidence(D_arr)
    M = D_arr.shape[1]
    tail = np.empty(M, dtype=np.int64)
    head = np.empty(M, dtype=np.int64)
    for m in range(M):
        tail[m] = int(np.flatnonzero(D_arr[:, m] < -0.5)[0])
        head[m] = int(np.flatnonzero(D_arr[:, m] > 0.5)[0])
    return tail, head


def build_pairwise_endpoint_polynomial_features(
    node_states: ArrayLike,
    D: ArrayLike,
    *,
    degree: int = 2,
    normalization: Literal["none", "rms", "maxabs"] = "rms",
    state_scale: Optional[float] = None,
) -> PairwiseEndpointPolynomialFeatures:
    """Build a dynamics-agnostic identifiable pairwise endpoint library.

    The incidence matrix is used only to gather the two endpoint states.  The
    complete endpoint information is retained; no assumption such as
    ``G=G(x_j-x_i)`` is made.  A common scalar scale is used for both endpoints,
    preserving swap equivariance.  The returned feature tensor already contains
    the two endpoint output values and can therefore represent non-conservative
    interactions such as SIS/LV-like pairwise terms without a fixed ``D q``
    assembly rule.

    The library is complete for polynomial swap-equivariant *pairwise* dynamics
    up to ``degree`` after quotienting out the node-local own-state sector.  The
    latter is a separate one-body component rather than an identifiable edge
    contribution.
    """

    X = np.asarray(node_states, dtype=float)
    D_arr = np.asarray(D, dtype=float)
    if X.ndim != 2:
        raise ValueError("node_states must have shape (n_observations, n_nodes).")
    if D_arr.ndim != 2 or X.shape[1] != D_arr.shape[0]:
        raise ValueError("D must have shape (n_nodes, n_candidate_edges).")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(D_arr)):
        raise ValueError("node_states and D must be finite.")
    degree = int(degree)
    if degree < 1:
        raise ValueError("degree must be >= 1.")

    tail, head = _pair_endpoints_from_incidence(D_arr)
    xs = X[:, tail]
    xr = X[:, head]
    endpoint = np.stack([xs, xr], axis=2)
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    c = inv_sqrt2 * (xs + xr)
    d = inv_sqrt2 * (xr - xs)
    local = np.stack([c, d], axis=2)

    normalization = str(normalization).lower()
    if normalization not in {"none", "rms", "maxabs"}:
        raise ValueError("normalization must be 'none', 'rms', or 'maxabs'.")
    if state_scale is not None:
        scale = float(state_scale)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("state_scale must be finite and positive.")
    elif normalization == "none":
        scale = 1.0
    elif normalization == "rms":
        scale = float(np.sqrt(np.mean(X * X)))
        if scale <= 1.0e-14:
            scale = 1.0
    else:
        scale = float(np.max(np.abs(X)))
        if scale <= 1.0e-14:
            scale = 1.0

    xsn = xs / scale
    xrn = xr / scale
    endpoint_scaled = np.stack([xsn, xrn], axis=2)

    powers: list[tuple[int, int]] = []
    labels: list[str] = []
    for total in range(1, degree + 1):
        for a in range(0, total):
            b = total - a  # neighbour power; always >= 1
            powers.append((int(a), int(b)))
            if a == 0:
                label = "neighbor" if b == 1 else f"neighbor^{b}"
            else:
                left = "self" if a == 1 else f"self^{a}"
                right = "neighbor" if b == 1 else f"neighbor^{b}"
                label = f"{left}*{right}"
            labels.append(label)

    E = np.empty((X.shape[0], D_arr.shape[1], len(powers), 2), dtype=float)
    for ell, (a, b) in enumerate(powers):
        tail_term = np.ones_like(xsn)
        head_term = np.ones_like(xrn)
        if a:
            tail_term *= np.power(xsn, a)
            head_term *= np.power(xrn, a)
        if b:
            tail_term *= np.power(xrn, b)
            head_term *= np.power(xsn, b)
        E[:, :, ell, 0] = tail_term
        E[:, :, ell, 1] = head_term

    return PairwiseEndpointPolynomialFeatures(
        local_states=np.asarray(local, dtype=float),
        endpoint_states=np.asarray(endpoint, dtype=float),
        normalized_endpoint_states=np.asarray(endpoint_scaled, dtype=float),
        endpoint_features=np.asarray(E, dtype=float),
        powers=tuple(powers),
        feature_labels=tuple(labels),
        component_labels=tuple(f"pair:{label}" for label in labels),
        state_scale=float(scale),
        degree=int(degree),
        normalization=str(normalization),
        feature_representation="endpoint_pairwise_irreducible",
    )


@dataclass(frozen=True)
class CumulativeChangeDesign:
    """Explicit global Step-2 design after preprocessing."""

    y: FloatArray
    X_base: FloatArray
    X_change: FloatArray
    y_perp: FloatArray
    X_change_perp: FloatArray
    group_columns: Mapping[GroupLabel, IntArray]
    group_order: tuple[GroupLabel, ...]
    baseline_rank: int
    baseline_singular_values: FloatArray
    baseline_rank_tolerance: float
    baseline_weakest_retained_relative_singular: float
    n_samples: int
    n_nodes: int
    n_edges: int
    n_edge_features: int
    feature_output_modes: tuple[str, ...]
    feature_representation: str
    n_transitions: int




@dataclass(frozen=True)
class ScalableCumulativeChangeDesign:
    """Lazy projected cumulative Step-2 design.

    The full ``X_change`` / ``X_change_perp`` arrays do not exist.  Projected
    group blocks are materialised only when requested by a scalable solver or
    the final prefix certificate.
    """

    y: FloatArray
    y_perp: FloatArray
    group_columns: Mapping[GroupLabel, IntArray]
    group_order: tuple[GroupLabel, ...]
    baseline_rank: int
    baseline_singular_values: FloatArray
    baseline_rank_tolerance: float
    baseline_weakest_retained_relative_singular: float
    n_samples: int
    n_nodes: int
    n_edges: int
    n_edge_features: int
    feature_output_modes: tuple[str, ...]
    feature_representation: str
    n_transitions: int
    n_base_columns: int
    anchor_nnz: int
    estimated_explicit_dense_bytes: int
    baseline_projection_backend: str
    _operator: object

    def group_block(self, label: GroupLabel) -> FloatArray:
        return self._operator.projected_group_block(label)

    def conditional_gain_provider(
        self,
        residual: FloatArray,
        Q: FloatArray,
        omitted: tuple[GroupLabel, ...],
    ) -> tuple[GroupLabel, float]:
        return self._operator.conditional_gain_provider(residual, Q, omitted)

    def group_adjoint_norm_provider(
        self,
        vector: FloatArray,
        labels: tuple[GroupLabel, ...],
    ) -> Mapping[GroupLabel, float]:
        return self._operator.group_adjoint_norm_provider(vector, labels)


class _SparseBaselineProjector:
    """Project onto the orthogonal complement of the stationary-anchor span.

    The scalable Step-2 path must preserve the *same profiled geometry* as the
    dense reference.  A small projection error can otherwise be interpreted as
    temporal signal when the uncertainty floor is very tight.

    Strategy
    --------
    1. If the scaled sparse anchor is small enough to densify, cache the same
       machine-rank SVD column-space basis used by the dense reference and apply

           P_perp b = b - Q (Q^T b).

       This is the preferred path for small/moderate anchors because it gives an
       explicitly orthogonal projector.

    2. Otherwise use sparse LSQR with a generous iteration budget.  The returned
       residual is *certified* by its maximum cosine with the unit-norm scaled
       anchor columns.  If needed, one or more correction solves remove the
       remaining anchor component.  Failure to meet the certificate raises an
       error rather than silently contaminating Step 2.

    The optional normal-Gram LU is retained only for the legacy conditional-
    screening self-Gram calculation; the KKT working-set backend does not need
    it for projection.
    """

    def __init__(
        self,
        X_base: sp.csc_matrix,
        *,
        projection_tol: float = 1.0e-16,
        projection_max_iter: Optional[int] = None,
        diagnostic_dense_bytes: int = 64 * 1024**2,
        projection_certificate_tol: float = 1.0e-12,
        projection_refinement_passes: int = 2,
        require_projected_gram: bool = True,
    ):
        X_base = sp.csc_matrix(X_base, dtype=float)
        self.n_rows, self.n_columns_original = X_base.shape
        self.projection_tol = float(projection_tol)
        if not np.isfinite(self.projection_tol) or self.projection_tol <= 0.0:
            raise ValueError("projection_tol must be finite and positive.")

        self.projection_certificate_tol = float(projection_certificate_tol)
        if (
            not np.isfinite(self.projection_certificate_tol)
            or self.projection_certificate_tol <= 0.0
        ):
            raise ValueError(
                "projection_certificate_tol must be finite and positive."
            )

        self.projection_refinement_passes = int(projection_refinement_passes)
        if self.projection_refinement_passes < 0:
            raise ValueError("projection_refinement_passes must be >= 0.")

        # Runtime diagnostics.  These are intentionally monotone so the caller
        # can inspect the worst projection encountered during one Step-2 solve.
        self.projection_solve_count = 0
        self.projection_refinement_count = 0
        self.max_projection_certificate = 0.0
        self.max_lsqr_iterations = 0
        self.last_lsqr_istop = 0

        norms = np.sqrt(np.asarray(X_base.power(2).sum(axis=0)).reshape(-1))
        good = norms > 1.0e-14
        self.good_columns = np.flatnonzero(good)
        self.zero_columns = np.flatnonzero(~good)
        self.column_scale = norms[good]

        self._orthogonal_basis: Optional[FloatArray] = None

        if self.good_columns.size == 0:
            self.X_scaled = sp.csc_matrix((self.n_rows, 0), dtype=float)
            self.gram_lu = None
            self.rank = 0
            self.singular_values = np.empty(0, dtype=float)
            self.rank_tolerance = 0.0
            self.weakest_relative_singular = np.nan
            self.max_iter = 0
            self.projection_backend = "empty-anchor"
            return

        X_good = X_base[:, self.good_columns]
        self.X_scaled = (
            X_good @ sp.diags(1.0 / self.column_scale, format="csc")
        ).tocsc()
        p = self.X_scaled.shape[1]

        # The previous default max(2000, 4*p) was not a convergence guarantee:
        # a well-defined N=8 endpoint benchmark required ~3700 iterations even
        # though p=84.  Give LSQR a true ceiling and let its own stopping rule
        # terminate early when the problem is easy.
        if projection_max_iter is None:
            self.max_iter = 20000
        else:
            self.max_iter = int(projection_max_iter)
            if self.max_iter < 1:
                raise ValueError("projection_max_iter must be >= 1.")

        dense_bytes = int(self.X_scaled.shape[0] * p * 8)
        if dense_bytes <= int(diagnostic_dense_bytes):
            A_dense = self.X_scaled.toarray()
            Q, rank, singular, rank_tol, weakest = _scaled_column_space_basis(
                A_dense,
                zero_column_tol=1.0e-14,
            )
            self._orthogonal_basis = np.asarray(Q, dtype=float)
            self.rank = int(rank)
            self.singular_values = np.asarray(singular, dtype=float)
            self.rank_tolerance = float(rank_tol)
            self.weakest_relative_singular = float(weakest)
            self.projection_backend = "dense-svd-anchor-orthogonal"
        else:
            # Large-anchor path: remain sparse/matrix-free for projection.
            # Rank is not inferred from LSQR.  The optional projected-Gram LU
            # below can certify full column rank for the legacy screen only.
            self.rank = -1
            self.singular_values = np.empty(0, dtype=float)
            self.rank_tolerance = np.nan
            self.weakest_relative_singular = np.nan
            self.projection_backend = "sparse-lsqr-certified"

        # Only the legacy conditional-gain screen needs A_g^T P_perp A_g.
        self.gram_lu = None
        if bool(require_projected_gram):
            gram = (self.X_scaled.T @ self.X_scaled).tocsc()
            try:
                self.gram_lu = splu(gram)
            except RuntimeError as exc:
                raise RuntimeError(
                    "Legacy screened scalable Step 2 requires the nonzero "
                    "stationary-anchor columns to be numerically independent "
                    "for its projected-Gram factorisation. Use "
                    "scalable_solver='working_set' to avoid this requirement."
                ) from exc
            if self.rank < 0:
                self.rank = int(p)

    def _projection_certificate(self, residual: FloatArray) -> float:
        """Maximum cosine with any scaled anchor column.

        ``X_scaled`` has unit-norm columns, so

            max_j |x_j^T r| / ||r||

        is exactly the maximum absolute anchor-column cosine.  A true orthogonal
        projection has value zero up to roundoff.
        """

        r = np.asarray(residual, dtype=float).reshape(-1)
        r_norm = float(np.linalg.norm(r))
        if r_norm <= np.finfo(float).tiny or self.X_scaled.shape[1] == 0:
            return 0.0
        normal = np.asarray(self.X_scaled.T @ r).reshape(-1)
        if normal.size == 0:
            return 0.0
        return float(np.max(np.abs(normal)) / r_norm)

    def _lsqr_once(
        self,
        rhs: FloatArray,
        *,
        btol: Optional[float] = None,
    ) -> tuple[FloatArray, tuple]:
        rhs = np.asarray(rhs, dtype=float).reshape(-1)
        out = lsqr(
            self.X_scaled,
            rhs,
            atol=self.projection_tol,
            btol=self.projection_tol if btol is None else float(btol),
            conlim=1.0e18,
            iter_lim=self.max_iter,
            show=False,
        )
        self.max_lsqr_iterations = max(self.max_lsqr_iterations, int(out[2]))
        self.last_lsqr_istop = int(out[1])
        return np.asarray(out[0], dtype=float), out

    def _project_vector_sparse_certified(self, b: FloatArray) -> FloatArray:
        b = np.asarray(b, dtype=float).reshape(-1)

        coef, _ = self._lsqr_once(b)
        residual = b - np.asarray(self.X_scaled @ coef).reshape(-1)
        certificate = self._projection_certificate(residual)

        # A second least-squares solve on the *remaining anchor component* is
        # effective only after the first solve has actually converged.  This is
        # different from the earlier diagnostic that repeatedly restarted an
        # under-converged 2000-iteration solve.
        for _ in range(self.projection_refinement_passes):
            if certificate <= self.projection_certificate_tol:
                break
            correction, _ = self._lsqr_once(residual, btol=0.0)
            residual = residual - np.asarray(
                self.X_scaled @ correction
            ).reshape(-1)
            self.projection_refinement_count += 1
            certificate = self._projection_certificate(residual)

        self.projection_solve_count += 1
        self.max_projection_certificate = max(
            self.max_projection_certificate,
            float(certificate),
        )

        if certificate > self.projection_certificate_tol:
            raise RuntimeError(
                "Scalable stationary-anchor projection failed its numerical "
                "orthogonality certificate: max anchor cosine="
                f"{certificate:.3e} exceeds tolerance "
                f"{self.projection_certificate_tol:.3e}. Increase "
                "scalable_projection_max_iter, relax the externally calibrated "
                "uncertainty floor if scientifically justified, or use the "
                "dense backend for this problem."
            )

        return np.asarray(residual, dtype=float)

    def project_vector(self, b: FloatArray) -> FloatArray:
        b = np.asarray(b, dtype=float).reshape(-1)
        if b.size != self.n_rows:
            raise ValueError("projection vector has incompatible length.")
        if self.X_scaled.shape[1] == 0:
            return b.copy()

        if self._orthogonal_basis is not None:
            Q = self._orthogonal_basis
            out = b - Q @ (Q.T @ b)
            certificate = self._projection_certificate(out)
            self.projection_solve_count += 1
            self.max_projection_certificate = max(
                self.max_projection_certificate,
                float(certificate),
            )
            return np.asarray(out, dtype=float)

        return self._project_vector_sparse_certified(b)

    def project_dense_block(self, G: FloatArray) -> FloatArray:
        G = np.asarray(G, dtype=float)
        if G.ndim != 2 or G.shape[0] != self.n_rows:
            raise ValueError("projection block has incompatible shape.")
        if self.X_scaled.shape[1] == 0:
            return G.copy()

        if self._orthogonal_basis is not None:
            Q = self._orthogonal_basis
            out = G - Q @ (Q.T @ G)
            # Track a certificate per column without re-projecting.
            for j in range(out.shape[1]):
                certificate = self._projection_certificate(out[:, j])
                self.max_projection_certificate = max(
                    self.max_projection_certificate,
                    float(certificate),
                )
            self.projection_solve_count += int(out.shape[1])
            return np.asarray(out, dtype=float)

        out = np.empty_like(G)
        for j in range(G.shape[1]):
            out[:, j] = self._project_vector_sparse_certified(G[:, j])
        return out

    def projected_gram_from_sparse_block(self, G: sp.csc_matrix) -> FloatArray:
        """Compute ``G^T P_perp G`` for legacy conditional screening."""

        G = sp.csc_matrix(G, dtype=float)
        raw = np.asarray((G.T @ G).toarray(), dtype=float)
        if self.X_scaled.shape[1] == 0:
            return raw

        # When an orthonormal basis is already cached, use it directly rather
        # than routing a small problem through normal equations.
        if self._orthogonal_basis is not None:
            projected = self.project_dense_block(G.toarray())
            H = projected.T @ projected
            return 0.5 * (H + H.T)

        if self.gram_lu is None:
            raise RuntimeError(
                "Projected self-Gram requested although the scalable design was "
                "built without the legacy screening Gram factorisation."
            )
        rhs = np.asarray((self.X_scaled.T @ G).toarray(), dtype=float)
        coef = np.asarray(self.gram_lu.solve(rhs), dtype=float)
        H = raw - rhs.T @ coef
        H = 0.5 * (H + H.T)

        # Normal equations are used only for legacy screening geometry.  Clip
        # negative roundoff modes while preserving positive directions.
        eig, vec = np.linalg.eigh(H)
        scale = max(float(np.max(np.abs(eig))) if eig.size else 0.0, 1.0e-300)
        tol = 500.0 * np.finfo(float).eps * scale
        eig = np.where(eig > tol, eig, 0.0)
        return (vec * eig[None, :]) @ vec.T

def _build_sparse_anchor_design(
    D: FloatArray,
    edge_features: FloatArray,
    feature_output_modes: Sequence[str],
) -> sp.csc_matrix:
    """Build the stationary edge-function design directly in CSC form."""

    T, M, L = edge_features.shape[:3]
    N = D.shape[0]
    index_dtype = np.int32 if T * N < np.iinfo(np.int32).max else np.int64

    if edge_features.ndim == 4:
        tail, head = _pair_endpoints_from_incidence(D)
        nnz_per_col = 2 * T
        total_nnz = M * L * nnz_per_col
        data = np.empty(total_nnz, dtype=float)
        indices = np.empty(total_nnz, dtype=index_dtype)
        indptr = np.arange(0, total_nnz + 1, nnz_per_col, dtype=np.int64)
        pos = 0
        time_base = np.arange(T, dtype=np.int64) * N
        for m in range(M):
            rows = np.column_stack([time_base + tail[m], time_base + head[m]]).reshape(-1)
            for ell in range(L):
                n = rows.size
                indices[pos:pos+n] = rows
                data[pos:pos+n] = edge_features[:, m, ell, :].reshape(-1)
                pos += n
        return sp.csc_matrix((data, indices, indptr), shape=(T * N, M * L))

    modes = tuple(feature_output_modes)
    directions = _feature_output_directions(D, modes)  # (N,M,L)
    nnz_per_feature = np.count_nonzero(directions, axis=0).astype(np.int64)  # (M,L)
    total_nnz = int(T * int(nnz_per_feature.sum()))

    data = np.empty(total_nnz, dtype=float)
    indices = np.empty(total_nnz, dtype=index_dtype)
    indptr = np.empty(M * L + 1, dtype=np.int64)
    indptr[0] = 0

    time_rows = (np.arange(T, dtype=np.int64) * N)[:, None]
    pos = 0
    col = 0
    for m in range(M):
        for ell in range(L):
            direction = directions[:, m, ell]
            nodes = np.flatnonzero(np.abs(direction) > 1.0e-14)
            values = direction[nodes]
            rows = (time_rows + nodes[None, :]).reshape(-1)
            n = rows.size
            indices[pos : pos + n] = rows
            data[pos : pos + n] = (
                edge_features[:, m, ell, None] * values[None, :]
            ).reshape(-1)
            pos += n
            col += 1
            indptr[col] = pos

    return sp.csc_matrix(
        (data, indices, indptr),
        shape=(T * N, M * L),
    )


class _LazyProjectedChangeOperator:
    """Cumulative temporal group blocks with feature-resolved output assembly."""

    def __init__(
        self,
        D: FloatArray,
        edge_features: FloatArray,
        stage: IntArray,
        projector: _SparseBaselineProjector,
        feature_output_modes: Sequence[str],
    ):
        self.D = np.asarray(D, dtype=float)
        self.edge_features = np.asarray(edge_features, dtype=float)
        self.stage = np.asarray(stage, dtype=np.int64)
        self.projector = projector
        self.T, self.M, self.L = self.edge_features.shape
        self.N = self.D.shape[0]
        self.K = int(np.max(self.stage))
        self.n_rows = int(self.T * self.N)
        self.feature_output_modes = tuple(feature_output_modes)
        if len(self.feature_output_modes) != self.L:
            raise ValueError("feature_output_modes has incompatible length.")

        self.output_directions = _feature_output_directions(
            self.D, self.feature_output_modes
        )  # (N,M,L)
        self.mode_matrices: dict[str, FloatArray] = {}
        inv_sqrt2 = 1.0 / np.sqrt(2.0)
        for mode in sorted(set(self.feature_output_modes)):
            if mode == "legacy_difference":
                self.mode_matrices[mode] = self.D
            elif mode == "difference":
                self.mode_matrices[mode] = inv_sqrt2 * self.D
            elif mode == "common":
                self.mode_matrices[mode] = inv_sqrt2 * np.abs(self.D)
            else:
                raise RuntimeError(f"Unknown feature output mode {mode!r}.")
        self.mode_indices = {
            mode: np.asarray(
                [ell for ell, value in enumerate(self.feature_output_modes) if value == mode],
                dtype=np.int64,
            )
            for mode in self.mode_matrices
        }

        self.active_indices = tuple(
            np.flatnonzero(self.stage >= (k + 1)) for k in range(self.K)
        )
        self.edge_nodes = tuple(
            np.flatnonzero(
                np.any(np.abs(self.output_directions[:, m, :]) > 1.0e-14, axis=1)
            )
            for m in range(self.M)
        )
        self.edge_values = tuple(
            self.output_directions[self.edge_nodes[m], m, :].copy()
            for m in range(self.M)
        )  # each is (n_endpoint_nodes, L)
        self._projected_block_cache: dict[GroupLabel, FloatArray] = {}
        self._projected_gram_cache: dict[GroupLabel, FloatArray] = {}

    def _validate_label(self, label: GroupLabel) -> tuple[int, int]:
        k, m = int(label[0]), int(label[1])
        if not (0 <= k < self.K and 0 <= m < self.M):
            raise KeyError(f"Unknown temporal group {(k, m)!r}.")
        return k, m

    def raw_sparse_group_block(self, label: GroupLabel) -> sp.csc_matrix:
        k, m = self._validate_label(label)
        times = self.active_indices[k]
        nodes = self.edge_nodes[m]
        values = self.edge_values[m]  # (nodes,L)
        nnz_per_col = int(times.size * nodes.size)

        data = np.empty(nnz_per_col * self.L, dtype=float)
        index_dtype = np.int32 if self.n_rows < np.iinfo(np.int32).max else np.int64
        indices = np.empty(nnz_per_col * self.L, dtype=index_dtype)
        indptr = np.arange(
            0,
            (self.L + 1) * nnz_per_col,
            nnz_per_col,
            dtype=np.int64,
        )
        rows = (times[:, None] * self.N + nodes[None, :]).reshape(-1)
        for ell in range(self.L):
            a = ell * nnz_per_col
            b = a + nnz_per_col
            indices[a:b] = rows
            data[a:b] = (
                self.edge_features[times, m, ell, None] * values[None, :, ell]
            ).reshape(-1)

        return sp.csc_matrix(
            (data, indices, indptr),
            shape=(self.n_rows, self.L),
        )

    def projected_group_block(self, label: GroupLabel) -> FloatArray:
        label = self._validate_label(label)
        cached = self._projected_block_cache.get(label)
        if cached is not None:
            return cached
        raw = self.raw_sparse_group_block(label).toarray()
        projected = self.projector.project_dense_block(raw)
        self._projected_block_cache[label] = projected
        return projected

    def projected_self_gram(self, label: GroupLabel) -> FloatArray:
        label = self._validate_label(label)
        cached = self._projected_gram_cache.get(label)
        if cached is not None:
            return cached
        raw = self.raw_sparse_group_block(label)
        H = self.projector.projected_gram_from_sparse_block(raw)
        self._projected_gram_cache[label] = H
        return H

    @staticmethod
    def _gain_from_gram(c: FloatArray, H: FloatArray) -> float:
        H = 0.5 * (H + H.T)
        eig, vec = np.linalg.eigh(H)
        if eig.size == 0:
            return 0.0
        scale = max(float(np.max(np.abs(eig))), 1.0e-300)
        tol = 1.0e3 * np.finfo(float).eps * scale
        keep = eig > tol
        if not np.any(keep):
            return 0.0
        coordinates = vec[:, keep].T @ c
        return float(np.sum((coordinates * coordinates) / eig[keep]))

    def _stage_feature_gradient(self, Vnode: FloatArray, ids: IntArray) -> FloatArray:
        """Return feature-resolved edge gradients for one stage."""

        G = np.zeros((self.M, self.L), dtype=float)
        if not ids.size:
            return G
        psi = self.edge_features[ids]
        for mode, H_mode in self.mode_matrices.items():
            ell = self.mode_indices[mode]
            if not ell.size:
                continue
            edge_signal = Vnode[ids] @ H_mode
            tmp = np.einsum(
                "tm,tml->ml",
                edge_signal,
                psi,
                optimize=True,
            )
            G[:, ell] = tmp[:, ell]
        return G

    def group_adjoint_norm_provider(
        self,
        vector: FloatArray,
        labels: tuple[GroupLabel, ...],
    ) -> Mapping[GroupLabel, float]:
        """Return ``||A_g^T vector||_2`` for requested temporal edge groups.

        Because the vector and every projected working-set column lie in the
        complement of the stationary anchor, ``A_g^T v = X_g^T v``.  The
        feature-wise output modes therefore require only a small number of
        batched node-to-edge contractions (legacy, common and/or difference),
        followed by the same temporal suffix accumulation as before.
        """

        labels = tuple(self._validate_label(label) for label in labels)
        if not labels:
            return {}
        v = np.asarray(vector, dtype=float).reshape(-1)
        if v.size != self.n_rows:
            raise ValueError("adjoint vector has incompatible length.")
        Vnode = v.reshape(self.T, self.N)
        R = self.K + 1

        stage_grad: list[FloatArray] = []
        for r in range(R):
            ids = np.flatnonzero(self.stage == r)
            stage_grad.append(self._stage_feature_gradient(Vnode, ids))

        suffix_after: list[FloatArray] = [
            np.zeros((self.M, self.L), dtype=float) for _ in range(self.K)
        ]
        suffix = np.zeros((self.M, self.L), dtype=float)
        for r in range(R - 1, 0, -1):
            suffix += stage_grad[r]
            suffix_after[r - 1] = suffix.copy()

        return {
            label: float(np.linalg.norm(suffix_after[label[0]][label[1], :]))
            for label in labels
        }

    def conditional_gain_provider(
        self,
        residual: FloatArray,
        Q: FloatArray,
        omitted: tuple[GroupLabel, ...],
    ) -> tuple[GroupLabel, float]:
        """Legacy exact conditional-gain scan for feature-resolved outputs."""

        residual = np.asarray(residual, dtype=float).reshape(self.T, self.N)
        Q = np.asarray(Q, dtype=float)
        qdim = int(Q.shape[1])
        Q3 = Q.reshape(self.T, self.N, qdim) if qdim else None

        best_label: Optional[GroupLabel] = None
        best_gain = -np.inf

        for raw_label in omitted:
            k, m = self._validate_label(raw_label)
            times = self.active_indices[k]
            nodes = self.edge_nodes[m]
            values = self.edge_values[m]  # (nodes,L)
            psi = self.edge_features[times, m, :]  # (times,L)

            local_node_residual = residual[np.ix_(times, nodes)]
            local_signal = local_node_residual @ values  # (times,L)
            c = np.sum(psi * local_signal, axis=0)

            H = self.projected_self_gram((k, m)).copy()
            if qdim:
                Q_local = Q3[times][:, nodes, :]  # (times,nodes,q)
                # Each feature has its own endpoint-output direction.
                local_Q = np.einsum(
                    "tnq,nl->tlq",
                    Q_local,
                    values,
                    optimize=True,
                )
                V = np.einsum(
                    "tl,tlq->lq",
                    psi,
                    local_Q,
                    optimize=True,
                )
                H -= V @ V.T

            gain = self._gain_from_gram(c, H)
            if gain > best_gain:
                best_gain = gain
                best_label = (k, m)

        if best_label is None:
            raise RuntimeError("Conditional gain scan received no candidate groups.")
        return best_label, float(best_gain)



class _LazyProjectedEndpointChangeOperator:
    """Lazy cumulative operator for generalized endpoint-output edge features."""

    def __init__(
        self,
        D: FloatArray,
        endpoint_features: FloatArray,
        stage: IntArray,
        projector: _SparseBaselineProjector,
    ):
        self.D = np.asarray(D, dtype=float)
        self.endpoint_features = np.asarray(endpoint_features, dtype=float)
        self.stage = np.asarray(stage, dtype=np.int64)
        self.projector = projector
        self.T, self.M, self.L, two = self.endpoint_features.shape
        if two != 2:
            raise ValueError("endpoint feature tensor must have final dimension 2.")
        self.N = self.D.shape[0]
        self.K = int(np.max(self.stage))
        self.n_rows = int(self.T * self.N)
        self.tail, self.head = _pair_endpoints_from_incidence(self.D)
        self.active_indices = tuple(
            np.flatnonzero(self.stage >= (k + 1)) for k in range(self.K)
        )
        self._projected_block_cache: dict[GroupLabel, FloatArray] = {}
        self._projected_gram_cache: dict[GroupLabel, FloatArray] = {}

    def _validate_label(self, label: GroupLabel) -> tuple[int, int]:
        k, m = int(label[0]), int(label[1])
        if not (0 <= k < self.K and 0 <= m < self.M):
            raise KeyError(f"Unknown temporal group {(k, m)!r}.")
        return k, m

    def raw_sparse_group_block(self, label: GroupLabel) -> sp.csc_matrix:
        k, m = self._validate_label(label)
        times = self.active_indices[k]
        nnz_per_col = int(2 * times.size)
        data = np.empty(nnz_per_col * self.L, dtype=float)
        index_dtype = np.int32 if self.n_rows < np.iinfo(np.int32).max else np.int64
        indices = np.empty(nnz_per_col * self.L, dtype=index_dtype)
        indptr = np.arange(
            0, (self.L + 1) * nnz_per_col, nnz_per_col, dtype=np.int64
        )
        rows = np.column_stack(
            [times * self.N + self.tail[m], times * self.N + self.head[m]]
        ).reshape(-1)
        for ell in range(self.L):
            a = ell * nnz_per_col
            b = a + nnz_per_col
            indices[a:b] = rows
            data[a:b] = self.endpoint_features[times, m, ell, :].reshape(-1)
        return sp.csc_matrix(
            (data, indices, indptr), shape=(self.n_rows, self.L)
        )

    def projected_group_block(self, label: GroupLabel) -> FloatArray:
        label = self._validate_label(label)
        cached = self._projected_block_cache.get(label)
        if cached is not None:
            return cached
        raw = self.raw_sparse_group_block(label).toarray()
        projected = self.projector.project_dense_block(raw)
        self._projected_block_cache[label] = projected
        return projected

    def projected_self_gram(self, label: GroupLabel) -> FloatArray:
        label = self._validate_label(label)
        cached = self._projected_gram_cache.get(label)
        if cached is not None:
            return cached
        raw = self.raw_sparse_group_block(label)
        H = self.projector.projected_gram_from_sparse_block(raw)
        self._projected_gram_cache[label] = H
        return H

    def _stage_feature_gradient(self, Vnode: FloatArray, ids: IntArray) -> FloatArray:
        if not ids.size:
            return np.zeros((self.M, self.L), dtype=float)
        Et = self.endpoint_features[ids, :, :, 0]
        Eh = self.endpoint_features[ids, :, :, 1]
        vt = Vnode[ids][:, self.tail]
        vh = Vnode[ids][:, self.head]
        return (
            np.einsum("tm,tml->ml", vt, Et, optimize=True)
            + np.einsum("tm,tml->ml", vh, Eh, optimize=True)
        )

    def group_adjoint_norm_provider(
        self,
        vector: FloatArray,
        labels: tuple[GroupLabel, ...],
    ) -> Mapping[GroupLabel, float]:
        labels = tuple(self._validate_label(label) for label in labels)
        if not labels:
            return {}
        v = np.asarray(vector, dtype=float).reshape(-1)
        if v.size != self.n_rows:
            raise ValueError("adjoint vector has incompatible length.")
        Vnode = v.reshape(self.T, self.N)
        R = self.K + 1
        stage_grad = [
            self._stage_feature_gradient(Vnode, np.flatnonzero(self.stage == r))
            for r in range(R)
        ]
        suffix_after = [
            np.zeros((self.M, self.L), dtype=float) for _ in range(self.K)
        ]
        suffix = np.zeros((self.M, self.L), dtype=float)
        for r in range(R - 1, 0, -1):
            suffix += stage_grad[r]
            suffix_after[r - 1] = suffix.copy()
        return {
            label: float(np.linalg.norm(suffix_after[label[0]][label[1], :]))
            for label in labels
        }

    def conditional_gain_provider(
        self,
        residual: FloatArray,
        Q: FloatArray,
        omitted: tuple[GroupLabel, ...],
    ) -> tuple[GroupLabel, float]:
        residual = np.asarray(residual, dtype=float).reshape(self.T, self.N)
        Q = np.asarray(Q, dtype=float)
        qdim = int(Q.shape[1])
        Q3 = Q.reshape(self.T, self.N, qdim) if qdim else None
        best_label: Optional[GroupLabel] = None
        best_gain = -np.inf
        for raw_label in omitted:
            k, m = self._validate_label(raw_label)
            times = self.active_indices[k]
            Et = self.endpoint_features[times, m, :, 0]
            Eh = self.endpoint_features[times, m, :, 1]
            rt = residual[times, self.tail[m]][:, None]
            rh = residual[times, self.head[m]][:, None]
            c = np.sum(rt * Et + rh * Eh, axis=0)

            H = self.projected_self_gram((k, m)).copy()
            if qdim:
                Qt = Q3[times, self.tail[m], :]
                Qh = Q3[times, self.head[m], :]
                V = (
                    np.einsum("tl,tq->lq", Et, Qt, optimize=True)
                    + np.einsum("tl,tq->lq", Eh, Qh, optimize=True)
                )
                H -= V @ V.T
            gain = _LazyProjectedChangeOperator._gain_from_gram(c, H)
            if gain > best_gain:
                best_gain = gain
                best_label = (k, m)
        if best_label is None:
            raise RuntimeError("Conditional gain scan received no candidate groups.")
        return best_label, float(best_gain)

def _estimate_explicit_dense_design_bytes(
    *,
    n_samples: int,
    n_nodes: int,
    n_edges: int,
    n_edge_features: int,
    n_transitions: int,
) -> int:
    """Conservative memory estimate for the current explicit dense path."""

    n_rows = int(n_samples * n_nodes)
    anchor_cols = int(n_edges * n_edge_features)
    change_cols = int(n_transitions * anchor_cols)
    # pair/anchor storage + X_change + projected X_change, ignoring SVD workspace.
    return int(8 * n_rows * (2 * anchor_cols + 2 * change_cols))


def build_scalable_cumulative_change_design(
    Y: ArrayLike,
    D: ArrayLike,
    edge_features: ArrayLike,
    stage_of_sample: Sequence[int],
    *,
    feature_output_modes: Optional[Sequence[str]] = None,
    stationary_nuisance: Optional[ArrayLike] = None,
    projection_tol: float = 1.0e-16,
    projection_max_iter: Optional[int] = None,
    diagnostic_dense_bytes: int = 64 * 1024**2,
    projection_certificate_tol: float = 1.0e-12,
    projection_refinement_passes: int = 2,
    require_projected_gram: bool = True,
) -> ScalableCumulativeChangeDesign:
    """Build the lazy/sparse Step-2 design used by ``backend='scalable'``."""

    Y, D, edge_features, stage, K = _validate_observations(
        Y, D, edge_features, stage_of_sample
    )
    T, N = Y.shape
    M = D.shape[1]
    L = edge_features.shape[2]
    if edge_features.ndim == 4:
        if feature_output_modes is not None:
            raise ValueError(
                "feature_output_modes must be omitted when edge_features already "
                "contains endpoint outputs with shape (T,M,L,2)."
            )
        output_modes = tuple("endpoint_pair" for _ in range(L))
        feature_representation = "endpoint_pairwise"
    else:
        output_modes = _resolve_feature_output_modes(L, feature_output_modes)
        feature_representation = "scalar_mode_resolved"

    X_anchor = _build_sparse_anchor_design(D, edge_features, output_modes)
    nuisance = _coerce_stationary_nuisance(
        stationary_nuisance,
        n_samples=T,
        n_nodes=N,
    )
    if nuisance is None:
        X_base = X_anchor
    else:
        X_base = sp.hstack(
            [X_anchor, sp.csc_matrix(nuisance)],
            format="csc",
        )

    projector = _SparseBaselineProjector(
        X_base,
        projection_tol=projection_tol,
        projection_max_iter=projection_max_iter,
        diagnostic_dense_bytes=diagnostic_dense_bytes,
        projection_certificate_tol=projection_certificate_tol,
        projection_refinement_passes=projection_refinement_passes,
        require_projected_gram=require_projected_gram,
    )

    y = Y.reshape(T * N)
    y_perp = projector.project_vector(y)

    group_columns: dict[GroupLabel, IntArray] = {}
    group_order: list[GroupLabel] = []
    for k in range(K):
        for m in range(M):
            start = (k * M + m) * L
            label = (int(k), int(m))
            group_columns[label] = start + np.arange(L, dtype=np.int64)
            group_order.append(label)

    if edge_features.ndim == 4:
        operator = _LazyProjectedEndpointChangeOperator(
            D, edge_features, stage, projector
        )
    else:
        operator = _LazyProjectedChangeOperator(
            D, edge_features, stage, projector, output_modes
        )

    return ScalableCumulativeChangeDesign(
        y=np.asarray(y, dtype=float),
        y_perp=np.asarray(y_perp, dtype=float),
        group_columns=group_columns,
        group_order=tuple(group_order),
        baseline_rank=int(projector.rank),
        baseline_singular_values=np.asarray(projector.singular_values, dtype=float),
        baseline_rank_tolerance=float(projector.rank_tolerance),
        baseline_weakest_retained_relative_singular=float(
            projector.weakest_relative_singular
        ),
        n_samples=int(T),
        n_nodes=int(N),
        n_edges=int(M),
        n_edge_features=int(L),
        feature_output_modes=tuple(output_modes),
        feature_representation=str(feature_representation),
        n_transitions=int(K),
        n_base_columns=int(X_base.shape[1]),
        anchor_nnz=int(X_anchor.nnz),
        estimated_explicit_dense_bytes=_estimate_explicit_dense_design_bytes(
            n_samples=T,
            n_nodes=N,
            n_edges=M,
            n_edge_features=L,
            n_transitions=K,
        ),
        baseline_projection_backend=str(projector.projection_backend),
        _operator=operator,
    )

@dataclass(frozen=True)
class SparseRowConstraint:
    """Rows of one ``Delta B^(k)`` block admitted by Step 2.

    ``projected_coefficients`` are the unpenalised coefficients from the final
    floor-certified projected refit.  They are diagnostics only; Step 3 refits
    the original unprojected cumulative model from scratch.
    """

    transition_ordinal: int
    transition_index: int
    transition_time: float
    support: IntArray
    support_labels: tuple[Hashable, ...]
    projected_coefficients: FloatArray


@dataclass(frozen=True)
class PrefixRefitPoint:
    """One nested-prefix support certification fit."""

    prefix_size: int
    added_group: GroupLabel
    groups: tuple[GroupLabel, ...]
    relative_residual: float
    rank: Optional[int]
    identifiable: Optional[bool]
    condition_number_scaled: float
    condition_number_raw: float


@dataclass(frozen=True)
class RankedPrefixSelectionResult:
    """Step-2 support selected from a convex group ranking.

    The first prefix whose unpenalised refit reaches the uncertainty floor is
    selected.  ``coefficients`` are the final projected-refit coefficients in
    the full change-coordinate vector, with zeros outside the selected groups.
    """

    coefficients: FloatArray
    selected_groups: tuple[GroupLabel, ...]
    selected_prefix_size: int
    relative_residual: float
    post_rank: Optional[int]
    post_identifiable: Optional[bool]
    post_condition_number_scaled: float
    post_condition_number_raw: float
    uncertainty_floor: float
    floor_reached: bool
    stop_reason: str
    prefix_path: tuple[PrefixRefitPoint, ...]
    selection_method: str = "group-bpdn-ranking-prefix-floor"


SparseSolverResult = (
    GroupBasisPursuitResult
    | ScreenedGroupBasisPursuitResult
    | WorkingSetGroupBasisPursuitResult
    | AdaptiveGroupLassoResult
    | ForwardBackwardGroupSearchResult
)


@dataclass(frozen=True)
class ChangeStructureResult:
    """Output of TIDES Step 2."""

    hypothesis: str
    constraints: tuple[SparseRowConstraint, ...]
    selected_groups: tuple[GroupLabel, ...]
    diagnostic_delta_B: FloatArray
    relative_residual: float
    solver_result: SparseSolverResult
    selection_result: Optional[RankedPrefixSelectionResult]
    metadata: dict
    design: Optional[CumulativeChangeDesign | "ScalableCumulativeChangeDesign"] = None

    @property
    def supports(self) -> tuple[IntArray, ...]:
        """Selected changed-edge row indices, one array per transition."""

        return tuple(c.support for c in self.constraints)

def _validate_observations(
    Y: ArrayLike,
    D: ArrayLike,
    edge_features: ArrayLike,
    stage_of_sample: Sequence[int],
) -> tuple[FloatArray, FloatArray, FloatArray, IntArray, int]:
    Y = np.asarray(Y, dtype=float)
    D = np.asarray(D, dtype=float)
    edge_features = np.asarray(edge_features, dtype=float)
    stage = np.asarray(stage_of_sample, dtype=np.int64).reshape(-1)

    if Y.ndim != 2:
        raise ValueError("Y must have shape (n_observations, n_nodes).")
    if D.ndim != 2:
        raise ValueError("D must have shape (n_nodes, n_candidate_edges).")
    if edge_features.ndim not in {3, 4}:
        raise ValueError(
            "edge_features must have shape (T,M,L) for scalar/mode-resolved "
            "features or (T,M,L,2) for generalized endpoint-output features."
        )
    if edge_features.ndim == 4 and edge_features.shape[3] != 2:
        raise ValueError(
            "General endpoint-output edge_features must have final dimension 2 "
            "ordered as (tail contribution, head contribution)."
        )

    T, N = Y.shape
    if D.shape[0] != N:
        raise ValueError("D.shape[0] must equal the node dimension of Y.")
    M = D.shape[1]
    if edge_features.shape[:2] != (T, M):
        raise ValueError(
            "edge_features.shape[:2] must equal (Y.shape[0], D.shape[1])."
        )
    if stage.size != T:
        raise ValueError("stage_of_sample must have one entry per observation.")
    if T == 0 or N == 0 or M == 0 or edge_features.shape[2] == 0:
        raise ValueError("Y, D, and edge_features must be non-empty.")
    if not np.all(np.isfinite(Y)):
        raise ValueError("Y must be finite.")
    if not np.all(np.isfinite(D)):
        raise ValueError("D must be finite.")
    if not np.all(np.isfinite(edge_features)):
        raise ValueError("edge_features must be finite.")
    if np.any(stage < 0):
        raise ValueError("stage_of_sample must contain non-negative stage labels.")

    if edge_features.ndim == 4:
        _validate_standard_pairwise_incidence(D)

    unique = np.unique(stage)
    if unique.size == 0 or unique[0] != 0:
        raise ValueError("stage_of_sample must start at stage 0.")
    expected = np.arange(int(unique[-1]) + 1, dtype=np.int64)
    if not np.array_equal(unique, expected):
        raise ValueError(
            "stage_of_sample must use contiguous labels 0,1,...,R-1."
        )

    n_transitions = int(unique[-1])
    return Y, D, edge_features, stage, n_transitions


def _coerce_stationary_nuisance(
    stationary_nuisance: Optional[ArrayLike],
    *,
    n_samples: int,
    n_nodes: int,
) -> Optional[FloatArray]:
    if stationary_nuisance is None:
        return None

    A = np.asarray(stationary_nuisance, dtype=float)
    n_rows = n_samples * n_nodes

    if A.ndim == 2 and A.shape[0] == n_rows:
        out = A
    elif A.ndim == 3 and A.shape[:2] == (n_samples, n_nodes):
        out = A.reshape(n_rows, A.shape[2])
    else:
        raise ValueError(
            "stationary_nuisance must have shape (T*N, P) or (T, N, P)."
        )

    if not np.all(np.isfinite(out)):
        raise ValueError("stationary_nuisance must be finite.")
    return np.asarray(out, dtype=float)


def _scaled_column_space_basis(
    A: FloatArray,
    *,
    zero_column_tol: float = 1e-14,
) -> tuple[FloatArray, int, FloatArray, float, float]:
    """Numerically stable orthonormal basis for ``col(A)``.

    Columns are first norm-scaled.  Scaling changes conditioning but not the
    column space.  Machine-rank SVD is then used to avoid assuming that every
    stationary nuisance direction is independently identifiable.
    """

    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise ValueError("A must be two-dimensional.")
    if A.shape[1] == 0:
        return (
            np.zeros((A.shape[0], 0), dtype=float),
            0,
            np.empty(0, dtype=float),
            0.0,
            np.nan,
        )

    scale = np.linalg.norm(A, axis=0)
    good = scale > zero_column_tol
    if not np.any(good):
        return (
            np.zeros((A.shape[0], 0), dtype=float),
            0,
            np.empty(0, dtype=float),
            0.0,
            np.nan,
        )

    As = A[:, good] / scale[good][None, :]
    U, singular, _ = np.linalg.svd(As, full_matrices=False)

    if singular.size == 0 or singular[0] <= 0.0:
        return (
            np.zeros((A.shape[0], 0), dtype=float),
            0,
            singular,
            0.0,
            np.nan,
        )

    rank_tol = float(
        np.finfo(float).eps * max(As.shape) * singular[0]
    )
    rank = int(np.sum(singular > rank_tol))
    Q = U[:, :rank]
    weakest = (
        float(singular[rank - 1] / singular[0])
        if rank > 0
        else np.nan
    )
    return Q, rank, singular, rank_tol, weakest


def _project_out(Q: FloatArray, A: FloatArray) -> FloatArray:
    A = np.asarray(A, dtype=float)
    if Q.shape[1] == 0:
        return A.copy()
    return A - Q @ (Q.T @ A)


def build_cumulative_change_design(
    Y: ArrayLike,
    D: ArrayLike,
    edge_features: ArrayLike,
    stage_of_sample: Sequence[int],
    *,
    feature_output_modes: Optional[Sequence[str]] = None,
    stationary_nuisance: Optional[ArrayLike] = None,
) -> CumulativeChangeDesign:
    """Build the blind global cumulative Step-2 regression.

    Parameters
    ----------
    Y
        Preprocessed node-space vector-field observations, shape ``(T, N)``.
        Any known intrinsic term must already have been removed.
    D
        Edge-to-node aggregation/incidence matrix, shape ``(N, M)``.
    edge_features
        Edge-function responses, shape ``(T, M, L)``.  Entry ``[t,m,l]`` is
        candidate feature ``l`` for edge ``m`` at observation ``t``.
    stage_of_sample
        Detected stage label for each observation, using contiguous integers
        ``0,...,K``.  This must come from Step 1 / preprocessing; no oracle
        topology information enters here.
    stationary_nuisance
        Optional extra *stationary* nuisance columns, either ``(T*N,P)`` or
        ``(T,N,P)``.  They are appended to the unrestricted anchor before the
        common projection.
    """

    Y, D, edge_features, stage, K = _validate_observations(
        Y, D, edge_features, stage_of_sample
    )
    T, N = Y.shape
    M = D.shape[1]
    L = edge_features.shape[2]
    if edge_features.ndim == 4:
        if feature_output_modes is not None:
            raise ValueError(
                "feature_output_modes must be omitted when edge_features already "
                "contains endpoint outputs with shape (T,M,L,2)."
            )
        output_modes = tuple("endpoint_pair" for _ in range(L))
        feature_representation = "endpoint_pairwise"
        tail, head = _pair_endpoints_from_incidence(D)
        pair_basis = np.zeros((T, N, M, L), dtype=float)
        for m in range(M):
            pair_basis[:, tail[m], m, :] = edge_features[:, m, :, 0]
            pair_basis[:, head[m], m, :] = edge_features[:, m, :, 1]
    else:
        output_modes = _resolve_feature_output_modes(L, feature_output_modes)
        feature_representation = "scalar_mode_resolved"
        # Historical calls (feature_output_modes=None) use D[:,m] exactly.
        # General orientation-equivariant scalar calls use |D|/sqrt(2) for
        # common/even features and D/sqrt(2) for difference/odd features.
        output_directions = _feature_output_directions(D, output_modes)
        pair_basis = output_directions[None, :, :, :] * edge_features[:, None, :, :]

    y = Y.reshape(T * N)
    X_anchor = pair_basis.reshape(T * N, M * L)

    nuisance = _coerce_stationary_nuisance(
        stationary_nuisance,
        n_samples=T,
        n_nodes=N,
    )
    if nuisance is None:
        X_base = X_anchor
    else:
        X_base = np.hstack([X_anchor, nuisance])

    # Cumulative temporal dictionary.  Transition k affects every stage r>k.
    # Column ordering is (k, m, l), with l the fastest index.
    X_change = np.zeros((T * N, K * M * L), dtype=float)
    group_columns: dict[GroupLabel, IntArray] = {}
    group_order: list[GroupLabel] = []

    for k in range(K):
        active = stage >= (k + 1)
        block = np.zeros_like(pair_basis)
        block[active] = pair_basis[active]
        block2d = block.reshape(T * N, M * L)

        start = k * M * L
        stop = start + M * L
        X_change[:, start:stop] = block2d

        for m in range(M):
            cols = start + m * L + np.arange(L, dtype=np.int64)
            label = (int(k), int(m))
            group_columns[label] = cols
            group_order.append(label)

    Q_base, rank_base, s_base, rank_tol, weakest = _scaled_column_space_basis(
        X_base
    )
    y_perp = _project_out(Q_base, y[:, None]).ravel()
    X_change_perp = _project_out(Q_base, X_change)

    return CumulativeChangeDesign(
        y=y,
        X_base=X_base,
        X_change=X_change,
        y_perp=y_perp,
        X_change_perp=X_change_perp,
        group_columns=group_columns,
        group_order=tuple(group_order),
        baseline_rank=int(rank_base),
        baseline_singular_values=np.asarray(s_base, dtype=float),
        baseline_rank_tolerance=float(rank_tol),
        baseline_weakest_retained_relative_singular=float(weakest),
        n_samples=int(T),
        n_nodes=int(N),
        n_edges=int(M),
        n_edge_features=int(L),
        feature_output_modes=tuple(output_modes),
        feature_representation=str(feature_representation),
        n_transitions=int(K),
    )


# -----------------------------------------------------------------------------
# Varying-structure inference
# -----------------------------------------------------------------------------


def _coerce_transition_metadata(
    n_transitions: int,
    transition_indices: Optional[Sequence[int]],
    transition_times: Optional[Sequence[float]],
) -> tuple[IntArray, FloatArray]:
    K = int(n_transitions)

    if transition_indices is None:
        indices = np.arange(K, dtype=np.int64)
    else:
        indices = np.asarray(transition_indices, dtype=np.int64).reshape(-1)
        if indices.size != K:
            raise ValueError(
                "transition_indices must have one entry per detected transition."
            )
        if indices.size and not np.all(np.diff(indices) > 0):
            raise ValueError("transition_indices must be strictly increasing.")

    if transition_times is None:
        times = np.full(K, np.nan, dtype=float)
    else:
        times = np.asarray(transition_times, dtype=float).reshape(-1)
        if times.size != K:
            raise ValueError(
                "transition_times must have one entry per detected transition."
            )
        if not np.all(np.isfinite(times)):
            raise ValueError("transition_times must be finite when supplied.")
        if times.size and not np.all(np.diff(times) > 0):
            raise ValueError("transition_times must be strictly increasing.")

    return indices, times



# -----------------------------------------------------------------------------
# Varying-structure inference
# -----------------------------------------------------------------------------


def _resolve_uncertainty_floor(
    *,
    uncertainty_floor: Optional[float],
    profile_floor: Optional[float],
) -> Optional[float]:
    """Resolve the new general name while preserving notebook compatibility."""

    if uncertainty_floor is not None:
        uncertainty_floor = float(uncertainty_floor)
        if not np.isfinite(uncertainty_floor) or uncertainty_floor < 0.0:
            raise ValueError("uncertainty_floor must be finite and non-negative.")

    if profile_floor is not None:
        profile_floor = float(profile_floor)
        if not np.isfinite(profile_floor) or profile_floor < 0.0:
            raise ValueError("profile_floor must be finite and non-negative.")

    if uncertainty_floor is not None and profile_floor is not None:
        if not np.isclose(
            uncertainty_floor,
            profile_floor,
            rtol=1e-12,
            atol=0.0,
        ):
            raise ValueError(
                "uncertainty_floor and profile_floor were both supplied with "
                "different values. Supply only one uncertainty budget."
            )
        return uncertainty_floor

    return uncertainty_floor if uncertainty_floor is not None else profile_floor


def _prefix_columns(
    groups: Sequence[GroupLabel],
    group_columns: Mapping[GroupLabel, IntArray],
) -> IntArray:
    if not groups:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(
        [np.asarray(group_columns[g], dtype=np.int64) for g in groups]
    )


def _certify_ranked_prefixes(
    design: CumulativeChangeDesign,
    ranking: Sequence[GroupLabel],
    *,
    uncertainty_floor: float,
    refit_method: Literal["dense_lstsq", "lsqr", "auto"] = "dense_lstsq",
    refit_kwargs: Optional[Mapping[str, object]] = None,
    max_prefix_groups: Optional[int] = None,
    verbose: bool = True,
) -> RankedPrefixSelectionResult:
    """Select the smallest ranked prefix that reaches the uncertainty floor."""

    ranking = tuple((int(k), int(m)) for k, m in ranking)
    if len(ranking) != len(design.group_order):
        raise ValueError("ranking must contain every candidate group exactly once.")
    if set(ranking) != set(design.group_order):
        raise ValueError("ranking is not a permutation of the candidate groups.")

    floor = float(uncertainty_floor)
    if not np.isfinite(floor) or floor < 0.0:
        raise ValueError("uncertainty_floor must be finite and non-negative.")

    if max_prefix_groups is None:
        prefix_cap = len(ranking)
    else:
        prefix_cap = int(min(max_prefix_groups, len(ranking)))
        if prefix_cap < 1:
            raise ValueError("max_prefix_groups must be >= 1 when supplied.")

    kwargs = {} if refit_kwargs is None else dict(refit_kwargs)
    forbidden = {"method", "verbose"}.intersection(kwargs)
    if forbidden:
        raise ValueError(
            "Pass refit method/verbosity through Step-2 arguments, not "
            f"refit_kwargs keys {sorted(forbidden)!r}."
        )

    # Prefixes are nested, so the optimal residual is non-increasing in prefix
    # size.  We therefore stop at the first feasible prefix; no support-size
    # oracle or coefficient threshold enters the decision.
    path: list[PrefixRefitPoint] = []
    selected_groups: tuple[GroupLabel, ...] = tuple()
    selected_fit: Optional[LinearRegressionResult] = None
    selected_cols = np.empty(0, dtype=np.int64)

    comparison_slack = max(
        100.0 * np.finfo(float).eps,
        1.0e-12 * max(floor, np.finfo(float).tiny),
    )

    if verbose:
        print("Starting ranked-prefix support certification...")
        print(
            f"  candidate groups={len(ranking)} | prefix cap={prefix_cap} | "
            f"relative uncertainty floor={floor:.3e}"
        )

    for j in range(1, prefix_cap + 1):
        groups_j = ranking[:j]
        cols_j = _prefix_columns(groups_j, design.group_columns)
        Xj = design.X_change_perp[:, cols_j]

        fit = solve_least_squares(
            Xj,
            design.y_perp,
            method=refit_method,
            verbose=False,
            **kwargs,
        )

        point = PrefixRefitPoint(
            prefix_size=int(j),
            added_group=groups_j[-1],
            groups=tuple(groups_j),
            relative_residual=float(fit.relative_residual),
            rank=None if fit.rank is None else int(fit.rank),
            identifiable=fit.identifiable,
            condition_number_scaled=float(fit.condition_number_scaled),
            condition_number_raw=float(fit.condition_number_raw),
        )
        path.append(point)

        if verbose and (
            j <= 5
            or j % 10 == 0
            or fit.relative_residual <= floor + comparison_slack
        ):
            print(
                f"  [prefix {j:4d}] rel-res={fit.relative_residual:.3e} | "
                f"added={groups_j[-1]!r}"
            )

        if fit.relative_residual <= floor + comparison_slack:
            selected_groups = tuple(groups_j)
            selected_fit = fit
            selected_cols = cols_j
            break

    p = design.X_change_perp.shape[1]
    coefficients = np.zeros(p, dtype=float)

    if selected_fit is None:
        # Preserve the best tested prefix for diagnostics when the caller allows
        # a non-feasible return.  Formal inference normally requests failure.
        if path:
            groups_last = ranking[: path[-1].prefix_size]
            cols_last = _prefix_columns(groups_last, design.group_columns)
            X_last = design.X_change_perp[:, cols_last]
            selected_fit = solve_least_squares(
                X_last,
                design.y_perp,
                method=refit_method,
                verbose=False,
                **kwargs,
            )
            selected_groups = tuple(groups_last)
            selected_cols = cols_last
            coefficients[selected_cols] = np.asarray(
                selected_fit.coefficients,
                dtype=float,
            ).reshape(-1)
            relative_residual = float(selected_fit.relative_residual)
            post_rank = selected_fit.rank
            post_identifiable = selected_fit.identifiable
            cond_scaled = float(selected_fit.condition_number_scaled)
            cond_raw = float(selected_fit.condition_number_raw)
        else:
            relative_residual = 1.0
            post_rank = 0
            post_identifiable = False
            cond_scaled = np.inf
            cond_raw = np.inf

        return RankedPrefixSelectionResult(
            coefficients=coefficients,
            selected_groups=selected_groups,
            selected_prefix_size=int(len(selected_groups)),
            relative_residual=float(relative_residual),
            post_rank=None if post_rank is None else int(post_rank),
            post_identifiable=post_identifiable,
            post_condition_number_scaled=cond_scaled,
            post_condition_number_raw=cond_raw,
            uncertainty_floor=floor,
            floor_reached=False,
            stop_reason=(
                "no ranked prefix reached the supplied uncertainty floor "
                f"within {prefix_cap} groups"
            ),
            prefix_path=tuple(path),
        )

    coefficients[selected_cols] = np.asarray(
        selected_fit.coefficients,
        dtype=float,
    ).reshape(-1)

    result = RankedPrefixSelectionResult(
        coefficients=coefficients,
        selected_groups=selected_groups,
        selected_prefix_size=int(len(selected_groups)),
        relative_residual=float(selected_fit.relative_residual),
        post_rank=None if selected_fit.rank is None else int(selected_fit.rank),
        post_identifiable=selected_fit.identifiable,
        post_condition_number_scaled=float(selected_fit.condition_number_scaled),
        post_condition_number_raw=float(selected_fit.condition_number_raw),
        uncertainty_floor=floor,
        floor_reached=True,
        stop_reason="first ranked prefix reached the supplied uncertainty floor",
        prefix_path=tuple(path),
    )

    if verbose:
        print("Ranked-prefix support certification complete.")
        print(
            f"  selected groups={result.selected_prefix_size} | "
            f"relative residual={result.relative_residual:.3e}"
        )

    return result




def _certify_ranked_prefixes_scalable(
    design: ScalableCumulativeChangeDesign,
    ranking: Sequence[GroupLabel],
    *,
    uncertainty_floor: float,
    refit_method: Literal["dense_lstsq", "lsqr", "auto"] = "dense_lstsq",
    refit_kwargs: Optional[Mapping[str, object]] = None,
    max_prefix_groups: Optional[int] = None,
    verbose: bool = True,
) -> RankedPrefixSelectionResult:
    """Lazy analogue of ``_certify_ranked_prefixes``.

    Only blocks occurring in tested prefixes are materialised, and they are
    cached by the scalable design operator.
    """

    ranking = tuple((int(k), int(m)) for k, m in ranking)
    if len(ranking) != len(design.group_order):
        raise ValueError("ranking must contain every candidate group exactly once.")
    if set(ranking) != set(design.group_order):
        raise ValueError("ranking is not a permutation of the candidate groups.")

    floor = float(uncertainty_floor)
    if not np.isfinite(floor) or floor < 0.0:
        raise ValueError("uncertainty_floor must be finite and non-negative.")

    if max_prefix_groups is None:
        prefix_cap = len(ranking)
    else:
        prefix_cap = int(min(max_prefix_groups, len(ranking)))
        if prefix_cap < 1:
            raise ValueError("max_prefix_groups must be >= 1 when supplied.")

    kwargs = {} if refit_kwargs is None else dict(refit_kwargs)
    forbidden = {"method", "verbose"}.intersection(kwargs)
    if forbidden:
        raise ValueError(
            "Pass refit method/verbosity through Step-2 arguments, not "
            f"refit_kwargs keys {sorted(forbidden)!r}."
        )

    path: list[PrefixRefitPoint] = []
    selected_groups: tuple[GroupLabel, ...] = tuple()
    selected_fit: Optional[LinearRegressionResult] = None
    selected_cols = np.empty(0, dtype=np.int64)

    comparison_slack = max(
        100.0 * np.finfo(float).eps,
        1.0e-12 * max(floor, np.finfo(float).tiny),
    )

    if verbose:
        print("Starting ranked-prefix support certification...")
        print(
            f"  candidate groups={len(ranking)} | prefix cap={prefix_cap} | "
            f"relative uncertainty floor={floor:.3e}"
        )

    blocks: list[FloatArray] = []
    for j in range(1, prefix_cap + 1):
        added = ranking[j - 1]
        blocks.append(design.group_block(added))
        Xj = np.column_stack(blocks)
        groups_j = ranking[:j]
        cols_j = _prefix_columns(groups_j, design.group_columns)

        fit = solve_least_squares(
            Xj,
            design.y_perp,
            method=refit_method,
            verbose=False,
            **kwargs,
        )
        point = PrefixRefitPoint(
            prefix_size=int(j),
            added_group=added,
            groups=tuple(groups_j),
            relative_residual=float(fit.relative_residual),
            rank=None if fit.rank is None else int(fit.rank),
            identifiable=fit.identifiable,
            condition_number_scaled=float(fit.condition_number_scaled),
            condition_number_raw=float(fit.condition_number_raw),
        )
        path.append(point)

        if verbose and (
            j <= 5
            or j % 10 == 0
            or fit.relative_residual <= floor + comparison_slack
        ):
            print(
                f"  [prefix {j:4d}] rel-res={fit.relative_residual:.3e} | "
                f"added={added!r}"
            )

        if fit.relative_residual <= floor + comparison_slack:
            selected_groups = tuple(groups_j)
            selected_fit = fit
            selected_cols = cols_j
            break

    p = design.n_transitions * design.n_edges * design.n_edge_features
    coefficients = np.zeros(p, dtype=float)

    if selected_fit is None:
        if path:
            groups_last = ranking[: path[-1].prefix_size]
            cols_last = _prefix_columns(groups_last, design.group_columns)
            X_last = np.column_stack(
                [design.group_block(g) for g in groups_last]
            )
            selected_fit = solve_least_squares(
                X_last,
                design.y_perp,
                method=refit_method,
                verbose=False,
                **kwargs,
            )
            selected_groups = tuple(groups_last)
            selected_cols = cols_last
            coefficients[selected_cols] = np.asarray(
                selected_fit.coefficients, dtype=float
            ).reshape(-1)
            relative_residual = float(selected_fit.relative_residual)
            post_rank = selected_fit.rank
            post_identifiable = selected_fit.identifiable
            cond_scaled = float(selected_fit.condition_number_scaled)
            cond_raw = float(selected_fit.condition_number_raw)
        else:
            relative_residual = 1.0
            post_rank = 0
            post_identifiable = False
            cond_scaled = np.inf
            cond_raw = np.inf

        return RankedPrefixSelectionResult(
            coefficients=coefficients,
            selected_groups=selected_groups,
            selected_prefix_size=int(len(selected_groups)),
            relative_residual=float(relative_residual),
            post_rank=None if post_rank is None else int(post_rank),
            post_identifiable=post_identifiable,
            post_condition_number_scaled=cond_scaled,
            post_condition_number_raw=cond_raw,
            uncertainty_floor=floor,
            floor_reached=False,
            stop_reason=(
                "no ranked prefix reached the supplied uncertainty floor "
                f"within {prefix_cap} groups"
            ),
            prefix_path=tuple(path),
        )

    coefficients[selected_cols] = np.asarray(
        selected_fit.coefficients, dtype=float
    ).reshape(-1)
    result = RankedPrefixSelectionResult(
        coefficients=coefficients,
        selected_groups=selected_groups,
        selected_prefix_size=int(len(selected_groups)),
        relative_residual=float(selected_fit.relative_residual),
        post_rank=None if selected_fit.rank is None else int(selected_fit.rank),
        post_identifiable=selected_fit.identifiable,
        post_condition_number_scaled=float(selected_fit.condition_number_scaled),
        post_condition_number_raw=float(selected_fit.condition_number_raw),
        uncertainty_floor=floor,
        floor_reached=True,
        stop_reason="first ranked prefix reached the supplied uncertainty floor",
        prefix_path=tuple(path),
    )

    if verbose:
        print("Ranked-prefix support certification complete.")
        print(
            f"  selected groups={result.selected_prefix_size} | "
            f"relative residual={result.relative_residual:.3e}"
        )
    return result

def infer_sparse_row_changes_from_observations(
    Y: ArrayLike,
    D: ArrayLike,
    edge_features: ArrayLike,
    stage_of_sample: Sequence[int],
    *,
    transition_indices: Optional[Sequence[int]] = None,
    transition_times: Optional[Sequence[float]] = None,
    edge_labels: Optional[Sequence[Hashable]] = None,
    stationary_nuisance: Optional[ArrayLike] = None,
    feature_output_modes: Optional[Sequence[str]] = None,
    uncertainty_floor: Optional[float] = None,
    profile_floor: Optional[float] = None,
    solver_method: Literal[
        "group_bpdn_prefix",
        "adaptive_group_lasso",
        "forward_backward_floor",
    ] = "group_bpdn_prefix",
    backend: Literal["dense", "scalable", "auto"] = "dense",
    scalable_solver: Literal["working_set", "screened"] = "working_set",
    dense_memory_limit_bytes: int = 512 * 1024**2,
    scalable_projection_tol: float = 1.0e-16,
    scalable_projection_max_iter: Optional[int] = None,
    solver_kwargs: Optional[Mapping[str, object]] = None,
    refit_method: Literal["dense_lstsq", "lsqr", "auto"] = "dense_lstsq",
    refit_kwargs: Optional[Mapping[str, object]] = None,
    max_prefix_groups: Optional[int] = None,
    require_solver_convergence: bool = True,
    require_floor_reached: bool = True,
    return_design: bool = False,
) -> ChangeStructureResult:
    """Infer row-sparse temporal changes from preprocessed observations.

    ``backend='dense'`` preserves the original correctness/reference path.
    ``backend='scalable'`` uses sparse anchor projection and lazy temporal
    group blocks.  By default ``scalable_solver='working_set'`` uses batched
    seeding plus global dual/KKT reactivation to certify the full grouped-BPDN
    optimum before the same floor-certified prefix selection.  The older
    ``scalable_solver='screened'`` path is retained for regression comparisons.
    ``backend='auto'`` dispatches only from a conservative explicit-design
    memory estimate; it never inspects an inference result to decide the backend.
    """

    floor = _resolve_uncertainty_floor(
        uncertainty_floor=uncertainty_floor,
        profile_floor=profile_floor,
    )
    kwargs = {} if solver_kwargs is None else dict(solver_kwargs)

    # Validate once up front so auto dispatch can estimate dimensions without
    # constructing the explicit dense dictionary.
    Y0, D0, edge0, stage0, K0 = _validate_observations(
        Y, D, edge_features, stage_of_sample
    )
    T0, N0 = Y0.shape
    M0 = D0.shape[1]
    L0 = edge0.shape[2]
    if edge0.ndim == 4:
        if feature_output_modes is not None:
            raise ValueError(
                "feature_output_modes must be omitted for endpoint-output features."
            )
        output_modes0 = None
    else:
        output_modes0 = _resolve_feature_output_modes(L0, feature_output_modes)
    estimated_dense_bytes = _estimate_explicit_dense_design_bytes(
        n_samples=T0,
        n_nodes=N0,
        n_edges=M0,
        n_edge_features=L0,
        n_transitions=K0,
    )

    backend_requested = str(backend)
    if backend_requested not in {"dense", "scalable", "auto"}:
        raise ValueError("backend must be 'dense', 'scalable', or 'auto'.")
    scalable_solver = str(scalable_solver).lower()
    if scalable_solver not in {"working_set", "screened"}:
        raise ValueError("scalable_solver must be 'working_set' or 'screened'.")
    if backend_requested == "auto":
        if solver_method != "group_bpdn_prefix":
            backend_used = "dense"
        else:
            backend_used = (
                "dense"
                if estimated_dense_bytes <= int(dense_memory_limit_bytes)
                else "scalable"
            )
    else:
        backend_used = backend_requested

    if backend_used == "dense":
        design: CumulativeChangeDesign | ScalableCumulativeChangeDesign = (
            build_cumulative_change_design(
                Y0,
                D0,
                edge0,
                stage0,
                feature_output_modes=output_modes0,
                stationary_nuisance=stationary_nuisance,
            )
        )
    else:
        if solver_method != "group_bpdn_prefix":
            raise ValueError(
                "backend='scalable' currently supports solver_method="
                "'group_bpdn_prefix' only. Legacy solvers remain available "
                "through backend='dense'."
            )
        design = build_scalable_cumulative_change_design(
            Y0,
            D0,
            edge0,
            stage0,
            feature_output_modes=output_modes0,
            stationary_nuisance=stationary_nuisance,
            projection_tol=scalable_projection_tol,
            projection_max_iter=scalable_projection_max_iter,
            projection_certificate_tol=max(
                100.0 * np.finfo(float).eps,
                0.05 * float(floor),
            ),
            projection_refinement_passes=2,
            require_projected_gram=(scalable_solver == "screened"),
        )

    K = design.n_transitions
    M = design.n_edges
    L = design.n_edge_features
    if K < 1:
        raise ValueError(
            "Step 2 received no detected transitions. The current change-"
            "support routine requires at least one transition."
        )

    transition_indices_arr, transition_times_arr = _coerce_transition_metadata(
        K, transition_indices, transition_times
    )
    if edge_labels is None:
        labels: tuple[Hashable, ...] = tuple(range(M))
    else:
        labels = tuple(edge_labels)
        if len(labels) != M:
            raise ValueError("edge_labels must have exactly D.shape[1] entries.")

    selection_result: Optional[RankedPrefixSelectionResult] = None

    if solver_method == "group_bpdn_prefix":
        if floor is None:
            raise ValueError(
                "solver_method='group_bpdn_prefix' requires uncertainty_floor "
                "(or the backward-compatible profile_floor alias)."
            )
        forbidden = {"target_relative_residual", "residual_radius"}.intersection(
            kwargs
        )
        if forbidden:
            raise ValueError(
                "The Step-2 uncertainty budget is supplied through "
                "uncertainty_floor/profile_floor, not solver_kwargs keys "
                f"{sorted(forbidden)!r}."
            )

        if backend_used == "dense":
            assert isinstance(design, CumulativeChangeDesign)
            solver_result = solve_group_basis_pursuit_denoising(
                design.X_change_perp,
                design.y_perp,
                design.group_columns,
                target_relative_residual=float(floor),
                **kwargs,
            )
            selection_result = _certify_ranked_prefixes(
                design,
                solver_result.ranking,
                uncertainty_floor=float(floor),
                refit_method=refit_method,
                refit_kwargs=refit_kwargs,
                max_prefix_groups=max_prefix_groups,
                verbose=bool(kwargs.get("verbose", True)),
            )
        else:
            assert isinstance(design, ScalableCumulativeChangeDesign)
            screened_kwargs = dict(kwargs)
            restricted = dict(
                screened_kwargs.pop("restricted_solver_kwargs", {}) or {}
            )
            # Preserve the familiar dense-solver keyword surface in notebooks:
            # numerical DR options are simply forwarded to the restricted solve.
            for key in (
                "max_iter",
                "tol",
                "check_every",
                "douglas_rachford_step",
                "lambda_bisection_iterations",
            ):
                if key in screened_kwargs:
                    restricted[key] = screened_kwargs.pop(key)
            verbose_flag = bool(screened_kwargs.pop("verbose", True))

            if scalable_solver == "working_set":
                # TIDES designs are highly coherent; a constant modest batch is
                # less prone to overshooting the useful working-set size than
                # geometric seed growth.  Callers may override either value.
                screened_kwargs.setdefault("seed_batch_size", 8)
                screened_kwargs.setdefault("seed_growth_factor", 1.0)
                solver_result = solve_working_set_group_basis_pursuit_denoising(
                    design.y_perp,
                    design.group_columns,
                    n_coefficients=K * M * L,
                    group_block_provider=design.group_block,
                    target_relative_residual=float(floor),
                    group_adjoint_norm_provider=design.group_adjoint_norm_provider,
                    restricted_solver_kwargs=restricted,
                    require_restricted_convergence=require_solver_convergence,
                    verbose=verbose_flag,
                    **screened_kwargs,
                )
            else:
                solver_result = solve_screened_group_basis_pursuit_denoising(
                    design.y_perp,
                    design.group_columns,
                    n_coefficients=K * M * L,
                    group_block_provider=design.group_block,
                    target_relative_residual=float(floor),
                    conditional_gain_provider=design.conditional_gain_provider,
                    restricted_solver_kwargs=restricted,
                    require_restricted_convergence=require_solver_convergence,
                    verbose=verbose_flag,
                    **screened_kwargs,
                )
            selection_result = _certify_ranked_prefixes_scalable(
                design,
                solver_result.ranking,
                uncertainty_floor=float(floor),
                refit_method=refit_method,
                refit_kwargs=refit_kwargs,
                max_prefix_groups=max_prefix_groups,
                verbose=verbose_flag,
            )

        if not solver_result.feasible:
            raise RuntimeError(
                "Grouped BPDN returned a solution outside the requested "
                "uncertainty ball."
            )
        if require_solver_convergence and not solver_result.converged:
            raise RuntimeError(
                "Grouped BPDN did not reach its numerical convergence "
                "criterion. Set require_solver_convergence=False only for "
                "diagnostic experiments."
            )
        if require_floor_reached and not selection_result.floor_reached:
            raise RuntimeError(selection_result.stop_reason)

        selected_groups = selection_result.selected_groups
        final_coefficients = selection_result.coefficients
        final_relative_residual = selection_result.relative_residual

    elif solver_method == "adaptive_group_lasso":
        assert isinstance(design, CumulativeChangeDesign)
        if "target_relative_floor" in kwargs:
            raise ValueError(
                "Pass the Step-2 uncertainty floor through uncertainty_floor/"
                "profile_floor, not solver_kwargs['target_relative_floor']."
            )
        solver_result = solve_adaptive_group_lasso(
            design.X_change_perp,
            design.y_perp,
            design.group_columns,
            target_relative_floor=floor,
            **kwargs,
        )
        selected_groups = tuple(
            (int(k), int(m)) for k, m in solver_result.selected_groups
        )
        final_coefficients = np.asarray(
            solver_result.coefficients, dtype=float
        ).copy()
        final_relative_residual = float(solver_result.post_relative_residual)

    elif solver_method == "forward_backward_floor":
        assert isinstance(design, CumulativeChangeDesign)
        if floor is None:
            raise ValueError(
                "solver_method='forward_backward_floor' requires an uncertainty floor."
            )
        if "target_relative_floor" in kwargs:
            raise ValueError(
                "Pass the Step-2 uncertainty floor through uncertainty_floor/"
                "profile_floor, not solver_kwargs['target_relative_floor']."
            )
        solver_result = solve_forward_backward_group_search(
            design.X_change_perp,
            design.y_perp,
            design.group_columns,
            target_relative_floor=float(floor),
            **kwargs,
        )
        selected_groups = tuple(
            (int(k), int(m)) for k, m in solver_result.selected_groups
        )
        final_coefficients = np.asarray(
            solver_result.coefficients, dtype=float
        ).copy()
        final_relative_residual = float(solver_result.post_relative_residual)
    else:
        raise ValueError(f"Unknown sparse solver_method: {solver_method!r}.")

    selected_groups = tuple((int(k), int(m)) for k, m in selected_groups)
    selected_set = set(selected_groups)

    diagnostic_delta_B = np.zeros((K, M, L), dtype=float)
    for label in selected_groups:
        k, m = label
        cols = design.group_columns[label]
        diagnostic_delta_B[k, m, :] = final_coefficients[cols]

    constraints: list[SparseRowConstraint] = []
    for k in range(K):
        support = np.asarray(
            [m for m in range(M) if (k, m) in selected_set], dtype=np.int64
        )
        projected = (
            diagnostic_delta_B[k, support, :].copy()
            if support.size
            else np.zeros((0, L), dtype=float)
        )
        constraints.append(
            SparseRowConstraint(
                transition_ordinal=int(k),
                transition_index=int(transition_indices_arr[k]),
                transition_time=float(transition_times_arr[k]),
                support=support,
                support_labels=tuple(labels[int(m)] for m in support),
                projected_coefficients=projected,
            )
        )

    y_perp_norm = float(np.linalg.norm(design.y_perp))
    n_base_columns = (
        int(design.X_base.shape[1])
        if isinstance(design, CumulativeChangeDesign)
        else int(design.n_base_columns)
    )
    metadata = {
        "backend": f"global-cumulative-{backend_used}-{solver_method}",
        "computational_backend_requested": backend_requested,
        "computational_backend_used": backend_used,
        "scalable_solver": scalable_solver if backend_used == "scalable" else None,
        "oracle_information_used": False,
        "solver_method": str(solver_method),
        "n_preprocessed_observations": int(design.n_samples),
        "n_scalar_rows": int(design.n_samples * design.n_nodes),
        "n_nodes": int(design.n_nodes),
        "n_candidate_edges": int(M),
        "n_edge_features": int(L),
        "feature_output_modes": tuple(design.feature_output_modes),
        "feature_representation": str(design.feature_representation),
        "output_generalization_active": bool(
            design.feature_representation != "scalar_mode_resolved"
            or any(mode != "legacy_difference" for mode in design.feature_output_modes)
        ),
        "n_transitions": int(K),
        "n_candidate_groups": int(K * M),
        "coefficients_per_group": int(L),
        "anchor_edge_coefficients": int(M * L),
        "stationary_columns": int(n_base_columns),
        "baseline_rank": int(design.baseline_rank),
        "baseline_weakest_retained_relative_singular": float(
            design.baseline_weakest_retained_relative_singular
        ),
        "projected_target_norm": y_perp_norm,
        "uncertainty_floor": None if floor is None else float(floor),
        "profile_floor": None if floor is None else float(floor),
        "selected_group_count": int(len(selected_groups)),
        "post_selection_relative_residual": float(final_relative_residual),
        "estimated_explicit_dense_bytes": int(estimated_dense_bytes),
        "global_change_design_materialized": bool(backend_used == "dense"),
    }

    if isinstance(design, ScalableCumulativeChangeDesign):
        metadata.update(
            {
                "baseline_projection_backend": design.baseline_projection_backend,
                "sparse_anchor_nnz": int(design.anchor_nnz),
                "scalable_projection_tol": float(scalable_projection_tol),
                "scalable_projection_certificate_tol": float(
                    design._operator.projector.projection_certificate_tol
                ),
                "max_projection_anchor_cosine": float(
                    design._operator.projector.max_projection_certificate
                ),
                "projection_solve_count": int(
                    design._operator.projector.projection_solve_count
                ),
                "projection_refinement_count": int(
                    design._operator.projector.projection_refinement_count
                ),
                "max_projection_lsqr_iterations": int(
                    design._operator.projector.max_lsqr_iterations
                ),
            }
        )

    if solver_method == "group_bpdn_prefix":
        assert selection_result is not None
        if isinstance(solver_result, GroupBasisPursuitResult):
            convex_iterations = int(solver_result.iterations)
            convex_fixed_point = float(solver_result.fixed_point_residual)
            convex_rank = int(solver_result.design_rank)
            projection_backend = str(solver_result.projection_backend)
            metadata.update(
                {
                    "convex_duality_gap": float(solver_result.duality_gap),
                    "convex_max_dual_group_ratio": float(
                        solver_result.max_dual_group_ratio
                    ),
                    "convex_dual_feasible": bool(solver_result.dual_feasible),
                }
            )
        elif isinstance(solver_result, WorkingSetGroupBasisPursuitResult):
            restricted_result = solver_result.restricted_result
            convex_iterations = int(restricted_result.iterations)
            convex_fixed_point = float(restricted_result.fixed_point_residual)
            convex_rank = int(restricted_result.design_rank)
            projection_backend = str(solver_result.projection_backend)
            metadata.update(
                {
                    "seed_group_count": int(len(solver_result.seed_groups)),
                    "working_set_group_count": int(len(solver_result.working_set)),
                    "seed_scans": int(solver_result.seed_scans),
                    "kkt_audits": int(solver_result.kkt_audits),
                    "working_set_expansions": int(
                        solver_result.working_set_expansions
                    ),
                    "total_reactivations": int(solver_result.total_reactivations),
                    "max_global_dual_ratio": float(
                        solver_result.max_global_dual_ratio
                    ),
                    "global_dual_feasible": bool(
                        solver_result.global_dual_feasible
                    ),
                    "global_duality_gap": float(solver_result.global_duality_gap),
                    "restricted_group_count": int(
                        solver_result.n_restricted_groups
                    ),
                    "peak_restricted_columns": int(
                        solver_result.peak_restricted_columns
                    ),
                }
            )
        else:
            assert isinstance(solver_result, ScreenedGroupBasisPursuitResult)
            restricted_result = solver_result.restricted_result
            convex_iterations = int(restricted_result.iterations)
            convex_fixed_point = float(restricted_result.fixed_point_residual)
            convex_rank = int(restricted_result.design_rank)
            projection_backend = str(solver_result.projection_backend)
            metadata.update(
                {
                    "screening_group_count": int(len(solver_result.screening_groups)),
                    "screening_steps": int(solver_result.screening_steps),
                    "screening_relative_residual": float(
                        solver_result.screening_relative_residual
                    ),
                    "screening_rank": int(solver_result.screening_rank),
                    "screening_condition_number": float(
                        solver_result.screening_condition_number
                    ),
                    "restricted_group_count": int(
                        solver_result.n_restricted_groups
                    ),
                    "peak_restricted_columns": int(
                        solver_result.peak_restricted_columns
                    ),
                }
            )

        metadata.update(
            {
                "convex_objective_value": float(solver_result.objective_value),
                "convex_relative_residual": float(solver_result.relative_residual),
                "convex_feasible": bool(solver_result.feasible),
                "convex_converged": bool(solver_result.converged),
                "convex_iterations": convex_iterations,
                "convex_fixed_point_residual": convex_fixed_point,
                "convex_design_rank": convex_rank,
                "convex_projection_backend": projection_backend,
                "selected_prefix_size": int(selection_result.selected_prefix_size),
                "prefix_fits_evaluated": int(len(selection_result.prefix_path)),
                "floor_reached": bool(selection_result.floor_reached),
                "selection_stop_reason": str(selection_result.stop_reason),
                "post_selection_rank": selection_result.post_rank,
                "post_selection_identifiable": selection_result.post_identifiable,
                "post_selection_condition_number_scaled": float(
                    selection_result.post_condition_number_scaled
                ),
                "post_selection_condition_number_raw": float(
                    selection_result.post_condition_number_raw
                ),
            }
        )
    else:
        metadata.update(
            {
                "selected_lambda": float(solver_result.selected_lambda),
                "post_selection_rank": int(solver_result.post_rank),
                "post_selection_condition_number": float(
                    solver_result.post_condition_number
                ),
                "floor_reached": bool(solver_result.floor_reached),
                "solver_stop_reason": str(solver_result.stop_reason),
                "all_path_kkt_certified": bool(
                    solver_result.all_path_kkt_certified
                ),
                "total_kkt_reactivations": int(
                    solver_result.total_kkt_reactivations
                ),
                "screening_group_count": int(
                    len(getattr(solver_result, "screening_groups", tuple()))
                ),
                "forward_steps": int(getattr(solver_result, "forward_steps", 0)),
                "backward_steps": int(getattr(solver_result, "backward_steps", 0)),
            }
        )

    return ChangeStructureResult(
        hypothesis="varying_structure",
        constraints=tuple(constraints),
        selected_groups=selected_groups,
        diagnostic_delta_B=diagnostic_delta_B,
        relative_residual=float(final_relative_residual),
        solver_result=solver_result,
        selection_result=selection_result,
        metadata=metadata,
        design=design if return_design else None,
    )


# -----------------------------------------------------------------------------
# Other hypothesis / public dispatch
# -----------------------------------------------------------------------------


def infer_coherent_dynamics_changes_from_observations(*args, **kwargs) -> ChangeStructureResult:
    """Planned Step-2 backend for fixed topology / varying dynamics."""

    raise NotImplementedError(
        "The coherent-dynamics Step-2 backend is intentionally pending its "
        "identifiability/change-model formulation. Use "
        "hypothesis='varying_structure' for the current TIDES branch."
    )


def infer_change_structure_from_observations(
    Y: ArrayLike,
    D: ArrayLike,
    edge_features: ArrayLike,
    stage_of_sample: Sequence[int],
    *,
    hypothesis: Literal["varying_structure", "varying_dynamics"] = "varying_structure",
    **kwargs,
) -> ChangeStructureResult:
    """Unified Step-2 interface for preprocessed observations."""

    if hypothesis == "varying_structure":
        return infer_sparse_row_changes_from_observations(
            Y,
            D,
            edge_features,
            stage_of_sample,
            **kwargs,
        )
    if hypothesis == "varying_dynamics":
        return infer_coherent_dynamics_changes_from_observations(
            Y,
            D,
            edge_features,
            stage_of_sample,
            **kwargs,
        )
    raise ValueError(f"Unknown Step-2 hypothesis: {hypothesis!r}")


infer_change_structure = infer_change_structure_from_observations


__all__ = [
    "PairwisePolynomialFeatures",
    "build_pairwise_polynomial_features",
    "PairwiseEndpointPolynomialFeatures",
    "build_pairwise_endpoint_polynomial_features",
    "CumulativeChangeDesign",
    "ScalableCumulativeChangeDesign",
    "SparseRowConstraint",
    "PrefixRefitPoint",
    "RankedPrefixSelectionResult",
    "ChangeStructureResult",
    "build_cumulative_change_design",
    "build_scalable_cumulative_change_design",
    "infer_sparse_row_changes_from_observations",
    "infer_coherent_dynamics_changes_from_observations",
    "infer_change_structure_from_observations",
    "infer_change_structure",
]
