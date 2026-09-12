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
    Conditional feasibility screening supplies a small working set to the same
    dense-SVD BPDN reference solver.  Final support size is still chosen only by
    the independent ranked-prefix uncertainty-floor certificate.

The scalable screen is an acceleration/ranking backend, not a claim that tiny
full-BPDN leakage coefficients outside the working set are exactly zero.  The
formal Step-2 output remains the floor-certified support.

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
        solve_adaptive_group_lasso,
        solve_forward_backward_group_search,
        solve_group_basis_pursuit_denoising,
        solve_screened_group_basis_pursuit_denoising,
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
        solve_adaptive_group_lasso,
        solve_forward_backward_group_search,
        solve_group_basis_pursuit_denoising,
        solve_screened_group_basis_pursuit_denoising,
    )
    from solvers_linear_regression import (
        LinearRegressionResult,
        solve_least_squares,
    )


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
GroupLabel = tuple[int, int]  # (transition ordinal, candidate-edge index)


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
    n_transitions: int




@dataclass(frozen=True)
class ScalableCumulativeChangeDesign:
    """Lazy projected cumulative Step-2 design.

    The full ``X_change`` / ``X_change_perp`` arrays do not exist.  Projected
    group blocks are materialised only when requested by the screened solver or
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


class _SparseBaselineProjector:
    """Sparse high-accuracy projector onto the complement of the anchor span.

    The anchor itself is sparse for edge-incidence models: one edge-function
    column touches only the nodes on that candidate edge.  Projection vectors
    and the small number of selected group columns are computed by LSQR.  A
    sparse factorisation of the scaled normal Gram is retained *only* for the
    conditional-screening self-Gram calculation; it is not used for the final
    projected data/columns.
    """

    def __init__(
        self,
        X_base: sp.csc_matrix,
        *,
        projection_tol: float = 1.0e-16,
        projection_max_iter: Optional[int] = None,
        diagnostic_dense_bytes: int = 64 * 1024**2,
    ):
        X_base = sp.csc_matrix(X_base, dtype=float)
        self.n_rows, self.n_columns_original = X_base.shape
        self.projection_tol = float(projection_tol)
        if not np.isfinite(self.projection_tol) or self.projection_tol <= 0.0:
            raise ValueError("projection_tol must be finite and positive.")

        norms = np.sqrt(np.asarray(X_base.power(2).sum(axis=0)).reshape(-1))
        good = norms > 1.0e-14
        self.good_columns = np.flatnonzero(good)
        self.zero_columns = np.flatnonzero(~good)
        self.column_scale = norms[good]

        if self.good_columns.size == 0:
            self.X_scaled = sp.csc_matrix((self.n_rows, 0), dtype=float)
            self.gram_lu = None
            self.rank = 0
            self.singular_values = np.empty(0, dtype=float)
            self.rank_tolerance = 0.0
            self.weakest_relative_singular = np.nan
            self.max_iter = 0
            return

        X_good = X_base[:, self.good_columns]
        self.X_scaled = (
            X_good @ sp.diags(1.0 / self.column_scale, format="csc")
        ).tocsc()
        p = self.X_scaled.shape[1]
        if projection_max_iter is None:
            self.max_iter = int(max(2000, min(20000, 4 * p)))
        else:
            self.max_iter = int(projection_max_iter)
            if self.max_iter < 1:
                raise ValueError("projection_max_iter must be >= 1.")

        # Screening repeatedly needs A_g^T P_perp A_g.  The sparse scaled Gram
        # is factorised once.  A singular Gram means the baseline representation
        # itself contains redundant nonzero columns; for now the scalable path
        # asks the caller to use the dense reference or remove that redundancy.
        gram = (self.X_scaled.T @ self.X_scaled).tocsc()
        try:
            self.gram_lu = splu(gram)
        except RuntimeError as exc:
            raise RuntimeError(
                "Scalable Step 2 requires the nonzero stationary-anchor columns "
                "to be numerically independent for its sparse screening "
                "factorisation. The dense backend can still handle a rank-"
                "deficient anchor."
            ) from exc

        dense_bytes = int(self.X_scaled.shape[0] * p * 8)
        if dense_bytes <= int(diagnostic_dense_bytes):
            A_dense = self.X_scaled.toarray()
            singular = np.linalg.svd(A_dense, compute_uv=False)
            if singular.size and singular[0] > 0.0:
                rank_tol = float(
                    np.finfo(float).eps * max(A_dense.shape) * singular[0]
                )
                rank = int(np.sum(singular > rank_tol))
                weakest = (
                    float(singular[rank - 1] / singular[0])
                    if rank > 0
                    else np.nan
                )
            else:
                rank_tol = 0.0
                rank = 0
                weakest = np.nan
            self.rank = rank
            self.singular_values = np.asarray(singular, dtype=float)
            self.rank_tolerance = rank_tol
            self.weakest_relative_singular = weakest
        else:
            # Successful sparse LU of X_s^T X_s certifies full column rank in
            # the arithmetic used by the scalable screening backend.  We avoid
            # an expensive dense SVD solely for diagnostics at large scale.
            self.rank = int(p)
            self.singular_values = np.empty(0, dtype=float)
            self.rank_tolerance = np.nan
            self.weakest_relative_singular = np.nan

    def _lsqr_projection_coefficients(self, b: FloatArray) -> FloatArray:
        if self.X_scaled.shape[1] == 0:
            return np.empty(0, dtype=float)
        out = lsqr(
            self.X_scaled,
            np.asarray(b, dtype=float).reshape(-1),
            atol=self.projection_tol,
            btol=self.projection_tol,
            conlim=1.0e18,
            iter_lim=self.max_iter,
            show=False,
        )
        return np.asarray(out[0], dtype=float)

    def project_vector(self, b: FloatArray) -> FloatArray:
        b = np.asarray(b, dtype=float).reshape(-1)
        if b.size != self.n_rows:
            raise ValueError("projection vector has incompatible length.")
        if self.X_scaled.shape[1] == 0:
            return b.copy()
        coef = self._lsqr_projection_coefficients(b)
        return b - np.asarray(self.X_scaled @ coef).reshape(-1)

    def project_dense_block(self, G: FloatArray) -> FloatArray:
        G = np.asarray(G, dtype=float)
        if G.ndim != 2 or G.shape[0] != self.n_rows:
            raise ValueError("projection block has incompatible shape.")
        if self.X_scaled.shape[1] == 0:
            return G.copy()
        out = G.copy()
        for j in range(G.shape[1]):
            coef = self._lsqr_projection_coefficients(G[:, j])
            out[:, j] -= np.asarray(self.X_scaled @ coef).reshape(-1)
        return out

    def projected_gram_from_sparse_block(self, G: sp.csc_matrix) -> FloatArray:
        """Compute G^T P_perp G for screening without forming P_perp G."""

        G = sp.csc_matrix(G, dtype=float)
        raw = np.asarray((G.T @ G).toarray(), dtype=float)
        if self.X_scaled.shape[1] == 0:
            return raw
        rhs = np.asarray((self.X_scaled.T @ G).toarray(), dtype=float)
        coef = np.asarray(self.gram_lu.solve(rhs), dtype=float)
        H = raw - rhs.T @ coef
        H = 0.5 * (H + H.T)

        # Normal equations are used only for screening geometry.  Clip negative
        # roundoff modes while preserving every numerically positive direction.
        eig, vec = np.linalg.eigh(H)
        scale = max(float(np.max(np.abs(eig))) if eig.size else 0.0, 1.0e-300)
        tol = 500.0 * np.finfo(float).eps * scale
        eig = np.where(eig > tol, eig, 0.0)
        return (vec * eig[None, :]) @ vec.T


def _build_sparse_anchor_design(
    D: FloatArray,
    edge_features: FloatArray,
) -> sp.csc_matrix:
    """Build the stationary edge-function design directly in CSC form."""

    T, M, L = edge_features.shape
    N = D.shape[0]
    edge_nnz = np.count_nonzero(D, axis=0).astype(np.int64)
    total_nnz = int(T * L * int(edge_nnz.sum()))

    data = np.empty(total_nnz, dtype=float)
    index_dtype = np.int32 if T * N < np.iinfo(np.int32).max else np.int64
    indices = np.empty(total_nnz, dtype=index_dtype)
    indptr = np.empty(M * L + 1, dtype=np.int64)
    indptr[0] = 0

    time_rows = (np.arange(T, dtype=np.int64) * N)[:, None]
    pos = 0
    col = 0
    for m in range(M):
        nodes = np.flatnonzero(D[:, m] != 0.0)
        values = D[nodes, m]
        rows = (time_rows + nodes[None, :]).reshape(-1)
        for ell in range(L):
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
    """Cumulative temporal group blocks with sparse-anchor projection."""

    def __init__(
        self,
        D: FloatArray,
        edge_features: FloatArray,
        stage: IntArray,
        projector: _SparseBaselineProjector,
    ):
        self.D = np.asarray(D, dtype=float)
        self.edge_features = np.asarray(edge_features, dtype=float)
        self.stage = np.asarray(stage, dtype=np.int64)
        self.projector = projector
        self.T, self.M, self.L = self.edge_features.shape
        self.N = self.D.shape[0]
        self.K = int(np.max(self.stage))
        self.n_rows = int(self.T * self.N)

        self.active_indices = tuple(
            np.flatnonzero(self.stage >= (k + 1)) for k in range(self.K)
        )
        self.edge_nodes = tuple(
            np.flatnonzero(self.D[:, m] != 0.0) for m in range(self.M)
        )
        self.edge_values = tuple(
            self.D[self.edge_nodes[m], m].copy() for m in range(self.M)
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
        nodes = self.edge_nodes[m]
        values = self.edge_values[m]
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
                self.edge_features[times, m, ell, None] * values[None, :]
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

    def conditional_gain_provider(
        self,
        residual: FloatArray,
        Q: FloatArray,
        omitted: tuple[GroupLabel, ...],
    ) -> tuple[GroupLabel, float]:
        """Exact conditional-gain formula up to sparse-Gram roundoff.

        Since both the current residual and selected-span basis already lie in
        the complement of the stationary anchor,

            A_g^T r = X_g^T r,
            A_g^T Q = X_g^T Q.

        Hence only the projected self-Gram ``A_g^T A_g`` must be precomputed.
        """

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
            values = self.edge_values[m]
            psi = self.edge_features[times, m, :]

            local_residual = residual[np.ix_(times, nodes)] @ values
            c = psi.T @ local_residual

            H = self.projected_self_gram((k, m)).copy()
            if qdim:
                local_Q = np.tensordot(
                    Q3[times][:, nodes, :],
                    values,
                    axes=(1, 0),
                )
                V = psi.T @ local_Q
                H -= V @ V.T

            gain = self._gain_from_gram(c, H)
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
    stationary_nuisance: Optional[ArrayLike] = None,
    projection_tol: float = 1.0e-16,
    projection_max_iter: Optional[int] = None,
    diagnostic_dense_bytes: int = 64 * 1024**2,
) -> ScalableCumulativeChangeDesign:
    """Build the lazy/sparse Step-2 design used by ``backend='scalable'``."""

    Y, D, edge_features, stage, K = _validate_observations(
        Y, D, edge_features, stage_of_sample
    )
    T, N = Y.shape
    M = D.shape[1]
    L = edge_features.shape[2]

    X_anchor = _build_sparse_anchor_design(D, edge_features)
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

    operator = _LazyProjectedChangeOperator(
        D,
        edge_features,
        stage,
        projector,
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
        baseline_projection_backend="sparse-lsqr",
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
    if edge_features.ndim != 3:
        raise ValueError(
            "edge_features must have shape "
            "(n_observations, n_candidate_edges, n_edge_features)."
        )

    T, N = Y.shape
    if D.shape[0] != N:
        raise ValueError("D.shape[0] must equal the node dimension of Y.")
    M = D.shape[1]
    if edge_features.shape[:2] != (T, M):
        raise ValueError(
            "edge_features.shape[:2] must equal "
            "(Y.shape[0], D.shape[1])."
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

    # Per-observation, per-edge, per-feature contribution to node space:
    #     d_m * psi_{t,m,l}
    # Shape: (T, N, M, L).
    pair_basis = D[None, :, :, None] * edge_features[:, None, :, :]

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
    uncertainty_floor: Optional[float] = None,
    profile_floor: Optional[float] = None,
    solver_method: Literal[
        "group_bpdn_prefix",
        "adaptive_group_lasso",
        "forward_backward_floor",
    ] = "group_bpdn_prefix",
    backend: Literal["dense", "scalable", "auto"] = "dense",
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
    ``backend='scalable'`` uses sparse anchor projection, lazy temporal group
    blocks, conditional feasibility screening, restricted BPDN ranking, and the
    same floor-certified prefix selection.  ``backend='auto'`` dispatches only
    from a conservative explicit-design memory estimate; it never inspects an
    inference result to decide the backend.
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
            stationary_nuisance=stationary_nuisance,
            projection_tol=scalable_projection_tol,
            projection_max_iter=scalable_projection_max_iter,
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
        "oracle_information_used": False,
        "solver_method": str(solver_method),
        "n_preprocessed_observations": int(design.n_samples),
        "n_scalar_rows": int(design.n_samples * design.n_nodes),
        "n_nodes": int(design.n_nodes),
        "n_candidate_edges": int(M),
        "n_edge_features": int(L),
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
            }
        )

    if solver_method == "group_bpdn_prefix":
        assert selection_result is not None
        if isinstance(solver_result, GroupBasisPursuitResult):
            convex_iterations = int(solver_result.iterations)
            convex_fixed_point = float(solver_result.fixed_point_residual)
            convex_rank = int(solver_result.design_rank)
            projection_backend = str(solver_result.projection_backend)
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
