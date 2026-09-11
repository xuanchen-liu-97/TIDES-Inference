"""TIDES Step 2: infer the structure/support of vector-field changes.

Step 2 is hypothesis-dependent.  The implemented ``varying_structure`` branch
uses a *global cumulative* edge-space regression rather than independent local
jump fits.

For preprocessed observations indexed by a detected stage r,

    y = X_base vec(B^(1)) + X_change beta_delta,

with

    B^(r) = B^(1) + sum_{k<r} Delta B^(k).

The stationary anchor ``B^(1)`` is an unpenalized nuisance.  It is projected
out before sparse selection,

    y_perp = (I - P_base) y,
    X_perp = (I - P_base) X_change,

and each candidate group is one ``(transition, edge)`` row of ``Delta B``.
If the edge-function dimension is L, every group contains L coefficients.
Adaptive Group LASSO then localizes the changed rows.  The coefficients from
this projected Step-2 fit are diagnostics only; Step 3 performs the final
unpenalized reconstruction of ``B^(1)`` and all selected ``Delta B`` rows in
the original, unprojected model.

This module owns TIDES-specific *problem construction and interpretation*.
Numerical sparse optimization lives in ``solvers_sparse_structure.py``.

Current scope
-------------
- continuous, preprocessed vector-field observations;
- piecewise-stationary stages supplied by Step 1 / preprocessing;
- conservative pairwise edge-space representation;
- row-sparse ``Delta B`` for ``hypothesis='varying_structure'``.

The fixed-topology / varying-dynamics branch remains intentionally separate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Literal, Mapping, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

try:  # package form
    from .solvers_sparse_structure import (
        AdaptiveGroupLassoResult,
        ForwardBackwardGroupSearchResult,
        solve_adaptive_group_lasso,
        solve_forward_backward_group_search,
    )
except ImportError:  # flat-file form
    from solvers_sparse_structure import (
        AdaptiveGroupLassoResult,
        ForwardBackwardGroupSearchResult,
        solve_adaptive_group_lasso,
        solve_forward_backward_group_search,
    )


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
GroupLabel = tuple[int, int]  # (transition ordinal, candidate-edge index)


@dataclass(frozen=True)
class CumulativeChangeDesign:
    """Explicit global Step-2 design after preprocessing.

    ``X_base`` contains the unrestricted stationary edge-space anchor together
    with any optional stationary nuisance columns.  ``X_change`` contains all
    candidate cumulative temporal-change columns.  ``y_perp`` and
    ``X_change_perp`` are obtained by orthogonally projecting out the numerical
    column space of ``X_base``.
    """

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
class SparseRowConstraint:
    """Rows of one ``Delta B^(k)`` block admitted by Step 2.

    ``projected_coefficients`` are the post-selection coefficients from the
    projected Step-2 regression, restricted to the selected rows.  They are
    retained only as diagnostics and are *not* the final Step-3 reconstruction.
    """

    transition_ordinal: int
    transition_index: int
    transition_time: float
    support: IntArray
    support_labels: tuple[Hashable, ...]
    projected_coefficients: FloatArray


@dataclass(frozen=True)
class ChangeStructureResult:
    """Output of TIDES Step 2."""

    hypothesis: str
    constraints: tuple[SparseRowConstraint, ...]
    selected_groups: tuple[GroupLabel, ...]
    diagnostic_delta_B: FloatArray
    relative_residual: float
    solver_result: AdaptiveGroupLassoResult | ForwardBackwardGroupSearchResult
    metadata: dict
    design: Optional[CumulativeChangeDesign] = None

    @property
    def supports(self) -> tuple[IntArray, ...]:
        """Selected changed-edge row indices, one array per transition."""

        return tuple(c.support for c in self.constraints)


# -----------------------------------------------------------------------------
# Validation / construction utilities
# -----------------------------------------------------------------------------


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
    profile_floor: Optional[float] = None,
    solver_method: Literal[
        "adaptive_group_lasso",
        "forward_backward_floor",
    ] = "adaptive_group_lasso",
    solver_kwargs: Optional[Mapping[str, object]] = None,
    return_design: bool = False,
) -> ChangeStructureResult:
    """Infer row-sparse temporal changes from preprocessed observations.

    The sparse solver is completely blind to the meaning of a group.  This
    function supplies groups ``(k,m)`` and interprets the selected labels as
    changed edge rows at transition ``k``.
    """

    design = build_cumulative_change_design(
        Y,
        D,
        edge_features,
        stage_of_sample,
        stationary_nuisance=stationary_nuisance,
    )

    K = design.n_transitions
    M = design.n_edges
    L = design.n_edge_features

    transition_indices_arr, transition_times_arr = _coerce_transition_metadata(
        K, transition_indices, transition_times
    )

    if edge_labels is None:
        labels: tuple[Hashable, ...] = tuple(range(M))
    else:
        labels = tuple(edge_labels)
        if len(labels) != M:
            raise ValueError("edge_labels must have exactly D.shape[1] entries.")

    kwargs = {} if solver_kwargs is None else dict(solver_kwargs)
    if "target_relative_floor" in kwargs:
        raise ValueError(
            "Pass the Step-2 numerical floor through profile_floor, not "
            "solver_kwargs['target_relative_floor']."
        )

    if solver_method == "adaptive_group_lasso":
        solver_result = solve_adaptive_group_lasso(
            design.X_change_perp,
            design.y_perp,
            design.group_columns,
            target_relative_floor=profile_floor,
            **kwargs,
        )
    elif solver_method == "forward_backward_floor":
        if profile_floor is None:
            raise ValueError(
                "solver_method='forward_backward_floor' requires profile_floor."
            )
        solver_result = solve_forward_backward_group_search(
            design.X_change_perp,
            design.y_perp,
            design.group_columns,
            target_relative_floor=float(profile_floor),
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown sparse solver_method: {solver_method!r}.")

    selected_groups = tuple(
        (int(k), int(m)) for k, m in solver_result.selected_groups
    )
    selected_set = set(selected_groups)

    # Diagnostic projected coefficients only.  Step 3 must reconstruct again in
    # the original unprojected global model after support has been selected.
    diagnostic_delta_B = np.zeros((K, M, L), dtype=float)
    for label in selected_groups:
        k, m = label
        cols = design.group_columns[label]
        diagnostic_delta_B[k, m, :] = solver_result.coefficients[cols]

    constraints: list[SparseRowConstraint] = []
    for k in range(K):
        support = np.asarray(
            [m for m in range(M) if (k, m) in selected_set],
            dtype=np.int64,
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
    metadata = {
        "backend": f"global-cumulative-{solver_method}",
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
        "stationary_columns": int(design.X_base.shape[1]),
        "baseline_rank": int(design.baseline_rank),
        "baseline_weakest_retained_relative_singular": float(
            design.baseline_weakest_retained_relative_singular
        ),
        "projected_target_norm": y_perp_norm,
        "profile_floor": None if profile_floor is None else float(profile_floor),
        "selected_group_count": int(len(selected_groups)),
        "selected_lambda": float(solver_result.selected_lambda),
        "post_selection_relative_residual": float(
            solver_result.post_relative_residual
        ),
        "post_selection_rank": int(solver_result.post_rank),
        "post_selection_condition_number": float(
            solver_result.post_condition_number
        ),
        "floor_reached": bool(solver_result.floor_reached),
        "solver_stop_reason": str(solver_result.stop_reason),
        "all_path_kkt_certified": bool(solver_result.all_path_kkt_certified),
        "total_kkt_reactivations": int(
            solver_result.total_kkt_reactivations
        ),
        "screening_group_count": int(
            len(getattr(solver_result, "screening_groups", tuple()))
        ),
        "forward_steps": int(
            getattr(solver_result, "forward_steps", 0)
        ),
        "backward_steps": int(
            getattr(solver_result, "backward_steps", 0)
        ),
    }

    return ChangeStructureResult(
        hypothesis="varying_structure",
        constraints=tuple(constraints),
        selected_groups=selected_groups,
        diagnostic_delta_B=diagnostic_delta_B,
        relative_residual=float(solver_result.post_relative_residual),
        solver_result=solver_result,
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


# Keep the short public name for interactive/package use.  The new Step-2 API
# is deliberately observation-based because arbitrary edge features cannot in
# general be reconstructed from a single scalar edge state X @ D.
infer_change_structure = infer_change_structure_from_observations


__all__ = [
    "CumulativeChangeDesign",
    "SparseRowConstraint",
    "ChangeStructureResult",
    "build_cumulative_change_design",
    "infer_sparse_row_changes_from_observations",
    "infer_coherent_dynamics_changes_from_observations",
    "infer_change_structure_from_observations",
    "infer_change_structure",
]
