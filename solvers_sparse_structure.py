"""
Generic grouped sparse solvers for TIDES.

This module intentionally contains no TIDES-specific modelling logic.
Problem-specific modules (for example ``step2_change_structure.py``) build the
linear inverse problem ``y ~= X @ beta`` and define the non-overlapping group
partition before calling these numerical routines.

Primary solver
--------------
``solve_group_basis_pursuit_denoising`` solves the noise-aware convex problem

    minimise    sum_g w_g ||beta_g||_2
    subject to  ||y - X beta||_2 <= delta,

where ``delta`` is an externally calibrated uncertainty radius.  The solver
returns a continuous grouped solution and a ranking of group norms; it does
*not* choose a support.  Support certification belongs to the problem-specific
Step-2 layer, which can refit ranked prefixes and stop at the first prefix that
reaches the independently supplied uncertainty floor.

For the current correctness benchmark the reference implementation uses
Douglas--Rachford splitting.  Projection onto the residual ball is evaluated
accurately from one thin SVD of the dense design.

A second large-scale ranking backend,
``solve_screened_group_basis_pursuit_denoising``, never requires the global
design matrix.  It accesses one group block at a time, builds a floor-feasible
working set by exact conditional least-squares screening, and runs the same
Douglas--Rachford BPDN solver only on that restricted matrix.  Problem-specific
operator layers may provide a batched conditional-gain callback so the global
scan can also be evaluated without materialising all candidate blocks.

Legacy baselines
----------------
The previous Adaptive Group LASSO and forward/backward floor search are retained
as diagnostic/baseline solvers so existing notebooks remain import-compatible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Hashable, Mapping, Optional, Sequence, Tuple

import numpy as np


Array = np.ndarray
GroupLabel = Hashable


@dataclass(frozen=True)
class AdaptiveGroupLassoPathPoint:
    """Diagnostics for one lambda value on the adaptive path."""

    path_index: int
    lambda_value: float
    lambda_over_max: float
    active_groups: Tuple[GroupLabel, ...]
    working_set: Tuple[GroupLabel, ...]
    n_active_groups: int
    working_set_size: int
    post_relative_residual: float
    post_rank: int
    post_condition_number: float
    solver_iterations: int
    solver_converged: bool
    all_intermediate_converged: bool
    restricted_kkt_ratio: float
    kkt_expansions: int
    kkt_reactivations: int
    max_kkt_ratio: float
    final_kkt_satisfied: bool
    next_entry_lambda: float
    next_entry_group: Optional[GroupLabel]
    adaptive_next_lambda: float


@dataclass(frozen=True)
class AdaptiveGroupLassoResult:
    """Result returned by :func:`solve_adaptive_group_lasso`."""

    coefficients: Array
    penalized_coefficients: Array
    selected_groups: Tuple[GroupLabel, ...]
    selected_lambda: float
    post_relative_residual: float
    post_rank: int
    post_condition_number: float
    floor_reached: bool
    stop_reason: str
    lambda_max: float
    ridge_alpha: float
    ridge_lambda_max: float
    ridge_lambda_min: float
    column_scale: Array
    pilot_coefficients_scaled: Array
    adaptive_weights: Dict[GroupLabel, float]
    path: Tuple[AdaptiveGroupLassoPathPoint, ...]
    total_kkt_reactivations: int
    all_path_kkt_certified: bool


@dataclass(frozen=True)
class ForwardBackwardGroupSearchResult:
    """Result of blind floor-driven forward/backward group selection."""

    coefficients: Array
    selected_groups: Tuple[GroupLabel, ...]
    post_relative_residual: float
    post_rank: int
    post_condition_number: float
    floor_reached: bool
    stop_reason: str
    screening_groups: Tuple[GroupLabel, ...]
    screening_relative_residual: float
    forward_steps: int
    backward_steps: int

    # Compatibility/diagnostic fields used by Step-2 reporting.
    selected_lambda: float = np.nan
    all_path_kkt_certified: bool = False
    total_kkt_reactivations: int = 0
    selection_method: str = "forward-backward-floor-search"


@dataclass(frozen=True)
class GroupBasisPursuitResult:
    """Continuous solution of grouped basis-pursuit denoising.

    ``coefficients`` are returned in the coordinates supplied by the caller.
    ``group_scores[label]`` is ``||coefficients[group]||_2`` unless explicit
    group weights are supplied; the ranking always uses these unweighted group
    norms.  The convex objective value itself uses ``group_weights``.

    The solver deliberately does not report a selected support.  A caller that
    needs model selection should use ``ranking`` and certify prefixes against
    the same residual radius with an unpenalised refit.
    """

    coefficients: Array
    group_scores: Dict[GroupLabel, float]
    ranking: Tuple[GroupLabel, ...]
    objective_value: float
    residual_norm: float
    relative_residual: float
    residual_radius: float
    target_relative_residual: float
    feasible: bool
    converged: bool
    stop_reason: str
    iterations: int
    fixed_point_residual: float
    douglas_rachford_step: float
    group_weights: Dict[GroupLabel, float]
    design_rank: int
    design_singular_values: Array
    irreducible_residual_norm: float
    projection_backend: str = "dense-svd"
    selection_method: str = "group-bpdn-ranking"


@dataclass(frozen=True)
class ScreenedGroupBasisPursuitResult:
    """Continuous grouped BPDN ranking from a memory-scalable screened solve.

    The global design matrix is never required.  A caller supplies one group
    block at a time through ``group_block_provider``.  A deterministic forward
    feasibility screen first constructs a small working set whose unpenalised
    refit reaches the requested residual ball.  The *same* dense-SVD
    Douglas--Rachford BPDN reference solver is then applied only to that
    restricted working set.

    This object deliberately reports a ranking rather than a selected support.
    Final support-size choice remains the responsibility of the caller (for
    TIDES, the independently calibrated prefix-floor certification in Step 2).

    Important
    ---------
    Screening is an acceleration device, not a proof that omitted groups have
    zero coefficients in the full convex optimum.  ``coefficients`` and
    ``group_scores`` therefore contain zeros outside ``screening_groups``.
    The intended regression criterion is equality of the *certified Step-2
    support* with the dense reference backend, not coefficient-by-coefficient
    equality of tiny leakage directions.
    """

    coefficients: Array
    group_scores: Dict[GroupLabel, float]
    ranking: Tuple[GroupLabel, ...]
    objective_value: float
    residual_norm: float
    relative_residual: float
    residual_radius: float
    target_relative_residual: float
    feasible: bool
    converged: bool
    stop_reason: str
    group_weights: Dict[GroupLabel, float]
    screening_groups: Tuple[GroupLabel, ...]
    screening_relative_residual: float
    screening_steps: int
    screening_rank: int
    screening_condition_number: float
    restricted_result: GroupBasisPursuitResult
    n_global_groups: int
    n_restricted_groups: int
    peak_restricted_columns: int
    projection_backend: str = "screened-dense-svd"
    selection_method: str = "screened-group-bpdn-ranking"


# -----------------------------------------------------------------------------
# Validation / utilities
# -----------------------------------------------------------------------------


def _validate_problem(
    X: Array,
    y: Array,
    groups: Mapping[GroupLabel, Sequence[int]],
) -> Tuple[Array, Array, Dict[GroupLabel, Array], Tuple[GroupLabel, ...]]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if y.ndim != 1:
        raise ValueError("y must be one-dimensional after flattening.")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y have incompatible sample counts: {X.shape[0]} vs {y.shape[0]}."
        )
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must contain at least one row and one column.")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("X and y must be finite.")
    if not groups:
        raise ValueError("groups must be non-empty.")

    p = X.shape[1]
    clean: Dict[GroupLabel, Array] = {}
    seen = np.zeros(p, dtype=int)

    for label, idx in groups.items():
        arr = np.asarray(idx, dtype=int).reshape(-1)
        if len(arr) == 0:
            raise ValueError(f"Group {label!r} is empty.")
        if np.any(arr < 0) or np.any(arr >= p):
            raise ValueError(f"Group {label!r} contains out-of-range coefficient indices.")
        arr = np.unique(arr)
        clean[label] = arr
        seen[arr] += 1

    if np.any(seen > 1):
        overlap = np.flatnonzero(seen > 1)
        raise ValueError(
            "Adaptive Group LASSO requires non-overlapping groups; "
            f"overlap at coefficient indices {overlap[:20].tolist()}."
        )
    if np.any(seen == 0):
        missing = np.flatnonzero(seen == 0)
        raise ValueError(
            "Every coefficient must belong to exactly one group; "
            f"ungrouped indices {missing[:20].tolist()}."
        )

    order = tuple(clean.keys())
    return X, y, clean, order


def _relative_residual(prediction: Array, target: Array) -> float:
    denom = max(float(np.linalg.norm(target)), np.finfo(float).tiny)
    return float(np.linalg.norm(prediction - target) / denom)


def _standardize_columns(X: Array, floor: float = 1e-14) -> Tuple[Array, Array]:
    """RMS-standardise columns, mirroring the scaling used before TSC v3.6."""
    scale = np.sqrt(np.mean(X * X, axis=0))
    bad = np.flatnonzero(scale <= floor)
    if len(bad):
        raise ValueError(
            "Near-zero design columns cannot be standardised; "
            f"indices={bad[:20].tolist()}."
        )
    return X / scale[None, :], scale


def _power_largest_eigenvalue_design(
    X: Array,
    *,
    max_iter: int = 80,
    tol: float = 1e-8,
) -> float:
    """Estimate lambda_max(X^T X / n) without explicitly forming it."""
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    if n == 0 or p == 0:
        return 0.0

    if p <= n:
        v = np.ones(p, dtype=float)
        v /= np.linalg.norm(v)
        last = 0.0
        for _ in range(max_iter):
            z = X.T @ (X @ v) / n
            norm = float(np.linalg.norm(z))
            if norm == 0.0:
                return 0.0
            v = z / norm
            rayleigh = float(v @ (X.T @ (X @ v)) / n)
            if abs(rayleigh - last) <= tol * max(abs(rayleigh), 1.0):
                return max(rayleigh, 0.0)
            last = rayleigh
        return max(last, 0.0)

    v = np.ones(n, dtype=float)
    v /= np.linalg.norm(v)
    last = 0.0
    for _ in range(max_iter):
        z = X @ (X.T @ v) / n
        norm = float(np.linalg.norm(z))
        if norm == 0.0:
            return 0.0
        v = z / norm
        rayleigh = float(v @ (X @ (X.T @ v)) / n)
        if abs(rayleigh - last) <= tol * max(abs(rayleigh), 1.0):
            return max(rayleigh, 0.0)
        last = rayleigh
    return max(last, 0.0)


def _adaptive_ridge_alpha(
    X: Array,
    *,
    condition_target: float,
    alpha_floor_ratio: float,
) -> Tuple[float, float, float]:
    """Smallest scale-aware Ridge alpha giving the requested Gram condition cap."""
    n, p = X.shape
    lambda_max = _power_largest_eigenvalue_design(X)
    if lambda_max <= 0.0:
        return float(alpha_floor_ratio), 0.0, 0.0

    if p >= n:
        lambda_min = 0.0
    else:
        gram = (X.T @ X) / n
        eigvals = np.linalg.eigvalsh(gram)
        lambda_min = max(float(eigvals[0]), 0.0)
        lambda_max = max(float(eigvals[-1]), lambda_max)

    kappa = float(condition_target)
    alpha_cond = max(
        0.0,
        (lambda_max - kappa * lambda_min) / (kappa - 1.0),
    )
    alpha_floor = float(alpha_floor_ratio) * max(lambda_max, 1.0)
    alpha = max(alpha_cond, alpha_floor)
    return float(alpha), float(lambda_max), float(lambda_min)


def _ridge_solve(X: Array, y: Array, alpha: float) -> Array:
    """Solve the normalised scalar-response Ridge problem."""
    n, p = X.shape
    if p <= n:
        lhs = X.T @ X
        lhs.flat[:: p + 1] += n * alpha
        rhs = X.T @ y
        return np.linalg.solve(lhs, rhs)

    lhs = X @ X.T
    lhs.flat[:: n + 1] += n * alpha
    dual = np.linalg.solve(lhs, y)
    return X.T @ dual


def _adaptive_weights(
    beta_pilot: Array,
    groups: Mapping[GroupLabel, Array],
    *,
    gamma: float,
    delta_ratio: float,
) -> Dict[GroupLabel, float]:
    norms = {
        label: float(np.linalg.norm(beta_pilot[idx]))
        for label, idx in groups.items()
    }
    max_norm = max(norms.values()) if norms else 0.0
    delta = max(1e-12, float(delta_ratio) * max_norm)
    return {
        label: float(np.sqrt(len(idx)) / (norms[label] + delta) ** gamma)
        for label, idx in groups.items()
    }


def _lambda_max(
    X: Array,
    y: Array,
    groups: Mapping[GroupLabel, Array],
    weights: Mapping[GroupLabel, float],
) -> float:
    grad0 = -(X.T @ y) / X.shape[0]
    return float(
        max(
            np.linalg.norm(grad0[idx]) / weights[label]
            for label, idx in groups.items()
        )
    )


# -----------------------------------------------------------------------------
# Restricted proximal solve + KKT machinery
# -----------------------------------------------------------------------------


def _ordered_subset(
    labels: Sequence[GroupLabel],
    group_order: Tuple[GroupLabel, ...],
) -> Tuple[GroupLabel, ...]:
    selected = set(labels)
    return tuple(label for label in group_order if label in selected)


def _active_columns(
    working_set: Sequence[GroupLabel],
    groups: Mapping[GroupLabel, Array],
) -> Array:
    if not working_set:
        return np.empty(0, dtype=int)
    return np.unique(np.concatenate([groups[label] for label in working_set]))


def _restricted_group_kkt_ratio(
    beta_local: Array,
    gram_local: Array,
    corr_local: Array,
    lam: float,
    weights: Mapping[GroupLabel, float],
    local_group_indices: Mapping[GroupLabel, Array],
    working_set: Sequence[GroupLabel],
    *,
    active_group_tol: float,
) -> float:
    """Maximum normalized KKT residual inside a restricted working set.

    For a nonzero group g, stationarity requires

        grad_g + lambda w_g beta_g / ||beta_g|| = 0.

    For a zero group already inside the working set, the subgradient condition is

        ||grad_g|| <= lambda w_g.

    The returned quantity is a dimensionless residual ratio.  It complements the
    omitted-group KKT audit performed by :func:`_kkt_violations`.
    """

    gradient = gram_local @ beta_local - corr_local
    max_ratio = 0.0

    for label in working_set:
        loc = local_group_indices[label]
        b = beta_local[loc]
        g = gradient[loc]
        threshold = float(lam * weights[label])
        norm_b = float(np.linalg.norm(b))

        if norm_b > max(0.1 * active_group_tol, 1e-12):
            residual = float(
                np.linalg.norm(g + threshold * b / norm_b)
            )
        else:
            residual = max(0.0, float(np.linalg.norm(g)) - threshold)

        scale = max(
            threshold,
            float(np.linalg.norm(corr_local[loc])),
            np.finfo(float).eps,
        )
        max_ratio = max(max_ratio, residual / scale)

    return float(max_ratio)


def _restricted_objective(
    beta_local: Array,
    gram_local: Array,
    corr_local: Array,
    lam: float,
    weights: Mapping[GroupLabel, float],
    local_group_indices: Mapping[GroupLabel, Array],
    working_set: Sequence[GroupLabel],
) -> float:
    """Restricted Group-LASSO objective up to an irrelevant additive constant."""

    smooth = 0.5 * float(beta_local @ (gram_local @ beta_local))
    smooth -= float(corr_local @ beta_local)
    penalty = float(
        lam
        * sum(
            weights[label]
            * np.linalg.norm(beta_local[local_group_indices[label]])
            for label in working_set
        )
    )
    return smooth + penalty


def _prox_group_blocks_local(
    proposal: Array,
    threshold_scale: float,
    weights: Mapping[GroupLabel, float],
    local_group_indices: Mapping[GroupLabel, Array],
    working_set: Sequence[GroupLabel],
) -> Array:
    """Group soft-thresholding on local restricted coordinates."""

    out = proposal.copy()
    for label in working_set:
        loc = local_group_indices[label]
        vec = proposal[loc]
        norm = float(np.linalg.norm(vec))
        threshold = float(threshold_scale * weights[label])
        if norm > threshold:
            out[loc] = vec * (1.0 - threshold / norm)
        else:
            out[loc] = 0.0
    return out


def _newton_polish_active_manifold(
    beta_local: Array,
    gram_local: Array,
    corr_local: Array,
    lam: float,
    weights: Mapping[GroupLabel, float],
    local_group_indices: Mapping[GroupLabel, Array],
    working_set: Sequence[GroupLabel],
    *,
    active_group_tol: float,
    kkt_tol: float,
    max_steps: int,
) -> Tuple[Array, int]:
    """Damped Newton polishing on the currently nonzero group manifold.

    The Group-LASSO objective is smooth as long as every currently active group
    stays away from zero.  Proximal iterations establish/change support; Newton
    then removes the severe slow-down caused by highly coherent restricted
    designs.  If a step approaches the nonsmooth boundary or fails a monotone
    line search, control simply returns to the proximal phase.
    """

    beta = beta_local.copy()
    if max_steps <= 0:
        return beta, 0

    objective = _restricted_objective(
        beta,
        gram_local,
        corr_local,
        lam,
        weights,
        local_group_indices,
        working_set,
    )

    steps_used = 0
    manifold_active_floor = max(10.0 * float(active_group_tol), 1e-9)
    manifold_crossing_floor = max(0.1 * float(active_group_tol), 1e-11)

    for _ in range(max_steps):
        active = tuple(
            label
            for label in working_set
            if np.linalg.norm(beta[local_group_indices[label]]) > manifold_active_floor
        )
        if not active:
            break

        active_cols = np.unique(
            np.concatenate([local_group_indices[label] for label in active])
        )
        pos = {int(col): j for j, col in enumerate(active_cols)}
        active_local = {
            label: np.asarray(
                [pos[int(col)] for col in local_group_indices[label]],
                dtype=int,
            )
            for label in active
        }

        b = beta[active_cols]
        G = gram_local[np.ix_(active_cols, active_cols)]
        c = corr_local[active_cols]
        gradient = G @ b - c
        hessian = G.copy()

        for label in active:
            loc = active_local[label]
            vec = b[loc]
            norm = float(np.linalg.norm(vec))
            threshold = float(lam * weights[label])

            gradient[loc] += threshold * vec / norm
            hessian[np.ix_(loc, loc)] += threshold * (
                np.eye(len(loc), dtype=float) / norm
                - np.outer(vec, vec) / (norm ** 3)
            )

        try:
            direction = np.linalg.solve(hessian, -gradient)
        except np.linalg.LinAlgError:
            direction = np.linalg.lstsq(hessian, -gradient, rcond=None)[0]

        grad_dot_direction = float(gradient @ direction)
        if (
            not np.all(np.isfinite(direction))
            or grad_dot_direction >= 0.0
            or np.linalg.norm(direction)
            <= 1e-14 * max(1.0, float(np.linalg.norm(b)))
        ):
            break

        accepted = False
        step = 1.0
        for _line_search in range(30):
            candidate = beta.copy()
            candidate[active_cols] = b + step * direction

            # Stay on the same smooth manifold during Newton polishing.
            if any(
                np.linalg.norm(candidate[local_group_indices[label]])
                <= manifold_crossing_floor
                for label in active
            ):
                step *= 0.5
                continue

            candidate_objective = _restricted_objective(
                candidate,
                gram_local,
                corr_local,
                lam,
                weights,
                local_group_indices,
                working_set,
            )
            if candidate_objective <= objective + 1e-4 * step * grad_dot_direction:
                beta = candidate
                objective = candidate_objective
                accepted = True
                break
            step *= 0.5

        steps_used += 1
        if not accepted:
            break

        ratio = _restricted_group_kkt_ratio(
            beta,
            gram_local,
            corr_local,
            lam,
            weights,
            local_group_indices,
            working_set,
            active_group_tol=active_group_tol,
        )
        if ratio <= kkt_tol:
            break

    return beta, int(steps_used)


def _solve_restricted(
    X: Array,
    y: Array,
    lam: float,
    weights: Mapping[GroupLabel, float],
    groups: Mapping[GroupLabel, Array],
    beta_start: Array,
    step_size: float,
    working_set: Sequence[GroupLabel],
    *,
    max_iter: int,
    tol: float,
    kkt_tol: float,
    active_group_tol: float,
):
    """Solve one working-set Group-LASSO problem without changing its objective.

    A purely first-order proximal solve can become extremely slow when temporal
    columns are highly coherent.  This routine therefore alternates two phases:

    1. locally preconditioned accelerated proximal-gradient iterations, which
       are allowed to add/drop groups inside the working set;
    2. damped Newton polishing on the current nonzero-group manifold.

    Convergence is certified by the *restricted* Group-LASSO KKT residual, not
    only by iterate-to-iterate change.  Omitted groups are audited separately by
    :func:`_kkt_violations`.
    """

    p = X.shape[1]
    working_set = tuple(working_set)
    beta = np.zeros(p, dtype=float)

    if not working_set:
        return beta, {
            "iterations": 0,
            "converged": True,
            "relative_change": 0.0,
            "restricted_kkt_ratio": 0.0,
        }

    active_cols = _active_columns(working_set, groups)
    beta[active_cols] = beta_start[active_cols]

    Xw = X[:, active_cols]
    col_to_local = {int(col): pos for pos, col in enumerate(active_cols)}
    local_group_indices = {
        label: np.asarray(
            [col_to_local[int(c)] for c in groups[label]],
            dtype=int,
        )
        for label in working_set
    }

    n = X.shape[0]
    gram_local = (Xw.T @ Xw) / n
    corr_local = (Xw.T @ y) / n

    # Local Lipschitz scaling is strictly no worse than the global step passed
    # by the outer path, and is often much better for a small working set.
    eigvals = np.linalg.eigvalsh(gram_local)
    local_lipschitz = max(float(eigvals[-1]), np.finfo(float).tiny)
    local_step = 1.0 / local_lipschitz
    if np.isfinite(step_size) and step_size > 0.0:
        local_step = max(local_step, float(step_size))

    b = beta[active_cols].copy()
    converged = False
    relative_change = np.inf
    restricted_kkt_ratio = np.inf
    iterations = 0

    # Short proximal bursts are enough to establish the active manifold; Newton
    # then deals with the ill-conditioned smooth directions.  If Newton cannot
    # safely continue, the next proximal burst is a valid fallback.
    proximal_burst = 150
    newton_burst = 20

    while iterations < max_iter:
        z = b.copy()
        momentum = 1.0
        burst = min(proximal_burst, max_iter - iterations)

        for _ in range(burst):
            gradient = gram_local @ z - corr_local
            proposal = z - local_step * gradient
            b_new = _prox_group_blocks_local(
                proposal,
                local_step * lam,
                weights,
                local_group_indices,
                working_set,
            )

            denom = max(float(np.linalg.norm(b)), 1e-12)
            relative_change = float(np.linalg.norm(b_new - b) / denom)

            momentum_new = 0.5 * (
                1.0 + np.sqrt(1.0 + 4.0 * momentum * momentum)
            )
            z_new = b_new + ((momentum - 1.0) / momentum_new) * (b_new - b)

            # Adaptive restart suppresses the common FISTA oscillation on highly
            # coherent designs while retaining acceleration on smooth segments.
            if float((z_new - b_new) @ (b_new - b)) > 0.0:
                z_new = b_new.copy()
                momentum_new = 1.0

            b = b_new
            z = z_new
            momentum = momentum_new
            iterations += 1

            if iterations % 10 == 0 or relative_change <= tol:
                restricted_kkt_ratio = _restricted_group_kkt_ratio(
                    b,
                    gram_local,
                    corr_local,
                    lam,
                    weights,
                    local_group_indices,
                    working_set,
                    active_group_tol=active_group_tol,
                )
                if restricted_kkt_ratio <= kkt_tol:
                    converged = True
                    break

        if converged or iterations >= max_iter:
            break

        remaining = max_iter - iterations
        b, newton_steps = _newton_polish_active_manifold(
            b,
            gram_local,
            corr_local,
            lam,
            weights,
            local_group_indices,
            working_set,
            active_group_tol=active_group_tol,
            kkt_tol=kkt_tol,
            max_steps=min(newton_burst, remaining),
        )
        iterations += int(newton_steps)

        restricted_kkt_ratio = _restricted_group_kkt_ratio(
            b,
            gram_local,
            corr_local,
            lam,
            weights,
            local_group_indices,
            working_set,
            active_group_tol=active_group_tol,
        )
        if restricted_kkt_ratio <= kkt_tol:
            converged = True
            break

        # If Newton had no admissible step, the next proximal phase still makes
        # progress and can change the active manifold.

    if not np.isfinite(restricted_kkt_ratio):
        restricted_kkt_ratio = _restricted_group_kkt_ratio(
            b,
            gram_local,
            corr_local,
            lam,
            weights,
            local_group_indices,
            working_set,
            active_group_tol=active_group_tol,
        )

    beta[active_cols] = b
    return beta, {
        "iterations": int(iterations),
        "converged": bool(converged),
        "relative_change": float(relative_change),
        "restricted_kkt_ratio": float(restricted_kkt_ratio),
    }

def _kkt_violations(
    X: Array,
    y: Array,
    beta: Array,
    lam: float,
    weights: Mapping[GroupLabel, float],
    groups: Mapping[GroupLabel, Array],
    group_order: Tuple[GroupLabel, ...],
    working_set: Sequence[GroupLabel],
    *,
    kkt_tol: float,
):
    residual = X @ beta - y
    gradient = (X.T @ residual) / X.shape[0]
    working = set(working_set)

    violations = []
    max_ratio = 0.0

    for label in group_order:
        if label in working:
            continue
        threshold = lam * weights[label]
        norm = float(np.linalg.norm(gradient[groups[label]]))
        if threshold <= 0.0:
            ratio = np.inf if norm > 0.0 else 0.0
        else:
            ratio = norm / threshold
        max_ratio = max(max_ratio, ratio)
        if ratio > 1.0 + kkt_tol:
            violations.append((label, ratio))

    violations.sort(key=lambda item: item[1], reverse=True)
    return violations, float(max_ratio)


def _solve_one_lambda(
    X: Array,
    y: Array,
    lam: float,
    weights: Mapping[GroupLabel, float],
    groups: Mapping[GroupLabel, Array],
    group_order: Tuple[GroupLabel, ...],
    beta_start: Array,
    step_size: float,
    working_set: Sequence[GroupLabel],
    *,
    max_iter: int,
    tol: float,
    kkt_tol: float,
    max_kkt_expansions: int,
    active_group_tol: float,
):
    working = set(working_set)
    beta = beta_start.copy()

    total_iterations = 0
    all_intermediate_converged = True
    final_restricted_converged = False
    final_relative_change = np.inf
    final_restricted_kkt_ratio = np.inf
    total_reactivated = 0
    expansions = 0
    final_max_kkt_ratio = np.inf

    while True:
        ordered_working = _ordered_subset(working, group_order)
        beta, info = _solve_restricted(
            X,
            y,
            lam,
            weights,
            groups,
            beta,
            step_size,
            ordered_working,
            max_iter=max_iter,
            tol=tol,
            kkt_tol=kkt_tol,
            active_group_tol=active_group_tol,
        )

        total_iterations += int(info["iterations"])
        final_restricted_converged = bool(info["converged"])
        all_intermediate_converged = (
            all_intermediate_converged and final_restricted_converged
        )
        final_relative_change = float(info["relative_change"])
        final_restricted_kkt_ratio = float(info["restricted_kkt_ratio"])

        violations, max_ratio = _kkt_violations(
            X,
            y,
            beta,
            lam,
            weights,
            groups,
            group_order,
            ordered_working,
            kkt_tol=kkt_tol,
        )
        final_max_kkt_ratio = max_ratio

        if not violations:
            break

        expansions += 1
        if expansions > max_kkt_expansions:
            raise RuntimeError(
                "Adaptive Group LASSO exceeded max_kkt_expansions="
                f"{max_kkt_expansions} at lambda={lam:.3e}."
            )

        new_groups = [label for label, _ in violations]
        total_reactivated += len(new_groups)
        working.update(new_groups)

    ordered_working = _ordered_subset(working, group_order)
    final_kkt_satisfied = bool(
        final_max_kkt_ratio <= 1.0 + kkt_tol
        and final_restricted_kkt_ratio <= kkt_tol
    )

    return beta, ordered_working, {
        "iterations": int(total_iterations),
        "converged": bool(final_restricted_converged),
        "all_intermediate_converged": bool(all_intermediate_converged),
        "relative_change": float(final_relative_change),
        "restricted_kkt_ratio": float(final_restricted_kkt_ratio),
        "kkt_expansions": int(expansions),
        "kkt_reactivations": int(total_reactivated),
        "working_set_size": int(len(ordered_working)),
        "max_kkt_ratio": float(final_max_kkt_ratio),
        "final_kkt_satisfied": bool(final_kkt_satisfied),
        "kkt_certified": bool(final_restricted_converged and final_kkt_satisfied),
    }


def _active_groups(
    beta: Array,
    groups: Mapping[GroupLabel, Array],
    group_order: Tuple[GroupLabel, ...],
    *,
    active_group_tol: float,
) -> Tuple[GroupLabel, ...]:
    return tuple(
        label
        for label in group_order
        if np.linalg.norm(beta[groups[label]]) > active_group_tol
    )


def _next_inactive_entry_score(
    X: Array,
    y: Array,
    beta: Array,
    weights: Mapping[GroupLabel, float],
    groups: Mapping[GroupLabel, Array],
    group_order: Tuple[GroupLabel, ...],
    working_set: Sequence[GroupLabel],
):
    residual = X @ beta - y
    gradient = (X.T @ residual) / X.shape[0]
    working = set(working_set)

    q_max = 0.0
    group_max = None
    for label in group_order:
        if label in working:
            continue
        w = float(weights[label])
        if w <= 0.0 or not np.isfinite(w):
            continue
        q = float(np.linalg.norm(gradient[groups[label]]) / w)
        if q > q_max:
            q_max = q
            group_max = label

    return float(q_max), group_max


def _adaptive_next_lambda(
    lam: float,
    lambda_max: float,
    q_max: float,
    *,
    lambda_min_ratio: float,
    adaptive_geometric_ratio: float,
    adaptive_entry_fraction: float,
    adaptive_max_jump_decades: float,
) -> float:
    safety = float(lambda_max * lambda_min_ratio)
    geometric = float(adaptive_geometric_ratio * lam)

    if np.isfinite(q_max) and q_max > 0.0:
        event = float(adaptive_entry_fraction * q_max)
        candidate = min(geometric, event)
    else:
        candidate = geometric

    max_jump_floor = float(lam * (10.0 ** (-adaptive_max_jump_decades)))
    candidate = max(candidate, max_jump_floor, safety)

    if candidate >= lam * (1.0 - 1e-12):
        candidate = max(geometric, safety)

    return float(candidate)


# -----------------------------------------------------------------------------
# Noise-aware grouped basis-pursuit denoising
# -----------------------------------------------------------------------------


def _validate_group_weights(
    group_order: Tuple[GroupLabel, ...],
    group_weights: Optional[Mapping[GroupLabel, float]],
) -> Dict[GroupLabel, float]:
    """Return a complete positive finite group-weight dictionary."""

    if group_weights is None:
        return {label: 1.0 for label in group_order}

    supplied = set(group_weights.keys())
    expected = set(group_order)
    missing = expected - supplied
    extra = supplied - expected
    if missing or extra:
        raise ValueError(
            "group_weights must contain exactly the same labels as groups; "
            f"missing={list(missing)[:10]!r}, extra={list(extra)[:10]!r}."
        )

    clean: Dict[GroupLabel, float] = {}
    for label in group_order:
        value = float(group_weights[label])
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"group weight for {label!r} must be positive and finite."
            )
        clean[label] = value
    return clean


def _group_soft_threshold(
    vector: Array,
    threshold: float,
    groups: Mapping[GroupLabel, Array],
    group_order: Tuple[GroupLabel, ...],
    weights: Mapping[GroupLabel, float],
) -> Array:
    """Proximal map of ``threshold * sum_g w_g ||beta_g||_2``."""

    out = np.asarray(vector, dtype=float).copy()
    for label in group_order:
        idx = groups[label]
        norm = float(np.linalg.norm(out[idx]))
        cutoff = float(threshold * weights[label])
        if norm <= cutoff:
            out[idx] = 0.0
        else:
            out[idx] *= 1.0 - cutoff / norm
    return out


class _DenseResidualBallProjector:
    """Euclidean projector onto ``{beta: ||X beta - y||_2 <= delta}``.

    One thin SVD is cached.  Each subsequent projection is reduced to a scalar
    monotone root solve for the KKT multiplier.  No TIDES-specific assumptions
    enter here.
    """

    def __init__(
        self,
        X: Array,
        y: Array,
        radius: float,
        *,
        lambda_bisection_iterations: int = 70,
    ) -> None:
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float).reshape(-1)
        self.radius = float(radius)
        self.lambda_bisection_iterations = int(lambda_bisection_iterations)

        if self.radius < 0.0 or not np.isfinite(self.radius):
            raise ValueError("residual radius must be non-negative and finite.")
        if self.lambda_bisection_iterations < 20:
            raise ValueError("lambda_bisection_iterations must be >= 20.")

        U_full, singular_full, Vt_full = np.linalg.svd(
            self.X,
            full_matrices=False,
        )
        self.singular_values_full = singular_full.copy()

        if singular_full.size == 0 or singular_full[0] <= 0.0:
            rank_tol = np.inf
            keep = np.zeros_like(singular_full, dtype=bool)
        else:
            rank_tol = float(
                np.finfo(float).eps
                * max(self.X.shape)
                * singular_full[0]
            )
            keep = singular_full > rank_tol

        self.design_rank = int(np.sum(keep))
        self.rank_tolerance = float(rank_tol)
        self.U = U_full[:, keep]
        self.singular = singular_full[keep]
        self.Vt = Vt_full[keep, :]
        self.singular_squared = self.singular * self.singular
        self.Uty = self.U.T @ self.y

        # Directions below the standard numerical-rank threshold are treated as
        # unresolved rather than exploited through enormous coefficients.  Their
        # target component therefore belongs to the irreducible residual.
        y_perp = self.y - self.U @ self.Uty
        self.irreducible_residual_norm = float(np.linalg.norm(y_perp))
        self.irreducible_residual_squared = float(y_perp @ y_perp)

        # Numerical feasibility of the residual ball.  The small tolerance only
        # protects against round-off in the orthogonal projection of y.
        feasibility_slack = 100.0 * np.finfo(float).eps * max(
            float(np.linalg.norm(self.y)),
            self.radius,
            1.0,
        )
        if self.irreducible_residual_norm > self.radius + feasibility_slack:
            raise ValueError(
                "The requested residual ball does not intersect the design "
                "column space: irreducible residual "
                f"{self.irreducible_residual_norm:.3e} exceeds radius "
                f"{self.radius:.3e}."
            )

    def project(self, beta: Array) -> Array:
        beta = np.asarray(beta, dtype=float).reshape(-1)
        coordinates = self.Vt @ beta
        active_residual = self.singular * coordinates - self.Uty
        residual_squared = float(
            active_residual @ active_residual
            + self.irreducible_residual_squared
        )

        if residual_squared <= self.radius * self.radius:
            return beta.copy()

        target_active_squared = max(
            self.radius * self.radius - self.irreducible_residual_squared,
            0.0,
        )
        if target_active_squared <= 0.0:
            # Exact projection onto the closest point in the column space.  The
            # formal infinite-multiplier limit is numerically represented by
            # removing all residual components associated with nonzero singular
            # values.  This branch is uncommon for TIDES because the supplied
            # uncertainty radius is positive.
            nonzero = self.singular > 0.0
            correction = np.zeros_like(active_residual)
            correction[nonzero] = (
                active_residual[nonzero] / self.singular[nonzero]
            )
            return beta - self.Vt.T @ correction

        def secular(lam: float) -> float:
            denominator = 1.0 + lam * self.singular_squared
            shrunk = active_residual / denominator
            return float(shrunk @ shrunk - target_active_squared)

        lo = 0.0
        hi = 1.0
        value_hi = secular(hi)
        # The secular function is monotone decreasing.  Geometric bracketing is
        # stable even when the requested radius is extremely small.
        for _ in range(320):
            if value_hi <= 0.0:
                break
            hi *= 10.0
            if not np.isfinite(hi):
                raise RuntimeError(
                    "Residual-ball projection failed to bracket the KKT "
                    "multiplier."
                )
            value_hi = secular(hi)
        else:
            raise RuntimeError(
                "Residual-ball projection exceeded multiplier bracketing limit."
            )

        for _ in range(self.lambda_bisection_iterations):
            if lo <= 0.0:
                mid = 0.5 * hi
            else:
                # Geometric bisection resolves many decades of multipliers much
                # more evenly than arithmetic bisection.
                mid = float(np.sqrt(lo * hi))
            if secular(mid) > 0.0:
                lo = mid
            else:
                hi = mid

        lam = hi
        denominator = 1.0 + lam * self.singular_squared
        correction_svd = (
            lam
            * self.singular
            / denominator
            * active_residual
        )
        return beta - self.Vt.T @ correction_svd


def _default_douglas_rachford_step(
    y: Array,
    largest_singular_value: float,
) -> float:
    """Scale-aware numerical step for Douglas--Rachford splitting.

    The parameter changes convergence speed but not the convex minimiser.  The
    scale ``||y|| / ||X||_2`` has coefficient units; multiplying it by 1e-3 has
    been conservative on the current high-accuracy benchmark and remains
    invariant under a common rescaling of the linear inverse problem.
    """

    coefficient_scale = float(np.linalg.norm(y)) / max(
        float(largest_singular_value),
        np.finfo(float).tiny,
    )
    return max(1.0e-3 * coefficient_scale, np.finfo(float).eps)


def solve_group_basis_pursuit_denoising(
    X: Array,
    y: Array,
    groups: Mapping[GroupLabel, Sequence[int]],
    *,
    residual_radius: Optional[float] = None,
    target_relative_residual: Optional[float] = None,
    group_weights: Optional[Mapping[GroupLabel, float]] = None,
    max_iter: int = 10000,
    tol: float = 1e-8,
    check_every: int = 50,
    douglas_rachford_step: Optional[float] = None,
    lambda_bisection_iterations: int = 70,
    verbose: bool = True,
) -> GroupBasisPursuitResult:
    """Solve noise-aware grouped basis-pursuit denoising.

    The optimisation problem is

    ``min_beta sum_g w_g ||beta_g||_2``

    subject to

    ``||y - X @ beta||_2 <= residual_radius``.

    Exactly one of ``residual_radius`` and ``target_relative_residual`` must be
    supplied.  The latter is converted to an absolute radius using ``||y||_2``.

    Notes
    -----
    * The returned continuous solution is *not* thresholded.
    * ``ranking`` orders groups by unweighted coefficient norm.  The caller may
      use that ordering for support certification, but support size selection is
      deliberately outside this generic solver.
    * ``group_weights`` are optional positive convex-penalty weights.  The
      default is the theoretically simplest unweighted group ``l2,1`` norm.
    * The current backend is dense-SVD Douglas--Rachford.  It is intended as the
      correctness/reference implementation before a future matrix-free backend.
    """

    X_raw, y, groups_clean, group_order = _validate_problem(X, y, groups)
    weights = _validate_group_weights(group_order, group_weights)

    if (residual_radius is None) == (target_relative_residual is None):
        raise ValueError(
            "Supply exactly one of residual_radius or target_relative_residual."
        )

    y_norm = float(np.linalg.norm(y))
    if target_relative_residual is not None:
        relative_target = float(target_relative_residual)
        if not np.isfinite(relative_target) or relative_target < 0.0:
            raise ValueError("target_relative_residual must be finite and >= 0.")
        if y_norm <= np.finfo(float).tiny:
            raise ValueError(
                "target_relative_residual is undefined for a zero target; "
                "supply residual_radius instead."
            )
        radius = relative_target * y_norm
    else:
        radius = float(residual_radius)
        if not np.isfinite(radius) or radius < 0.0:
            raise ValueError("residual_radius must be finite and >= 0.")
        relative_target = (
            radius / y_norm if y_norm > np.finfo(float).tiny else np.nan
        )

    if max_iter < 1:
        raise ValueError("max_iter must be >= 1.")
    if tol <= 0.0 or not np.isfinite(tol):
        raise ValueError("tol must be positive and finite.")
    if check_every < 1:
        raise ValueError("check_every must be >= 1.")

    projector = _DenseResidualBallProjector(
        X_raw,
        y,
        radius,
        lambda_bisection_iterations=lambda_bisection_iterations,
    )

    if projector.design_rank == 0:
        raise ValueError("The design has numerical rank zero.")
    largest_singular = float(projector.singular[0])
    if douglas_rachford_step is None:
        dr_step = _default_douglas_rachford_step(y, largest_singular)
    else:
        dr_step = float(douglas_rachford_step)
        if not np.isfinite(dr_step) or dr_step <= 0.0:
            raise ValueError("douglas_rachford_step must be positive and finite.")

    if verbose:
        print("Starting grouped basis-pursuit denoising...")
        print(
            f"  samples={X_raw.shape[0]} | coefficients={X_raw.shape[1]} | "
            f"groups={len(group_order)}"
        )
        print(
            f"  residual radius={radius:.3e} | "
            f"relative target={relative_target:.3e}"
        )
        print(
            f"  backend=dense-svd Douglas-Rachford | rank="
            f"{projector.design_rank}/{min(X_raw.shape)} | "
            f"DR step={dr_step:.3e}"
        )
        print(
            "  support selection=disabled here; solver returns group ranking only"
        )

    # Douglas--Rachford iterate.  The shadow sequence ``primal`` is always in
    # the residual ball because it is an exact projection onto the constraint.
    state = np.zeros(X_raw.shape[1], dtype=float)
    primal = projector.project(state)
    fixed_point_residual = np.inf
    iterations = 0
    converged = False
    stop_reason = f"maximum iterations reached ({max_iter})"

    for iteration in range(1, max_iter + 1):
        primal = projector.project(state)
        reflected = 2.0 * primal - state
        proximal = _group_soft_threshold(
            reflected,
            dr_step,
            groups_clean,
            group_order,
            weights,
        )
        displacement = proximal - primal
        state = state + displacement
        iterations = iteration

        if iteration == 1 or iteration % check_every == 0:
            fixed_point_residual = float(
                np.linalg.norm(displacement)
                / max(1.0, float(np.linalg.norm(primal)))
            )
            if verbose and (
                iteration == 1
                or iteration % max(check_every, 10 * check_every) == 0
            ):
                prediction = X_raw @ primal
                rel = _relative_residual(prediction, y)
                print(
                    f"  [iter {iteration:6d}] fixed-point="
                    f"{fixed_point_residual:.3e} | rel-res={rel:.3e}"
                )

            if fixed_point_residual <= tol:
                converged = True
                stop_reason = "Douglas-Rachford fixed-point tolerance reached"
                break

    # Recompute the final shadow point after the last state update so all
    # reported quantities correspond to the returned coefficient vector.
    coefficients = projector.project(state)
    prediction = X_raw @ coefficients
    residual_norm = float(np.linalg.norm(prediction - y))
    relative_residual = (
        residual_norm / y_norm if y_norm > np.finfo(float).tiny else np.nan
    )
    feasibility_tol = max(
        100.0 * np.finfo(float).eps * max(y_norm, radius, 1.0),
        1.0e-12 * max(radius, 1.0),
    )
    feasible = bool(residual_norm <= radius + feasibility_tol)

    scores = {
        label: float(np.linalg.norm(coefficients[groups_clean[label]]))
        for label in group_order
    }
    order_index = {label: i for i, label in enumerate(group_order)}
    ranking = tuple(
        sorted(
            group_order,
            key=lambda label: (-scores[label], order_index[label]),
        )
    )
    objective_value = float(
        sum(weights[label] * scores[label] for label in group_order)
    )

    if not feasible:
        converged = False
        stop_reason = "returned shadow iterate violates residual radius"

    if verbose:
        print("Grouped basis-pursuit denoising complete.")
        print(f"  stop reason={stop_reason}")
        print(
            f"  iterations={iterations} | fixed-point="
            f"{fixed_point_residual:.3e} | feasible={feasible}"
        )
        print(
            f"  relative residual={relative_residual:.3e} | "
            f"objective={objective_value:.6e}"
        )

    return GroupBasisPursuitResult(
        coefficients=coefficients.copy(),
        group_scores=dict(scores),
        ranking=tuple(ranking),
        objective_value=float(objective_value),
        residual_norm=float(residual_norm),
        relative_residual=float(relative_residual),
        residual_radius=float(radius),
        target_relative_residual=float(relative_target),
        feasible=bool(feasible),
        converged=bool(converged),
        stop_reason=str(stop_reason),
        iterations=int(iterations),
        fixed_point_residual=float(fixed_point_residual),
        douglas_rachford_step=float(dr_step),
        group_weights=dict(weights),
        design_rank=int(projector.design_rank),
        design_singular_values=projector.singular_values_full.copy(),
        irreducible_residual_norm=float(projector.irreducible_residual_norm),
    )


# Concise alias used by notebooks and future Step-2 orchestration.
solve_group_bpdn = solve_group_basis_pursuit_denoising



# -----------------------------------------------------------------------------
# Memory-scalable screened BPDN
# -----------------------------------------------------------------------------


def _validate_block_problem(
    y: Array,
    groups: Mapping[GroupLabel, Sequence[int]],
    *,
    n_coefficients: int,
) -> Tuple[Array, Dict[GroupLabel, Array], Tuple[GroupLabel, ...]]:
    """Validate a grouped problem without requiring a global design matrix."""

    y = np.asarray(y, dtype=float).reshape(-1)
    if y.ndim != 1 or y.size == 0:
        raise ValueError("y must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(y)):
        raise ValueError("y must be finite.")

    p = int(n_coefficients)
    if p < 1:
        raise ValueError("n_coefficients must be positive.")
    if not groups:
        raise ValueError("groups must be non-empty.")

    clean: Dict[GroupLabel, Array] = {}
    seen = np.zeros(p, dtype=int)
    for label, idx in groups.items():
        arr = np.asarray(idx, dtype=int).reshape(-1)
        if arr.size == 0:
            raise ValueError(f"Group {label!r} is empty.")
        if np.any(arr < 0) or np.any(arr >= p):
            raise ValueError(
                f"Group {label!r} contains out-of-range coefficient indices."
            )
        arr = np.unique(arr)
        clean[label] = arr
        seen[arr] += 1

    if np.any(seen > 1):
        overlap = np.flatnonzero(seen > 1)
        raise ValueError(
            "Screened grouped BPDN requires non-overlapping groups; "
            f"overlap at coefficient indices {overlap[:20].tolist()}."
        )
    if np.any(seen == 0):
        missing = np.flatnonzero(seen == 0)
        raise ValueError(
            "Every coefficient must belong to exactly one group; "
            f"ungrouped indices {missing[:20].tolist()}."
        )

    return y, clean, tuple(clean.keys())


def _coerce_group_block(
    label: GroupLabel,
    block: Array,
    *,
    n_rows: int,
    expected_width: int,
) -> Array:
    """Validate one lazily materialised group block."""

    G = np.asarray(block, dtype=float)
    if G.ndim != 2:
        raise ValueError(
            f"group_block_provider({label!r}) must return a 2D array."
        )
    if G.shape != (n_rows, expected_width):
        raise ValueError(
            f"group_block_provider({label!r}) returned shape {G.shape}; "
            f"expected {(n_rows, expected_width)}."
        )
    if not np.all(np.isfinite(G)):
        raise ValueError(f"Group block {label!r} contains non-finite values.")
    return G


def _fit_materialised_group_blocks(
    y: Array,
    selected: Sequence[GroupLabel],
    selected_blocks: Mapping[GroupLabel, Array],
):
    """Stable OLS refit and selected-span basis for lazily materialised groups."""

    n = y.size
    if not selected:
        residual = y.copy()
        return (
            np.empty(0, dtype=float),
            residual,
            1.0,
            0,
            np.inf,
            np.zeros((n, 0), dtype=float),
            np.zeros((n, 0), dtype=float),
        )

    A = np.column_stack([selected_blocks[label] for label in selected])
    scale = np.linalg.norm(A, axis=0)
    scale = np.where(scale > 1e-14, scale, 1.0)
    As = A / scale[None, :]

    U, singular, Vt = np.linalg.svd(As, full_matrices=False)
    if singular.size == 0 or singular[0] <= 0.0:
        residual = y.copy()
        return (
            np.zeros(A.shape[1], dtype=float),
            residual,
            1.0,
            0,
            np.inf,
            np.zeros((n, 0), dtype=float),
            A,
        )

    rank_tol = float(np.finfo(float).eps * max(As.shape) * singular[0])
    rank = int(np.sum(singular > rank_tol))
    if rank == 0:
        residual = y.copy()
        return (
            np.zeros(A.shape[1], dtype=float),
            residual,
            1.0,
            0,
            np.inf,
            np.zeros((n, 0), dtype=float),
            A,
        )

    Ur = U[:, :rank]
    sr = singular[:rank]
    Vr = Vt[:rank, :].T
    coef_scaled = Vr @ ((Ur.T @ y) / sr)
    coef = coef_scaled / scale
    prediction = A @ coef
    residual = y - prediction
    rel = _relative_residual(prediction, y)
    cond = (
        float(singular[0] / singular[rank - 1])
        if singular[rank - 1] > 0.0
        else np.inf
    )
    return coef, residual, float(rel), int(rank), float(cond), Ur, A


def _conditional_group_gain(
    G: Array,
    residual: Array,
    Q: Array,
) -> float:
    """Exact conditional least-squares residual reduction for one group."""

    if Q.shape[1]:
        G_perp = G - Q @ (Q.T @ G)
    else:
        G_perp = G

    coef, _, rank, _ = np.linalg.lstsq(G_perp, residual, rcond=None)
    if rank == 0:
        return 0.0
    fitted = G_perp @ coef
    return float(fitted @ fitted)


def solve_screened_group_basis_pursuit_denoising(
    y: Array,
    groups: Mapping[GroupLabel, Sequence[int]],
    *,
    n_coefficients: int,
    group_block_provider: Callable[[GroupLabel], Array],
    residual_radius: Optional[float] = None,
    target_relative_residual: Optional[float] = None,
    group_weights: Optional[Mapping[GroupLabel, float]] = None,
    initial_groups: Optional[Sequence[GroupLabel]] = None,
    max_screening_groups: Optional[int] = None,
    min_gain_ratio: float = 1e-14,
    conditional_gain_provider: Optional[
        Callable[[Array, Array, Tuple[GroupLabel, ...]], Tuple[GroupLabel, float]]
    ] = None,
    restricted_solver_kwargs: Optional[Mapping[str, object]] = None,
    require_restricted_convergence: bool = True,
    verbose: bool = True,
) -> ScreenedGroupBasisPursuitResult:
    """Solve grouped BPDN after a memory-scalable conditional feasibility screen.

    The global convex target is the same grouped BPDN used by
    :func:`solve_group_basis_pursuit_denoising`, but the full design matrix is
    never materialised.  Instead, ``group_block_provider(label)`` returns only
    the columns of one group.  Forward screening repeatedly adds the omitted
    group giving the largest *conditional* least-squares residual reduction,
    refitting the current working set after every addition.  Screening stops at
    the first working set whose unpenalised refit reaches the requested
    uncertainty ball.  The dense-SVD Douglas--Rachford BPDN reference solver is
    then run only on that restricted matrix.

    Parameters
    ----------
    y
        Target vector.
    groups
        Non-overlapping partition of the global coefficient vector.  The
        returned ``coefficients`` use exactly these global coordinates.
    n_coefficients
        Length of the global coefficient vector.  This is cheap to allocate
        even when the corresponding global design matrix is too large to form.
    group_block_provider
        Callable returning an ``(len(y), len(groups[label]))`` block.  The
        provider may construct/project a block lazily and discard it after the
        score is computed.  Only selected blocks are cached by this solver.
    residual_radius, target_relative_residual
        Exactly one must be supplied, with the same semantics as the dense BPDN
        solver.
    conditional_gain_provider
        Optional accelerated screening callback.  It receives ``(residual, Q,
        omitted_labels)`` and must return ``(best_label, best_gain)`` using the
        same exact conditional-gain criterion.  Problem-specific operator
        layers can use this hook to batch/vectorise the global scan.
    restricted_solver_kwargs
        Keyword arguments forwarded to the dense reference BPDN solve on the
        screened working set.  By default the restricted tolerance is ``1e-7``;
        the final support is still certified independently by the caller.

    Notes
    -----
    This routine is designed as a large-scale *ranking backend*.  It guarantees
    that the screened span can explain the data to the requested uncertainty
    radius before convex ranking is attempted, but it does not claim that tiny
    coefficients outside the screen are exactly zero in the full BPDN optimum.
    For TIDES the externally certified ranked-prefix refit remains the decisive
    support-selection stage.
    """

    y, groups_clean, group_order = _validate_block_problem(
        y,
        groups,
        n_coefficients=n_coefficients,
    )
    weights = _validate_group_weights(group_order, group_weights)

    if (residual_radius is None) == (target_relative_residual is None):
        raise ValueError(
            "Supply exactly one of residual_radius or target_relative_residual."
        )

    y_norm = float(np.linalg.norm(y))
    if target_relative_residual is not None:
        relative_target = float(target_relative_residual)
        if not np.isfinite(relative_target) or relative_target < 0.0:
            raise ValueError("target_relative_residual must be finite and >= 0.")
        if y_norm <= np.finfo(float).tiny:
            raise ValueError(
                "target_relative_residual is undefined for a zero target; "
                "supply residual_radius instead."
            )
        radius = relative_target * y_norm
    else:
        radius = float(residual_radius)
        if not np.isfinite(radius) or radius < 0.0:
            raise ValueError("residual_radius must be finite and >= 0.")
        relative_target = (
            radius / y_norm if y_norm > np.finfo(float).tiny else np.nan
        )

    if min_gain_ratio <= 0.0 or not np.isfinite(min_gain_ratio):
        raise ValueError("min_gain_ratio must be finite and positive.")

    if max_screening_groups is None:
        max_screening_groups = len(group_order)
    max_screening_groups = int(min(max_screening_groups, len(group_order)))
    if max_screening_groups < 1:
        raise ValueError("max_screening_groups must be >= 1.")

    selected: list[GroupLabel] = []
    selected_set: set[GroupLabel] = set()
    selected_blocks: Dict[GroupLabel, Array] = {}

    if initial_groups is not None:
        for label in initial_groups:
            if label not in groups_clean:
                raise KeyError(f"Unknown initial group {label!r}.")
            if label in selected_set:
                continue
            if len(selected) >= max_screening_groups:
                raise ValueError(
                    "initial_groups contains more entries than max_screening_groups."
                )
            G = _coerce_group_block(
                label,
                group_block_provider(label),
                n_rows=y.size,
                expected_width=len(groups_clean[label]),
            )
            selected.append(label)
            selected_set.add(label)
            selected_blocks[label] = G

    coef_screen, residual, rel, rank, cond, Q, A_selected = (
        _fit_materialised_group_blocks(y, selected, selected_blocks)
    )
    initial_energy = float(y @ y)
    screening_steps = 0

    if verbose:
        print("Starting screened grouped basis-pursuit denoising...")
        print(
            f"  samples={y.size} | global coefficients={int(n_coefficients)} | "
            f"global groups={len(group_order)}"
        )
        print(
            f"  residual radius={radius:.3e} | "
            f"relative target={relative_target:.3e}"
        )
        print(
            "  global design=lazy group blocks | "
            "restricted convex backend=dense-svd Douglas-Rachford"
        )

    floor_slack = max(
        100.0 * np.finfo(float).eps,
        1.0e-12 * max(relative_target, np.finfo(float).tiny),
    )

    while (
        rel > relative_target + floor_slack
        and len(selected) < max_screening_groups
    ):
        omitted = tuple(label for label in group_order if label not in selected_set)
        if not omitted:
            break

        if conditional_gain_provider is not None:
            best_group, best_gain = conditional_gain_provider(residual, Q, omitted)
            if best_group not in groups_clean or best_group in selected_set:
                raise ValueError(
                    "conditional_gain_provider returned an invalid/already-selected group."
                )
            best_gain = float(best_gain)
            best_block = _coerce_group_block(
                best_group,
                group_block_provider(best_group),
                n_rows=y.size,
                expected_width=len(groups_clean[best_group]),
            )
        else:
            best_group = None
            best_gain = -np.inf
            best_block = None
            for label in omitted:
                G = _coerce_group_block(
                    label,
                    group_block_provider(label),
                    n_rows=y.size,
                    expected_width=len(groups_clean[label]),
                )
                gain = _conditional_group_gain(G, residual, Q)
                if gain > best_gain:
                    best_group = label
                    best_gain = gain
                    best_block = G

        if (
            best_group is None
            or best_block is None
            or not np.isfinite(best_gain)
            or best_gain <= min_gain_ratio * max(initial_energy, 1.0)
        ):
            break

        selected.append(best_group)
        selected_set.add(best_group)
        selected_blocks[best_group] = best_block
        screening_steps += 1

        coef_screen, residual, rel, rank, cond, Q, A_selected = (
            _fit_materialised_group_blocks(y, selected, selected_blocks)
        )

        if verbose and (
            screening_steps <= 5
            or screening_steps % 5 == 0
            or rel <= relative_target + floor_slack
        ):
            print(
                f"  [screen {screening_steps:4d}] "
                f"groups={len(selected):5d} | post-rel={rel:.3e} | "
                f"added={best_group!r}"
            )

    screening_groups = tuple(selected)
    screening_rel = float(rel)
    screening_floor_reached = bool(rel <= relative_target + floor_slack)

    if not screening_floor_reached:
        raise RuntimeError(
            "Conditional screening did not reach the requested residual ball: "
            f"post-refit relative residual={screening_rel:.6e}, "
            f"target={relative_target:.6e}, groups={len(screening_groups)}."
        )

    # Build the restricted matrix in *screening order*.  Only these columns are
    # retained in memory; the global design never exists as a dense array.
    A_work = np.column_stack([selected_blocks[label] for label in screening_groups])
    local_groups: Dict[GroupLabel, Array] = {}
    local_weights: Dict[GroupLabel, float] = {}
    offset = 0
    for label in screening_groups:
        width = len(groups_clean[label])
        local_groups[label] = np.arange(offset, offset + width, dtype=int)
        local_weights[label] = float(weights[label])
        offset += width

    restricted_kwargs = (
        {} if restricted_solver_kwargs is None else dict(restricted_solver_kwargs)
    )
    forbidden = {"residual_radius", "target_relative_residual", "group_weights"}.intersection(
        restricted_kwargs
    )
    if forbidden:
        raise ValueError(
            "restricted_solver_kwargs must not override the uncertainty radius "
            f"or group weights; forbidden keys={sorted(forbidden)!r}."
        )
    restricted_kwargs.setdefault("max_iter", 10000)
    restricted_kwargs.setdefault("tol", 1.0e-7)
    restricted_kwargs.setdefault("check_every", 50)
    restricted_kwargs.setdefault("verbose", verbose)

    if verbose:
        print("Conditional feasibility screen complete.")
        print(
            f"  screened groups={len(screening_groups)} | "
            f"restricted columns={A_work.shape[1]} | post-rel={screening_rel:.3e}"
        )
        print("Starting restricted convex BPDN ranking...")

    restricted = solve_group_basis_pursuit_denoising(
        A_work,
        y,
        local_groups,
        residual_radius=radius,
        group_weights=local_weights,
        **restricted_kwargs,
    )

    if require_restricted_convergence and not restricted.converged:
        raise RuntimeError(
            "Restricted grouped BPDN did not converge.  Increase the restricted "
            "iteration budget or relax only the numerical fixed-point tolerance."
        )
    if not restricted.feasible:
        raise RuntimeError(
            "Restricted grouped BPDN returned a point outside the requested "
            "uncertainty ball."
        )

    coefficients = np.zeros(int(n_coefficients), dtype=float)
    scores = {label: 0.0 for label in group_order}
    for label in screening_groups:
        local = local_groups[label]
        global_idx = groups_clean[label]
        coefficients[global_idx] = restricted.coefficients[local]
        scores[label] = float(np.linalg.norm(restricted.coefficients[local]))

    order_index = {label: i for i, label in enumerate(group_order)}
    ranking = tuple(
        sorted(
            group_order,
            key=lambda label: (-scores[label], order_index[label]),
        )
    )
    objective = float(
        sum(weights[label] * scores[label] for label in group_order)
    )

    # Prediction can be evaluated from the restricted matrix directly.
    prediction = A_work @ restricted.coefficients
    residual_norm = float(np.linalg.norm(prediction - y))
    relative_residual = (
        residual_norm / y_norm if y_norm > np.finfo(float).tiny else np.nan
    )
    feasibility_tol = max(
        100.0 * np.finfo(float).eps * max(y_norm, radius, 1.0),
        1.0e-12 * max(radius, 1.0),
    )
    feasible = bool(residual_norm <= radius + feasibility_tol)
    converged = bool(restricted.converged and feasible)
    stop_reason = (
        "conditional screen reached uncertainty ball; restricted BPDN converged"
        if converged
        else "restricted screened BPDN incomplete"
    )

    if verbose:
        print("Screened grouped basis-pursuit denoising complete.")
        print(f"  stop reason={stop_reason}")
        print(
            f"  global groups={len(group_order)} | screened={len(screening_groups)} | "
            f"restricted columns={A_work.shape[1]}"
        )
        print(
            f"  relative residual={relative_residual:.3e} | "
            f"objective={objective:.6e} | feasible={feasible}"
        )

    return ScreenedGroupBasisPursuitResult(
        coefficients=coefficients.copy(),
        group_scores=dict(scores),
        ranking=tuple(ranking),
        objective_value=float(objective),
        residual_norm=float(residual_norm),
        relative_residual=float(relative_residual),
        residual_radius=float(radius),
        target_relative_residual=float(relative_target),
        feasible=bool(feasible),
        converged=bool(converged),
        stop_reason=str(stop_reason),
        group_weights=dict(weights),
        screening_groups=tuple(screening_groups),
        screening_relative_residual=float(screening_rel),
        screening_steps=int(screening_steps),
        screening_rank=int(rank),
        screening_condition_number=float(cond),
        restricted_result=restricted,
        n_global_groups=int(len(group_order)),
        n_restricted_groups=int(len(screening_groups)),
        peak_restricted_columns=int(A_work.shape[1]),
    )


# Concise alias for the scalable screened ranking backend.
solve_screened_group_bpdn = solve_screened_group_basis_pursuit_denoising


# -----------------------------------------------------------------------------
# Post-selection refit
# -----------------------------------------------------------------------------


def _post_selection_refit(
    X_raw: Array,
    y: Array,
    active_groups: Sequence[GroupLabel],
    groups: Mapping[GroupLabel, Array],
):
    p = X_raw.shape[1]
    beta = np.zeros(p, dtype=float)

    if not active_groups:
        rel = _relative_residual(np.zeros_like(y), y)
        return beta, rel, 0, np.inf

    cols = np.unique(np.concatenate([groups[label] for label in active_groups]))
    A = X_raw[:, cols]

    # Scale only for numerical conditioning of the OLS solve; transform the
    # coefficients back to the original X scale afterwards.
    scale = np.linalg.norm(A, axis=0)
    scale = np.where(scale > 1e-14, scale, 1.0)
    As = A / scale[None, :]

    coef_scaled, _, rank, singular = np.linalg.lstsq(As, y, rcond=None)
    coef = coef_scaled / scale
    beta[cols] = coef

    pred = A @ coef
    rel = _relative_residual(pred, y)

    if len(singular) == 0 or singular[-1] <= 0.0:
        cond = np.inf
    else:
        cond = float(singular[0] / singular[-1])

    return beta, float(rel), int(rank), float(cond)



def _fit_support_with_residual(
    X_raw: Array,
    y: Array,
    active_groups: Sequence[GroupLabel],
    groups: Mapping[GroupLabel, Array],
):
    """Stable OLS refit plus residual and an orthonormal basis for one support."""

    p = X_raw.shape[1]
    beta = np.zeros(p, dtype=float)

    if not active_groups:
        residual = y.copy()
        rel = _relative_residual(np.zeros_like(y), y)
        return (
            beta,
            residual,
            float(rel),
            0,
            np.inf,
            np.zeros((X_raw.shape[0], 0), dtype=float),
        )

    cols = np.unique(np.concatenate([groups[label] for label in active_groups]))
    A = X_raw[:, cols]

    scale = np.linalg.norm(A, axis=0)
    scale = np.where(scale > 1e-14, scale, 1.0)
    As = A / scale[None, :]

    # SVD gives both a stable minimum-norm refit and the selected column-space
    # basis needed by the next forward screening step.
    U, singular, Vt = np.linalg.svd(As, full_matrices=False)
    if singular.size == 0 or singular[0] <= 0.0:
        residual = y.copy()
        return (
            beta,
            residual,
            1.0,
            0,
            np.inf,
            np.zeros((X_raw.shape[0], 0), dtype=float),
        )

    rank_tol = float(np.finfo(float).eps * max(As.shape) * singular[0])
    rank = int(np.sum(singular > rank_tol))
    if rank == 0:
        residual = y.copy()
        return (
            beta,
            residual,
            1.0,
            0,
            np.inf,
            np.zeros((X_raw.shape[0], 0), dtype=float),
        )

    Ur = U[:, :rank]
    sr = singular[:rank]
    Vr = Vt[:rank, :].T

    coef_scaled = Vr @ ((Ur.T @ y) / sr)
    coef = coef_scaled / scale
    beta[cols] = coef

    prediction = A @ coef
    residual = y - prediction
    rel = _relative_residual(prediction, y)

    cond = (
        float(singular[0] / singular[rank - 1])
        if singular[rank - 1] > 0.0
        else np.inf
    )

    return beta, residual, float(rel), int(rank), float(cond), Ur


def solve_forward_backward_group_search(
    X: Array,
    y: Array,
    groups: Mapping[GroupLabel, Sequence[int]],
    *,
    target_relative_floor: float,
    max_forward_groups: Optional[int] = None,
    min_gain_ratio: float = 1e-14,
    verbose: bool = True,
) -> ForwardBackwardGroupSearchResult:
    """Blind floor-driven grouped sparse search.

    The algorithm is designed for coherent grouped designs when convex
    Group-LASSO screening can follow surrogate supports.

    Forward screening adds the omitted group giving the largest exact
    conditional least-squares reduction of the current OLS residual after
    residualising that candidate group against the selected span.

    Once the independently supplied numerical resolution floor is reached,
    backward pruning repeatedly removes the group whose deletion yields the
    smallest refitted residual, accepting the deletion only while the floor
    remains satisfied.

    The procedure uses no oracle support size, topology, or benchmark truth.
    """

    X_raw, y, groups_clean, group_order = _validate_problem(X, y, groups)

    if target_relative_floor <= 0.0:
        raise ValueError("target_relative_floor must be positive.")
    if max_forward_groups is None:
        max_forward_groups = len(group_order)
    max_forward_groups = int(min(max_forward_groups, len(group_order)))
    if max_forward_groups < 1:
        raise ValueError("max_forward_groups must be >= 1.")
    if min_gain_ratio <= 0.0:
        raise ValueError("min_gain_ratio must be positive.")

    selected: list[GroupLabel] = []
    initial_energy = float(y @ y)
    forward_steps = 0

    beta, residual, rel, rank, cond, Q = _fit_support_with_residual(
        X_raw, y, selected, groups_clean
    )

    if verbose:
        print("Starting forward/backward group search...")
        print(
            f"  samples={X_raw.shape[0]} | coefficients={X_raw.shape[1]} | "
            f"groups={len(group_order)}"
        )
        print(f"  requested post-refit relative floor={target_relative_floor:.3e}")

    while (
        rel > target_relative_floor
        and len(selected) < max_forward_groups
    ):
        selected_set = set(selected)
        best_group = None
        best_gain = -np.inf

        for label in group_order:
            if label in selected_set:
                continue

            G = X_raw[:, groups_clean[label]]
            if Q.shape[1]:
                G_perp = G - Q @ (Q.T @ G)
            else:
                G_perp = G

            coef_g, _, rank_g, _ = np.linalg.lstsq(
                G_perp,
                residual,
                rcond=None,
            )
            if rank_g == 0:
                gain = 0.0
            else:
                fitted = G_perp @ coef_g
                gain = float(fitted @ fitted)

            if gain > best_gain:
                best_gain = gain
                best_group = label

        if (
            best_group is None
            or not np.isfinite(best_gain)
            or best_gain <= min_gain_ratio * max(initial_energy, 1.0)
        ):
            break

        selected.append(best_group)
        forward_steps += 1
        beta, residual, rel, rank, cond, Q = _fit_support_with_residual(
            X_raw,
            y,
            selected,
            groups_clean,
        )

        if verbose:
            print(
                f"  [forward {forward_steps:03d}] "
                f"groups={len(selected):4d} | post-rel={rel:.3e} | "
                f"added={best_group!r}"
            )

    screening_groups = tuple(selected)
    screening_rel = float(rel)
    floor_reached = bool(rel <= target_relative_floor)

    if not floor_reached:
        stop_reason = "forward screening did not reach target floor"
        if verbose:
            print("Forward/backward group search complete.")
            print(f"  stop reason={stop_reason}")
            print(
                f"  groups={len(selected)} | post-rel={rel:.3e} | "
                f"rank={rank} | cond={cond:.3e}"
            )
        return ForwardBackwardGroupSearchResult(
            coefficients=beta,
            selected_groups=tuple(selected),
            post_relative_residual=float(rel),
            post_rank=int(rank),
            post_condition_number=float(cond),
            floor_reached=False,
            stop_reason=stop_reason,
            screening_groups=screening_groups,
            screening_relative_residual=screening_rel,
            forward_steps=int(forward_steps),
            backward_steps=0,
        )

    backward_steps = 0

    while len(selected) > 1:
        best_group = None
        best_trial = None

        for label in tuple(selected):
            trial = [g for g in selected if g != label]
            trial_fit = _fit_support_with_residual(
                X_raw,
                y,
                trial,
                groups_clean,
            )
            rel_trial = float(trial_fit[2])

            if best_trial is None or rel_trial < float(best_trial[2]):
                best_group = label
                best_trial = trial_fit

        if best_trial is None or float(best_trial[2]) > target_relative_floor:
            break

        selected.remove(best_group)
        backward_steps += 1
        beta, residual, rel, rank, cond, Q = best_trial

        if verbose:
            print(
                f"  [prune   {backward_steps:03d}] "
                f"groups={len(selected):4d} | post-rel={rel:.3e} | "
                f"removed={best_group!r}"
            )

    stop_reason = (
        "forward screening reached floor; "
        "backward floor-preserving pruning complete"
    )

    if verbose:
        print("Forward/backward group search complete.")
        print(f"  stop reason={stop_reason}")
        print(
            f"  screening groups={len(screening_groups)} -> "
            f"selected groups={len(selected)}"
        )
        print(
            f"  post-selection relative residual={rel:.3e} | "
            f"rank={rank} | cond={cond:.3e}"
        )

    return ForwardBackwardGroupSearchResult(
        coefficients=beta,
        selected_groups=tuple(selected),
        post_relative_residual=float(rel),
        post_rank=int(rank),
        post_condition_number=float(cond),
        floor_reached=bool(rel <= target_relative_floor),
        stop_reason=stop_reason,
        screening_groups=screening_groups,
        screening_relative_residual=screening_rel,
        forward_steps=int(forward_steps),
        backward_steps=int(backward_steps),
    )


# -----------------------------------------------------------------------------
# Public solver
# -----------------------------------------------------------------------------


def solve_adaptive_group_lasso(
    X: Array,
    y: Array,
    groups: Mapping[GroupLabel, Sequence[int]],
    *,
    target_relative_floor: Optional[float] = None,
    gamma: float = 1.0,
    delta_ratio: float = 1e-8,
    max_iter: int = 5000,
    tol: float = 1e-8,
    active_group_tol: float = 1e-10,
    ridge_condition_target: float = 1e6,
    ridge_alpha_floor_ratio: float = 1e-12,
    kkt_tol: float = 1e-7,
    max_kkt_expansions: int = 100,
    lambda_min_ratio: float = 1e-10,
    adaptive_geometric_ratio: float = 0.75,
    adaptive_entry_fraction: float = 0.98,
    adaptive_max_jump_decades: float = 2.0,
    adaptive_max_evals: int = 60,
    verbose: bool = True,
) -> AdaptiveGroupLassoResult:
    """
    Solve a grouped sparse linear regression problem with Adaptive Group LASSO.

    Parameters
    ----------
    X, y
        Linear regression problem ``y ~= X @ beta``.
    groups
        Mapping ``label -> coefficient indices``. Groups must form a complete,
        non-overlapping partition of all columns of ``X``.
    target_relative_floor
        Optional post-selection relative-residual floor. If provided, adaptive
        continuation stops at the first KKT-certified path point whose
        unpenalised refit reaches this floor. The floor itself is supplied by the
        problem builder (for TIDES Step 2, the data-derived profile floor).

    Notes
    -----
    The solver owns numerical optimisation only. It does not know what a group
    means physically, how X/y were constructed, or how selected groups should be
    interpreted.
    """

    if gamma <= 0.0:
        raise ValueError("gamma must be positive.")
    if delta_ratio <= 0.0:
        raise ValueError("delta_ratio must be positive.")
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1.")
    if tol <= 0.0:
        raise ValueError("tol must be positive.")
    if active_group_tol < 0.0:
        raise ValueError("active_group_tol must be non-negative.")
    if ridge_condition_target <= 1.0:
        raise ValueError("ridge_condition_target must be > 1.")
    if ridge_alpha_floor_ratio <= 0.0:
        raise ValueError("ridge_alpha_floor_ratio must be positive.")
    if kkt_tol < 0.0:
        raise ValueError("kkt_tol must be non-negative.")
    if max_kkt_expansions < 1:
        raise ValueError("max_kkt_expansions must be >= 1.")
    if not (0.0 < lambda_min_ratio <= 1.0):
        raise ValueError("lambda_min_ratio must lie in (0, 1].")
    if not (0.0 < adaptive_geometric_ratio < 1.0):
        raise ValueError("adaptive_geometric_ratio must lie in (0, 1).")
    if not (0.0 < adaptive_entry_fraction < 1.0):
        raise ValueError("adaptive_entry_fraction must lie in (0, 1).")
    if adaptive_max_jump_decades <= 0.0:
        raise ValueError("adaptive_max_jump_decades must be positive.")
    if adaptive_max_evals < 2:
        raise ValueError("adaptive_max_evals must be >= 2.")
    if target_relative_floor is not None and target_relative_floor <= 0.0:
        raise ValueError("target_relative_floor must be positive when supplied.")

    X_raw, y, groups_clean, group_order = _validate_problem(X, y, groups)
    X_scaled, column_scale = _standardize_columns(X_raw)

    ridge_alpha, ridge_lambda_max, ridge_lambda_min = _adaptive_ridge_alpha(
        X_scaled,
        condition_target=ridge_condition_target,
        alpha_floor_ratio=ridge_alpha_floor_ratio,
    )
    pilot_scaled = _ridge_solve(X_scaled, y, ridge_alpha)
    weights = _adaptive_weights(
        pilot_scaled,
        groups_clean,
        gamma=gamma,
        delta_ratio=delta_ratio,
    )

    lambda_max = _lambda_max(X_scaled, y, groups_clean, weights)
    if not np.isfinite(lambda_max) or lambda_max <= 0.0:
        raise RuntimeError("Adaptive Group LASSO produced a non-positive lambda_max.")

    lipschitz = _power_largest_eigenvalue_design(X_scaled)
    if not np.isfinite(lipschitz) or lipschitz <= 0.0:
        raise RuntimeError("Adaptive Group LASSO produced a non-positive Lipschitz constant.")
    step_size = 1.0 / lipschitz

    safety_lambda = float(lambda_max * lambda_min_ratio)

    if verbose:
        print("Starting Adaptive Group LASSO (KKT-event-driven path)...")
        print(f"  samples={X_raw.shape[0]} | coefficients={X_raw.shape[1]} | groups={len(group_order)}")
        print(
            "  pilot=condition-adaptive Ridge | "
            f"alpha={ridge_alpha:.3e} | "
            f"Gram eig range=[{ridge_lambda_min:.3e}, {ridge_lambda_max:.3e}]"
        )
        print(
            f"  lambda_max={lambda_max:.3e} | emergency floor="
            f"{lambda_min_ratio:.0e} * lambda_max"
        )
        print(
            f"  continuation: geometric={adaptive_geometric_ratio:.3f} | "
            f"entry fraction={adaptive_entry_fraction:.3f} | "
            f"max jump={adaptive_max_jump_decades:.2f} decades"
        )
        if target_relative_floor is not None:
            print(f"  requested post-refit relative floor={target_relative_floor:.3e}")

    path = []
    beta_states_scaled = []
    beta_warm = np.zeros(X_raw.shape[1], dtype=float)
    working_set: Tuple[GroupLabel, ...] = tuple()
    total_reactivations = 0
    lam = float(lambda_max)
    stop_reason = ""
    selected_index: Optional[int] = None

    for path_index in range(adaptive_max_evals):
        beta_warm, working_set, info = _solve_one_lambda(
            X_scaled,
            y,
            lam,
            weights,
            groups_clean,
            group_order,
            beta_warm,
            step_size,
            working_set,
            max_iter=max_iter,
            tol=tol,
            kkt_tol=kkt_tol,
            max_kkt_expansions=max_kkt_expansions,
            active_group_tol=active_group_tol,
        )
        total_reactivations += int(info["kkt_reactivations"])

        active = _active_groups(
            beta_warm,
            groups_clean,
            group_order,
            active_group_tol=active_group_tol,
        )
        beta_refit, post_rel, post_rank, post_cond = _post_selection_refit(
            X_raw,
            y,
            active,
            groups_clean,
        )

        q_max, q_group = _next_inactive_entry_score(
            X_scaled,
            y,
            beta_warm,
            weights,
            groups_clean,
            group_order,
            working_set,
        )
        next_lam = _adaptive_next_lambda(
            lam,
            lambda_max,
            q_max,
            lambda_min_ratio=lambda_min_ratio,
            adaptive_geometric_ratio=adaptive_geometric_ratio,
            adaptive_entry_fraction=adaptive_entry_fraction,
            adaptive_max_jump_decades=adaptive_max_jump_decades,
        )

        point = AdaptiveGroupLassoPathPoint(
            path_index=int(path_index),
            lambda_value=float(lam),
            lambda_over_max=float(lam / lambda_max),
            active_groups=tuple(active),
            working_set=tuple(working_set),
            n_active_groups=int(len(active)),
            working_set_size=int(info["working_set_size"]),
            post_relative_residual=float(post_rel),
            post_rank=int(post_rank),
            post_condition_number=float(post_cond),
            solver_iterations=int(info["iterations"]),
            solver_converged=bool(info["converged"]),
            all_intermediate_converged=bool(info["all_intermediate_converged"]),
            restricted_kkt_ratio=float(info["restricted_kkt_ratio"]),
            kkt_expansions=int(info["kkt_expansions"]),
            kkt_reactivations=int(info["kkt_reactivations"]),
            max_kkt_ratio=float(info["max_kkt_ratio"]),
            final_kkt_satisfied=bool(info["final_kkt_satisfied"]),
            next_entry_lambda=float(q_max),
            next_entry_group=q_group,
            adaptive_next_lambda=float(next_lam),
        )
        path.append(point)
        beta_states_scaled.append(beta_warm.copy())

        if verbose:
            entry_ratio = q_max / lam if lam > 0.0 else np.nan
            next_ratio = next_lam / lam if lam > 0.0 else np.nan
            print(
                f"  [{path_index + 1:03d}/{adaptive_max_evals}] "
                f"lambda={lam:.3e} | groups={len(active):4d} | "
                f"WS={len(working_set):4d} | post-rel={post_rel:.3e} | "
                f"entry/lambda={entry_ratio:.3e} | next/lambda={next_ratio:.3e} | "
                f"KKT+={info['kkt_reactivations']:4d} | iter={info['iterations']:5d}"
            )

        if not bool(info["kkt_certified"]):
            stop_reason = "current lambda failed KKT certification"
            selected_index = path_index
            break

        if (
            target_relative_floor is not None
            and len(active) > 0
            and post_rel <= target_relative_floor
        ):
            stop_reason = "post-selection relative-residual floor reached"
            selected_index = path_index
            break

        if lam <= safety_lambda * (1.0 + 1e-12):
            stop_reason = "emergency lambda safety floor reached"
            selected_index = path_index
            break

        if next_lam >= lam * (1.0 - 1e-12):
            stop_reason = "adaptive continuation could not decrease lambda"
            selected_index = path_index
            break

        lam = float(next_lam)

    else:
        stop_reason = f"adaptive path reached max_evals={adaptive_max_evals}"

    if len(path) == 0:
        raise RuntimeError("Adaptive Group LASSO produced an empty path.")

    if selected_index is None:
        if target_relative_floor is not None:
            floor_indices = [
                i
                for i, point in enumerate(path)
                if point.post_relative_residual <= target_relative_floor
            ]
            if floor_indices:
                selected_index = floor_indices[0]
            else:
                selected_index = int(
                    np.argmin([point.post_relative_residual for point in path])
                )
        else:
            selected_index = len(path) - 1

    selected_point = path[selected_index]
    selected_groups = selected_point.active_groups
    beta_refit, post_rel, post_rank, post_cond = _post_selection_refit(
        X_raw,
        y,
        selected_groups,
        groups_clean,
    )

    # Transform the selected penalised solution back to the raw-X coefficient scale.
    # beta_scaled satisfies y ~= X_scaled @ beta_scaled = X_raw @ (beta_scaled/scale).
    beta_pen_raw = beta_states_scaled[selected_index] / column_scale

    floor_reached = bool(
        target_relative_floor is not None
        and post_rel <= target_relative_floor
    )
    all_path_kkt = bool(
        all(point.final_kkt_satisfied and point.solver_converged for point in path)
    )

    if verbose:
        print("Adaptive Group LASSO complete.")
        print(f"  stop reason={stop_reason}")
        print(
            f"  selected lambda={selected_point.lambda_value:.3e} | "
            f"groups={len(selected_groups)} | WS={selected_point.working_set_size}"
        )
        print(
            f"  post-selection relative residual={post_rel:.3e} | "
            f"rank={post_rank} | cond={post_cond:.3e}"
        )
        print(
            f"  selected KKT={selected_point.final_kkt_satisfied} | "
            f"all-path KKT={all_path_kkt} | total reactivations={total_reactivations}"
        )

    return AdaptiveGroupLassoResult(
        coefficients=beta_refit,
        penalized_coefficients=beta_pen_raw,
        selected_groups=tuple(selected_groups),
        selected_lambda=float(selected_point.lambda_value),
        post_relative_residual=float(post_rel),
        post_rank=int(post_rank),
        post_condition_number=float(post_cond),
        floor_reached=bool(floor_reached),
        stop_reason=str(stop_reason),
        lambda_max=float(lambda_max),
        ridge_alpha=float(ridge_alpha),
        ridge_lambda_max=float(ridge_lambda_max),
        ridge_lambda_min=float(ridge_lambda_min),
        column_scale=column_scale.copy(),
        pilot_coefficients_scaled=pilot_scaled.copy(),
        adaptive_weights=dict(weights),
        path=tuple(path),
        total_kkt_reactivations=int(total_reactivations),
        all_path_kkt_certified=bool(all_path_kkt),
    )


__all__ = [
    "GroupBasisPursuitResult",
    "ScreenedGroupBasisPursuitResult",
    "solve_group_basis_pursuit_denoising",
    "solve_group_bpdn",
    "solve_screened_group_basis_pursuit_denoising",
    "solve_screened_group_bpdn",
    "AdaptiveGroupLassoPathPoint",
    "AdaptiveGroupLassoResult",
    "ForwardBackwardGroupSearchResult",
    "solve_adaptive_group_lasso",
    "solve_forward_backward_group_search",
]
