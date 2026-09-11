"""TIF Step 2: infer the admissible structure of each vector-field change.

Step 2 is hypothesis-dependent.

Current implemented branch
--------------------------
``varying_structure`` / ``sparse_row``:
    the change in the lifted field, ΔB^(k), is assumed to be sparse in edge
    rows.  For the conservative pairwise representation, a local derivative
    jump is written as

        Δx_dot_k = D rho_k,

    where one effective amplitude rho_{k,m} is associated with each candidate
    edge.  Sparse recovery localizes the changed edge rows without attempting
    to recover the full ΔB row; Step 3 does that globally from the trajectory.

Planned branch
--------------
``varying_dynamics`` / ``coherent_dynamics``:
    a fixed topology with changing dynamics generally produces a coherent,
    non-row-sparse ΔB.  Its solver is intentionally left as a separate backend
    because its identifiability structure is different from sparse topology
    changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.linear_model import Lasso


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SparseRowConstraint:
    """Rows of ΔB^(k) allowed to be nonzero."""

    transition_index: int
    transition_time: float
    support: NDArray[np.int64]
    local_amplitudes: FloatArray
    jump: FloatArray
    relative_residual: float
    rank_on_support: int
    sigma_min_on_support: float
    condition_number_on_support: float


@dataclass(frozen=True)
class ChangeStructureResult:
    """Output of TIF Step 2."""

    hypothesis: str
    constraints: tuple[SparseRowConstraint, ...]
    metadata: dict

    @property
    def supports(self) -> tuple[NDArray[np.int64], ...]:
        return tuple(c.support for c in self.constraints)


def _validate_inputs(
    X: ArrayLike,
    t: ArrayLike,
    D: ArrayLike,
    transition_indices: Sequence[int],
) -> tuple[FloatArray, FloatArray, FloatArray, NDArray[np.int64]]:
    X = np.asarray(X, dtype=float)
    t = np.asarray(t, dtype=float)
    D = np.asarray(D, dtype=float)
    transitions = np.asarray(transition_indices, dtype=np.int64)

    if X.ndim != 2:
        raise ValueError("X must have shape (n_samples, n_nodes).")
    if t.ndim != 1 or t.size != X.shape[0]:
        raise ValueError("t must have len(t) == X.shape[0].")
    if D.ndim != 2 or D.shape[0] != X.shape[1]:
        raise ValueError("D must have shape (n_nodes, n_candidate_edges).")
    if not np.all(np.diff(t) > 0):
        raise ValueError("t must be strictly increasing.")
    if transitions.size and (
        np.any(transitions <= 0) or np.any(transitions >= X.shape[0] - 1)
    ):
        raise ValueError("Each transition index must have samples on both sides.")
    return X, t, D, transitions


def estimate_local_jump(
    X: ArrayLike,
    t: ArrayLike,
    transition_index: int,
    *,
    window: int = 1,
) -> FloatArray:
    """Estimate x_dot(t*+) - x_dot(t*-) from one-sided secants."""

    X = np.asarray(X, dtype=float)
    t = np.asarray(t, dtype=float)
    i = int(transition_index)
    if window < 1:
        raise ValueError("window must be >= 1.")
    if i - window < 0 or i + window >= X.shape[0]:
        raise ValueError("Not enough samples around transition for requested window.")

    dt = np.diff(t)
    V = np.diff(X, axis=0) / dt[:, None]
    left = V[i - window : i].mean(axis=0)
    right = V[i : i + window].mean(axis=0)
    return np.asarray(right - left, dtype=float)


def _ridge_pilot(A: FloatArray, y: FloatArray, ridge: float) -> FloatArray:
    gram = A.T @ A
    scale = max(float(np.linalg.norm(gram, ord=2)), 1.0)
    lam = float(ridge) * scale
    return np.linalg.solve(gram + lam * np.eye(gram.shape[0]), A.T @ y)


def _debiased_fit(A: FloatArray, y: FloatArray, support: NDArray[np.int64]) -> FloatArray:
    beta = np.zeros(A.shape[1], dtype=float)
    if support.size:
        beta[support], *_ = np.linalg.lstsq(A[:, support], y, rcond=None)
    return beta


def adaptive_lasso_support(
    A: ArrayLike,
    y: ArrayLike,
    *,
    ridge: float = 1e-8,
    gamma: float = 1.0,
    weight_floor: float = 1e-10,
    n_lambda: int = 80,
    lambda_ratio: float = 1e-8,
    residual_tol: float = 1e-9,
    coefficient_tol: float = 1e-10,
    max_support: Optional[int] = None,
    max_iter: int = 50_000,
    external_solver: Optional[
        Callable[[FloatArray, FloatArray], tuple[FloatArray, dict]]
    ] = None,
) -> tuple[FloatArray, NDArray[np.int64], dict]:
    """A transparent adaptive-L1 baseline for local sparse-row localization.

    The production TIF solver can later inject the existing KKT-certified
    Adaptive Group LASSO through ``external_solver``.  This baseline is useful
    for the current noiseless scale experiments and keeps Step 2 executable as a
    standalone module.
    """

    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if A.ndim != 2 or A.shape[0] != y.size:
        raise ValueError("A must be 2-D with A.shape[0] == len(y).")

    if external_solver is not None:
        beta, meta = external_solver(A, y)
        beta = np.asarray(beta, dtype=float).reshape(-1)
        support = np.flatnonzero(np.abs(beta) > coefficient_tol).astype(np.int64)
        return beta, support, dict(meta)

    # Ridge pilot gives adaptive weights.  In underdetermined incidence problems
    # the ridge solution is deliberately used only as a weighting device.
    pilot = _ridge_pilot(A, y, ridge=ridge)
    weights = 1.0 / np.maximum(np.abs(pilot), weight_floor) ** gamma

    # Transform weighted L1: theta_j = w_j beta_j, so A' = A / w_j.
    A_scaled = A / weights[None, :]
    n_obs = A.shape[0]
    alpha_max = float(np.max(np.abs(A_scaled.T @ y)) / max(n_obs, 1))
    if alpha_max == 0.0:
        beta = np.zeros(A.shape[1], dtype=float)
        return beta, np.empty(0, dtype=np.int64), {
            "selected_alpha": 0.0,
            "selection": "zero_jump",
            "pilot": pilot,
            "weights": weights,
        }

    alphas = alpha_max * np.geomspace(1.0, lambda_ratio, n_lambda)
    y_norm = max(float(np.linalg.norm(y)), np.finfo(float).eps)

    candidates: list[tuple[float, float, int, FloatArray, NDArray[np.int64]]] = []
    exact_candidates: list[tuple[int, float, FloatArray, NDArray[np.int64]]] = []

    for alpha in alphas:
        model = Lasso(
            alpha=float(alpha),
            fit_intercept=False,
            max_iter=max_iter,
            tol=1e-10,
            selection="cyclic",
        )
        model.fit(A_scaled, y)
        beta = model.coef_ / weights
        support = np.flatnonzero(np.abs(beta) > coefficient_tol).astype(np.int64)
        if max_support is not None and support.size > max_support:
            continue

        debiased = _debiased_fit(A, y, support)
        rel = float(np.linalg.norm(A @ debiased - y) / y_norm)
        rss = float(np.sum((A @ debiased - y) ** 2))
        k = int(support.size)
        # Small finite floor keeps BIC defined in the noiseless exact-fit regime.
        bic = n_obs * np.log(max(rss / max(n_obs, 1), 1e-30)) + k * np.log(max(n_obs, 2))
        candidates.append((bic, rel, k, debiased, support))
        if rel <= residual_tol:
            exact_candidates.append((k, rel, debiased, support))

    if exact_candidates:
        # Prefer the sparsest exact/near-exact explanation; break ties by residual.
        exact_candidates.sort(key=lambda x: (x[0], x[1]))
        k, rel, beta, support = exact_candidates[0]
        selection = "sparsest_residual_tolerance"
        selected_alpha = np.nan
    elif candidates:
        candidates.sort(key=lambda x: x[0])
        _, rel, k, beta, support = candidates[0]
        selection = "bic"
        selected_alpha = np.nan
    else:
        raise RuntimeError("No admissible adaptive-lasso candidate was produced.")

    meta = {
        "selection": selection,
        "selected_alpha": selected_alpha,
        "relative_residual": rel,
        "support_size": int(k),
        "pilot": pilot,
        "weights": weights,
        "alpha_grid": alphas,
    }
    return beta, support, meta


def infer_sparse_row_changes(
    X: ArrayLike,
    t: ArrayLike,
    D: ArrayLike,
    transition_indices: Sequence[int],
    *,
    window: int = 1,
    observable_projector: Optional[ArrayLike] = None,
    residual_tol: float = 1e-9,
    max_support: Optional[int] = None,
    external_solver: Optional[
        Callable[[FloatArray, FloatArray], tuple[FloatArray, dict]]
    ] = None,
) -> ChangeStructureResult:
    """Infer sparse changed rows of ΔB at all detected transitions.

    ``observable_projector`` may be used to remove known conservation/null
    directions before sparse recovery.  It should map node-space vectors into
    the effective observable space, i.e. have shape ``(d_obs, n_nodes)``.
    """

    X, t, D, transitions = _validate_inputs(X, t, D, transition_indices)
    P = None if observable_projector is None else np.asarray(observable_projector, dtype=float)
    if P is not None and (P.ndim != 2 or P.shape[1] != D.shape[0]):
        raise ValueError("observable_projector must have shape (d_obs, n_nodes).")

    A = D if P is None else P @ D
    d_obs = int(np.linalg.matrix_rank(A))
    if max_support is None:
        # Necessary known-support amplitude bound.  Blind support recovery may
        # require a stricter regime, but it should never exceed this dimension.
        max_support = d_obs

    constraints: list[SparseRowConstraint] = []
    solver_meta: list[dict] = []

    for i in transitions:
        jump_node = estimate_local_jump(X, t, int(i), window=window)
        y = jump_node if P is None else P @ jump_node

        beta, support, meta = adaptive_lasso_support(
            A,
            y,
            residual_tol=residual_tol,
            max_support=max_support,
            external_solver=external_solver,
        )

        y_norm = max(float(np.linalg.norm(y)), np.finfo(float).eps)
        rel = float(np.linalg.norm(A @ beta - y) / y_norm)

        if support.size:
            As = A[:, support]
            svals = np.linalg.svd(As, compute_uv=False)
            rank = int(np.linalg.matrix_rank(As))
            sigma_min = float(svals[-1])
            cond = float(svals[0] / svals[-1]) if svals[-1] > 0 else np.inf
        else:
            rank, sigma_min, cond = 0, np.nan, np.nan

        constraints.append(
            SparseRowConstraint(
                transition_index=int(i),
                transition_time=float(t[int(i)]),
                support=support,
                local_amplitudes=beta,
                jump=jump_node,
                relative_residual=rel,
                rank_on_support=rank,
                sigma_min_on_support=sigma_min,
                condition_number_on_support=cond,
            )
        )
        solver_meta.append(meta)

    return ChangeStructureResult(
        hypothesis="varying_structure",
        constraints=tuple(constraints),
        metadata={
            "backend": "sparse_row_adaptive_lasso",
            "observable_rank": d_obs,
            "max_support": int(max_support),
            "window": int(window),
            "solver": solver_meta,
        },
    )


def infer_coherent_dynamics_changes(*args, **kwargs) -> ChangeStructureResult:
    """Planned Step-2 backend for fixed topology / varying dynamics.

    A global dynamics switch generally changes many active rows of B coherently,
    so the sparse-row boundary does not apply in the same way.  This branch must
    exploit a low-dimensional/coherent change model across edges and potentially
    multiple stages.  It is deliberately not approximated by sparse AGLASSO.
    """

    raise NotImplementedError(
        "The coherent-dynamics Step-2 backend is intentionally pending the "
        "duality/identifiability formulation. Use hypothesis='varying_structure' "
        "for the current scale experiments."
    )


def infer_change_structure(
    X: ArrayLike,
    t: ArrayLike,
    D: ArrayLike,
    transition_indices: Sequence[int],
    *,
    hypothesis: Literal["varying_structure", "varying_dynamics"] = "varying_structure",
    **kwargs,
) -> ChangeStructureResult:
    """Unified public interface for TIF Step 2."""

    if hypothesis == "varying_structure":
        return infer_sparse_row_changes(X, t, D, transition_indices, **kwargs)
    if hypothesis == "varying_dynamics":
        return infer_coherent_dynamics_changes(
            X, t, D, transition_indices, **kwargs
        )
    raise ValueError(f"Unknown Step-2 hypothesis: {hypothesis!r}")
