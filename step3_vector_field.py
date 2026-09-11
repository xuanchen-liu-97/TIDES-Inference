"""TIF Step 3: reconstruct stage-wise lifted vector fields B.

Step 3 is the hypothesis-neutral reconstruction engine.  It receives transition
locations and admissible temporal-change constraints from Steps 1--2, but does
not interpret why those constraints hold microscopically.

For the currently implemented sparse-row change constraint,

    B^(r) = B^(1) + sum_{k<r} ΔB^(k),

where only selected rows of each ΔB^(k) are free.  The regression is solved with
an implicit scipy LinearOperator, avoiding explicit construction of the very
large stacked trajectory design matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse.linalg import LinearOperator, lsqr


FloatArray = NDArray[np.float64]
LibraryFunction = Callable[[FloatArray], FloatArray]


@dataclass(frozen=True)
class VectorFieldReconstructionResult:
    """Output of TIF Step 3."""

    B_anchor: FloatArray
    delta_B: tuple[FloatArray, ...]
    B_stages: FloatArray
    relative_residual: float
    residual_norm: float
    parameter_count: int
    n_observations: int
    used_sample_mask: NDArray[np.bool_]
    stage_of_sample: NDArray[np.int64]
    lsqr_info: dict


def _validate(
    X: ArrayLike,
    t: ArrayLike,
    D: ArrayLike,
    transition_indices: Sequence[int],
    change_supports: Sequence[Sequence[int]],
) -> tuple[FloatArray, FloatArray, FloatArray, NDArray[np.int64], tuple[NDArray[np.int64], ...]]:
    X = np.asarray(X, dtype=float)
    t = np.asarray(t, dtype=float)
    D = np.asarray(D, dtype=float)
    transitions = np.asarray(transition_indices, dtype=np.int64)
    supports = tuple(np.asarray(s, dtype=np.int64) for s in change_supports)

    if X.ndim != 2:
        raise ValueError("X must have shape (n_samples, n_nodes).")
    if t.ndim != 1 or t.size != X.shape[0]:
        raise ValueError("t must have len(t) == X.shape[0].")
    if D.ndim != 2 or D.shape[0] != X.shape[1]:
        raise ValueError("D must have shape (n_nodes, n_candidate_edges).")
    if len(supports) != transitions.size:
        raise ValueError("There must be one change support per transition.")
    if transitions.size and not np.all(np.diff(transitions) > 0):
        raise ValueError("transition_indices must be strictly increasing.")
    M = D.shape[1]
    for s in supports:
        if s.size and (np.min(s) < 0 or np.max(s) >= M):
            raise ValueError("A change support contains an invalid edge index.")
    return X, t, D, transitions, supports


def stagewise_derivative(
    X: ArrayLike,
    t: ArrayLike,
    transition_indices: Sequence[int],
) -> FloatArray:
    """Estimate derivatives without taking finite differences across switches."""

    X = np.asarray(X, dtype=float)
    t = np.asarray(t, dtype=float)
    transitions = np.asarray(transition_indices, dtype=int)
    n = X.shape[0]
    dX = np.empty_like(X, dtype=float)

    # Boundary sample itself is shared geometrically by the two stages.  For the
    # derivative estimate, give it to the stage on the right; Step 3 normally
    # excludes exact transition samples from regression anyway.
    starts = np.r_[0, transitions]
    stops = np.r_[transitions, n - 1]

    for r, (a, b) in enumerate(zip(starts, stops)):
        if r == 0:
            idx = np.arange(a, b + 1)
        else:
            idx = np.arange(a, b + 1)
        if idx.size < 3:
            raise ValueError("Each stage needs at least three samples for derivative estimation.")
        dX[idx] = np.gradient(X[idx], t[idx], axis=0, edge_order=2)

    return dX


def evaluate_library(Z: FloatArray, library: Sequence[LibraryFunction]) -> FloatArray:
    """Evaluate candidate functions on edge states.

    Parameters
    ----------
    Z : (n_samples, M)
    Returns
    -------
    Psi : (n_samples, M, L)
    """

    vals = []
    for psi in library:
        v = np.asarray(psi(Z), dtype=float)
        if v.shape != Z.shape:
            try:
                v = np.broadcast_to(v, Z.shape).astype(float, copy=False)
            except ValueError as exc:
                raise ValueError("Each library function must return/broadcast to Z.shape.") from exc
        vals.append(v)
    if not vals:
        raise ValueError("library must contain at least one candidate function.")
    return np.stack(vals, axis=-1)


def _extract_supports(change_constraints_or_supports) -> tuple[NDArray[np.int64], ...]:
    supports = []
    for item in change_constraints_or_supports:
        if hasattr(item, "support"):
            item = item.support
        supports.append(np.asarray(item, dtype=np.int64))
    return tuple(supports)


def reconstruct_vector_field(
    X: ArrayLike,
    t: ArrayLike,
    D: ArrayLike,
    library: Sequence[LibraryFunction],
    transition_indices: Sequence[int],
    change_constraints_or_supports,
    *,
    dXdt: Optional[ArrayLike] = None,
    intrinsic: Optional[Callable[[FloatArray], FloatArray] | ArrayLike] = None,
    exclude_radius: int = 1,
    atol: float = 1e-11,
    btol: float = 1e-11,
    iter_lim: Optional[int] = None,
) -> VectorFieldReconstructionResult:
    """Globally reconstruct B^(1) and all admissible ΔB^(k).

    The currently supported change constraint is a set of free edge rows for
    each ΔB block.  The anchor B^(1) is unrestricted in the candidate edge-
    function representation.
    """

    supports = _extract_supports(change_constraints_or_supports)
    X, t, D, transitions, supports = _validate(
        X, t, D, transition_indices, supports
    )
    n_samples, N = X.shape
    M = D.shape[1]
    L = len(library)
    R = transitions.size + 1

    if dXdt is None:
        dX = stagewise_derivative(X, t, transitions)
    else:
        dX = np.asarray(dXdt, dtype=float)
        if dX.shape != X.shape:
            raise ValueError("dXdt must have the same shape as X.")

    if intrinsic is None:
        F = np.zeros_like(X)
    elif callable(intrinsic):
        F = np.asarray(intrinsic(X), dtype=float)
    else:
        F = np.asarray(intrinsic, dtype=float)
    if F.shape != X.shape:
        raise ValueError("intrinsic must evaluate to the same shape as X.")
    Y_all = dX - F

    sample_ids = np.arange(n_samples)
    stage_all = np.searchsorted(transitions, sample_ids, side="right").astype(np.int64)

    mask = np.ones(n_samples, dtype=bool)
    if exclude_radius < 0:
        raise ValueError("exclude_radius must be >= 0.")
    for i in transitions:
        lo = max(0, int(i) - exclude_radius)
        hi = min(n_samples, int(i) + exclude_radius + 1)
        mask[lo:hi] = False

    X_use = X[mask]
    Y_use = Y_all[mask]
    stage_use = stage_all[mask]
    if X_use.shape[0] == 0:
        raise ValueError("No samples remain after transition exclusion.")

    # Candidate edge states and library responses.  The large node-time by
    # parameter design matrix is never explicitly formed.
    Z = X_use @ D
    Psi = evaluate_library(Z, library)  # (T, M, L)

    anchor_size = M * L
    delta_slices: list[slice] = []
    p = anchor_size
    for s in supports:
        q = p + int(s.size) * L
        delta_slices.append(slice(p, q))
        p = q
    n_params = p
    n_obs = X_use.shape[0] * N

    stage_indices = tuple(np.flatnonzero(stage_use == r) for r in range(R))

    def unpack(beta: FloatArray) -> tuple[FloatArray, list[FloatArray]]:
        B0 = beta[:anchor_size].reshape(M, L)
        deltas: list[FloatArray] = []
        for s, sl in zip(supports, delta_slices):
            dB = np.zeros((M, L), dtype=float)
            if s.size:
                dB[s, :] = beta[sl].reshape(s.size, L)
            deltas.append(dB)
        return B0, deltas

    def matvec(beta: FloatArray) -> FloatArray:
        beta = np.asarray(beta, dtype=float)
        B0, deltas = unpack(beta)
        B_stage: list[FloatArray] = [B0]
        current = B0.copy()
        for dB in deltas:
            current = current + dB
            B_stage.append(current.copy())

        pred = np.zeros_like(Y_use)
        for r, ids in enumerate(stage_indices):
            if ids.size == 0:
                continue
            # Edge response u_tm = sum_l B_ml psi_l(z_tm)
            U = np.einsum("ml,tml->tm", B_stage[r], Psi[ids], optimize=True)
            pred[ids] = U @ D.T
        return pred.ravel()

    def rmatvec(v: FloatArray) -> FloatArray:
        Rnode = np.asarray(v, dtype=float).reshape(-1, N)
        stage_grad: list[FloatArray] = []
        for r, ids in enumerate(stage_indices):
            if ids.size == 0:
                stage_grad.append(np.zeros((M, L), dtype=float))
                continue
            edge_residual = Rnode[ids] @ D  # (T_r, M)
            G = np.einsum(
                "tm,tml->ml", edge_residual, Psi[ids], optimize=True
            )
            stage_grad.append(G)

        grad = np.zeros(n_params, dtype=float)
        grad[:anchor_size] = np.sum(stage_grad, axis=0).ravel()

        # ΔB^(k) affects every stage r > k.  Build suffix sums of stage gradients.
        suffix = np.zeros((M, L), dtype=float)
        suffix_after: list[FloatArray] = [np.zeros((M, L), dtype=float) for _ in supports]
        for r in range(R - 1, 0, -1):
            suffix += stage_grad[r]
            suffix_after[r - 1] = suffix.copy()

        for k, (s, sl) in enumerate(zip(supports, delta_slices)):
            if s.size:
                grad[sl] = suffix_after[k][s, :].ravel()
        return grad

    Aop = LinearOperator(
        shape=(n_obs, n_params),
        matvec=matvec,
        rmatvec=rmatvec,
        dtype=float,
    )
    y = Y_use.ravel()

    if iter_lim is None:
        iter_lim = max(500, min(20_000, 4 * n_params))

    sol = lsqr(Aop, y, atol=atol, btol=btol, iter_lim=iter_lim)
    beta = np.asarray(sol[0], dtype=float)
    B0, deltas = unpack(beta)

    B_stages = np.empty((R, M, L), dtype=float)
    B_stages[0] = B0
    for r in range(1, R):
        B_stages[r] = B_stages[r - 1] + deltas[r - 1]

    residual = matvec(beta) - y
    residual_norm = float(np.linalg.norm(residual))
    y_norm = max(float(np.linalg.norm(y)), np.finfo(float).eps)

    info = {
        "istop": int(sol[1]),
        "iterations": int(sol[2]),
        "r1norm": float(sol[3]),
        "r2norm": float(sol[4]),
        "anorm": float(sol[5]),
        "acond": float(sol[6]),
        "arnorm": float(sol[7]),
        "xnorm": float(sol[8]),
    }

    return VectorFieldReconstructionResult(
        B_anchor=B0,
        delta_B=tuple(deltas),
        B_stages=B_stages,
        relative_residual=residual_norm / y_norm,
        residual_norm=residual_norm,
        parameter_count=n_params,
        n_observations=n_obs,
        used_sample_mask=mask,
        stage_of_sample=stage_all,
        lsqr_info=info,
    )
