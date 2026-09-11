"""
solvers_linear_regression.py
============================

Generic linear-regression solvers for TIDES.

This module contains numerical machinery only. It does not know about
stages, edges, vector fields, B^(1), Delta B, or any other TIDES-specific
semantics.

Primary public entry point
--------------------------
solve_least_squares(X, y, ...)

Backends
--------
1. dense_lstsq
   Column-scaled SVD least squares via numpy.linalg.lstsq. This is the
   correctness/reference solver for modest dense problems.

2. lsqr
   Iterative least squares via scipy.sparse.linalg.lsqr. This supports
   scipy sparse matrices and LinearOperator objects and is intended for
   larger-scale problems.

Column scaling is numerical preconditioning only. Returned coefficients are
always mapped back to the original, unscaled coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import scipy.sparse as sp
    from scipy.sparse.linalg import LinearOperator, aslinearoperator, lsqr
except Exception:  # scipy is optional until the iterative backend is used
    sp = None
    LinearOperator = None
    aslinearoperator = None
    lsqr = None


Array = np.ndarray


@dataclass(frozen=True)
class LinearRegressionResult:
    """Result of an unregularized linear least-squares fit."""

    coefficients: Array
    fitted_values: Array
    residual: Array

    residual_norm: float
    relative_residual: float
    relative_residual_by_output: Tuple[float, ...]

    n_samples: int
    n_features: int
    n_outputs: int

    method: str
    column_scaling: bool
    column_scales: Array

    # Exact dense diagnostics when available.
    rank: Optional[int]
    identifiable: Optional[bool]
    singular_values_scaled: Optional[Array]
    singular_values_raw: Optional[Array]
    condition_number_scaled: float
    condition_number_raw: float

    # Iterative-solver diagnostics.
    converged: bool
    iterations: int
    solver_status: Tuple[int, ...]
    normal_equation_relative_residual: float

    @property
    def full_column_rank(self) -> Optional[bool]:
        """Alias for identifiability of the supplied parameterization."""
        return self.identifiable


def _as_2d_target(y: Array, n_samples: int):
    y = np.asarray(y, dtype=float)

    if y.ndim == 1:
        if y.shape[0] != n_samples:
            raise ValueError(
                f"y has length {y.shape[0]}, expected {n_samples}."
            )
        return y[:, None], True

    if y.ndim == 2:
        if y.shape[0] != n_samples:
            raise ValueError(
                f"y has shape {y.shape}; first dimension must be {n_samples}."
            )
        return y, False

    raise ValueError("y must be one- or two-dimensional.")


def _safe_relative_norm(residual: Array, target: Array) -> float:
    denom = max(float(np.linalg.norm(target)), np.finfo(float).tiny)
    return float(np.linalg.norm(residual) / denom)


def _condition_from_singular_values(
    singular_values: Optional[Array],
    rank: Optional[int],
    n_features: int,
) -> float:
    if singular_values is None or len(singular_values) == 0:
        return np.nan

    if rank is not None and rank < n_features:
        return np.inf

    smax = float(singular_values[0])
    smin = float(singular_values[-1])

    if smin <= 0.0:
        return np.inf

    return float(smax / smin)


def _dense_column_scales(X: Array) -> Array:
    scales = np.sqrt(np.mean(np.square(X), axis=0))
    good = np.isfinite(scales) & (scales > 0.0)

    out = np.ones(X.shape[1], dtype=float)
    out[good] = scales[good]
    return out


def _sparse_column_scales(X) -> Array:
    squared_mean = np.asarray(X.power(2).mean(axis=0)).ravel()
    scales = np.sqrt(np.maximum(squared_mean, 0.0))
    good = np.isfinite(scales) & (scales > 0.0)

    out = np.ones(X.shape[1], dtype=float)
    out[good] = scales[good]
    return out


def _validate_user_scales(scales: Array, n_features: int) -> Array:
    scales = np.asarray(scales, dtype=float)

    if scales.shape != (n_features,):
        raise ValueError(
            f"column_scales must have shape ({n_features},), got {scales.shape}."
        )
    if not np.all(np.isfinite(scales)):
        raise ValueError("column_scales must be finite.")
    if np.any(scales <= 0.0):
        raise ValueError("column_scales must be strictly positive.")

    return scales.copy()


def _scaled_linear_operator(A, scales: Array):
    if LinearOperator is None or aslinearoperator is None:
        raise ImportError("scipy is required for LinearOperator least squares.")

    Aop = aslinearoperator(A)
    inv = 1.0 / scales

    def matvec(v):
        return Aop.matvec(inv * v)

    def rmatvec(u):
        return inv * Aop.rmatvec(u)

    def matmat(V):
        return Aop.matmat(inv[:, None] * V)

    def rmatmat(U):
        return inv[:, None] * Aop.rmatmat(U)

    return LinearOperator(
        shape=Aop.shape,
        matvec=matvec,
        rmatvec=rmatvec,
        matmat=matmat,
        rmatmat=rmatmat,
        dtype=float,
    )


def _normal_equation_relative_residual(X, residual: Array) -> float:
    """Small diagnostic for first-order least-squares stationarity."""
    if residual.ndim == 1:
        residual = residual[:, None]

    if isinstance(X, np.ndarray):
        grad = X.T @ residual
        denom = max(
            float(np.linalg.norm(X, ord="fro") * np.linalg.norm(residual)),
            np.finfo(float).tiny,
        )
        return float(np.linalg.norm(grad) / denom)

    if sp is not None and sp.issparse(X):
        grad = X.T @ residual
        xnorm = float(np.sqrt(X.multiply(X).sum()))
        denom = max(
            xnorm * float(np.linalg.norm(residual)),
            np.finfo(float).tiny,
        )
        return float(np.linalg.norm(grad) / denom)

    if aslinearoperator is not None:
        Aop = aslinearoperator(X)
        grad_cols = [
            Aop.rmatvec(residual[:, j])
            for j in range(residual.shape[1])
        ]
        grad = np.column_stack(grad_cols)
        denom = max(
            float(np.linalg.norm(grad)),
            float(np.linalg.norm(residual)),
            1.0,
        )
        return float(np.linalg.norm(grad) / denom)

    return np.nan


def _solve_dense_lstsq(
    X: Array,
    y2: Array,
    *,
    column_scaling: bool,
    column_scales: Optional[Array],
    rcond: Optional[float],
    compute_raw_svd_diagnostics: bool,
) -> LinearRegressionResult:

    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be two-dimensional.")

    n_samples, n_features = X.shape
    n_outputs = y2.shape[1]

    if not np.all(np.isfinite(X)):
        raise ValueError("X contains non-finite values.")
    if not np.all(np.isfinite(y2)):
        raise ValueError("y contains non-finite values.")

    if column_scales is not None:
        scales = _validate_user_scales(column_scales, n_features)
    elif column_scaling:
        scales = _dense_column_scales(X)
    else:
        scales = np.ones(n_features, dtype=float)

    Xs = X / scales[None, :]

    beta_scaled, _, rank, s_scaled = np.linalg.lstsq(
        Xs,
        y2,
        rcond=rcond,
    )

    beta = beta_scaled / scales[:, None]
    fitted = X @ beta
    residual = y2 - fitted

    residual_norm = float(np.linalg.norm(residual))
    relative_residual = _safe_relative_norm(residual, y2)

    rel_by_output = tuple(
        _safe_relative_norm(residual[:, j], y2[:, j])
        for j in range(n_outputs)
    )

    if compute_raw_svd_diagnostics:
        s_raw = np.linalg.svd(X, full_matrices=False, compute_uv=False)

        if len(s_raw):
            if rcond is None:
                cutoff = (
                    np.finfo(float).eps
                    * max(X.shape)
                    * float(s_raw[0])
                )
            else:
                cutoff = float(rcond) * float(s_raw[0])
            rank_raw = int(np.count_nonzero(s_raw > cutoff))
        else:
            rank_raw = 0

        cond_raw = _condition_from_singular_values(
            s_raw,
            rank_raw,
            n_features,
        )
    else:
        s_raw = None
        cond_raw = np.nan

    rank = int(rank)
    cond_scaled = _condition_from_singular_values(
        np.asarray(s_scaled, dtype=float),
        rank,
        n_features,
    )

    normal_rel = _normal_equation_relative_residual(X, residual)

    return LinearRegressionResult(
        coefficients=beta[:, 0] if n_outputs == 1 else beta,
        fitted_values=fitted[:, 0] if n_outputs == 1 else fitted,
        residual=residual[:, 0] if n_outputs == 1 else residual,
        residual_norm=residual_norm,
        relative_residual=relative_residual,
        relative_residual_by_output=rel_by_output,
        n_samples=n_samples,
        n_features=n_features,
        n_outputs=n_outputs,
        method="dense_lstsq",
        column_scaling=bool(column_scaling or column_scales is not None),
        column_scales=scales,
        rank=rank,
        identifiable=bool(rank == n_features),
        singular_values_scaled=np.asarray(s_scaled, dtype=float),
        singular_values_raw=(
            None if s_raw is None else np.asarray(s_raw, dtype=float)
        ),
        condition_number_scaled=cond_scaled,
        condition_number_raw=cond_raw,
        converged=True,
        iterations=1,
        solver_status=(0,) * n_outputs,
        normal_equation_relative_residual=normal_rel,
    )


def _solve_lsqr(
    X,
    y2: Array,
    *,
    column_scaling: bool,
    column_scales: Optional[Array],
    atol: float,
    btol: float,
    conlim: float,
    iter_lim: Optional[int],
) -> LinearRegressionResult:

    if sp is None or lsqr is None or aslinearoperator is None:
        raise ImportError("scipy is required for method='lsqr'.")

    n_samples, n_features = X.shape
    n_outputs = y2.shape[1]

    if column_scales is not None:
        scales = _validate_user_scales(column_scales, n_features)
    elif column_scaling:
        if sp.issparse(X):
            scales = _sparse_column_scales(X)
        elif isinstance(X, np.ndarray):
            scales = _dense_column_scales(np.asarray(X, dtype=float))
        else:
            raise ValueError(
                "For a LinearOperator with column_scaling=True, provide "
                "explicit column_scales."
            )
    else:
        scales = np.ones(n_features, dtype=float)

    if isinstance(X, np.ndarray):
        Xs = np.asarray(X, dtype=float) / scales[None, :]
    elif sp.issparse(X):
        Xs = X @ sp.diags(1.0 / scales)
    else:
        Xs = _scaled_linear_operator(X, scales)

    beta_scaled = np.empty((n_features, n_outputs), dtype=float)
    statuses = []
    iterations = []
    cond_estimates = []
    converged_flags = []

    for j in range(n_outputs):
        out = lsqr(
            Xs,
            y2[:, j],
            atol=float(atol),
            btol=float(btol),
            conlim=float(conlim),
            iter_lim=iter_lim,
            show=False,
        )

        beta_scaled[:, j] = out[0]
        istop = int(out[1])
        itn = int(out[2])
        acond = float(out[6])

        statuses.append(istop)
        iterations.append(itn)
        cond_estimates.append(acond)

        # scipy LSQR stop codes:
        # 1/2 = tolerance conditions reached;
        # 4/5 = machine-precision variants.
        converged_flags.append(istop in (1, 2, 4, 5))

    beta = beta_scaled / scales[:, None]

    if isinstance(X, np.ndarray) or sp.issparse(X):
        fitted = X @ beta
    else:
        Aop = aslinearoperator(X)
        fitted = np.column_stack(
            [Aop.matvec(beta[:, j]) for j in range(n_outputs)]
        )

    fitted = np.asarray(fitted, dtype=float)
    residual = y2 - fitted

    residual_norm = float(np.linalg.norm(residual))
    relative_residual = _safe_relative_norm(residual, y2)
    rel_by_output = tuple(
        _safe_relative_norm(residual[:, j], y2[:, j])
        for j in range(n_outputs)
    )

    normal_rel = _normal_equation_relative_residual(X, residual)

    finite_cond = [
        c for c in cond_estimates
        if np.isfinite(c) and c >= 0.0
    ]
    cond_est = max(finite_cond) if finite_cond else np.nan

    return LinearRegressionResult(
        coefficients=beta[:, 0] if n_outputs == 1 else beta,
        fitted_values=fitted[:, 0] if n_outputs == 1 else fitted,
        residual=residual[:, 0] if n_outputs == 1 else residual,
        residual_norm=residual_norm,
        relative_residual=relative_residual,
        relative_residual_by_output=rel_by_output,
        n_samples=n_samples,
        n_features=n_features,
        n_outputs=n_outputs,
        method="lsqr",
        column_scaling=bool(column_scaling or column_scales is not None),
        column_scales=scales,
        rank=None,
        identifiable=None,
        singular_values_scaled=None,
        singular_values_raw=None,
        condition_number_scaled=float(cond_est),
        condition_number_raw=np.nan,
        converged=bool(all(converged_flags)),
        iterations=max(iterations, default=0),
        solver_status=tuple(statuses),
        normal_equation_relative_residual=normal_rel,
    )


def solve_least_squares(
    X,
    y,
    *,
    method: str = "auto",
    column_scaling: bool = True,
    column_scales: Optional[Array] = None,
    rcond: Optional[float] = None,
    compute_raw_svd_diagnostics: bool = True,
    atol: float = 1e-12,
    btol: float = 1e-12,
    conlim: float = 1e12,
    iter_lim: Optional[int] = None,
    verbose: bool = False,
) -> LinearRegressionResult:
    """
    Solve an unregularized linear least-squares problem.

    Parameters
    ----------
    X
        Design matrix of shape (n_samples, n_features). Dense numpy arrays,
        scipy sparse matrices, and scipy LinearOperator objects are supported.

    y
        Target of shape (n_samples,) or (n_samples, n_outputs).

    method
        "auto", "dense_lstsq", or "lsqr". "auto" uses dense_lstsq for a dense
        numpy array and lsqr otherwise.

    column_scaling
        If True, solve after RMS column normalization and map coefficients back
        to the original coordinates.

    column_scales
        Optional positive scale vector supplied by the caller. Required for
        generic LinearOperator problems when column_scaling=True.

    rcond
        Rank cutoff passed to numpy.linalg.lstsq for the dense backend.

    compute_raw_svd_diagnostics
        Dense backend only. If True, also report singular values / condition
        number of the original unscaled design.

    atol, btol, conlim, iter_lim
        scipy.sparse.linalg.lsqr controls.

    Returns
    -------
    LinearRegressionResult
        Coefficients are expressed in the original unscaled design coordinates.
    """

    if not hasattr(X, "shape") or len(X.shape) != 2:
        raise ValueError("X must expose a two-dimensional shape.")

    n_samples, n_features = int(X.shape[0]), int(X.shape[1])

    if n_samples < 1 or n_features < 1:
        raise ValueError("X must have at least one sample and one feature.")

    y2, _ = _as_2d_target(y, n_samples)

    method = str(method).lower()

    if method == "auto":
        method = "dense_lstsq" if isinstance(X, np.ndarray) else "lsqr"

    if method not in {"dense_lstsq", "lsqr"}:
        raise ValueError(
            "method must be one of {'auto', 'dense_lstsq', 'lsqr'}."
        )

    if verbose:
        print("Starting linear least-squares solve...")
        print(
            f"  samples={n_samples} | features={n_features} | "
            f"outputs={y2.shape[1]} | method={method}"
        )

    if method == "dense_lstsq":
        if not isinstance(X, np.ndarray):
            if sp is not None and sp.issparse(X):
                raise ValueError(
                    "dense_lstsq does not densify sparse matrices implicitly; "
                    "use method='lsqr' or convert explicitly."
                )
            raise ValueError("dense_lstsq requires a dense numpy.ndarray.")

        result = _solve_dense_lstsq(
            X,
            y2,
            column_scaling=column_scaling,
            column_scales=column_scales,
            rcond=rcond,
            compute_raw_svd_diagnostics=compute_raw_svd_diagnostics,
        )

    else:
        result = _solve_lsqr(
            X,
            y2,
            column_scaling=column_scaling,
            column_scales=column_scales,
            atol=atol,
            btol=btol,
            conlim=conlim,
            iter_lim=iter_lim,
        )

    if verbose:
        print("Linear least-squares solve complete.")
        print(
            f"  relative residual={result.relative_residual:.3e} | "
            f"normal-eq residual={result.normal_equation_relative_residual:.3e}"
        )

        if result.rank is not None:
            print(
                f"  rank={result.rank}/{result.n_features} | "
                f"cond(scaled)={result.condition_number_scaled:.3e} | "
                f"cond(raw)={result.condition_number_raw:.3e}"
            )
        else:
            print(
                f"  LSQR converged={result.converged} | "
                f"iterations={result.iterations} | "
                f"cond-est(scaled)={result.condition_number_scaled:.3e}"
            )

    return result


__all__ = [
    "LinearRegressionResult",
    "solve_least_squares",
]
