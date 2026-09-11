"""TIF Step 1: change-point detection in a continuous trajectory.

Scientific role
---------------
Step 1 answers only one question: *when does the observed vector field change?*
It does not assume whether the source is a network/topology change or a change in
interaction dynamics.

The current implementation is intentionally simple and transparent for the
piecewise-stationary, noiseless/low-noise setting used by the first TIF
benchmarks.  More sophisticated change-point detectors can later replace this
module without changing Steps 2--4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import find_peaks


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ChangeDetectionResult:
    """Output of TIF Step 1."""

    transition_indices: NDArray[np.int64]
    transition_times: FloatArray
    score_indices: NDArray[np.int64]
    score_times: FloatArray
    scores: FloatArray
    secant_velocities: FloatArray
    stage_of_sample: NDArray[np.int64]
    method: str

    @property
    def n_transitions(self) -> int:
        return int(self.transition_indices.size)

    @property
    def n_stages(self) -> int:
        return self.n_transitions + 1


def _validate_trajectory(X: ArrayLike, t: ArrayLike) -> tuple[FloatArray, FloatArray]:
    X = np.asarray(X, dtype=float)
    t = np.asarray(t, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must have shape (n_samples, n_nodes).")
    if t.ndim != 1 or t.size != X.shape[0]:
        raise ValueError("t must be one-dimensional with len(t) == X.shape[0].")
    if X.shape[0] < 4:
        raise ValueError("At least four samples are required for change detection.")
    if not np.all(np.diff(t) > 0):
        raise ValueError("t must be strictly increasing.")
    return X, t


def secant_velocities(X: ArrayLike, t: ArrayLike) -> FloatArray:
    """Return interval-wise secant velocities.

    ``V[n]`` approximates the velocity on the interval ``[t[n], t[n+1]]``.
    """

    X, t = _validate_trajectory(X, t)
    dt = np.diff(t)
    return np.diff(X, axis=0) / dt[:, None]


def change_scores(
    X: ArrayLike,
    t: ArrayLike,
    *,
    method: Literal["secant", "window"] = "secant",
    window: int = 1,
    norm: Literal["l2", "l1", "linf"] = "l2",
) -> tuple[NDArray[np.int64], FloatArray, FloatArray]:
    """Compute a vector-field change score at candidate sample boundaries.

    A candidate boundary ``i`` lies at ``t[i]``.  The left velocity is estimated
    from secants ending at ``i`` and the right velocity from secants starting at
    ``i``.  ``window=1`` gives the sharp secant-jump score.
    """

    X, t = _validate_trajectory(X, t)
    if window < 1:
        raise ValueError("window must be >= 1.")

    V = secant_velocities(X, t)
    n = X.shape[0]

    if method == "secant":
        window = 1
    elif method != "window":
        raise ValueError(f"Unknown method: {method!r}")

    indices: list[int] = []
    jumps: list[np.ndarray] = []
    for i in range(window, n - window):
        left = V[i - window : i].mean(axis=0)
        right = V[i : i + window].mean(axis=0)
        indices.append(i)
        jumps.append(right - left)

    J = np.asarray(jumps, dtype=float)
    if norm == "l2":
        scores = np.linalg.norm(J, axis=1)
    elif norm == "l1":
        scores = np.linalg.norm(J, ord=1, axis=1)
    elif norm == "linf":
        scores = np.linalg.norm(J, ord=np.inf, axis=1)
    else:
        raise ValueError(f"Unknown norm: {norm!r}")

    idx = np.asarray(indices, dtype=np.int64)
    return idx, t[idx], scores


def _robust_threshold(scores: FloatArray, mad_multiplier: float) -> float:
    median = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median)))
    # 1.4826 converts MAD to a Gaussian-consistent scale estimate.
    scale = 1.4826 * mad
    if scale == 0.0:
        positive = scores[scores > median]
        if positive.size == 0:
            return np.inf
        return float(np.min(positive))
    return median + mad_multiplier * scale


def detect_changes(
    X: ArrayLike,
    t: ArrayLike,
    *,
    method: Literal["secant", "window"] = "secant",
    window: int = 1,
    n_changes: Optional[int] = None,
    threshold: Optional[float] = None,
    mad_multiplier: float = 8.0,
    min_separation: int = 1,
    norm: Literal["l2", "l1", "linf"] = "l2",
) -> ChangeDetectionResult:
    """Detect piecewise-stationary transition times.

    Parameters
    ----------
    n_changes:
        If known for a controlled benchmark, select the strongest ``n_changes``
        separated peaks.  If omitted, a robust MAD threshold is used unless an
        explicit ``threshold`` is supplied.
    min_separation:
        Minimum separation between selected candidate boundaries, in samples.

    Notes
    -----
    Supplying ``n_changes`` is useful for development/oracle diagnostics but is
    not required by the public TIF pipeline.
    """

    X, t = _validate_trajectory(X, t)
    score_idx, score_t, scores = change_scores(
        X, t, method=method, window=window, norm=norm
    )
    V = secant_velocities(X, t)

    if min_separation < 1:
        raise ValueError("min_separation must be >= 1.")

    # Peaks are found in score-array coordinates; map back to trajectory indices.
    peak_pos, _ = find_peaks(scores, distance=min_separation)
    if peak_pos.size == 0 and scores.size:
        peak_pos = np.array([int(np.argmax(scores))], dtype=int)

    if n_changes is not None:
        if n_changes < 0:
            raise ValueError("n_changes must be non-negative.")
        if n_changes == 0:
            chosen_pos = np.empty(0, dtype=int)
        else:
            # If find_peaks yields too few points, rank every candidate score and
            # greedily enforce separation.
            ranked = np.argsort(scores)[::-1]
            chosen: list[int] = []
            for p in ranked:
                if all(abs(int(p) - q) >= min_separation for q in chosen):
                    chosen.append(int(p))
                if len(chosen) == n_changes:
                    break
            if len(chosen) < n_changes:
                raise ValueError("Could not select n_changes with the requested separation.")
            chosen_pos = np.asarray(sorted(chosen), dtype=int)
    else:
        cut = _robust_threshold(scores, mad_multiplier) if threshold is None else float(threshold)
        chosen_pos = peak_pos[scores[peak_pos] >= cut]

    transition_indices = np.sort(score_idx[chosen_pos]).astype(np.int64)
    transition_times = t[transition_indices]

    # A boundary sample is assigned to the stage on its right.  Step 3 normally
    # excludes the exact transition sample from derivative regression anyway.
    sample_ids = np.arange(X.shape[0])
    stage_of_sample = np.searchsorted(transition_indices, sample_ids, side="right")

    return ChangeDetectionResult(
        transition_indices=transition_indices,
        transition_times=np.asarray(transition_times, dtype=float),
        score_indices=score_idx,
        score_times=score_t,
        scores=np.asarray(scores, dtype=float),
        secant_velocities=V,
        stage_of_sample=stage_of_sample.astype(np.int64),
        method=method,
    )
