"""Stage-local polynomial smoothing of states and their time derivatives.

Fit in time, independently of the interaction-law library. Each midpoint uses
one least-squares polynomial for both state and derivative. Windows never cross
a supplied change point. A boundary state may be used by both adjacent stages:
the state is assumed continuous even when its derivative jumps.

No global smoothing matrix is formed. The two local weight vectors are solved
by SVD-backed least squares and applied to all nodes at once. Noise diagnostics
assume independent additive measurement noise with known per-node standard
deviation; they exclude smoothing bias and do not define an uncertainty floor.
"""
from __future__ import annotations

from dataclasses import dataclass
from operator import index
from typing import Literal, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class SegmentedSmoothingResult:
    t_mid: np.ndarray
    x_mid: np.ndarray
    velocity_mid: np.ndarray
    observation_interval_indices: np.ndarray
    stage_of_observation: np.ndarray
    window_start_indices: np.ndarray
    window_stop_indices: np.ndarray  # exclusive state-sample index
    fit_condition_numbers: np.ndarray
    state_noise_gain: np.ndarray  # standard deviation per unit measurement SD
    velocity_noise_gain: np.ndarray
    state_noise_std: Optional[np.ndarray]
    velocity_noise_std: Optional[np.ndarray]
    window_size: int
    degree: int
    boundary: str
    trim_intervals: int

    @property
    def window_sizes(self) -> np.ndarray:
        return self.window_stop_indices - self.window_start_indices

    @property
    def expected_velocity_noise_rms(self) -> Optional[float]:
        """sqrt(E[mean(noise**2)]), excluding fit bias and library error."""
        if self.velocity_noise_std is None:
            return None
        return float(np.sqrt(np.mean(self.velocity_noise_std**2)))


def _integer(value, name, minimum):
    try:
        result = index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if isinstance(value, (bool, np.bool_)) or result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return result


def smooth_segmented_trajectory(
    X: ArrayLike,
    t: ArrayLike,
    transition_indices: Sequence[int],
    *,
    window_size: int = 12,
    degree: int = 3,
    boundary: Literal["shift", "drop"] = "shift",
    trim_intervals: int = 1,
    measurement_noise_std: Optional[ArrayLike] = None,
) -> SegmentedSmoothingResult:
    """Fit local polynomials within stages and evaluate at interval midpoints.

    Parameters
    ----------
    X, t
        Finite states of shape (n_samples, n_nodes) and strictly increasing
        sample times. Actual times are used, including on nonuniform grids.
    transition_indices
        State-sample boundaries, with the same convention as TIDES Step 1.
    window_size
        Number of adjacent state samples (even or odd). Even sizes give
        symmetric interior midpoint windows on uniform grids. A stage shorter
        than the requested window uses all its states; degree is never reduced.
    degree
        Degree of a local polynomial in TIME, not of the interaction law.
        At least degree+1 states are required in every stage.
    boundary
        'shift': shift the sample window inside the stage at its boundaries.
        'drop': omit midpoints whose centered-by-index window would cross a
        stage boundary. Windows are selected by sample index, not time radius.
    trim_intervals
        Omit this many intervals at each end of every stage. The default 1
        matches the existing four-point TIDES observation locations with shift.
    measurement_noise_std
        Optional nonnegative scalar or one SD per node, constant over time.
        Used only for diagnostics; never changes the fit or infers epsilon.

    Notes
    -----
    Local weights reproduce time polynomials up to the specified degree.
    Neighboring outputs are correlated because they reuse noisy measurements.
    Reported pointwise SDs do not account for that covariance, segmentation
    uncertainty, smoothing bias, or state errors in the interaction library.
    """
    X = np.asarray(X, dtype=float)
    t = np.asarray(t, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 1:
        raise ValueError("X must have shape (n_samples >= 2, n_nodes >= 1).")
    if t.ndim != 1 or t.size != X.shape[0]:
        raise ValueError("t must be a 1D vector matching X's sample count.")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(t)):
        raise ValueError("X and t must contain only finite values.")
    if not np.all(np.diff(t) > 0.):
        raise ValueError("t must be strictly increasing.")
    window_size = _integer(window_size, "window_size", 2)
    degree = _integer(degree, "degree", 1)
    trim_intervals = _integer(trim_intervals, "trim_intervals", 0)
    if window_size < degree + 1:
        raise ValueError("window_size must be >= degree + 1.")
    if boundary not in {"shift", "drop"}:
        raise ValueError("boundary must be 'shift' or 'drop'.")
    transitions = np.asarray(transition_indices)
    if transitions.ndim != 1:
        raise ValueError("transition_indices must be one-dimensional.")
    if transitions.size and transitions.dtype.kind not in "iu":
        raise ValueError("transition_indices must contain integers.")
    if transitions.size and (
        np.any(transitions < 1) or np.any(transitions >= t.size - 1)
        or np.any(transitions[1:] <= transitions[:-1])
    ):
        raise ValueError("transition_indices must be strictly increasing interior indices.")
    bounds = np.r_[0, transitions.astype(np.int64), t.size - 1]
    sigma = None
    if measurement_noise_std is not None:
        sigma = np.asarray(measurement_noise_std, dtype=float)
        if sigma.ndim == 0:
            sigma = np.full(X.shape[1], float(sigma))
        if sigma.shape != (X.shape[1],) or not np.all(np.isfinite(sigma)) or np.any(sigma < 0.):
            raise ValueError("measurement_noise_std must be a nonnegative scalar or one SD per node.")

    obs, stages, starts, stops = [], [], [], []
    for stage, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
        width = min(window_size, int(hi - lo + 1))
        if width < degree + 1:
            raise ValueError(f"Stage {stage} has fewer than degree+1 state samples.")
        count_before = len(obs)
        for n in range(int(lo) + trim_intervals, int(hi) - trim_intervals):
            start = n - (width - 1) // 2
            if boundary == "drop" and (start < lo or start + width > hi + 1):
                continue
            start = max(int(lo), min(start, int(hi) + 1 - width))
            obs.append(n)
            stages.append(stage)
            starts.append(start)
            stops.append(start + width)
        if len(obs) == count_before:
            raise ValueError(f"Stage {stage} has no observations after trimming/boundary handling.")

    obs = np.asarray(obs, dtype=np.int64)
    stages = np.asarray(stages, dtype=np.int64)
    starts, stops = np.asarray(starts, dtype=np.int64), np.asarray(stops, dtype=np.int64)
    mid = t[obs] + 0.5 * (t[obs + 1] - t[obs])
    states, velocity = np.empty((len(obs), X.shape[1])), np.empty((len(obs), X.shape[1]))
    conditions, gain_x, gain_v = np.empty(len(obs)), np.empty(len(obs)), np.empty(len(obs))
    for k, (tm, start, stop) in enumerate(zip(mid, starts, stops)):
        offsets = t[start:stop] - tm
        scale = float(np.max(np.abs(offsets)))
        z = offsets / scale
        V = np.vander(z, N=degree + 1, increasing=True)
        # V.T w_x = e_0 and V.T w_v = e_1/scale. Minimum-norm weights
        # equal evaluation/derivative of the ordinary least-squares fit.
        rhs = np.zeros((degree + 1, 2))
        rhs[0, 0], rhs[1, 1] = 1., 1. / scale
        weights, _, rank, singular = np.linalg.lstsq(V.T, rhs, rcond=None)
        if rank != degree + 1:
            raise ValueError(f"Numerically rank-deficient local polynomial at interval {obs[k]}.")
        fitted = weights.T @ X[start:stop]
        states[k], velocity[k] = fitted[0], fitted[1]
        conditions[k] = singular[0] / singular[-1]
        gain_x[k], gain_v[k] = np.linalg.norm(weights, axis=0)

    return SegmentedSmoothingResult(
        mid, states, velocity, obs, stages, starts, stops, conditions, gain_x, gain_v,
        None if sigma is None else gain_x[:, None] * sigma[None, :],
        None if sigma is None else gain_v[:, None] * sigma[None, :],
        window_size, degree, boundary, trim_intervals,
    )


__all__ = ["SegmentedSmoothingResult", "smooth_segmented_trajectory"]
