"""Public orchestration API for the three-layer TIDES framework.

TIDES = Temporal Inference and Decomposition via Edge Space.

The public pipeline is

    Step 1  temporal segmentation
        x(t) -> tau_(1:K-1)

    preprocessing / lifting coordinates
        trajectory -> midpoint states and vector-field observations
        midpoint states -> evaluated local interaction library Psi

    Step 2  edge-space field-family reconstruction
        (Y, D, Psi, stages) -> B_epsilon

    Step 3  physical factorization + conditional-MDL compression
        (B_epsilon, H) -> (W, Theta)^*_H

This module owns orchestration and trajectory-to-observation preprocessing.  The
scientific solvers remain in the independent Step modules, while the local
interaction-library implementation remains in
``pairwise_local_interaction_library.py`` and is called here through one public
workflow.

Default midpoint reconstruction
-------------------------------
For the clean piecewise-stationary benchmarks, TIDES uses the four-point
midpoint interpolant employed by the N=8 Kuramoto benchmark.  For an interval
[n,n+1], the stencil uses states n-1,...,n+2 and is retained only when the
three covered intervals all belong to the same detected stage.  On a uniform
grid this reduces to

    x_(n+1/2) = (-x_(n-1) + 9x_n + 9x_(n+1) - x_(n+2)) / 16,

    dx/dt|_(n+1/2)
      = (x_(n-1) - 27x_n + 27x_(n+1) - x_(n+2)) / (24 dt).

The implementation computes interpolation / derivative weights from the actual
time coordinates, so the same API also supports a nonuniform strictly
increasing sampling grid.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import factorial
from typing import Any, Callable, Literal, Mapping, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

try:  # package imports
    from .step1_change_detection import ChangeDetectionResult, detect_changes
    from .pairwise_local_interaction_library import (
        PairwiseLocalInteractionAtom,
        PairwiseLocalInteractionLibrary,
        build_candidate_pairwise_local_interaction_library,
        build_pairwise_local_interaction_library,
        pair_endpoints_from_incidence,
    )
    from .step2_edge_space_reconstruction import (
        EdgeSpaceFieldFamily,
        reconstruct_edge_space_family_from_observations,
    )
    from .step3_physical_compression import (
        PhysicalCompressionResult,
        compress_physical_representation,
    )
except ImportError:  # flat-file imports
    from step1_change_detection import ChangeDetectionResult, detect_changes
    from pairwise_local_interaction_library import (
        PairwiseLocalInteractionAtom,
        PairwiseLocalInteractionLibrary,
        build_candidate_pairwise_local_interaction_library,
        build_pairwise_local_interaction_library,
        pair_endpoints_from_incidence,
    )
    from step2_edge_space_reconstruction import (
        EdgeSpaceFieldFamily,
        reconstruct_edge_space_family_from_observations,
    )
    from step3_physical_compression import (
        PhysicalCompressionResult,
        compress_physical_representation,
    )


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
IntrinsicVectorField = Callable[[FloatArray, FloatArray], ArrayLike]


# -----------------------------------------------------------------------------
# Public result containers
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class TrajectoryPreprocessingResult:
    """Trajectory-derived observations passed to Step 2.

    ``velocity_mid`` is the reconstructed total node-space vector field.
    ``pairwise_target`` is the actual Step-2 target after subtraction of an
    optional known node-local / intrinsic vector field.
    """

    t_state: FloatArray
    x_state: FloatArray

    transition_indices: IntArray
    transition_times: FloatArray
    stage_of_interval: IntArray
    segment_bounds: IntArray
    segment_interval_counts: IntArray

    observation_interval_indices: IntArray
    t_mid: FloatArray
    x_mid: FloatArray
    velocity_mid: FloatArray
    pairwise_target: FloatArray
    stage_of_observation: IntArray

    intrinsic_vector_field: FloatArray

    method: str
    resolution_state_relative: FloatArray
    resolution_velocity_relative: FloatArray
    resolution_relative_max: Optional[float]
    estimated_uncertainty_floor: Optional[float]
    uncertainty_multiplier: Optional[float]

    @property
    def n_observations(self) -> int:
        return int(self.x_mid.shape[0])

    @property
    def n_stages(self) -> int:
        return int(self.transition_indices.size + 1)

    @property
    def samples_per_stage(self) -> IntArray:
        return np.bincount(
            self.stage_of_observation,
            minlength=self.n_stages,
        ).astype(np.int64)


@dataclass(frozen=True)
class TIDESResult:
    """Output of the complete three-layer TIDES pipeline."""

    transition_indices: IntArray
    transition_times: FloatArray
    segmentation_source: str

    step1: Optional[ChangeDetectionResult]
    preprocessing: TrajectoryPreprocessingResult
    interaction_library: PairwiseLocalInteractionLibrary
    step2: EdgeSpaceFieldFamily
    step3: Optional[PhysicalCompressionResult]

    physical_hypothesis: Optional[str]
    uncertainty_floor: float
    metadata: Mapping[str, Any]

    @property
    def field_family(self) -> EdgeSpaceFieldFamily:
        return self.step2

    @property
    def reference_B_stages(self) -> FloatArray:
        return self.step2.reference_B_stages

    @property
    def reference_delta_B(self) -> FloatArray:
        return self.step2.reference_delta_B

    @property
    def W_stages(self) -> Optional[FloatArray]:
        return None if self.step3 is None else self.step3.W_stages

    @property
    def theta(self) -> Optional[FloatArray]:
        return None if self.step3 is None else self.step3.theta

    @property
    def B_stages(self) -> FloatArray:
        """Selected physical B when Step 3 ran; otherwise the Step-2 reference."""
        return (
            self.step2.reference_B_stages
            if self.step3 is None
            else self.step3.B_stages
        )

    @property
    def delta_B(self) -> FloatArray:
        return np.diff(self.B_stages, axis=0)

    @property
    def mdl_score(self):
        return None if self.step3 is None else self.step3.mdl_score


# -----------------------------------------------------------------------------
# Validation and temporal segmentation helpers
# -----------------------------------------------------------------------------


def _validate_trajectory(X: ArrayLike, t: ArrayLike) -> tuple[FloatArray, FloatArray]:
    X_arr = np.asarray(X, dtype=float)
    t_arr = np.asarray(t, dtype=float).reshape(-1)
    if X_arr.ndim != 2:
        raise ValueError("X must have shape (n_samples, n_nodes).")
    if t_arr.ndim != 1 or t_arr.size != X_arr.shape[0]:
        raise ValueError("t must be one-dimensional with len(t) == X.shape[0].")
    if X_arr.shape[0] < 4:
        raise ValueError("At least four trajectory samples are required.")
    if X_arr.shape[1] < 1:
        raise ValueError("X must contain at least one node.")
    if not np.all(np.isfinite(X_arr)) or not np.all(np.isfinite(t_arr)):
        raise ValueError("X and t must contain only finite values.")
    if not np.all(np.diff(t_arr) > 0.0):
        raise ValueError("t must be strictly increasing.")
    return X_arr, t_arr


def _validate_transition_indices(
    transition_indices: Sequence[int],
    *,
    n_state_samples: int,
) -> IntArray:
    idx = np.asarray(transition_indices, dtype=np.int64).reshape(-1)
    if idx.size and not np.all(np.diff(idx) > 0):
        raise ValueError("transition_indices must be strictly increasing.")
    # A transition index i denotes the state-sample boundary t[i], with interval
    # i-1 on the left and interval i on the right.
    if idx.size and (int(idx[0]) <= 0 or int(idx[-1]) >= n_state_samples - 1):
        raise ValueError(
            "transition_indices must lie strictly inside the trajectory: "
            "1 <= i <= n_samples-2."
        )
    return idx


def _interval_stage_labels(
    n_intervals: int,
    transition_indices: IntArray,
) -> tuple[IntArray, IntArray, IntArray]:
    bounds = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.asarray(transition_indices, dtype=np.int64),
            np.array([n_intervals], dtype=np.int64),
        )
    )
    counts = np.diff(bounds).astype(np.int64)
    if np.any(counts <= 0):
        raise ValueError("Every temporal stage must contain at least one interval.")
    stage = np.empty(n_intervals, dtype=np.int64)
    for r in range(len(bounds) - 1):
        stage[bounds[r] : bounds[r + 1]] = r
    return stage, bounds, counts


# -----------------------------------------------------------------------------
# Midpoint state / derivative reconstruction
# -----------------------------------------------------------------------------


def _interpolation_weights(
    nodes: FloatArray,
    target: float,
    derivative_order: int,
) -> FloatArray:
    """Polynomial interpolation weights at ``target`` on arbitrary nodes."""

    x = np.asarray(nodes, dtype=float).reshape(-1)
    d = int(derivative_order)
    if x.size < 1 or d < 0 or d >= x.size:
        raise ValueError("Require 0 <= derivative_order < len(nodes).")
    if np.unique(x).size != x.size:
        raise ValueError("Interpolation nodes must be distinct.")

    centered = x - float(target)
    scale = float(np.max(np.abs(centered)))
    if scale <= np.finfo(float).tiny:
        raise ValueError("Interpolation stencil has zero time scale.")
    z = centered / scale
    V = np.vstack([z**k for k in range(x.size)])
    rhs = np.zeros(x.size, dtype=float)
    rhs[d] = float(factorial(d))
    weights_z = np.linalg.solve(V, rhs)
    return np.asarray(weights_z / (scale**d), dtype=float)


def _midpoint_from_stencil(
    X: FloatArray,
    t: FloatArray,
    state_indices: Sequence[int],
    target_time: float,
) -> tuple[FloatArray, FloatArray]:
    ids = np.asarray(state_indices, dtype=np.int64)
    nodes = t[ids]
    values = X[ids]

    # Preserve the canonical benchmark arithmetic on a uniform grid.  Using
    # dimensionless half-step coordinates avoids magnifying roundoff by solving
    # a Vandermonde system directly in very small physical time units.
    dnodes = np.diff(nodes)
    uniform = bool(
        dnodes.size
        and np.allclose(dnodes, dnodes[0], rtol=1e-12, atol=1e-15)
        and np.isclose(
            float(target_time),
            0.5 * float(nodes[0] + nodes[-1]),
            rtol=1e-12,
            atol=1e-15,
        )
    )
    if uniform:
        h = float(dnodes[0])
        z = np.arange(ids.size, dtype=float) - 0.5 * (ids.size - 1)
        V = np.vstack([z**k for k in range(ids.size)])
        rhs0 = np.zeros(ids.size, dtype=float)
        rhs0[0] = 1.0
        rhs1 = np.zeros(ids.size, dtype=float)
        rhs1[1] = 1.0
        wx = np.linalg.solve(V, rhs0)
        wv = np.linalg.solve(V, rhs1) / h
    else:
        wx = _interpolation_weights(nodes, target_time, 0)
        wv = _interpolation_weights(nodes, target_time, 1)

    x_mid = np.einsum("s,sn->n", wx, values, optimize=True)
    v_mid = np.einsum("s,sn->n", wv, values, optimize=True)
    return np.asarray(x_mid, dtype=float), np.asarray(v_mid, dtype=float)


def _coerce_intrinsic_vector_field(
    intrinsic_vector_field: Optional[IntrinsicVectorField | ArrayLike],
    *,
    x_mid: FloatArray,
    t_mid: FloatArray,
) -> FloatArray:
    if intrinsic_vector_field is None:
        return np.zeros_like(x_mid)

    if callable(intrinsic_vector_field):
        value = intrinsic_vector_field(x_mid, t_mid)
    else:
        value = intrinsic_vector_field

    A = np.asarray(value, dtype=float)
    if A.shape == (x_mid.shape[1],):
        A = np.broadcast_to(A[None, :], x_mid.shape).copy()
    if A.shape != x_mid.shape:
        raise ValueError(
            "intrinsic_vector_field must return/provide shape "
            f"{x_mid.shape} or ({x_mid.shape[1]},); got {A.shape}."
        )
    if not np.all(np.isfinite(A)):
        raise ValueError("intrinsic_vector_field contains non-finite values.")
    return A


def preprocess_trajectory(
    X: ArrayLike,
    t: ArrayLike,
    transition_indices: Sequence[int],
    *,
    method: Literal["four_point_midpoint", "secant_midpoint"] = "four_point_midpoint",
    intrinsic_vector_field: Optional[IntrinsicVectorField | ArrayLike] = None,
    resolution_audit: bool = True,
    uncertainty_multiplier: float = 20.0,
    resolution_floor: float = 1.0e-14,
) -> TrajectoryPreprocessingResult:
    """Convert a segmented raw trajectory into Step-2 observations.

    Parameters
    ----------
    method
        ``'four_point_midpoint'`` is the reference high-accuracy TIDES
        preprocessing.  ``'secant_midpoint'`` uses arithmetic midpoint states
        and interval secant velocities and is mainly useful for diagnostics or
        noisier pipelines with their own externally supplied uncertainty floor.
    intrinsic_vector_field
        Optional known node-local term to subtract before Step 2.  Supply either
        an array with shape ``(n_observations,n_nodes)``, a constant vector with
        shape ``(n_nodes,)``, or a callable ``f(x_mid, t_mid)`` returning the
        observation-shaped array.
    resolution_audit
        In four-point mode, compare 4-point and 6-point midpoint reconstructions
        on every same-stage six-point stencil.
    uncertainty_multiplier
        If the audit is available, the reported estimated uncertainty floor is
        ``uncertainty_multiplier * max(relative 4-vs-6 discrepancy)``.
    """

    X_arr, t_arr = _validate_trajectory(X, t)
    transitions = _validate_transition_indices(
        transition_indices,
        n_state_samples=X_arr.shape[0],
    )
    n_intervals = X_arr.shape[0] - 1
    stage_interval, bounds, counts = _interval_stage_labels(
        n_intervals,
        transitions,
    )
    transition_times = t_arr[transitions]

    method = str(method).lower()
    if method not in {"four_point_midpoint", "secant_midpoint"}:
        raise ValueError(
            "method must be 'four_point_midpoint' or 'secant_midpoint'."
        )

    obs_ids: list[int] = []
    x_obs: list[FloatArray] = []
    v_obs: list[FloatArray] = []
    t_obs: list[float] = []
    s_obs: list[int] = []

    if method == "secant_midpoint":
        dt = np.diff(t_arr)
        X_mid = 0.5 * (X_arr[:-1] + X_arr[1:])
        V_mid = np.diff(X_arr, axis=0) / dt[:, None]
        obs_ids = list(range(n_intervals))
        x_obs = [row for row in X_mid]
        v_obs = [row for row in V_mid]
        t_obs = list(0.5 * (t_arr[:-1] + t_arr[1:]))
        s_obs = list(stage_interval)
    else:
        # The 4-state stencil around interval n covers intervals n-1,n,n+1.
        # Keeping only same-stage stencils prevents a derivative estimate from
        # averaging across a detected discontinuity of the vector field.
        for n in range(1, n_intervals - 1):
            if not (
                stage_interval[n - 1]
                == stage_interval[n]
                == stage_interval[n + 1]
            ):
                continue
            tm = 0.5 * (t_arr[n] + t_arr[n + 1])
            xm, vm = _midpoint_from_stencil(
                X_arr,
                t_arr,
                (n - 1, n, n + 1, n + 2),
                tm,
            )
            obs_ids.append(int(n))
            x_obs.append(xm)
            v_obs.append(vm)
            t_obs.append(float(tm))
            s_obs.append(int(stage_interval[n]))

    X_mid = np.asarray(x_obs, dtype=float)
    V_mid = np.asarray(v_obs, dtype=float)
    T_mid = np.asarray(t_obs, dtype=float)
    OBS_stage = np.asarray(s_obs, dtype=np.int64)
    obs_index = np.asarray(obs_ids, dtype=np.int64)

    if X_mid.ndim != 2 or X_mid.shape[0] == 0:
        raise ValueError(
            "Midpoint preprocessing produced no observations.  Temporal stages "
            "may be too short for the selected stencil."
        )
    K = transitions.size + 1
    present = np.unique(OBS_stage)
    if not np.array_equal(present, np.arange(K, dtype=np.int64)):
        raise ValueError(
            "At least one stage has no valid midpoint observations under the "
            "selected preprocessing stencil."
        )

    intrinsic = _coerce_intrinsic_vector_field(
        intrinsic_vector_field,
        x_mid=X_mid,
        t_mid=T_mid,
    )
    Y = np.asarray(V_mid - intrinsic, dtype=float)

    rel_state = np.zeros(0, dtype=float)
    rel_velocity = np.zeros(0, dtype=float)
    resolution_max: Optional[float] = None
    estimated_floor: Optional[float] = None

    if method == "four_point_midpoint" and resolution_audit:
        state_err: list[float] = []
        vel_err: list[float] = []
        for n in range(2, n_intervals - 2):
            if len(set(int(stage_interval[n + j]) for j in (-2, -1, 0, 1, 2))) != 1:
                continue
            tm = 0.5 * (t_arr[n] + t_arr[n + 1])
            x4, v4 = _midpoint_from_stencil(
                X_arr,
                t_arr,
                (n - 1, n, n + 1, n + 2),
                tm,
            )
            x6, v6 = _midpoint_from_stencil(
                X_arr,
                t_arr,
                (n - 2, n - 1, n, n + 1, n + 2, n + 3),
                tm,
            )
            state_err.append(
                float(np.linalg.norm(x6 - x4) / max(np.linalg.norm(x6), 1e-15))
            )
            vel_err.append(
                float(np.linalg.norm(v6 - v4) / max(np.linalg.norm(v6), 1e-15))
            )

        rel_state = np.asarray(state_err, dtype=float)
        rel_velocity = np.asarray(vel_err, dtype=float)
        if rel_state.size and rel_velocity.size:
            rf = float(resolution_floor)
            if not np.isfinite(rf) or rf < 0.0:
                raise ValueError("resolution_floor must be finite and non-negative.")
            resolution_max = max(
                float(np.max(rel_state)),
                float(np.max(rel_velocity)),
                rf,
            )
            multiplier = float(uncertainty_multiplier)
            if not np.isfinite(multiplier) or multiplier <= 0.0:
                raise ValueError("uncertainty_multiplier must be finite and positive.")
            estimated_floor = float(multiplier * resolution_max)

    return TrajectoryPreprocessingResult(
        t_state=t_arr.copy(),
        x_state=X_arr.copy(),
        transition_indices=transitions.copy(),
        transition_times=np.asarray(transition_times, dtype=float),
        stage_of_interval=stage_interval,
        segment_bounds=bounds,
        segment_interval_counts=counts,
        observation_interval_indices=obs_index,
        t_mid=T_mid,
        x_mid=X_mid,
        velocity_mid=V_mid,
        pairwise_target=Y,
        stage_of_observation=OBS_stage,
        intrinsic_vector_field=np.asarray(intrinsic, dtype=float),
        method=method,
        resolution_state_relative=rel_state,
        resolution_velocity_relative=rel_velocity,
        resolution_relative_max=resolution_max,
        estimated_uncertainty_floor=estimated_floor,
        uncertainty_multiplier=(
            float(uncertainty_multiplier)
            if method == "four_point_midpoint" and resolution_audit
            else None
        ),
    )


# -----------------------------------------------------------------------------
# Local interaction library construction
# -----------------------------------------------------------------------------


def _build_relative_polynomial_library(
    node_states: FloatArray,
    D: FloatArray,
    *,
    degree: int,
    difference_scale: Optional[float],
    check_sampled_swap_equivariance: bool,
) -> PairwiseLocalInteractionLibrary:
    degree = int(degree)
    if degree < 1:
        raise ValueError("library_degree must be >= 1.")

    tail, head = pair_endpoints_from_incidence(D)
    physical_difference = node_states[:, head] - node_states[:, tail]
    if difference_scale is None:
        scale = float(np.max(np.abs(physical_difference)))
        if scale <= 1.0e-14:
            scale = 1.0
    else:
        scale = float(difference_scale)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("library_state_scale must be finite and positive.")

    atoms: list[PairwiseLocalInteractionAtom] = []
    for power in range(1, degree + 1):
        p = int(power)

        def evaluate(xs, xr, *, _p=p, _scale=scale):
            z = (xr - xs) / _scale
            return np.stack([z**_p, (-z) ** _p], axis=-1)

        atoms.append(
            PairwiseLocalInteractionAtom(
                name=f"rel_d^{p}",
                evaluate=evaluate,
                description=(
                    f"Relative-coordinate polynomial atom z^{p}, "
                    "z=(x_head-x_tail)/scale; head receives (-z)^p."
                ),
                swap_equivariant=True,
                metadata={"power": p, "difference_scale": scale},
            )
        )

    result = build_candidate_pairwise_local_interaction_library(
        node_states,
        D,
        tuple(atoms),
        check_sampled_swap_equivariance=check_sampled_swap_equivariance,
    )
    # This convenience mode uses candidate evaluators but retains polynomial coding.
    return replace(result, metadata={**result.metadata, "mdl_mode": "polynomial"})


def build_interaction_library(
    node_states: ArrayLike,
    D: ArrayLike,
    *,
    library: Optional[PairwiseLocalInteractionLibrary] = None,
    mode: Literal["polynomial", "relative_polynomial", "candidate"] = "polynomial",
    degree: int = 2,
    normalization: Literal["none", "rms", "maxabs"] = "rms",
    state_scale: Optional[float] = None,
    atoms: Optional[Sequence[PairwiseLocalInteractionAtom]] = None,
    check_sampled_swap_equivariance: bool = False,
) -> PairwiseLocalInteractionLibrary:
    """Build/evaluate the local interaction library on Step-2 midpoint states.

    ``relative_polynomial`` is a TIDES convenience mode for translation-invariant
    pairwise laws.  It creates powers of

        z = (x_neighbor - x_self) / scale,

    with powers 1,...,degree and max-absolute sampled scaling by default.
    """

    X = np.asarray(node_states, dtype=float)
    D_arr = np.asarray(D, dtype=float)

    if library is not None:
        if atoms is not None:
            raise ValueError("atoms must be omitted when a pre-evaluated library is supplied.")
        F = np.asarray(library.endpoint_features, dtype=float)
        if F.shape[0] != X.shape[0] or F.shape[1] != D_arr.shape[1]:
            raise ValueError(
                "A supplied library must already be evaluated on the current "
                "midpoint observations and candidate edges."
            )
        return library

    mode = str(mode).lower()
    if mode == "relative_polynomial":
        if atoms is not None:
            raise ValueError("atoms must be omitted in relative_polynomial mode.")
        return _build_relative_polynomial_library(
            X,
            D_arr,
            degree=degree,
            difference_scale=state_scale,
            check_sampled_swap_equivariance=check_sampled_swap_equivariance,
        )

    return build_pairwise_local_interaction_library(
        X,
        D_arr,
        mode=mode,
        degree=degree,
        normalization=normalization,
        state_scale=state_scale,
        atoms=atoms,
        check_sampled_swap_equivariance=check_sampled_swap_equivariance,
    )


# -----------------------------------------------------------------------------
# Step-2/3 lower-level entry point for already prepared observations
# -----------------------------------------------------------------------------


def run_tides_from_observations(
    Y: ArrayLike,
    D: ArrayLike,
    library: Any,
    stage_of_sample: Sequence[int],
    *,
    uncertainty_floor: float,
    physical_hypothesis: str = "fixed_dynamics",
    run_physical_compression: bool = True,
    step2_kwargs: Optional[Mapping[str, Any]] = None,
    step3_kwargs: Optional[Mapping[str, Any]] = None,
) -> tuple[EdgeSpaceFieldFamily, Optional[PhysicalCompressionResult]]:
    """Run Steps 2--3 when preprocessing has already been performed."""

    s2 = {} if step2_kwargs is None else dict(step2_kwargs)
    forbidden2 = {"Y", "D", "library", "stage_of_sample", "uncertainty_floor"} & set(s2)
    if forbidden2:
        raise ValueError(
            "These Step-2 arguments are supplied by run_tides_from_observations: "
            + ", ".join(sorted(forbidden2))
        )

    family = reconstruct_edge_space_family_from_observations(
        Y,
        D,
        library,
        stage_of_sample,
        uncertainty_floor=float(uncertainty_floor),
        **s2,
    )

    if not run_physical_compression:
        return family, None

    s3 = {} if step3_kwargs is None else dict(step3_kwargs)
    if "hypothesis" in s3:
        raise ValueError(
            "Pass physical_hypothesis through run_tides_from_observations, not step3_kwargs."
        )
    physical = compress_physical_representation(
        family,
        hypothesis=physical_hypothesis,
        **s3,
    )
    return family, physical


# -----------------------------------------------------------------------------
# Complete trajectory -> Step 1 -> preprocessing/library -> Step 2 -> Step 3
# -----------------------------------------------------------------------------


def run_tides(
    X: ArrayLike,
    t: ArrayLike,
    D: ArrayLike,
    *,
    # Step 1.  Supplying transition_indices is an explicit oracle/manual bypass.
    transition_indices: Optional[Sequence[int]] = None,
    step1_kwargs: Optional[Mapping[str, Any]] = None,
    # Trajectory -> midpoint vector-field observations.
    preprocessing_method: Literal[
        "four_point_midpoint", "secant_midpoint"
    ] = "four_point_midpoint",
    intrinsic_vector_field: Optional[IntrinsicVectorField | ArrayLike] = None,
    resolution_audit: bool = True,
    uncertainty_floor: Optional[float] = None,
    uncertainty_multiplier: float = 20.0,
    resolution_floor: float = 1.0e-14,
    # Local interaction library Psi.
    interaction_library: Optional[PairwiseLocalInteractionLibrary] = None,
    library_mode: Literal[
        "polynomial", "relative_polynomial", "candidate"
    ] = "polynomial",
    library_degree: int = 2,
    library_normalization: Literal["none", "rms", "maxabs"] = "rms",
    library_state_scale: Optional[float] = None,
    interaction_atoms: Optional[Sequence[PairwiseLocalInteractionAtom]] = None,
    check_sampled_swap_equivariance: bool = False,
    # Step 2 / Step 3.
    step2_kwargs: Optional[Mapping[str, Any]] = None,
    physical_hypothesis: str = "fixed_dynamics",
    run_physical_compression: bool = True,
    step3_kwargs: Optional[Mapping[str, Any]] = None,
) -> TIDESResult:
    """Run the complete three-layer TIDES pipeline from a raw trajectory.

    The default path performs Step 1, four-point midpoint state/derivative
    reconstruction, a 4-vs-6 resolution audit, local-library evaluation, Step 2
    family reconstruction, and Step 3 physical MDL compression.

    ``uncertainty_floor`` may be supplied explicitly.  If omitted, the default
    four-point path uses the resolution-audit estimate

        epsilon = uncertainty_multiplier * resolution_relative_max.

    An explicit floor is required when that audit is unavailable.
    """

    X_arr, t_arr = _validate_trajectory(X, t)
    D_arr = np.asarray(D, dtype=float)
    if D_arr.ndim != 2 or D_arr.shape[0] != X_arr.shape[1]:
        raise ValueError(
            "D must have shape (n_nodes,n_candidate_edges) with n_nodes=X.shape[1]."
        )
    if not np.all(np.isfinite(D_arr)):
        raise ValueError("D must contain only finite values.")

    s1_kwargs = {} if step1_kwargs is None else dict(step1_kwargs)
    if transition_indices is None:
        step1_result = detect_changes(X_arr, t_arr, **s1_kwargs)
        transitions = np.asarray(step1_result.transition_indices, dtype=np.int64)
        segmentation_source = "step1_detected"
    else:
        if s1_kwargs:
            raise ValueError(
                "step1_kwargs must be empty when transition_indices are supplied explicitly."
            )
        step1_result = None
        transitions = _validate_transition_indices(
            transition_indices,
            n_state_samples=X_arr.shape[0],
        )
        segmentation_source = "caller_supplied"

    prep = preprocess_trajectory(
        X_arr,
        t_arr,
        transitions,
        method=preprocessing_method,
        intrinsic_vector_field=intrinsic_vector_field,
        resolution_audit=resolution_audit,
        uncertainty_multiplier=uncertainty_multiplier,
        resolution_floor=resolution_floor,
    )

    if uncertainty_floor is None:
        if prep.estimated_uncertainty_floor is None:
            raise ValueError(
                "uncertainty_floor was not supplied and the selected preprocessing "
                "path did not produce a resolution-audit estimate."
            )
        epsilon = float(prep.estimated_uncertainty_floor)
        floor_source = "resolution_audit"
    else:
        epsilon = float(uncertainty_floor)
        if not np.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("uncertainty_floor must be finite and strictly positive.")
        floor_source = "caller_supplied"

    local_library = build_interaction_library(
        prep.x_mid,
        D_arr,
        library=interaction_library,
        mode=library_mode,
        degree=library_degree,
        normalization=library_normalization,
        state_scale=library_state_scale,
        atoms=interaction_atoms,
        check_sampled_swap_equivariance=check_sampled_swap_equivariance,
    )

    family, physical = run_tides_from_observations(
        prep.pairwise_target,
        D_arr,
        local_library,
        prep.stage_of_observation,
        uncertainty_floor=epsilon,
        physical_hypothesis=physical_hypothesis,
        run_physical_compression=run_physical_compression,
        step2_kwargs=step2_kwargs,
        step3_kwargs=step3_kwargs,
    )

    return TIDESResult(
        transition_indices=transitions.copy(),
        transition_times=t_arr[transitions].copy(),
        segmentation_source=segmentation_source,
        step1=step1_result,
        preprocessing=prep,
        interaction_library=local_library,
        step2=family,
        step3=physical,
        physical_hypothesis=(
            str(physical_hypothesis) if run_physical_compression else None
        ),
        uncertainty_floor=epsilon,
        metadata={
            "architecture": "three_layer_tides",
            "preprocessing_method": str(preprocessing_method),
            "uncertainty_floor_source": floor_source,
            "library_mode": str(library_mode),
            "n_state_samples": int(X_arr.shape[0]),
            "n_midpoint_observations": int(prep.n_observations),
            "n_nodes": int(X_arr.shape[1]),
            "n_candidate_edges": int(D_arr.shape[1]),
            "n_stages": int(prep.n_stages),
            "n_library_atoms": int(local_library.n_features),
        },
    )


# Short interactive alias.
run = run_tides


__all__ = [
    # Complete pipeline.
    "TIDESResult",
    "TrajectoryPreprocessingResult",
    "run_tides",
    "run_tides_from_observations",
    "run",
    # Step 1.
    "ChangeDetectionResult",
    "detect_changes",
    # Preprocessing / library.
    "preprocess_trajectory",
    "build_interaction_library",
    "PairwiseLocalInteractionAtom",
    "PairwiseLocalInteractionLibrary",
    # Step 2.
    "EdgeSpaceFieldFamily",
    "reconstruct_edge_space_family_from_observations",
    # Step 3.
    "PhysicalCompressionResult",
    "compress_physical_representation",
]
