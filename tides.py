"""
tides.py
========

Public orchestration / convenience API for TIDES.

TIDES = Temporal Inference and Decomposition via Edge Space.

The scientific steps remain independently callable:

    Step 1  detect vector-field change times
    Step 2  infer the structure/support of each change
    Step 3  reconstruct stage-wise edge-space vector fields B
    Step 4  decompose B into microscopic sources under a source hypothesis

Current implemented branch
--------------------------
Step 2 change hypothesis:
    ``varying_structure``

with row-sparse changes in unrestricted edge-space coefficients,

    Delta B^(k).

Step 4 source hypothesis:
    ``shared_interaction_law``

with

    B^(r) = W^(r) theta^T.

Important pipeline note
-----------------------
The generic Step-2/3 APIs operate on preprocessed vector-field observations

    Y, D, edge_features, stage_of_sample.

Step 1 acts on the raw trajectory X(t), while the derivative / midpoint
preprocessing and edge-feature construction between Step 1 and Step 2 are
data/model-specific.  Therefore ``run_tides`` currently orchestrates Steps
2--4 from already preprocessed observations.  Step 1 is re-exported and can
be run independently before that preprocessing stage.

This module contains no scientific solver logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


# -----------------------------------------------------------------------------
# Imports: package form and flat-file form
# -----------------------------------------------------------------------------

try:  # package-relative imports
    from .step1_change_detection import (
        ChangeDetectionResult,
        detect_changes,
    )
    from .step2_change_structure import (
        ChangeStructureResult,
        infer_change_structure,
        infer_change_structure_from_observations,
    )
    from .step3_vector_field import (
        VectorFieldReconstructionResult,
        reconstruct_vector_field,
        reconstruct_vector_field_from_observations,
    )
    from .step4_source_decomposition import (
        SourceDecompositionResult,
        decompose_shared_interaction_law,
        decompose_vector_field_sources,
        decompose_sources,
    )

except ImportError:  # flat-directory imports
    from step1_change_detection import (
        ChangeDetectionResult,
        detect_changes,
    )
    from step2_change_structure import (
        ChangeStructureResult,
        infer_change_structure,
        infer_change_structure_from_observations,
    )
    from step3_vector_field import (
        VectorFieldReconstructionResult,
        reconstruct_vector_field,
        reconstruct_vector_field_from_observations,
    )
    from step4_source_decomposition import (
        SourceDecompositionResult,
        decompose_shared_interaction_law,
        decompose_vector_field_sources,
        decompose_sources,
    )


IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]

ChangeHypothesis = Literal[
    "varying_structure",
    "varying_dynamics",
]

SourceHypothesis = Literal[
    "shared_interaction_law",
]


# -----------------------------------------------------------------------------
# Pipeline result
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class TIDESResult:
    """Output of the current preprocessed-observation TIDES pipeline.

    ``step1`` is optional provenance: Step 1 is normally run before the
    preprocessing that constructs ``Y`` and ``edge_features``.

    ``step2`` is ``None`` only when manual/oracle change constraints were
    supplied.  Steps 3 and 4 never use Step-2 diagnostic coefficients; Step 3
    refits the full unprojected selected-support model.
    """

    change_hypothesis: str
    source_hypothesis: Optional[str]

    transition_indices: Optional[IntArray]
    transition_times: Optional[FloatArray]
    change_constraints: Any

    step1: Optional[ChangeDetectionResult]
    step2: Optional[ChangeStructureResult]
    step3: VectorFieldReconstructionResult
    step4: Optional[SourceDecompositionResult]

    @property
    def B_stages(self):
        """Stage-wise reconstructed edge-space vector fields."""
        return self.step3.B_stages

    @property
    def B_anchor(self):
        return self.step3.B_anchor

    @property
    def delta_B(self):
        return self.step3.delta_B

    @property
    def source_decomposition(self):
        """Step-4 source-decomposition result."""
        return self.step4

    @property
    def decomposition(self):
        """Backward-compatible alias for ``source_decomposition``."""
        return self.step4

    @property
    def theta(self):
        """Shared interaction-law coefficients when Step 4 was run."""
        return None if self.step4 is None else self.step4.theta

    @property
    def W_stages(self):
        """Microscopic edge amplitudes when Step 4 was run."""
        return None if self.step4 is None else self.step4.W_stages


# -----------------------------------------------------------------------------
# Small orchestration helpers
# -----------------------------------------------------------------------------


def _as_kwargs(values: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    return {} if values is None else dict(values)


def _coerce_optional_int_vector(
    values: Optional[Sequence[int]],
    *,
    name: str,
) -> Optional[IntArray]:

    if values is None:
        return None

    out = np.asarray(values, dtype=np.int64).reshape(-1)

    if out.size and not np.all(np.diff(out) > 0):
        raise ValueError(f"{name} must be strictly increasing.")

    return out


def _coerce_optional_float_vector(
    values: Optional[Sequence[float]],
    *,
    name: str,
) -> Optional[FloatArray]:

    if values is None:
        return None

    out = np.asarray(values, dtype=float).reshape(-1)

    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must contain only finite values.")

    if out.size and not np.all(np.diff(out) > 0):
        raise ValueError(f"{name} must be strictly increasing.")

    return out


def _transition_metadata_from_constraints(
    constraints,
) -> tuple[Optional[IntArray], Optional[FloatArray]]:
    """Best-effort extraction for manual Step-2 constraints."""

    if constraints is None:
        return None, None

    if hasattr(constraints, "constraints"):
        constraints = constraints.constraints

    try:
        items = tuple(constraints)
    except TypeError:
        return None, None

    if not items:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=float),
        )

    if not all(hasattr(c, "transition_index") for c in items):
        return None, None

    indices = np.asarray(
        [int(c.transition_index) for c in items],
        dtype=np.int64,
    )

    if all(hasattr(c, "transition_time") for c in items):
        times = np.asarray(
            [float(c.transition_time) for c in items],
            dtype=float,
        )
        if not np.all(np.isfinite(times)):
            times = None
    else:
        times = None

    return indices, times


# -----------------------------------------------------------------------------
# Current formal orchestration: preprocessed observations -> Steps 2--4
# -----------------------------------------------------------------------------


def run_tides(
    Y: ArrayLike,
    D: ArrayLike,
    edge_features: ArrayLike,
    stage_of_sample: Sequence[int],
    *,
    change_hypothesis: ChangeHypothesis = "varying_structure",
    source_hypothesis: SourceHypothesis = "shared_interaction_law",
    profile_floor: Optional[float] = None,
    transition_indices: Optional[Sequence[int]] = None,
    transition_times: Optional[Sequence[float]] = None,
    edge_labels: Optional[Sequence[Any]] = None,
    change_constraints: Any = None,
    step1_result: Optional[ChangeDetectionResult] = None,
    run_source_decomposition: bool = True,
    step2_kwargs: Optional[Mapping[str, Any]] = None,
    step3_kwargs: Optional[Mapping[str, Any]] = None,
    step4_kwargs: Optional[Mapping[str, Any]] = None,
) -> TIDESResult:
    """
    Run the current TIDES pipeline from preprocessed observations.

    Parameters
    ----------
    Y
        Preprocessed node-space vector-field observations, shape ``(T,N)``.

    D
        Candidate-edge incidence / aggregation matrix, shape ``(N,M)``.

    edge_features
        Edge-space feature responses, shape ``(T,M,L)``.

    stage_of_sample
        Detected stage label of each preprocessed observation.

    change_hypothesis
        Step-2 structural hypothesis.  The current implemented branch is
        ``'varying_structure'``.

    source_hypothesis
        Step-4 microscopic source hypothesis.  The current implemented branch
        is ``'shared_interaction_law'``.

    profile_floor
        Data-derived numerical-resolution floor used by the current formal
        forward/backward Step-2 sparse search.

    transition_indices, transition_times
        Optional Step-1 metadata.  They label Step-2 constraints but are not
        used to fit coefficients.

    edge_labels
        Optional human-readable labels for candidate edges.

    change_constraints
        Optional manual/oracle Step-2 output.  If supplied, Step 2 is skipped
        and the constraints are passed directly to Step 3.

    step1_result
        Optional Step-1 result retained for provenance.  If transition metadata
        are not supplied explicitly, they are copied from this object.

    run_source_decomposition
        If False, stop after Step 3.

    stepN_kwargs
        Additional keyword arguments forwarded only to that step.

    Notes
    -----
    For ``varying_structure``, this wrapper defaults Step 2 to the currently
    validated ``forward_backward_floor`` sparse search.  It does not use the
    old Adaptive Group LASSO path unless explicitly requested in
    ``step2_kwargs``.
    """

    s2_kwargs = _as_kwargs(step2_kwargs)
    s3_kwargs = _as_kwargs(step3_kwargs)
    s4_kwargs = _as_kwargs(step4_kwargs)

    # Step-1 metadata are provenance / labeling information only.
    if step1_result is not None:
        if transition_indices is None:
            transition_indices = step1_result.transition_indices
        if transition_times is None:
            transition_times = step1_result.transition_times

    transitions = _coerce_optional_int_vector(
        transition_indices,
        name="transition_indices",
    )
    transition_t = _coerce_optional_float_vector(
        transition_times,
        name="transition_times",
    )

    if (
        transitions is not None
        and transition_t is not None
        and transitions.size != transition_t.size
    ):
        raise ValueError(
            "transition_indices and transition_times must have equal length."
        )

    # ------------------------------------------------------------------ Step 2
    if change_constraints is None:

        forbidden_step2 = {
            "hypothesis",
            "transition_indices",
            "transition_times",
            "edge_labels",
            "profile_floor",
        } & set(s2_kwargs)

        if forbidden_step2:
            raise ValueError(
                "These Step-2 arguments are supplied by run_tides and must "
                "not be repeated in step2_kwargs: "
                + ", ".join(sorted(forbidden_step2))
            )

        if change_hypothesis == "varying_structure":
            s2_kwargs.setdefault(
                "solver_method",
                "forward_backward_floor",
            )

            if (
                s2_kwargs["solver_method"] == "forward_backward_floor"
                and profile_floor is None
            ):
                raise ValueError(
                    "The formal varying_structure Step-2 solver "
                    "'forward_backward_floor' requires profile_floor."
                )

        step2 = infer_change_structure_from_observations(
            Y,
            D,
            edge_features,
            stage_of_sample,
            hypothesis=change_hypothesis,
            transition_indices=transitions,
            transition_times=transition_t,
            edge_labels=edge_labels,
            profile_floor=profile_floor,
            **s2_kwargs,
        )

        constraints_for_step3: Any = step2

        if transitions is None:
            transitions = np.asarray(
                [c.transition_index for c in step2.constraints],
                dtype=np.int64,
            )
        if transition_t is None:
            times = np.asarray(
                [c.transition_time for c in step2.constraints],
                dtype=float,
            )
            if np.all(np.isfinite(times)):
                transition_t = times

    else:
        step2 = None
        constraints_for_step3 = change_constraints

        if transitions is None or transition_t is None:
            inferred_idx, inferred_t = _transition_metadata_from_constraints(
                change_constraints
            )
            if transitions is None:
                transitions = inferred_idx
            if transition_t is None:
                transition_t = inferred_t

    # ------------------------------------------------------------------ Step 3
    forbidden_step3 = {
        "Y",
        "D",
        "edge_features",
        "stage_of_sample",
        "change_constraints_or_supports",
    } & set(s3_kwargs)

    if forbidden_step3:
        raise ValueError(
            "These Step-3 arguments are supplied by run_tides and must not "
            "be repeated in step3_kwargs: "
            + ", ".join(sorted(forbidden_step3))
        )

    step3 = reconstruct_vector_field_from_observations(
        Y,
        D,
        edge_features,
        stage_of_sample,
        constraints_for_step3,
        **s3_kwargs,
    )

    # ------------------------------------------------------------------ Step 4
    if run_source_decomposition:

        forbidden_step4 = {"hypothesis"} & set(s4_kwargs)
        if forbidden_step4:
            raise ValueError(
                "Pass source_hypothesis through run_tides, not step4_kwargs."
            )

        step4 = decompose_vector_field_sources(
            step3,
            hypothesis=source_hypothesis,
            **s4_kwargs,
        )
        source_name: Optional[str] = str(source_hypothesis)

    else:
        step4 = None
        source_name = None

    return TIDESResult(
        change_hypothesis=str(change_hypothesis),
        source_hypothesis=source_name,
        transition_indices=(
            None if transitions is None else transitions.copy()
        ),
        transition_times=(
            None if transition_t is None else transition_t.copy()
        ),
        change_constraints=constraints_for_step3,
        step1=step1_result,
        step2=step2,
        step3=step3,
        step4=step4,
    )


# Explicit name that documents where the current wrapper starts.
run_tides_from_observations = run_tides

# Short interactive alias.
run = run_tides


__all__ = [
    # Pipeline result / orchestration.
    "TIDESResult",
    "run_tides",
    "run_tides_from_observations",
    "run",

    # Step 1.
    "ChangeDetectionResult",
    "detect_changes",

    # Step 2.
    "ChangeStructureResult",
    "infer_change_structure",
    "infer_change_structure_from_observations",

    # Step 3.
    "VectorFieldReconstructionResult",
    "reconstruct_vector_field",
    "reconstruct_vector_field_from_observations",

    # Step 4: source decomposition.
    "SourceDecompositionResult",
    "decompose_shared_interaction_law",
    "decompose_vector_field_sources",
    "decompose_sources",
]
