"""TIDES: end-to-end orchestration for the four-step inference pipeline.

TIDES = Temporal Inference and Decomposition via Edge Space.

This module intentionally contains *no new scientific solver logic*.  It only
connects the four independently callable steps:

    Step 1  detect changes
    Step 2  infer the admissible structure of each change
    Step 3  reconstruct the stage-wise edge-resolved vector fields B
    Step 4  decompose B under a physical source hypothesis

The same functions can therefore be used manually, while :func:`run_tides`
provides a convenient full-pipeline entry point.

Current scope
-------------
The currently implemented end-to-end branch is

    varying structure + shared dynamics

for piecewise-stationary systems with locally sparse row changes in ΔB.
The ``varying_dynamics`` hypothesis is already supported by Step 4, but its
Step-2 coherent-change backend is deliberately not yet implemented.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Support both usages:
#   1) package form:     from tides.tides import run_tides
#   2) flat-file form:   from tides import run_tides
try:  # package-relative imports
    from .step1_change_detection import ChangeDetectionResult, detect_changes
    from .step2_change_structure import ChangeStructureResult, infer_change_structure
    from .step3_vector_field import (
        LibraryFunction,
        VectorFieldReconstructionResult,
        reconstruct_vector_field,
    )
    from .step4_decomposition import (
        VaryingDynamicsDecomposition,
        VaryingStructureDecomposition,
        decompose_vector_field,
    )
except ImportError:  # flat directory imports
    from step1_change_detection import ChangeDetectionResult, detect_changes
    from step2_change_structure import ChangeStructureResult, infer_change_structure
    from step3_vector_field import (
        LibraryFunction,
        VectorFieldReconstructionResult,
        reconstruct_vector_field,
    )
    from step4_decomposition import (
        VaryingDynamicsDecomposition,
        VaryingStructureDecomposition,
        decompose_vector_field,
    )


IntArray = NDArray[np.int64]
Hypothesis = Literal["varying_structure", "varying_dynamics"]
DecompositionResult = VaryingStructureDecomposition | VaryingDynamicsDecomposition


@dataclass(frozen=True)
class TIDESResult:
    """Complete output of :func:`run_tides`.

    ``step1`` or ``step2`` is ``None`` when that stage was bypassed by a manual
    / oracle input.  ``transition_indices`` and ``change_constraints`` always
    record what was actually supplied to Step 3.
    """

    hypothesis: str
    transition_indices: IntArray
    change_constraints: Any
    step1: Optional[ChangeDetectionResult]
    step2: Optional[ChangeStructureResult]
    step3: VectorFieldReconstructionResult
    step4: Optional[DecompositionResult]

    @property
    def B_stages(self):
        """Convenience alias for the reconstructed stage-wise B matrices."""

        return self.step3.B_stages

    @property
    def B_anchor(self):
        return self.step3.B_anchor

    @property
    def delta_B(self):
        return self.step3.delta_B

    @property
    def decomposition(self):
        return self.step4



def _as_kwargs(values: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    return {} if values is None else dict(values)



def run_tides(
    X: ArrayLike,
    t: ArrayLike,
    D: ArrayLike,
    library: Sequence[LibraryFunction],
    *,
    hypothesis: Hypothesis,
    transition_indices: Optional[Sequence[int]] = None,
    change_constraints: Any = None,
    run_decomposition: bool = True,
    step1_kwargs: Optional[Mapping[str, Any]] = None,
    step2_kwargs: Optional[Mapping[str, Any]] = None,
    step3_kwargs: Optional[Mapping[str, Any]] = None,
    step4_kwargs: Optional[Mapping[str, Any]] = None,
) -> TIDESResult:
    """Run the current four-step TIDES pipeline.

    Parameters
    ----------
    X, t
        Observed trajectory. ``X`` has shape ``(n_samples, n_nodes)``.
    D
        Candidate-edge incidence/aggregation matrix with shape
        ``(n_nodes, n_candidate_edges)`` for the current conservative pairwise
        representation.
    library
        Candidate interaction-function library used by Step 3.
    hypothesis
        Physical branch used by hypothesis-dependent Steps 2 and 4.
        Currently ``"varying_structure"`` is the implemented end-to-end branch.

    transition_indices
        Optional manual/oracle Step-1 output.  If supplied, Step 1 is skipped.
        Indices refer to sample boundaries in the same convention as
        :func:`detect_changes`.

    change_constraints
        Optional manual/oracle Step-2 output.  It may be either the Step-2
        constraint objects or a sequence of raw row-support index arrays, both
        of which are accepted by Step 3.  If supplied, Step 2 is skipped.

    run_decomposition
        If ``False``, stop after Step 3.  This is useful for vector-field-only
        regression tests.

    stepN_kwargs
        Keyword dictionaries forwarded only to the corresponding independent
        step.  Core pipeline arguments are supplied by this function and should
        not be repeated in these dictionaries.

    Returns
    -------
    TIDESResult
        All intermediate results plus the inputs actually passed between steps.

    Notes
    -----
    This wrapper deliberately does not infer the physical hypothesis from data.
    The hypothesis must be supplied explicitly because it changes the Step-2
    change model and the Step-4 source decomposition.
    """

    s1_kwargs = _as_kwargs(step1_kwargs)
    s2_kwargs = _as_kwargs(step2_kwargs)
    s3_kwargs = _as_kwargs(step3_kwargs)
    s4_kwargs = _as_kwargs(step4_kwargs)

    # ------------------------------------------------------------------ Step 1
    if transition_indices is None:
        step1 = detect_changes(X, t, **s1_kwargs)
        transitions = np.asarray(step1.transition_indices, dtype=np.int64)
    else:
        step1 = None
        transitions = np.asarray(transition_indices, dtype=np.int64)
        if transitions.ndim != 1:
            raise ValueError("transition_indices must be one-dimensional.")
        if transitions.size and not np.all(np.diff(transitions) > 0):
            raise ValueError("transition_indices must be strictly increasing.")

    # ------------------------------------------------------------------ Step 2
    if change_constraints is None:
        step2 = infer_change_structure(
            X,
            t,
            D,
            transitions,
            hypothesis=hypothesis,
            **s2_kwargs,
        )
        constraints_for_step3: Any = step2.constraints
    else:
        step2 = None
        constraints_for_step3 = change_constraints

    # ------------------------------------------------------------------ Step 3
    step3 = reconstruct_vector_field(
        X,
        t,
        D,
        library,
        transitions,
        constraints_for_step3,
        **s3_kwargs,
    )

    # ------------------------------------------------------------------ Step 4
    if run_decomposition:
        step4 = decompose_vector_field(
            step3.B_stages,
            hypothesis=hypothesis,
            **s4_kwargs,
        )
    else:
        step4 = None

    return TIDESResult(
        hypothesis=hypothesis,
        transition_indices=transitions,
        change_constraints=constraints_for_step3,
        step1=step1,
        step2=step2,
        step3=step3,
        step4=step4,
    )


# Short alias for interactive notebooks.
run = run_tides


__all__ = [
    "TIDESResult",
    "run_tides",
    "run",
    # Re-export the four independent public steps so a single import is enough.
    "detect_changes",
    "infer_change_structure",
    "reconstruct_vector_field",
    "decompose_vector_field",
]
