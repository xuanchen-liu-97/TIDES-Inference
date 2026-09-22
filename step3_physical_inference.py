"""
TIDES Step 3: physical inference orchestration.

Architecture
------------
Step 3 has TWO outer computations and THREE inner computations.

Outer computation 1 — Dynamics Support Enumeration
    enumerate J independently
        -> in1: weight-category distribution inference at fixed (J, S)
        -> in2: structural support exploration at fixed J
                 [internally alternates accepted structural moves with in1]
        -> in3: category-relaxed basin-retention compression at fixed J
                 [delete-only; no in1 interleaving]
        -> terminal in1 re-profiling on each compressed support

Outer computation 2 — Cross-J Model Collapse
    collect terminal fixed-J models
        -> identify cross-J expansion sequences representing the same
           underlying physical model at different dynamics resolutions
        -> collapse each sequence into a resolution-consistent physical branch
        -> evaluate final posterior/MDL evidence across collapsed branches
        -> select/report one or multiple physical explanations

Important scheduling rule
-------------------------
in1 and in2 may alternate during reversible exploration.
After structural coverage is reached, in3 runs in category-relaxed/free-W
topology geometry and therefore does not interleave with in1.  Once an in3
lineage terminates, in1 is re-run from a fully released category seed on the
terminal support.

in2 and in3 are NOT arbitrarily interleaved.

The module is intentionally orchestration-only.  All problem-specific
numerics (VarPro, category MDL, destroy-repair proposals, category-relaxed basin
mass, cross-J branch matching, and final posterior/MDL evaluation) are
provided through backend interfaces.

No truth labels are used anywhere in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Hashable, Iterable, Literal, Mapping, Protocol, Sequence

try:  # package imports
    from .step3in1_weight_category_distribution import (
        CategoryBackend,
        In1Config,
        In1Result,
        In1InfeasibleError,
        ProfiledState,
        run_weight_category_distribution,
    )
    from .step3in2_structural_support_exploration import (
        BirthSeedBuilder,
        CapacityDiagnostic,
        In2Config,
        In2Result,
        StructuralExplorationBackend,
        run_structural_support_exploration,
    )
    from .step3in3_basin_preserving_compression import (
        BranchSelector,
        CompressionLineage,
        BasinBackend,
        In3Config,
        In3Result,
        StopRule,
        best_retention_selector,
        no_automatic_stop,
        run_basin_preserving_compression,
    )
except ImportError:  # flat-file imports
    from step3in1_weight_category_distribution import (
        CategoryBackend,
        In1Config,
        In1Result,
        In1InfeasibleError,
        ProfiledState,
        run_weight_category_distribution,
    )
    from step3in2_structural_support_exploration import (
        BirthSeedBuilder,
        CapacityDiagnostic,
        In2Config,
        In2Result,
        StructuralExplorationBackend,
        run_structural_support_exploration,
    )
    from step3in3_basin_preserving_compression import (
        BranchSelector,
        CompressionLineage,
        BasinBackend,
        In3Config,
        In3Result,
        StopRule,
        best_retention_selector,
        no_automatic_stop,
        run_basin_preserving_compression,
    )


DynamicsSupport = Hashable
BranchId = Hashable


class CoverageUnresolvedError(RuntimeError):
    """in2 stopped without an explicit structural-coverage certificate."""

    def __init__(self, message: str, *, partial_result=None):
        super().__init__(message)
        self.partial_result = partial_result


class CertifiedJInfeasibleError(RuntimeError):
    """Backend-supplied certificate that a tested J has no admissible model."""


class TerminalReprofileUnresolvedError(RuntimeError):
    """All archived compression checkpoints failed terminal category re-profiling."""

    def __init__(self, message: str, *, partial_result=None):
        super().__init__(message)
        self.partial_result = partial_result


# ---------------------------------------------------------------------------
# Fixed-J construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FixedJComponents:
    """Everything needed to run the three inner stages for one fixed J.

    Each J must receive an independently constructed instance.  In particular,
    a category partition or structural support learned under another J must not
    be silently reused here.

    ``initial_state`` is the starting physical representation for this J.
    It may be broad or imperfect; in2 is responsible for support composition,
    while optional birth can later relax an under-complete support assumption.
    """

    initial_state: ProfiledState
    category_backend: CategoryBackend
    structural_backend: StructuralExplorationBackend
    basin_backend: BasinBackend

    # Optional in2 capacity-expansion hooks.  These are needed only when
    # In2Config.allow_birth=True.
    capacity_diagnostic: CapacityDiagnostic | None = None
    build_birth_seed: BirthSeedBuilder | None = None

    # in3 navigation hooks.  Defaults reproduce the currently frozen local
    # basin-retention logic; production runs can inject adaptive cliff and
    # near-tie branching rules.
    stop_rule: StopRule | None = None
    branch_selector: BranchSelector = best_retention_selector

    metadata: Mapping[str, Any] = field(default_factory=dict)


class FixedJFactory(Protocol):
    """Build an independent fixed-J problem.

    Return None when the chosen J cannot produce an admissible initial
    representation under the current physical model class / uncertainty floor.
    """

    def build(self, dynamics_support: DynamicsSupport) -> FixedJComponents | None:
        ...


CheckpointReprofileStatus = Literal[
    "not_attempted",
    "success",
    "profile_infeasible",
]


@dataclass(frozen=True)
class TerminalReprofileFailure:
    """Stable failure record for one raw compression checkpoint."""

    lineage_id: int
    checkpoint_id: int
    deletion_depth: int
    support: frozenset[Hashable]
    reason: str

    # Compatibility aliases for older analysis code.  These are deliberately
    # properties, so the stable identity remains lineage_id/checkpoint_id.
    @property
    def lineage_index(self) -> int:
        return self.lineage_id

    @property
    def checkpoint_index(self) -> int:
        return self.checkpoint_id


@dataclass(frozen=True)
class CompressionCheckpointRecord:
    """Immutable provenance record for one raw in3 checkpoint.

    ``checkpoint_id`` is the checkpoint's position in the original raw lineage
    and never changes after terminal category re-profiling.  Because in3 is
    monotone and performs exactly one deletion per step, ``deletion_depth`` is
    the number of accepted deletions preceding this checkpoint.

    Raw and re-profiled states are stored separately.  Failed or unattempted
    re-profiling therefore cannot re-index the surviving candidates or destroy
    the original deletion path.
    """

    dynamics_support: DynamicsSupport
    lineage_id: int
    checkpoint_id: int
    deletion_depth: int
    raw_state: ProfiledState
    reprofile_status: CheckpointReprofileStatus
    reprofiled_state: ProfiledState | None = None
    selected_for_model_comparison: bool = False
    is_raw_terminal: bool = False
    failure_reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def candidate_id(self) -> tuple[DynamicsSupport, int, int]:
        return (self.dynamics_support, self.lineage_id, self.checkpoint_id)


@dataclass(frozen=True)
class FixedJInferenceResult:
    dynamics_support: DynamicsSupport
    initial_state: ProfiledState
    initial_in1: In1Result
    in2: In2Result
    # ``in3`` is the unmodified raw compression result.  Terminal category
    # re-profiling lives in ``checkpoint_archive`` instead of rewriting the
    # raw lineage/checkpoint coordinates.
    in3: In3Result
    checkpoint_archive: tuple[CompressionCheckpointRecord, ...] = ()
    # Compatibility/convenience view: every successful archived model state,
    # without support-based de-duplication.
    model_candidates: tuple[ProfiledState, ...] = ()
    terminal_reprofile_failures: tuple[TerminalReprofileFailure, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def broad_state(self) -> ProfiledState:
        """Coverage-stage output passed from in2 to in3."""
        return self.in2.state

    @property
    def terminal_lineages(self) -> tuple[CompressionLineage, ...]:
        """Raw in3 lineages; identities are never compacted after re-profiling."""
        return self.in3.lineages

    @property
    def successful_checkpoint_records(self) -> tuple[CompressionCheckpointRecord, ...]:
        return tuple(
            record
            for record in self.checkpoint_archive
            if record.reprofile_status == "success"
            and record.selected_for_model_comparison
        )


@dataclass(frozen=True)
class FixedJPartialResult:
    """Recoverable fixed-J work produced before an unresolved handoff.

    This object is deliberately archival rather than inferential: it preserves
    already-computed feasible states, exploration history, raw compression
    lineages, and local failures without promoting an unresolved run to a
    completed physical model.  It is suitable for diagnostics and restart.
    """

    dynamics_support: DynamicsSupport
    stage: str
    initial_state: ProfiledState
    initial_in1: In1Result | None = None
    in2: In2Result | None = None
    raw_in3: In3Result | None = None
    recoverable_states: tuple[ProfiledState, ...] = ()
    checkpoint_archive: tuple[CompressionCheckpointRecord, ...] = ()
    terminal_reprofile_failures: tuple[TerminalReprofileFailure, ...] = ()
    reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class JInferenceIssue:
    dynamics_support: DynamicsSupport
    kind: str
    reason: str
    partial_result: FixedJPartialResult | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InfeasibleJ:
    """A J with an explicit backend-supplied infeasibility certificate."""
    dynamics_support: DynamicsSupport
    reason: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DynamicsSupportEnumerationResult:
    """Result of Dynamics Support Enumeration."""
    fixed_j_results: tuple[FixedJInferenceResult, ...]
    unresolved_j: tuple[JInferenceIssue, ...] = ()
    certified_infeasible_j: tuple[InfeasibleJ, ...] = ()

    @property
    def infeasible_j(self) -> tuple[InfeasibleJ, ...]:
        return self.certified_infeasible_j

    @property
    def has_feasible_pairwise_model(self) -> bool:
        return bool(self.fixed_j_results)

    @property
    def partial_j_results(self) -> tuple[FixedJPartialResult, ...]:
        """Recoverable work from unresolved fixed-J runs."""
        return tuple(
            issue.partial_result
            for issue in self.unresolved_j
            if issue.partial_result is not None
        )


# ---------------------------------------------------------------------------
# Cross-J Model Collapse: cross-J branches + final posterior/MDL selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TerminalCandidate:
    """One successfully re-profiled fixed-J checkpoint candidate.

    Identity is inherited from the raw compression path.  It is never based on
    the position of a filtered/re-profiled list, so failures cannot renumber
    later checkpoints.
    """

    dynamics_support: DynamicsSupport
    lineage_id: int
    checkpoint_id: int
    deletion_depth: int
    lineage: CompressionLineage
    fixed_j_result: FixedJInferenceResult
    checkpoint_record: CompressionCheckpointRecord
    is_terminal: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def state(self) -> ProfiledState:
        state = self.checkpoint_record.reprofiled_state
        if state is None:
            raise RuntimeError("TerminalCandidate requires a successful re-profiled checkpoint.")
        return state

    @property
    def candidate_id(self) -> tuple[DynamicsSupport, int, int]:
        return (self.dynamics_support, self.lineage_id, self.checkpoint_id)

    # Compatibility aliases.
    @property
    def lineage_index(self) -> int:
        return self.lineage_id

    @property
    def checkpoint_index(self) -> int:
        return self.checkpoint_id


@dataclass(frozen=True)
class CrossJBranch:
    """A resolution-consistent physical branch produced by model collapse.

    Members are J-specific terminal models interpreted as different
    truncation/resolution representations of the same underlying physical
    interaction law. Matching should compare reconstructed interaction
    functions over the observed state domain together with structural
    consistency, rather than raw coefficient-vector distances across J.
    """

    branch_id: BranchId
    members: tuple[TerminalCandidate, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FinalBranchScore:
    """Final branch-level evidence used only in Cross-J Model Collapse.

    ``description_length`` and ``log_posterior_mass`` are optional because the
    exact final posterior/MDL decomposition remains backend-specific.

    The backend must preserve the TIDES hard-floor philosophy: residual depth
    inside B_epsilon is not a generic structural ranking score.  If basin mass
    is already included in the posterior/MDL term, it must not be double-counted
    through a separate precision penalty.
    """

    branch_id: BranchId
    description_length: float | None = None
    log_posterior_mass: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class CrossJModelCollapseBackend(Protocol):
    """Backend for Cross-J Model Collapse.

    discover_branches()
        Identify J-specific terminal models that form the same expansion
        sequence across dynamics resolutions.  Matching may use structural
        consistency together with reconstructed interaction-law agreement over
        the observed state domain.  It must not use benchmark truth labels.

    score_branch()
        Evaluate final posterior/MDL evidence only after resolution-specific
        models have been collapsed into physical branches.  This is final
        checkpoint selection, not local topology navigation.

    select()
        Return the collapsed branch IDs to retain/report.  Multiple branches
        may remain when the trajectory does not uniquely identify one physical
        explanation.
    """

    def discover_branches(
        self,
        candidates: Sequence[TerminalCandidate],
    ) -> Iterable[CrossJBranch]:
        ...

    def score_branch(
        self,
        branch: CrossJBranch,
    ) -> FinalBranchScore:
        ...

    def select(
        self,
        branches: Sequence[CrossJBranch],
        scores: Sequence[FinalBranchScore],
    ) -> Sequence[BranchId]:
        ...


@dataclass(frozen=True)
class CrossJModelCollapseResult:
    """Result of Cross-J Model Collapse."""

    candidates: tuple[TerminalCandidate, ...]
    branches: tuple[CrossJBranch, ...]
    scores: tuple[FinalBranchScore, ...]
    selected_branch_ids: tuple[BranchId, ...]


@dataclass(frozen=True)
class Step3Config:
    """Configuration shared by the Step-3 orchestrator."""
    in1: In1Config
    in2: In2Config = In2Config()
    in3: In3Config = In3Config()
    allow_unresolved_compression: bool = False
    compression_archive_tail: int = 3


@dataclass(frozen=True)
class Step3Result:
    """Complete Step-3 output."""

    outer1: DynamicsSupportEnumerationResult
    outer2: CrossJModelCollapseResult | None
    status: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Inner-pipeline orchestration at fixed J
# ---------------------------------------------------------------------------


def _fully_released_category_seed(state: ProfiledState) -> ProfiledState:
    """Give every active structural coordinate its own category.

    This is the safe bridge from category-relaxed in3 back to in1: whenever the
    terminal support has nonzero free-W basin mass, the fully released category
    representation has the same continuous flexibility and therefore cannot be
    blocked merely by stale category sharing.
    """

    ordered = tuple(sorted(state.support, key=repr))
    categories = {g: k for k, g in enumerate(ordered)}
    return ProfiledState(
        support=frozenset(ordered),
        categories=categories,
        description_length=state.description_length,
        relative_residual=state.relative_residual,
        weights=state.weights,
        theta=state.theta,
        metadata={
            **dict(state.metadata),
            "category_seed": "fully_released_after_in3",
        },
    )


def _validate_raw_compression_lineage(lineage: CompressionLineage) -> tuple[ProfiledState, ...]:
    """Validate the one-deletion/one-checkpoint provenance invariant."""

    checkpoints = lineage.checkpoints or (lineage.state,)
    if len(checkpoints) != len(lineage.history) + 1:
        raise RuntimeError(
            "Raw in3 lineage provenance is inconsistent: expected exactly one "
            "checkpoint before the first deletion and one after every deletion. "
            f"got checkpoints={len(checkpoints)}, history={len(lineage.history)}."
        )
    if frozenset(checkpoints[-1].support) != frozenset(lineage.state.support):
        raise RuntimeError("Raw in3 lineage.state does not match its final checkpoint.")

    for depth, step in enumerate(lineage.history, start=1):
        before = frozenset(checkpoints[depth - 1].support)
        after = frozenset(checkpoints[depth].support)
        if before != frozenset(step.before_support):
            raise RuntimeError(
                f"Compression history/checkpoint mismatch before deletion depth {depth}."
            )
        if after != frozenset(step.after_support):
            raise RuntimeError(
                f"Compression history/checkpoint mismatch after deletion depth {depth}."
            )
        if after != before - {step.deleted}:
            raise RuntimeError(
                f"Compression lineage is not one-delete monotone at depth {depth}."
            )
    return tuple(checkpoints)


def _archive_compression_checkpoints(
    dynamics_support: DynamicsSupport,
    raw_lineages: Sequence[CompressionLineage],
    *,
    relax_in1,
    archive_tail: int,
) -> tuple[
    tuple[CompressionCheckpointRecord, ...],
    tuple[ProfiledState, ...],
    tuple[TerminalReprofileFailure, ...],
]:
    """Re-profile selected tail checkpoints without mutating raw provenance.

    Every raw checkpoint receives an archive record.  Earlier checkpoints are
    retained as ``not_attempted`` recovery/provenance records; the configured
    tail is independently re-profiled and marked success/failure.  Successful
    states are *not* de-duplicated by support.
    """

    if archive_tail < 1:
        raise ValueError("compression_archive_tail must be >= 1.")

    archive: list[CompressionCheckpointRecord] = []
    model_candidates: list[ProfiledState] = []
    failures: list[TerminalReprofileFailure] = []

    for lineage_id, lineage in enumerate(raw_lineages):
        checkpoints = _validate_raw_compression_lineage(lineage)
        start_id = max(0, len(checkpoints) - archive_tail)
        terminal_id = len(checkpoints) - 1

        for checkpoint_id, raw_state in enumerate(checkpoints):
            selected = checkpoint_id >= start_id
            base_metadata = {
                "raw_lineage_status": lineage.status,
                "raw_stopped_reason": lineage.stopped_reason,
            }

            if not selected:
                archive.append(
                    CompressionCheckpointRecord(
                        dynamics_support=dynamics_support,
                        lineage_id=lineage_id,
                        checkpoint_id=checkpoint_id,
                        deletion_depth=checkpoint_id,
                        raw_state=raw_state,
                        reprofile_status="not_attempted",
                        selected_for_model_comparison=False,
                        is_raw_terminal=(checkpoint_id == terminal_id),
                        metadata=base_metadata,
                    )
                )
                continue

            try:
                profiled = relax_in1(_fully_released_category_seed(raw_state))
            except In1InfeasibleError as exc:
                failure = TerminalReprofileFailure(
                    lineage_id=lineage_id,
                    checkpoint_id=checkpoint_id,
                    deletion_depth=checkpoint_id,
                    support=frozenset(raw_state.support),
                    reason=str(exc),
                )
                failures.append(failure)
                archive.append(
                    CompressionCheckpointRecord(
                        dynamics_support=dynamics_support,
                        lineage_id=lineage_id,
                        checkpoint_id=checkpoint_id,
                        deletion_depth=checkpoint_id,
                        raw_state=raw_state,
                        reprofile_status="profile_infeasible",
                        selected_for_model_comparison=True,
                        is_raw_terminal=(checkpoint_id == terminal_id),
                        failure_reason=str(exc),
                        metadata=base_metadata,
                    )
                )
                continue

            archive.append(
                CompressionCheckpointRecord(
                    dynamics_support=dynamics_support,
                    lineage_id=lineage_id,
                    checkpoint_id=checkpoint_id,
                    deletion_depth=checkpoint_id,
                    raw_state=raw_state,
                    reprofile_status="success",
                    reprofiled_state=profiled,
                    selected_for_model_comparison=True,
                    is_raw_terminal=(checkpoint_id == terminal_id),
                    metadata=base_metadata,
                )
            )
            # Intentionally no support-based de-duplication: the same support
            # reached in different lineages/basins remains a distinct model
            # candidate for later human/cross-J analysis.
            model_candidates.append(profiled)

    return tuple(archive), tuple(model_candidates), tuple(failures)


def run_fixed_j_inference(
    dynamics_support: DynamicsSupport,
    components: FixedJComponents,
    *,
    config: Step3Config,
) -> FixedJInferenceResult:
    """Run in1 -> (in2 <-> in1) -> in3 -> terminal in1 for one fixed J.

    Phase boundary:
      * reversible structural exploration finishes before compression starts;
      * in3 measures topology necessity in category-relaxed/free-W geometry;
      * category sharing is re-inferred only after a compression lineage stops;
      * in3 never hands control back to in2.
    """

    in1_initial = run_weight_category_distribution(
        components.initial_state,
        components.category_backend,
        config=config.in1,
    )

    def relax_in1(state: ProfiledState) -> ProfiledState:
        return run_weight_category_distribution(
            state,
            components.category_backend,
            config=config.in1,
        ).state

    in2_result = run_structural_support_exploration(
        in1_initial.state,
        components.structural_backend,
        relax_in1=relax_in1,
        config=config.in2,
        capacity_diagnostic=components.capacity_diagnostic,
        build_birth_seed=components.build_birth_seed,
    )

    if in2_result.coverage_status != "coverage_saturated" and not config.allow_unresolved_compression:
        recoverable = in2_result.representative_states or (in2_result.state,)
        message = (
            f"in2 ended with status={in2_result.coverage_status!r}; compression is withheld "
            "until structural coverage is explicitly certified."
        )
        partial = FixedJPartialResult(
            dynamics_support=dynamics_support,
            stage="in2",
            initial_state=components.initial_state,
            initial_in1=in1_initial,
            in2=in2_result,
            recoverable_states=tuple(recoverable),
            reason=message,
            metadata={
                **dict(components.metadata),
                "in2_coverage_status": in2_result.coverage_status,
                "n_recoverable_states": len(recoverable),
            },
        )
        raise CoverageUnresolvedError(message, partial_result=partial)
    if config.compression_archive_tail < 1:
        raise ValueError("compression_archive_tail must be >= 1.")

    broad_representatives = in2_result.representative_states or (in2_result.state,)
    raw_lineages: list[CompressionLineage] = []
    for broad_state in broad_representatives:
        raw = run_basin_preserving_compression(
            broad_state, components.basin_backend, stop_rule=components.stop_rule,
            branch_selector=components.branch_selector, config=config.in3,
        )
        raw_lineages.extend(raw.lineages)

    raw_in3 = In3Result(lineages=tuple(raw_lineages))
    checkpoint_archive, model_candidates, terminal_failures = _archive_compression_checkpoints(
        dynamics_support,
        raw_lineages,
        relax_in1=relax_in1,
        archive_tail=config.compression_archive_tail,
    )

    if not model_candidates:
        message = (
            "Compression produced no archived tail checkpoint that remained feasible after "
            "fully released terminal category re-profiling."
        )
        # Broad representatives are already feasible outputs of reversible
        # exploration and therefore remain valid recovery/restart points even
        # when every selected terminal tail checkpoint fails re-profiling.
        partial = FixedJPartialResult(
            dynamics_support=dynamics_support,
            stage="terminal_reprofile",
            initial_state=components.initial_state,
            initial_in1=in1_initial,
            in2=in2_result,
            raw_in3=raw_in3,
            recoverable_states=tuple(broad_representatives),
            checkpoint_archive=checkpoint_archive,
            terminal_reprofile_failures=terminal_failures,
            reason=message,
            metadata={
                **dict(components.metadata),
                "in2_coverage_status": in2_result.coverage_status,
                "n_in2_representatives": len(broad_representatives),
                "n_raw_compression_lineages": len(raw_lineages),
                "n_archived_checkpoints": len(checkpoint_archive),
                "n_terminal_reprofile_failures": len(terminal_failures),
            },
        )
        raise TerminalReprofileUnresolvedError(message, partial_result=partial)

    return FixedJInferenceResult(
        dynamics_support=dynamics_support,
        initial_state=components.initial_state,
        initial_in1=in1_initial,
        in2=in2_result,
        in3=raw_in3,
        checkpoint_archive=checkpoint_archive,
        model_candidates=model_candidates,
        terminal_reprofile_failures=terminal_failures,
        metadata={
            **dict(components.metadata),
            "in2_coverage_status": in2_result.coverage_status,
            "n_in2_representatives": len(broad_representatives),
            "n_raw_compression_lineages": len(raw_lineages),
            "n_archived_checkpoints": len(checkpoint_archive),
            "n_compression_candidates": len(model_candidates),
            "n_terminal_reprofile_failures": len(terminal_failures),
        },
    )


# ---------------------------------------------------------------------------
# DYNAMICS SUPPORT ENUMERATION
# ---------------------------------------------------------------------------


def dynamics_support_enumeration(
    dynamics_supports: Sequence[DynamicsSupport],
    factory: FixedJFactory,
    *,
    config: Step3Config,
) -> DynamicsSupportEnumerationResult:
    """Dynamics Support Enumeration: enumerate J independently and run the complete three-inner-stage pipeline.

    There is deliberately no warm-start transfer of S, c, W, or Theta from one
    J to the next.  Each J receives its own full in1/in2/in3 inference.

    A J for which ``factory.build(J)`` returns None is recorded as unresolved
    (initialization not found), not scientifically infeasible.  Only an explicit
    ``CertifiedJInfeasibleError`` is stored as certified model-class infeasibility.
    """

    feasible: list[FixedJInferenceResult] = []
    unresolved: list[JInferenceIssue] = []
    certified_infeasible: list[InfeasibleJ] = []

    for J in dynamics_supports:
        try:
            components = factory.build(J)
        except CertifiedJInfeasibleError as exc:
            certified_infeasible.append(InfeasibleJ(dynamics_support=J, reason=str(exc)))
            continue

        if components is None:
            unresolved.append(JInferenceIssue(
                dynamics_support=J, kind="initialization_not_found",
                reason="factory returned no admissible fixed-J initialization",
            ))
            continue

        try:
            result = run_fixed_j_inference(J, components, config=config)
        except In1InfeasibleError as exc:
            unresolved.append(JInferenceIssue(
                dynamics_support=J, kind="initialization_infeasible", reason=str(exc)
            ))
            continue
        except CoverageUnresolvedError as exc:
            unresolved.append(JInferenceIssue(
                dynamics_support=J,
                kind="coverage_unresolved",
                reason=str(exc),
                partial_result=exc.partial_result,
            ))
            continue
        except TerminalReprofileUnresolvedError as exc:
            unresolved.append(JInferenceIssue(
                dynamics_support=J,
                kind="terminal_reprofile_unresolved",
                reason=str(exc),
                partial_result=exc.partial_result,
            ))
            continue

        feasible.append(result)

    return DynamicsSupportEnumerationResult(
        fixed_j_results=tuple(feasible),
        unresolved_j=tuple(unresolved),
        certified_infeasible_j=tuple(certified_infeasible),
    )


# ---------------------------------------------------------------------------
# CROSS-J MODEL COLLAPSE
# ---------------------------------------------------------------------------


def _collect_terminal_candidates(
    outer1: DynamicsSupportEnumerationResult,
) -> tuple[TerminalCandidate, ...]:
    """Collect successful archive records using their stable raw identities."""

    candidates: list[TerminalCandidate] = []
    for fixed_j in outer1.fixed_j_results:
        lineages = fixed_j.in3.lineages
        for record in fixed_j.successful_checkpoint_records:
            if not (0 <= record.lineage_id < len(lineages)):
                raise RuntimeError(
                    f"Archive record refers to unknown lineage_id={record.lineage_id}."
                )
            lineage = lineages[record.lineage_id]
            candidates.append(
                TerminalCandidate(
                    dynamics_support=fixed_j.dynamics_support,
                    lineage_id=record.lineage_id,
                    checkpoint_id=record.checkpoint_id,
                    deletion_depth=record.deletion_depth,
                    lineage=lineage,
                    fixed_j_result=fixed_j,
                    checkpoint_record=record,
                    is_terminal=record.is_raw_terminal,
                    metadata={
                        "reprofile_status": record.reprofile_status,
                        "selected_for_model_comparison": record.selected_for_model_comparison,
                    },
                )
            )
    return tuple(candidates)


def cross_J_model_collapse(
    outer1: DynamicsSupportEnumerationResult,
    backend: CrossJModelCollapseBackend,
) -> CrossJModelCollapseResult:
    """Cross-J Model Collapse: collapse resolution-specific terminal models into physical branches.

    This is the only place where candidates inferred under different dynamics
    supports J are compared.

    Local in1/in2/in3 decisions are already finished.  Therefore final
    posterior/MDL evidence cannot retroactively drive greedy structural moves
    inside an individual J.
    """

    candidates = _collect_terminal_candidates(outer1)
    if not candidates:
        return CrossJModelCollapseResult(
            candidates=(),
            branches=(),
            scores=(),
            selected_branch_ids=(),
        )

    branches = tuple(backend.discover_branches(candidates))

    # Validate branch membership and IDs.
    known_candidate_ids = {
        (c.dynamics_support, c.lineage_id, c.checkpoint_id) for c in candidates
    }
    seen_branch_ids: set[BranchId] = set()

    for branch in branches:
        if branch.branch_id in seen_branch_ids:
            raise ValueError(f"Duplicate cross-J branch id: {branch.branch_id!r}")
        seen_branch_ids.add(branch.branch_id)

        for member in branch.members:
            key = (member.dynamics_support, member.lineage_id, member.checkpoint_id)
            if key not in known_candidate_ids:
                raise ValueError(
                    "Cross-J backend returned a branch containing an unknown "
                    f"terminal candidate: {key!r}"
                )

    membership_count = {key: 0 for key in known_candidate_ids}
    for branch in branches:
        for member in branch.members:
            key = (member.dynamics_support, member.lineage_id, member.checkpoint_id)
            membership_count[key] += 1
    missing_members = {k for k, n in membership_count.items() if n == 0}
    duplicated_members = {k for k, n in membership_count.items() if n > 1}
    if missing_members or duplicated_members:
        raise ValueError(
            "Cross-J collapse must form a complete, non-overlapping partition of archived candidates. "
            f"missing={missing_members}, duplicated={duplicated_members}"
        )

    scores = tuple(backend.score_branch(branch) for branch in branches)
    score_ids = {score.branch_id for score in scores}
    branch_ids = {branch.branch_id for branch in branches}
    if score_ids != branch_ids:
        raise ValueError(
            "Cross-J scoring must return exactly one score for every branch. "
            f"missing={branch_ids - score_ids}, extra={score_ids - branch_ids}"
        )

    selected = tuple(backend.select(branches, scores))
    unknown = set(selected) - branch_ids
    if unknown:
        raise ValueError(f"Cross-J selector returned unknown branch ids: {unknown!r}")

    return CrossJModelCollapseResult(
        candidates=candidates,
        branches=branches,
        scores=scores,
        selected_branch_ids=selected,
    )


# ---------------------------------------------------------------------------
# COMPLETE STEP 3
# ---------------------------------------------------------------------------


def run_step3(
    dynamics_supports: Sequence[DynamicsSupport],
    fixed_j_factory: FixedJFactory,
    cross_j_backend: CrossJModelCollapseBackend | None,
    *,
    config: Step3Config,
) -> Step3Result:
    """Run complete TIDES Step 3.

    Flow
    ----
    Dynamics Support Enumeration
        enumerate J
            for each J:
                in1
                in2 <-> in1
                in3 (free-W basin compression)
                terminal/near-terminal in1 re-profiling

    Cross-J Model Collapse
        collect terminal fixed-J models
            -> identify cross-J expansion sequences
            -> collapse each sequence into one physical branch
            -> final posterior/MDL scoring across collapsed branches
            -> retain one or multiple physical branches

    If ``cross_j_backend`` is None, Dynamics Support Enumeration is still run and returned.  This is
    useful while the exact cross-J posterior/MDL formula is being finalized.
    """

    if not dynamics_supports:
        raise ValueError("At least one dynamics support J must be supplied.")

    outer1 = dynamics_support_enumeration(
        dynamics_supports,
        fixed_j_factory,
        config=config,
    )

    if not outer1.has_feasible_pairwise_model:
        return Step3Result(
            outer1=outer1, outer2=None, status="no_completed_feasible_model_found",
            metadata={
                "n_tested_J": len(dynamics_supports), "n_completed_J": 0,
                "n_unresolved_J": len(outer1.unresolved_j),
                "n_partial_J": len(outer1.partial_j_results),
                "n_certified_infeasible_J": len(outer1.certified_infeasible_j),
            },
        )

    if cross_j_backend is None:
        return Step3Result(
            outer1=outer1,
            outer2=None,
            status="outer1_complete_cross_j_selection_pending",
            metadata={
                "n_tested_J": len(dynamics_supports),
                "n_completed_J": len(outer1.fixed_j_results),
                "n_unresolved_J": len(outer1.unresolved_j),
                "n_partial_J": len(outer1.partial_j_results),
                "n_certified_infeasible_J": len(outer1.certified_infeasible_j),
            },
        )

    outer2 = cross_J_model_collapse(
        outer1,
        cross_j_backend,
    )

    return Step3Result(
        outer1=outer1,
        outer2=outer2,
        status="complete",
        metadata={
            "n_tested_J": len(dynamics_supports),
            "n_completed_J": len(outer1.fixed_j_results),
            "n_unresolved_J": len(outer1.unresolved_j),
            "n_certified_infeasible_J": len(outer1.certified_infeasible_j),
            "n_archived_candidates": len(outer2.candidates),
            "n_cross_j_branches": len(outer2.branches),
            "n_selected_branches": len(outer2.selected_branch_ids),
        },
    )


# Public aliases.
run = run_step3
infer_physical_models = run_step3


__all__ = [
    # Types.
    "DynamicsSupport",
    "BranchId",
    "CoverageUnresolvedError",
    "TerminalReprofileUnresolvedError",
    "CertifiedJInfeasibleError",
    # Fixed-J construction.
    "FixedJComponents",
    "FixedJFactory",
    "CheckpointReprofileStatus",
    "TerminalReprofileFailure",
    "CompressionCheckpointRecord",
    "FixedJInferenceResult",
    "FixedJPartialResult",
    "InfeasibleJ",
    "JInferenceIssue",
    # Dynamics Support Enumeration.
    "DynamicsSupportEnumerationResult",
    "run_fixed_j_inference",
    "dynamics_support_enumeration",
    # Cross-J Model Collapse.
    "TerminalCandidate",
    "CrossJBranch",
    "FinalBranchScore",
    "CrossJModelCollapseBackend",
    "CrossJModelCollapseResult",
    "cross_J_model_collapse",
    # Complete Step 3.
    "Step3Config",
    "Step3Result",
    "run_step3",
    "infer_physical_models",
    "run",
]
