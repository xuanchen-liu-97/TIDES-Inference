"""
TIDES Step 3 inner stage 2: structural support exploration.

Frozen role
-----------
At fixed dynamics support J, explore the composition of the structural support
without performing sparsifying compression.

Validated core:
    destroy -> repair proposal -> provisional replacement -> in1 relaxation

Thus the default move preserves support cardinality:
    E -> E

Residual / conditional-LS repair gains are proposal signals only. They are not
truth scores and are never used here as a final model-selection objective.

Optional generalization:
    persistent fixed-capacity competition -> capacity diagnostic -> birth
    E -> E + 1

Birth is disabled by default because the already-validated J2 path used fixed-E
replacement only. The hook is present so the generalized algorithm can be
tested without changing the validated core.

This module intentionally does not implement E -> E-1 compression. That is the
exclusive responsibility of Step 3 inner stage 3.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Iterable, Mapping, Protocol, Sequence
import random

try:
    from .step3in1_weight_category_distribution import ProfiledState, In1InfeasibleError
except ImportError:
    from step3in1_weight_category_distribution import ProfiledState, In1InfeasibleError

Coordinate = Hashable


@dataclass(frozen=True)
class RepairProposal:
    """One destroy-repair proposal.

    score may be the conditional least-squares repair gain G or an equivalent
    cheap score. It is used only for proposal ordering/screening.
    """

    removed: Coordinate
    added: Coordinate
    score: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BirthProposal:
    """Optional capacity-expansion proposal, produced only after diagnosis."""

    added: Coordinate
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class In2Config:
    max_sweeps: int = 100
    no_move_patience: int = 3
    random_seed: int | None = None
    allow_birth: bool = False
    recurrence_trigger: int = 2
    max_representatives: int = 8
    coverage_check_interval: int = 5


@dataclass(frozen=True)
class In2Move:
    kind: str  # replacement or birth
    before_support: frozenset[Coordinate]
    after_support: frozenset[Coordinate]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class In2CandidateFailure:
    kind: str
    support: frozenset[Coordinate]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CoverageAssessment:
    """Backend certificate for the in2 -> in3 phase boundary.

    ``saturated=True`` means the backend certifies that the currently explored
    structural region has sufficient coverage for irreversible compression to
    begin.  This is intentionally distinct from temporary lack of accepted
    moves or exhaustion of a computational budget.
    """

    saturated: bool
    reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CoverageCheck:
    """One explicit structural-coverage assessment performed during in2."""

    sweep: int
    trigger: str  # periodic, stall, or budget_end
    support: frozenset[Coordinate]
    assessment: CoverageAssessment


@dataclass(frozen=True)
class In2Result:
    state: ProfiledState
    history: tuple[In2Move, ...]
    visited_supports: tuple[frozenset[Coordinate], ...]
    representative_states: tuple[ProfiledState, ...]
    candidate_failures: tuple[In2CandidateFailure, ...]
    coverage_status: str
    coverage_checks: tuple[CoverageCheck, ...] = ()


class StructuralExplorationBackend(Protocol):
    """Model-specific destroy-first proposal engine.

    The implementation should preserve the tested proposal geometry:
      1. destroy an active coordinate;
      2. reprofile the destroyed state, releasing stale categories as needed;
      3. build repair directions from the destroyed residual;
      4. return a high-ranked working set PLUS a global stochastic tail.

    No truth labels may be used.
    """

    def replacement_proposals(
        self,
        state: ProfiledState,
        *,
        rng: random.Random,
    ) -> Iterable[RepairProposal]:
        ...

    def build_replacement_seed(
        self,
        state: ProfiledState,
        proposal: RepairProposal,
    ) -> ProfiledState:
        """Build a provisional E-preserving state before in1 relaxation."""
        ...

    def choose_replacement(
        self,
        current: ProfiledState,
        candidates: Sequence[tuple[RepairProposal, ProfiledState]],
        history: Sequence[In2Move],
        *,
        rng: random.Random,
    ) -> int | None:
        """Choose a navigation move.

        This must NOT default to greedy local MDL or residual-depth ranking.
        Returning None rejects all currently proposed replacements.
        """
        ...

    def coverage_saturated(
        self,
        state: ProfiledState,
        history: Sequence[In2Move],
        visited_supports: Sequence[frozenset[Coordinate]],
    ) -> bool | CoverageAssessment:
        """Assess whether reversible exploration has sufficient coverage.

        This is a scientific phase-boundary certificate, not a convergence
        surrogate.  It may return ``True`` only when the backend-specific
        coverage/capacity diagnostics justify handing the current representative
        set to irreversible in3 compression.
        """
        ...


In1RelaxFn = Callable[[ProfiledState], ProfiledState]
CapacityDiagnostic = Callable[
    [ProfiledState, Sequence[In2Move], random.Random],
    BirthProposal | None,
]
BirthSeedBuilder = Callable[[ProfiledState, BirthProposal], ProfiledState]


def _fingerprint(state: ProfiledState) -> frozenset[Coordinate]:
    return frozenset(state.support)


def _assert_replacement(
    before: ProfiledState,
    after: ProfiledState,
    proposal: RepairProposal,
) -> None:
    s0 = frozenset(before.support)
    s1 = frozenset(after.support)
    if len(s1) != len(s0):
        raise RuntimeError("in2 replacement must preserve E.")
    if proposal.removed not in s0:
        raise RuntimeError("Replacement tried to remove an inactive coordinate.")
    if proposal.added in s0:
        raise RuntimeError("Replacement tried to add an already-active coordinate.")
    expected = (s0 - {proposal.removed}) | {proposal.added}
    if s1 != expected:
        raise RuntimeError("Replacement seed does not match destroy-repair proposal.")


def _assert_birth(
    before: ProfiledState,
    after: ProfiledState,
    proposal: BirthProposal,
) -> None:
    s0 = frozenset(before.support)
    s1 = frozenset(after.support)
    if proposal.added in s0:
        raise RuntimeError("Birth candidate is already active.")
    if s1 != s0 | {proposal.added}:
        raise RuntimeError("Birth must add exactly one structural coordinate.")


def _representative_states(
    current: ProfiledState,
    archive: Mapping[frozenset[Coordinate], ProfiledState],
    visit_order: Sequence[frozenset[Coordinate]],
    *,
    max_representatives: int,
) -> tuple[ProfiledState, ...]:
    """Return diverse feasible states without ranking them by residual/MDL."""
    if max_representatives < 1:
        raise ValueError("max_representatives must be >= 1.")
    chosen: list[ProfiledState] = [current]
    seen = {frozenset(current.support)}
    if len(chosen) >= max_representatives:
        return tuple(chosen)
    for fp in reversed(tuple(visit_order)):
        if fp in seen or fp not in archive:
            continue
        if len(chosen) >= max_representatives:
            break
        chosen.append(archive[fp])
        seen.add(fp)
    return tuple(chosen)



def _normalise_coverage_assessment(
    raw: bool | CoverageAssessment,
) -> CoverageAssessment:
    if isinstance(raw, CoverageAssessment):
        return raw
    if isinstance(raw, bool):
        return CoverageAssessment(saturated=raw)
    raise TypeError(
        "coverage_saturated() must return bool or CoverageAssessment, "
        f"got {type(raw).__name__}."
    )


def _coverage_check(
    *,
    backend: StructuralExplorationBackend,
    state: ProfiledState,
    history: Sequence[In2Move],
    visited: Sequence[frozenset[Coordinate]],
    sweep: int,
    trigger: str,
    log: list[CoverageCheck],
) -> CoverageAssessment:
    assessment = _normalise_coverage_assessment(
        backend.coverage_saturated(state, tuple(history), tuple(visited))
    )
    log.append(
        CoverageCheck(
            sweep=int(sweep),
            trigger=str(trigger),
            support=frozenset(state.support),
            assessment=assessment,
        )
    )
    return assessment


def run_structural_support_exploration(
    initial: ProfiledState,
    backend: StructuralExplorationBackend,
    *,
    relax_in1: In1RelaxFn,
    config: In2Config = In2Config(),
    capacity_diagnostic: CapacityDiagnostic | None = None,
    build_birth_seed: BirthSeedBuilder | None = None,
) -> In2Result:
    """Run reversible structural exploration at fixed J.

    The routine never accepts E -> E-1. Ordinary accepted moves are provisional
    replacements. Optional E -> E+1 birth is available only through the
    explicit capacity-diagnostic hook.

    Coverage saturation is checked independently of temporary search stalling:
    periodically during reversible exploration, immediately when stalled, and
    once more at budget exhaustion. Only an explicit backend certificate can
    return ``coverage_saturated`` and authorize the in2 -> in3 handoff.
    """

    if config.max_sweeps < 1:
        raise ValueError("max_sweeps must be >= 1.")
    if config.no_move_patience < 1:
        raise ValueError("no_move_patience must be >= 1.")
    if config.recurrence_trigger < 2:
        raise ValueError("recurrence_trigger must be >= 2.")
    if config.max_representatives < 1:
        raise ValueError("max_representatives must be >= 1.")
    if config.coverage_check_interval < 1:
        raise ValueError("coverage_check_interval must be >= 1.")
    if config.allow_birth and (capacity_diagnostic is None or build_birth_seed is None):
        raise ValueError(
            "Birth is enabled, but capacity_diagnostic/build_birth_seed was not supplied."
        )

    rng = random.Random(config.random_seed)
    current = initial
    history: list[In2Move] = []
    visited: list[frozenset[Coordinate]] = [_fingerprint(current)]
    visit_count: Counter[frozenset[Coordinate]] = Counter(visited)
    state_archive: dict[frozenset[Coordinate], ProfiledState] = {_fingerprint(current): current}
    failures: list[In2CandidateFailure] = []
    coverage_checks: list[CoverageCheck] = []
    no_move = 0

    for _sweep in range(config.max_sweeps):
        raw = list(backend.replacement_proposals(current, rng=rng))
        evaluated: list[tuple[RepairProposal, ProfiledState]] = []

        for proposal in raw:
            seed = backend.build_replacement_seed(current, proposal)
            _assert_replacement(current, seed, proposal)
            try:
                relaxed = relax_in1(seed)
            except In1InfeasibleError as exc:
                failures.append(In2CandidateFailure(
                    kind="replacement_infeasible",
                    support=frozenset(seed.support),
                    metadata={"removed": proposal.removed, "added": proposal.added, "reason": str(exc)},
                ))
                continue
            _assert_replacement(current, relaxed, proposal)
            evaluated.append((proposal, relaxed))

        idx = backend.choose_replacement(current, evaluated, history, rng=rng)
        if idx is not None:
            if idx < 0 or idx >= len(evaluated):
                raise IndexError("choose_replacement returned an invalid index.")
            proposal, accepted = evaluated[idx]
            before = frozenset(current.support)
            after = frozenset(accepted.support)
            history.append(
                In2Move(
                    kind="replacement",
                    before_support=before,
                    after_support=after,
                    metadata={
                        "removed": proposal.removed,
                        "added": proposal.added,
                        "proposal_score": proposal.score,
                        **dict(proposal.metadata),
                    },
                )
            )
            current = accepted
            visited.append(after)
            visit_count[after] += 1
            state_archive[after] = accepted
            no_move = 0
        else:
            no_move += 1

        # Optional extension: recurrence only triggers the expensive capacity
        # diagnostic. It is not itself evidence for birth.
        if config.allow_birth:
            fp = _fingerprint(current)
            if visit_count[fp] >= config.recurrence_trigger:
                birth = capacity_diagnostic(current, history, rng)  # type: ignore[misc]
                if birth is not None:
                    seed = build_birth_seed(current, birth)  # type: ignore[misc]
                    _assert_birth(current, seed, birth)
                    try:
                        accepted = relax_in1(seed)
                    except In1InfeasibleError as exc:
                        failures.append(In2CandidateFailure(
                            kind="birth_infeasible",
                            support=frozenset(seed.support),
                            metadata={"added": birth.added, "reason": str(exc)},
                        ))
                        accepted = None

                    if accepted is not None:
                        _assert_birth(current, accepted, birth)
                        before = frozenset(current.support)
                        after = frozenset(accepted.support)
                        history.append(
                            In2Move(
                                kind="birth",
                                before_support=before,
                                after_support=after,
                                metadata={"added": birth.added, **dict(birth.metadata)},
                            )
                        )
                        current = accepted
                        visited.append(after)
                        visit_count[after] += 1
                        state_archive[after] = accepted
                        no_move = 0

        sweep_number = _sweep + 1

        # Coverage is a scientific phase-boundary diagnostic, independent of
        # whether the reversible Markov/search dynamics are still moving.
        if sweep_number % config.coverage_check_interval == 0:
            assessment = _coverage_check(
                backend=backend,
                state=current,
                history=history,
                visited=visited,
                sweep=sweep_number,
                trigger="periodic",
                log=coverage_checks,
            )
            if assessment.saturated:
                return In2Result(
                    state=current,
                    history=tuple(history),
                    visited_supports=tuple(visited),
                    representative_states=_representative_states(
                        current, state_archive, visited,
                        max_representatives=config.max_representatives,
                    ),
                    candidate_failures=tuple(failures),
                    coverage_status="coverage_saturated",
                    coverage_checks=tuple(coverage_checks),
                )

        if no_move >= config.no_move_patience:
            # Stalling triggers an immediate assessment but does not itself
            # certify coverage. Avoid duplicating a periodic check on this sweep.
            if not coverage_checks or coverage_checks[-1].sweep != sweep_number:
                assessment = _coverage_check(
                    backend=backend,
                    state=current,
                    history=history,
                    visited=visited,
                    sweep=sweep_number,
                    trigger="stall",
                    log=coverage_checks,
                )
            else:
                assessment = coverage_checks[-1].assessment
            return In2Result(
                state=current,
                history=tuple(history),
                visited_supports=tuple(visited),
                representative_states=_representative_states(
                    current, state_archive, visited,
                    max_representatives=config.max_representatives,
                ),
                candidate_failures=tuple(failures),
                coverage_status=(
                    "coverage_saturated"
                    if assessment.saturated
                    else "unresolved_stall"
                ),
                coverage_checks=tuple(coverage_checks),
            )

    # Budget exhaustion is not equivalent to incomplete coverage: perform one
    # final explicit assessment before returning an unresolved result.
    final_assessment = _coverage_check(
        backend=backend,
        state=current,
        history=history,
        visited=visited,
        sweep=config.max_sweeps,
        trigger="budget_end",
        log=coverage_checks,
    )
    return In2Result(
        state=current,
        history=tuple(history),
        visited_supports=tuple(visited),
        representative_states=_representative_states(
            current, state_archive, visited, max_representatives=config.max_representatives
        ),
        candidate_failures=tuple(failures),
        coverage_status=(
            "coverage_saturated"
            if final_assessment.saturated
            else "budget_exhausted"
        ),
        coverage_checks=tuple(coverage_checks),
    )


run_in2 = run_structural_support_exploration

__all__ = [
    "Coordinate",
    "RepairProposal",
    "BirthProposal",
    "In2Config",
    "In2Move",
    "In2CandidateFailure",
    "CoverageAssessment",
    "CoverageCheck",
    "In2Result",
    "StructuralExplorationBackend",
    "In1RelaxFn",
    "CapacityDiagnostic",
    "BirthSeedBuilder",
    "run_structural_support_exploration",
    "run_in2",
]
