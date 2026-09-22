"""
TIDES Step 3 inner stage 3: basin-retention structural compression.

Scientific role
---------------
Start from a deliberately broad support and perform monotone delete-only
compression,

    S_{t+1} subset S_t,   |S_{t+1}| = |S_t| - 1.

The basin diagnostic is deliberately CATEGORY-RELAXED: for each fixed dynamics
coordinate Theta, all active structural amplitudes W on the candidate support
are profiled independently.  Thus the measured feasible dynamics basin depends
on the topology/support, not on a possibly stale weight-category partition c.
This is the geometry used by the validated J2 basin-width calculation.

The category assignment carried in ``ProfiledState`` is only inherited as a
warm-start/bookkeeping object while compression is running.  It is NOT used by
the basin backend.  After compression terminates, the Step-3 outer orchestrator
re-runs in1 on the terminal support from a released (independent-weight)
category seed.  Consequently category locking cannot masquerade as structural
necessity during in3.

No repair and no birth are permitted in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Hashable, Mapping, Protocol, Sequence, Literal
import math

try:
    from .step3in1_weight_category_distribution import ProfiledState
except ImportError:
    from step3in1_weight_category_distribution import ProfiledState

Coordinate = Hashable


class BasinNestingError(RuntimeError):
    """A free-W child basin exceeded its parent beyond numerical tolerance."""


class BasinBackendContractError(RuntimeError):
    """The basin backend returned an invalid or internally inconsistent audit."""


class BasinNumericalError(RuntimeError):
    """The basin estimator returned a non-finite numerical result."""


BasinMassStatus = Literal["ok", "zero_detected", "inconclusive"]


@dataclass(frozen=True)
class BasinMassEstimate:
    """One basin-mass estimate with explicit estimator semantics.

    ``zero_detected`` means that no positive mass was detected above the
    configured numerical/sampling floor.  It is deliberately NOT a proof that
    the mathematical basin is empty.

    ``inconclusive`` means that the estimator did not produce a usable mass
    estimate (for example insufficient effective sample size).  In3 must stop
    that lineage as unresolved rather than interpreting it as a basin cliff.

    ``sample_id`` identifies the common Theta sample cloud used by stochastic
    parent/child estimates.  Stochastic batch backends should provide it so the
    free-W nesting comparison is performed on the same sample set.
    """

    mass: float | None
    status: BasinMassStatus = "ok"
    standard_error: float | None = None
    effective_sample_size: float | None = None
    sample_id: Hashable | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DeletionDiagnostic:
    coordinate: Coordinate
    parent_mass: float
    child_mass: float
    retention: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BasinDeletionAudit:
    """One parent-basin measurement plus all single-deletion child masses.

    A batch interface is important computationally: in J2 one alpha scan can
    evaluate every single deletion, and in higher-dimensional implementations
    one parent feasible-particle cloud can be reused for all deletions.

    Deterministic/exact backends may populate only ``parent_mass`` and the
    scalar masses in ``deletions``.  Stochastic backends should additionally
    provide ``parent_estimate`` and ``child_estimates`` with one shared
    ``sample_id`` so parent and all children are evaluated on the same Theta
    particles.
    """

    parent_mass: float
    deletions: tuple[DeletionDiagnostic, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    parent_estimate: BasinMassEstimate | None = None
    child_estimates: Mapping[Coordinate, BasinMassEstimate] = field(default_factory=dict)


@dataclass(frozen=True)
class CompressionStep:
    before_support: frozenset[Coordinate]
    after_support: frozenset[Coordinate]
    deleted: Coordinate
    retention: float
    parent_mass: float
    child_mass: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompressionLineage:
    state: ProfiledState
    history: tuple[CompressionStep, ...] = ()
    checkpoints: tuple[ProfiledState, ...] = ()
    status: Literal["active", "completed", "unresolved"] = "active"
    stopped_reason: str | None = None
    stop_metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class In3Config:
    max_deletions: int | None = None
    mass_floor: float = 0.0
    retention_tolerance: float = 1e-8


@dataclass(frozen=True)
class In3Result:
    lineages: tuple[CompressionLineage, ...]


class BasinBackend(Protocol):
    """Category-relaxed dynamics-basin backend.

    ``basin_mass`` must profile all active W independently at each fixed Theta.
    It must not impose the current ``state.categories`` sharing relations.

    Exact/deterministic implementations may return a float.  Estimators that
    can be inconclusive should return ``BasinMassEstimate`` so zero detection,
    insufficient sampling, and numerical failure are not conflated.
    """

    def basin_mass(self, state: ProfiledState) -> float | BasinMassEstimate:
        ...


class BatchDeletionBasinBackend(Protocol):
    """Efficient one-parent/all-deletion evaluation.

    For stochastic/Monte-Carlo estimators this is the preferred interface.
    Parent and every single-deletion child must be evaluated in the same Theta
    coordinate system, under the same prior/measure, epsilon and residual
    definition.  They should reuse one common Theta sample cloud and expose a
    shared ``sample_id`` through ``BasinMassEstimate``.
    """

    def audit_single_deletions(self, state: ProfiledState) -> BasinDeletionAudit:
        ...


StopRule = callable
BranchSelector = callable


def inherited_deletion_seed(
    state: ProfiledState,
    coordinate: Coordinate,
) -> ProfiledState:
    """Delete exactly one coordinate; continuous fields become stale warm starts."""

    if coordinate not in state.support:
        raise KeyError(f"Coordinate {coordinate!r} is not active.")

    support = frozenset(set(state.support) - {coordinate})
    categories = {
        g: label for g, label in state.categories.items() if g != coordinate
    }
    if frozenset(categories) != support:
        raise RuntimeError("Inherited category assignment no longer matches support.")

    return ProfiledState(
        support=support,
        categories=categories,
        description_length=state.description_length,
        relative_residual=state.relative_residual,
        weights=state.weights,
        theta=state.theta,
        metadata={
            **dict(state.metadata),
            "deleted_coordinate": coordinate,
            "continuous_state_stale_after_in3_deletion": True,
        },
    )


def best_retention_selector(
    lineage: CompressionLineage,
    diagnostics: Sequence[DeletionDiagnostic],
) -> Sequence[Coordinate]:
    """Reference local navigation: continue along the largest retention."""

    if not diagnostics:
        return ()
    return (max(diagnostics, key=lambda d: d.retention).coordinate,)


def no_automatic_stop(
    lineage: CompressionLineage,
    diagnostics: Sequence[DeletionDiagnostic],
) -> bool:
    """Compatibility rule that never declares a basin cliff on its own.

    Empty diagnostics are handled before the stop rule and receive the explicit
    reason ``no_positive_mass_deletion``.  Production Step 3 still requires an
    explicit adaptive stop rule.
    """

    return False


@dataclass(frozen=True)
class BasinDiagnosticAssessment:
    """Validated parent/child basin audit for one compression state."""

    status: Literal["ok", "parent_zero_detected", "inconclusive"]
    diagnostics: tuple[DeletionDiagnostic, ...] = ()
    zero_mass_coordinates: tuple[Coordinate, ...] = ()
    parent_estimate: BasinMassEstimate | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _coerce_mass_estimate(
    value: float | BasinMassEstimate,
    *,
    label: str,
    mass_floor: float,
) -> BasinMassEstimate:
    """Normalize one backend mass result without conflating failure modes."""

    if isinstance(value, BasinMassEstimate):
        est = value
        if est.status not in {"ok", "zero_detected", "inconclusive"}:
            raise BasinBackendContractError(
                f"{label} returned unknown BasinMassEstimate.status={est.status!r}."
            )
        if est.status == "inconclusive":
            return est
        if est.mass is None:
            raise BasinBackendContractError(
                f"{label} status={est.status!r} requires a numeric mass."
            )
        mass = float(est.mass)
    else:
        mass = float(value)
        est = BasinMassEstimate(mass=mass, status="ok")

    if math.isnan(mass) or math.isinf(mass):
        raise BasinNumericalError(f"{label} returned non-finite basin mass {mass!r}.")
    if mass < 0.0:
        raise BasinBackendContractError(
            f"{label} returned negative basin mass {mass:.16g}."
        )

    # A finite value at/below the configured detection floor is a zero-mass
    # *estimate*, not a proof of mathematical emptiness.
    if mass <= mass_floor:
        return replace(est, mass=mass, status="zero_detected")
    if est.status == "zero_detected":
        raise BasinBackendContractError(
            f"{label} reports status='zero_detected' but mass={mass:.16g} "
            f"exceeds mass_floor={mass_floor:.16g}."
        )
    return replace(est, mass=mass, status="ok")


def _validated_retention(
    parent: float,
    child: float,
    *,
    tolerance: float,
) -> tuple[float, float]:
    raw = child / parent
    if raw > 1.0 + tolerance:
        raise BasinNestingError(
            "Child free-W basin exceeds parent basin beyond tolerance: "
            f"parent={parent:.16g}, child={child:.16g}, raw_R={raw:.16g}. "
            "For stochastic estimators, parent and children must be evaluated "
            "on one common Theta sample cloud."
        )
    if raw < -tolerance:
        raise BasinNestingError(f"Negative basin retention {raw:.16g}.")
    return min(max(raw, 0.0), 1.0), raw


def _validate_common_sample_ids(
    parent: BasinMassEstimate,
    children: Mapping[Coordinate, BasinMassEstimate],
) -> None:
    """Enforce common random numbers when stochastic IDs are provided."""

    ids = [parent.sample_id] + [est.sample_id for est in children.values()]
    provided = [x for x in ids if x is not None]
    if not provided:
        return  # deterministic/exact or legacy backend
    if len(provided) != len(ids):
        raise BasinBackendContractError(
            "A stochastic batch audit supplied sample_id for only part of the "
            "parent/child estimates. Provide one common sample_id for all."
        )
    if len(set(provided)) != 1:
        raise BasinBackendContractError(
            "Parent and child basin estimates use different Theta sample clouds. "
            "A stochastic batch audit must reuse one common sample_id."
        )


def _fallback_diagnostics(
    state: ProfiledState,
    backend: BasinBackend,
    *,
    mass_floor: float,
    retention_tolerance: float,
) -> BasinDiagnosticAssessment:
    parent_est = _coerce_mass_estimate(
        backend.basin_mass(state), label="parent basin", mass_floor=mass_floor
    )
    if parent_est.status == "inconclusive":
        return BasinDiagnosticAssessment(
            status="inconclusive",
            parent_estimate=parent_est,
            metadata={"reason": "parent_mass_estimation_inconclusive"},
        )
    if parent_est.status == "zero_detected":
        return BasinDiagnosticAssessment(
            status="parent_zero_detected",
            parent_estimate=parent_est,
            metadata={"reason": "no_positive_parent_mass_detected"},
        )

    parent = float(parent_est.mass)
    out: list[DeletionDiagnostic] = []
    zeros: list[Coordinate] = []
    child_estimates: dict[Coordinate, BasinMassEstimate] = {}

    for g in state.support:
        child_seed = inherited_deletion_seed(state, g)
        child_est = _coerce_mass_estimate(
            backend.basin_mass(child_seed),
            label=f"child basin after deleting {g!r}",
            mass_floor=mass_floor,
        )
        child_estimates[g] = child_est
        if child_est.status == "inconclusive":
            return BasinDiagnosticAssessment(
                status="inconclusive",
                diagnostics=tuple(out),
                zero_mass_coordinates=tuple(zeros),
                parent_estimate=parent_est,
                metadata={
                    "reason": "child_mass_estimation_inconclusive",
                    "coordinate": g,
                },
            )
        if child_est.status == "zero_detected":
            zeros.append(g)
            continue

        child = float(child_est.mass)
        r, raw_r = _validated_retention(
            parent, child, tolerance=retention_tolerance
        )
        out.append(
            DeletionDiagnostic(
                coordinate=g,
                parent_mass=parent,
                child_mass=child,
                retention=r,
                metadata={"raw_retention": raw_r},
            )
        )

    # If this legacy/fallback route is used stochastically and explicit sample
    # IDs are supplied, still enforce common random numbers.
    _validate_common_sample_ids(parent_est, child_estimates)
    out.sort(key=lambda d: d.retention, reverse=True)
    return BasinDiagnosticAssessment(
        status="ok",
        diagnostics=tuple(out),
        zero_mass_coordinates=tuple(zeros),
        parent_estimate=parent_est,
    )


def _diagnostics(
    state: ProfiledState,
    backend: BasinBackend,
    *,
    mass_floor: float,
    retention_tolerance: float,
) -> BasinDiagnosticAssessment:
    # Prefer the batch audit whenever a backend provides it.
    batch = getattr(backend, "audit_single_deletions", None)
    if not callable(batch):
        return _fallback_diagnostics(
            state,
            backend,
            mass_floor=mass_floor,
            retention_tolerance=retention_tolerance,
        )

    audit = batch(state)
    support = frozenset(state.support)
    coordinates = [d.coordinate for d in audit.deletions]
    coord_set = set(coordinates)

    if len(coord_set) != len(coordinates):
        duplicates = sorted(
            {g for g in coordinates if coordinates.count(g) > 1},
            key=repr,
        )
        raise BasinBackendContractError(
            f"Batch deletion audit contains duplicate coordinates: {duplicates!r}."
        )
    extra = coord_set - set(support)
    missing = set(support) - coord_set
    if extra or missing:
        raise BasinBackendContractError(
            "Batch deletion audit must contain exactly one record for every "
            f"active coordinate. missing={missing!r}, extra={extra!r}."
        )

    parent_source: float | BasinMassEstimate = (
        audit.parent_estimate if audit.parent_estimate is not None else audit.parent_mass
    )
    parent_est = _coerce_mass_estimate(
        parent_source, label="batch parent basin", mass_floor=mass_floor
    )
    if parent_est.status == "inconclusive":
        return BasinDiagnosticAssessment(
            status="inconclusive",
            parent_estimate=parent_est,
            metadata={"reason": "parent_mass_estimation_inconclusive", **dict(audit.metadata)},
        )
    if parent_est.status == "zero_detected":
        return BasinDiagnosticAssessment(
            status="parent_zero_detected",
            parent_estimate=parent_est,
            metadata={"reason": "no_positive_parent_mass_detected", **dict(audit.metadata)},
        )

    parent = float(parent_est.mass)
    if math.isfinite(float(audit.parent_mass)) and abs(float(audit.parent_mass) - parent) > retention_tolerance * max(1.0, abs(parent)):
        raise BasinBackendContractError(
            "BasinDeletionAudit.parent_mass disagrees with parent_estimate.mass."
        )

    # If rich child estimates are supplied, they must cover exactly the same
    # deletion coordinates as the scalar audit records.
    if audit.child_estimates:
        child_keys = set(audit.child_estimates)
        if child_keys != set(support):
            raise BasinBackendContractError(
                "child_estimates must contain exactly one estimate for every "
                f"active coordinate. missing={set(support)-child_keys!r}, "
                f"extra={child_keys-set(support)!r}."
            )

    child_estimates: dict[Coordinate, BasinMassEstimate] = {}
    for d in audit.deletions:
        source: float | BasinMassEstimate = audit.child_estimates.get(d.coordinate, d.child_mass)
        child_estimates[d.coordinate] = _coerce_mass_estimate(
            source,
            label=f"batch child basin after deleting {d.coordinate!r}",
            mass_floor=mass_floor,
        )

    _validate_common_sample_ids(parent_est, child_estimates)

    out: list[DeletionDiagnostic] = []
    zeros: list[Coordinate] = []
    for d in audit.deletions:
        child_est = child_estimates[d.coordinate]
        if child_est.status == "inconclusive":
            return BasinDiagnosticAssessment(
                status="inconclusive",
                diagnostics=tuple(out),
                zero_mass_coordinates=tuple(zeros),
                parent_estimate=parent_est,
                metadata={
                    "reason": "child_mass_estimation_inconclusive",
                    "coordinate": d.coordinate,
                    **dict(audit.metadata),
                },
            )
        if child_est.status == "zero_detected":
            zeros.append(d.coordinate)
            continue

        child = float(child_est.mass)
        scalar_child = float(d.child_mass)
        if math.isfinite(scalar_child) and abs(scalar_child - child) > retention_tolerance * max(1.0, abs(child)):
            raise BasinBackendContractError(
                f"Scalar child_mass for {d.coordinate!r} disagrees with its rich estimate."
            )

        retention, raw = _validated_retention(
            parent, child, tolerance=retention_tolerance
        )

        # The batch backend may provide retention for convenience, but the
        # authoritative value is recomputed from the validated parent/child
        # masses.  Large disagreement signals an inconsistent audit.
        supplied_r = float(d.retention)
        if not math.isfinite(supplied_r):
            raise BasinNumericalError(
                f"Batch retention for {d.coordinate!r} is non-finite."
            )
        if abs(supplied_r - retention) > retention_tolerance:
            raise BasinBackendContractError(
                f"Batch retention for {d.coordinate!r} is inconsistent with "
                f"parent/child masses: supplied={supplied_r:.16g}, "
                f"recomputed={retention:.16g}."
            )

        out.append(
            DeletionDiagnostic(
                coordinate=d.coordinate,
                parent_mass=parent,
                child_mass=child,
                retention=retention,
                metadata={**dict(d.metadata), "raw_retention": raw},
            )
        )

    out.sort(key=lambda d: d.retention, reverse=True)
    return BasinDiagnosticAssessment(
        status="ok",
        diagnostics=tuple(out),
        zero_mass_coordinates=tuple(zeros),
        parent_estimate=parent_est,
        metadata=dict(audit.metadata),
    )


def run_basin_preserving_compression(
    initial: ProfiledState,
    backend: BasinBackend,
    *,
    stop_rule=None,
    branch_selector=best_retention_selector,
    config: In3Config = In3Config(),
) -> In3Result:
    """Run category-relaxed, delete-only basin-retention compression.

    in1 is intentionally NOT called between deletions: basin navigation is
    defined in the free-W/topology geometry and therefore does not depend on c.
    Step 3 re-runs in1 only after an in3 lineage terminates.
    """

    if config.max_deletions is not None and config.max_deletions < 0:
        raise ValueError("max_deletions must be nonnegative or None.")
    if config.mass_floor < 0:
        raise ValueError("mass_floor must be nonnegative.")
    if config.retention_tolerance < 0:
        raise ValueError("retention_tolerance must be nonnegative.")
    if stop_rule is None:
        raise ValueError(
            "in3 requires an explicit adaptive stop_rule; continuing until no "
            "positive-mass deletion is not a safe production default."
        )

    active: list[CompressionLineage] = [CompressionLineage(state=initial, checkpoints=(initial,))]
    terminal: list[CompressionLineage] = []

    while active:
        lineage = active.pop(0)
        state = lineage.state

        if config.max_deletions is not None and len(lineage.history) >= config.max_deletions:
            terminal.append(replace(
                lineage, status="unresolved", stopped_reason="max_deletions_reached"
            ))
            continue
        if len(state.support) == 0:
            terminal.append(replace(
                lineage, status="completed", stopped_reason="empty_support"
            ))
            continue

        assessment = _diagnostics(
            state, backend, mass_floor=config.mass_floor,
            retention_tolerance=config.retention_tolerance,
        )

        if assessment.status == "inconclusive":
            terminal.append(replace(
                lineage,
                status="unresolved",
                stopped_reason="basin_estimation_inconclusive",
                stop_metadata={
                    **dict(assessment.metadata),
                    "zero_mass_coordinates": assessment.zero_mass_coordinates,
                },
            ))
            continue
        if assessment.status == "parent_zero_detected":
            terminal.append(replace(
                lineage,
                status="unresolved",
                stopped_reason="parent_mass_zero_detected",
                stop_metadata=dict(assessment.metadata),
            ))
            continue

        diagnostics = assessment.diagnostics
        if not diagnostics:
            terminal.append(replace(
                lineage,
                status="completed",
                stopped_reason="no_positive_mass_deletion",
                stop_metadata={
                    **dict(assessment.metadata),
                    "zero_mass_coordinates": assessment.zero_mass_coordinates,
                },
            ))
            continue

        if stop_rule(lineage, diagnostics):
            terminal.append(replace(
                lineage,
                status="completed",
                stopped_reason="basin_cliff",
                stop_metadata={
                    **dict(assessment.metadata),
                    "zero_mass_coordinates": assessment.zero_mass_coordinates,
                },
            ))
            continue

        chosen = tuple(branch_selector(lineage, diagnostics))
        if not chosen:
            terminal.append(replace(lineage, status="unresolved", stopped_reason="selector_returned_no_branch"))
            continue

        diag_by_g = {d.coordinate: d for d in diagnostics}
        for g in chosen:
            if g not in diag_by_g:
                raise KeyError(
                    f"Branch selector chose {g!r}, which has no deletion diagnostic."
                )
            d = diag_by_g[g]
            child = inherited_deletion_seed(state, g)
            before = frozenset(state.support)
            after = frozenset(child.support)
            if after != before - {g}:
                raise RuntimeError("in3 must perform exactly one deletion per step.")

            step = CompressionStep(
                before_support=before,
                after_support=after,
                deleted=g,
                retention=d.retention,
                parent_mass=d.parent_mass,
                child_mass=d.child_mass,
                metadata=dict(d.metadata),
            )
            active.append(
                CompressionLineage(
                    state=child,
                    history=lineage.history + (step,),
                    checkpoints=lineage.checkpoints + (child,),
                    status="active",
                )
            )

    return In3Result(lineages=tuple(terminal))


run_in3 = run_basin_preserving_compression

# Compatibility alias for the previous draft API name.
ConditionalBasinBackend = BasinBackend

__all__ = [
    "Coordinate",
    "BasinNestingError",
    "BasinBackendContractError",
    "BasinNumericalError",
    "BasinMassStatus",
    "BasinMassEstimate",
    "DeletionDiagnostic",
    "BasinDeletionAudit",
    "BasinDiagnosticAssessment",
    "CompressionStep",
    "CompressionLineage",
    "In3Config",
    "In3Result",
    "BasinBackend",
    "BatchDeletionBasinBackend",
    "ConditionalBasinBackend",
    "inherited_deletion_seed",
    "best_retention_selector",
    "no_automatic_stop",
    "run_basin_preserving_compression",
    "run_in3",
]
