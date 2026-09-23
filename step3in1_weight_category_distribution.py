"""
TIDES Step 3 inner stage 1: weight-category distribution inference.

Frozen role
-----------
For fixed dynamics support J and fixed structural support S, infer a compact
weight-category representation c while re-profiling the continuous variables
(W, Theta). This module DOES NOT add, delete, or replace structural coordinates.

The search is representation-level coarse graining:
    fine active weights <-> category-shared active weights

The category partition remains reversible. When S changes later, the caller
must allow reassignment / split / merge-split so that an old partition cannot
lock a new topology.

The shared continuous calculations are delegated to
``step3func_physical_profiling``.  Description-length definitions and category
proposal rules remain separate from the physical profiler.  ``CategoryBackend``
is retained as an interface for custom/validated implementations, while
``PhysicalCategoryComputation`` is the first-party implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Iterable, Mapping, Protocol
import math
import random

try:
    from .step3func_physical_profiling import (
        FixedDynamicsProblem,
        CategoryProfile,
        profile_category_weights_and_dynamics,
    )
except ImportError:
    from step3func_physical_profiling import (
        FixedDynamicsProblem,
        CategoryProfile,
        profile_category_weights_and_dynamics,
    )

Coordinate = Hashable
CategoryLabel = Hashable


class In1InfeasibleError(RuntimeError):
    """The supplied fixed-(J,S,c) representation has no feasible profiled witness.

    This is deliberately distinct from ValueError/configuration errors so a
    failed local candidate cannot be promoted to a scientific claim that the
    entire dynamics support J is infeasible.
    """


@dataclass(frozen=True)
class ProfiledState:
    """Fully profiled representation at fixed (J,S,c)."""

    support: frozenset[Coordinate]
    categories: Mapping[Coordinate, CategoryLabel]
    description_length: float
    relative_residual: float
    weights: Any = None
    theta: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CategoryMove:
    kind: str
    categories: Mapping[Coordinate, CategoryLabel]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CategoryProposalFailure:
    """Expected failure of one local category proposal.

    These failures are part of the search record, not stage-level control-flow
    failures. In particular, an infeasible merge/split/reassignment does not
    invalidate the current feasible fixed-(J,S) representation.
    """

    sweep: int
    move_kind: str
    categories: Mapping[Coordinate, CategoryLabel]
    failure_kind: str
    reason: str
    relative_residual: float | None = None
    description_length: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class In1Config:
    uncertainty_floor: float
    move_order: tuple[str, ...] = (
        "reassign",
        "merge",
        "split",
        "merge_split",
    )
    max_sweeps: int = 50
    mdl_improvement_tol: float = 1e-12
    random_seed: int | None = None


@dataclass(frozen=True)
class In1Result:
    state: ProfiledState
    accepted_moves: tuple[CategoryMove, ...]
    n_sweeps: int
    proposal_failures: tuple[CategoryProposalFailure, ...] = ()


class CategoryBackend(Protocol):
    """Numerical backend for the existing category-relaxation solver.

    Bind the chosen dynamics support J into the backend/closure before calling
    run_weight_category_distribution.

    profile() must optimize the continuous category values / W and Theta for
    the supplied fixed support and category assignment.

    propose() should expose the existing proposal machinery for the requested
    move kind. It may use cheap fixed-Theta screening internally and return only
    serious candidates for expensive joint profiling.
    """

    def profile(
        self,
        support: frozenset[Coordinate],
        categories: Mapping[Coordinate, CategoryLabel],
        *,
        warm_start: ProfiledState | None = None,
    ) -> ProfiledState:
        ...

    def propose(
        self,
        state: ProfiledState,
        move_kind: str,
        *,
        rng: random.Random,
    ) -> Iterable[CategoryMove]:
        ...


# ---------------------------------------------------------------------------
# First-party physical-profiling implementation
# ---------------------------------------------------------------------------


DescriptionLengthScorer = Callable[
    [frozenset[Coordinate], Mapping[Coordinate, CategoryLabel], CategoryProfile],
    float,
]
CategoryProposalFn = Callable[
    [ProfiledState, str, random.Random],
    Iterable[CategoryMove],
]


class PhysicalCategoryComputation:
    """Category-stage computation using ``step3func_physical_profiling``.

    The class deliberately keeps the two conceptually different pieces
    separate:

    * physical feasibility / (W, Theta) fitting is performed here by the shared
      Step-3 profiling module;
    * description length and category proposals remain injected model/search
      definitions.

    Consequently in1 contains no duplicate VarPro, SVD, or least-squares code.
    """

    def __init__(
        self,
        problem: FixedDynamicsProblem,
        *,
        active_atoms: Iterable[int],
        description_length_scorer: DescriptionLengthScorer,
        proposal_fn: CategoryProposalFn | None = None,
        max_nfev: int = 250,
        compute_linear_relaxation: bool = False,
    ) -> None:
        self.problem = problem
        self.active_atoms = tuple(sorted(set(int(a) for a in active_atoms)))
        if not self.active_atoms:
            raise ValueError("active_atoms must be non-empty.")
        self.description_length_scorer = description_length_scorer
        self.proposal_fn = proposal_fn
        self.max_nfev = int(max_nfev)
        if self.max_nfev < 1:
            raise ValueError("max_nfev must be >= 1.")
        self.compute_linear_relaxation = bool(compute_linear_relaxation)

    def profile(
        self,
        support: frozenset[Coordinate],
        categories: Mapping[Coordinate, CategoryLabel],
        *,
        warm_start: ProfiledState | None = None,
    ) -> ProfiledState:
        warm_theta = None if warm_start is None else warm_start.theta
        profiled = profile_category_weights_and_dynamics(
            self.problem,
            support,
            categories,
            active_atoms=self.active_atoms,
            warm_theta=warm_theta,
            max_nfev=self.max_nfev,
            compute_linear_relaxation=self.compute_linear_relaxation,
            include_default_start=True,
        )

        fit = profiled.fit
        if not fit.feasible_witness:
            raise In1InfeasibleError(
                "Fixed-(J,S,c) physical profile lies outside B_epsilon: "
                f"rho={fit.relative_residual:.6e}, "
                f"eps={fit.uncertainty_floor:.6e}."
            )

        dl = float(
            self.description_length_scorer(
                frozenset(profiled.support),
                dict(profiled.categories),
                profiled,
            )
        )
        if not math.isfinite(dl):
            raise ValueError(
                "description_length_scorer must return a finite value for a "
                "feasible physical profile."
            )

        return ProfiledState(
            support=frozenset(profiled.support),
            categories=dict(profiled.categories),
            description_length=dl,
            relative_residual=float(fit.relative_residual),
            weights=dict(profiled.weights),
            theta=fit.theta.copy(),
            metadata={
                **({} if warm_start is None else dict(warm_start.metadata)),
                "physical_profiling": True,
                "n_categories": len(profiled.category_order),
                "category_values": dict(profiled.category_values),
                "optimizer_nfev": int(fit.optimizer_nfev),
                "optimizer_success": bool(fit.optimizer_success),
                "relaxed_relative_residual": float(fit.relaxed_relative_residual),
            },
        )

    def propose(
        self,
        state: ProfiledState,
        move_kind: str,
        *,
        rng: random.Random,
    ) -> Iterable[CategoryMove]:
        if self.proposal_fn is None:
            return ()
        return self.proposal_fn(state, move_kind, rng)


def _validate_partition(
    support: frozenset[Coordinate],
    categories: Mapping[Coordinate, CategoryLabel],
) -> None:
    keys = frozenset(categories)
    if keys != support:
        missing = support - keys
        extra = keys - support
        raise ValueError(
            "Category assignment must cover exactly the active support. "
            f"missing={sorted(map(repr, missing))}, "
            f"extra={sorted(map(repr, extra))}"
        )


def _is_feasible(state: ProfiledState, eps: float) -> bool:
    return (
        math.isfinite(state.relative_residual)
        and state.relative_residual <= eps
        and math.isfinite(state.description_length)
    )


def run_weight_category_distribution(
    initial: ProfiledState,
    backend: CategoryBackend,
    *,
    config: In1Config,
) -> In1Result:
    """Relax the weight-category distribution at fixed (J,S).

    Acceptance:
      * S remains exactly fixed;
      * the candidate remains inside the hard uncertainty floor;
      * conditional description length strictly improves.

    Failure semantics:
      * failure of the initial profile is stage-level and propagates;
      * expected infeasibility of one local category proposal is recorded and
        rejected locally, after which the search continues;
      * configuration/programming errors are not swallowed.

    Residual depth inside the uncertainty floor is never used for ranking.
    """

    if config.max_sweeps < 1:
        raise ValueError("max_sweeps must be >= 1.")
    if config.uncertainty_floor < 0:
        raise ValueError("uncertainty_floor must be nonnegative.")

    support0 = frozenset(initial.support)
    _validate_partition(support0, initial.categories)

    current = backend.profile(
        support0,
        dict(initial.categories),
        warm_start=initial,
    )
    if frozenset(current.support) != support0:
        raise RuntimeError("in1 backend changed S; in1 must keep S fixed.")
    _validate_partition(support0, current.categories)
    if not _is_feasible(current, config.uncertainty_floor):
        raise In1InfeasibleError("Initial (J,S,c) representation is outside B_epsilon.")

    rng = random.Random(config.random_seed)
    accepted: list[CategoryMove] = []
    proposal_failures: list[CategoryProposalFailure] = []

    for sweep in range(1, config.max_sweeps + 1):
        improved_this_sweep = False

        for move_kind in config.move_order:
            best_state: ProfiledState | None = None
            best_move: CategoryMove | None = None

            for move in backend.propose(current, move_kind, rng=rng):
                if move.kind != move_kind:
                    raise ValueError(
                        f"Backend returned move kind {move.kind!r} "
                        f"while {move_kind!r} was requested."
                    )

                _validate_partition(support0, move.categories)
                try:
                    candidate = backend.profile(
                        support0,
                        dict(move.categories),
                        warm_start=current,
                    )
                except In1InfeasibleError as exc:
                    proposal_failures.append(
                        CategoryProposalFailure(
                            sweep=sweep,
                            move_kind=move_kind,
                            categories=dict(move.categories),
                            failure_kind="profile_infeasible",
                            reason=str(exc),
                            metadata=dict(move.metadata),
                        )
                    )
                    continue

                if frozenset(candidate.support) != support0:
                    raise RuntimeError(
                        "in1 proposal changed S; category moves must not alter support."
                    )
                _validate_partition(support0, candidate.categories)

                if not _is_feasible(candidate, config.uncertainty_floor):
                    proposal_failures.append(
                        CategoryProposalFailure(
                            sweep=sweep,
                            move_kind=move_kind,
                            categories=dict(move.categories),
                            failure_kind="outside_uncertainty_floor",
                            reason=(
                                "Profiled category proposal is not a feasible "
                                "finite-description representation inside B_epsilon."
                            ),
                            relative_residual=float(candidate.relative_residual),
                            description_length=float(candidate.description_length),
                            metadata=dict(move.metadata),
                        )
                    )
                    continue

                if (
                    candidate.description_length
                    < current.description_length - config.mdl_improvement_tol
                    and (
                        best_state is None
                        or candidate.description_length < best_state.description_length
                    )
                ):
                    best_state = candidate
                    best_move = move

            if best_state is not None and best_move is not None:
                current = best_state
                accepted.append(best_move)
                improved_this_sweep = True

        if not improved_this_sweep:
            return In1Result(
                state=current,
                accepted_moves=tuple(accepted),
                n_sweeps=sweep,
                proposal_failures=tuple(proposal_failures),
            )

    return In1Result(
        state=current,
        accepted_moves=tuple(accepted),
        n_sweeps=config.max_sweeps,
        proposal_failures=tuple(proposal_failures),
    )


run_in1 = run_weight_category_distribution

__all__ = [
    "Coordinate",
    "CategoryLabel",
    "In1InfeasibleError",
    "ProfiledState",
    "CategoryMove",
    "CategoryProposalFailure",
    "In1Config",
    "In1Result",
    "CategoryBackend",
    "DescriptionLengthScorer",
    "CategoryProposalFn",
    "PhysicalCategoryComputation",
    "run_weight_category_distribution",
    "run_in1",
]
