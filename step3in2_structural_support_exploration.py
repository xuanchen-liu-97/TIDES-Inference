"""
TIDES Step 3 inner stage 2: category-aware fixed-E structural equilibration.

Scientific role
---------------
At fixed dynamics support J and fixed structural cardinality E, explore the
composition of the structural support without performing irreversible
compression.  The validated proposal kernel is

    destroy -> repair -> destroy -> repair

where a repair is a *joint* choice of

    (structural coordinate, weight category),

not a topology-only insertion.  Existing categories are inherited through the
proposal; a repair may either join an existing category or create one new
category subject to a small category-growth allowance.  A destroyed coordinate
may therefore be re-added immediately in a different category.  This is an
intentional local category release/reassignment move, not a no-op.

The proposal geometry is evaluated at the parent shared law Theta.  A complete
two-exchange leaf must satisfy the TIDES hard observation floor before any
expensive exact calculation is performed.  Floor-feasible leaves are then
jointly re-profiled in (z, Theta) and passed through full Step-3 in1 category
relaxation.

A persistent support archive is used for exploration.  Local residual depth and
local MDL are proposal/navigation diagnostics only; they are not structural
truth scores and do not greedily select topology.  In particular, archive
replacement deliberately excludes description length.  Irreversible E -> E-1
compression remains the exclusive responsibility of in3.

This is the TIDES adaptation of Peixoto's coupled topology/weight-category
search: topology and category membership remain coupled latent variables, while
the hard uncertainty set B_epsilon replaces ordinary likelihood differences
inside the observational tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Iterable, Mapping, Protocol, Sequence
import math

import numpy as np

try:  # package form
    from .step3in1_weight_category_distribution import (
        CategoryBackend,
        In1Config,
        In1InfeasibleError,
        PhysicalCategoryComputation,
        ProfiledState,
        compact_labels,
        run_weight_category_distribution,
    )
    from .step3func_physical_profiling import FixedDynamicsProblem
except ImportError:  # flat-file form
    from step3in1_weight_category_distribution import (
        CategoryBackend,
        In1Config,
        In1InfeasibleError,
        PhysicalCategoryComputation,
        ProfiledState,
        compact_labels,
        run_weight_category_distribution,
    )
    from step3func_physical_profiling import FixedDynamicsProblem

Coordinate = Hashable
CategoryLabel = Hashable


# ---------------------------------------------------------------------------
# Public state / result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TopologyMoveLeg:
    destroyed_coordinate: Coordinate
    destroy_rank: int
    destroy_source: str
    destroyed_relative_residual: float
    repaired_coordinate: Coordinate
    repair_category: int
    repair_kind: str  # existing_category or new_category
    repair_rank: int
    repair_source: str
    repaired_relative_residual: float


@dataclass(frozen=True)
class RepairProposal:
    """One complete destroy-repair-destroy-repair fixed-E proposal."""

    support: frozenset[Coordinate]
    categories: Mapping[Coordinate, int]
    fixed_theta_relative_residual: float
    path_score: float
    legs: tuple[TopologyMoveLeg, TopologyMoveLeg]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    # Compatibility conveniences for older diagnostics that expected a single
    # removed/added pair.  They refer to the second leg only and should not be
    # used to reconstruct the proposal semantics.
    @property
    def removed(self) -> Coordinate:
        return self.legs[-1].destroyed_coordinate

    @property
    def added(self) -> Coordinate:
        return self.legs[-1].repaired_coordinate

    @property
    def score(self) -> float:
        return -float(self.path_score)


@dataclass(frozen=True)
class BirthProposal:
    """Compatibility placeholder; birth is not part of the validated in2 core."""

    added: Coordinate
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class In2Config:
    # Persistent-archive search budget.
    max_sweeps: int = 6
    proposals_per_parent: int = 12

    # Peixoto-style category-aware destroy/repair kernel.
    category_growth: int = 2
    destroy_rank_temperature: float = 10.0
    repair_rank_temperature: float = 25.0
    global_tail_probability: float = 0.08
    ls_rcond: float = 1.0e-11

    # Archive parent mixture.
    residual_parents: int = 4
    path_parents: int = 6
    diversity_parents: int = 2
    random_tail_parents: int = 1
    max_archive_size: int | None = None

    # in1 relaxation performed after exact joint profiling of each leaf.
    in1_max_sweeps: int = 5
    in1_random_seed: int = 1

    # Reproducibility / phase-boundary diagnostics.
    random_seed: int | None = 404
    max_representatives: int = 8
    coverage_check_interval: int = 1

    # Retained only so older callers fail explicitly instead of silently using
    # a different algorithm.  Fixed-E in2 does not perform births.
    allow_birth: bool = False

    def __post_init__(self) -> None:
        if self.max_sweeps < 0 or self.proposals_per_parent < 1:
            raise ValueError("invalid in2 sweep/proposal budget.")
        if self.category_growth < 0:
            raise ValueError("category_growth must be nonnegative.")
        if self.destroy_rank_temperature <= 0 or self.repair_rank_temperature <= 0:
            raise ValueError("rank temperatures must be positive.")
        if not 0.0 <= self.global_tail_probability <= 1.0:
            raise ValueError("global_tail_probability must lie in [0,1].")
        if self.ls_rcond < 0:
            raise ValueError("ls_rcond must be nonnegative.")
        if self.max_representatives < 1:
            raise ValueError("max_representatives must be >= 1.")
        if self.coverage_check_interval < 1:
            raise ValueError("coverage_check_interval must be >= 1.")
        if self.allow_birth:
            raise ValueError(
                "Validated in2 is fixed-E. Birth/capacity expansion must be tested "
                "as a separate extension, not mixed into the reference kernel."
            )


@dataclass(frozen=True)
class In2Move:
    kind: str
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
    saturated: bool
    reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CoverageCheck:
    sweep: int
    trigger: str
    support: frozenset[Coordinate]
    assessment: CoverageAssessment


@dataclass(frozen=True)
class TopologyArchiveRecord:
    state: ProfiledState
    path_score: float
    parent_support: frozenset[Coordinate] | None
    proposal_legs: tuple[TopologyMoveLeg, ...]
    discovery_sweep: int


@dataclass(frozen=True)
class In2Sweep:
    sweep: int
    parents_expanded: int
    proposals_generated: int
    floor_feasible_leaves: int
    exact_candidates: int
    archive_size: int
    next_parent_count: int
    best_relative_residual_ls: float
    best_checkpoint_description_length: float


@dataclass(frozen=True)
class In2Result:
    # ``state`` is the best fully relaxed checkpoint by description length.
    state: ProfiledState
    history: tuple[In2Move, ...]
    visited_supports: tuple[frozenset[Coordinate], ...]
    representative_states: tuple[ProfiledState, ...]
    candidate_failures: tuple[In2CandidateFailure, ...]
    coverage_status: str
    coverage_checks: tuple[CoverageCheck, ...] = ()
    archive: tuple[TopologyArchiveRecord, ...] = ()
    sweep_history: tuple[In2Sweep, ...] = ()
    best_residual_state: ProfiledState | None = None
    expanded_supports: int = 0


# ---------------------------------------------------------------------------
# Category/support canonicalisation
# ---------------------------------------------------------------------------


def _ordered_support(
    problem: FixedDynamicsProblem,
    support: Sequence[Coordinate] | frozenset[Coordinate],
) -> tuple[Coordinate, ...]:
    return tuple(problem.normalise_support(support))


def _compact_category_map(
    problem: FixedDynamicsProblem,
    support: Sequence[Coordinate] | frozenset[Coordinate],
    categories: Mapping[Coordinate, CategoryLabel],
) -> dict[Coordinate, int]:
    ordered = _ordered_support(problem, support)
    if set(categories) != set(ordered):
        missing = set(ordered) - set(categories)
        extra = set(categories) - set(ordered)
        raise ValueError(
            "Category assignment must cover exactly the support. "
            f"missing={sorted(map(repr, missing))}, extra={sorted(map(repr, extra))}"
        )
    mapping: dict[CategoryLabel, int] = {}
    raw: list[int] = []
    for g in ordered:
        label = categories[g]
        if label not in mapping:
            mapping[label] = len(mapping)
        raw.append(mapping[label])
    lab = compact_labels(raw)
    return {g: int(k) for g, k in zip(ordered, lab)}


def _category_count(
    problem: FixedDynamicsProblem,
    support: Sequence[Coordinate] | frozenset[Coordinate],
    categories: Mapping[Coordinate, CategoryLabel],
) -> int:
    cats = _compact_category_map(problem, support, categories)
    return 0 if not cats else max(cats.values()) + 1


def _delete_coordinate(
    problem: FixedDynamicsProblem,
    support: Sequence[Coordinate] | frozenset[Coordinate],
    categories: Mapping[Coordinate, CategoryLabel],
    coordinate: Coordinate,
) -> tuple[frozenset[Coordinate], dict[Coordinate, int]]:
    s = frozenset(support)
    if coordinate not in s:
        raise ValueError("destroy coordinate is not active.")
    s2 = frozenset(g for g in s if g != coordinate)
    if not s2:
        raise ValueError("cannot destroy the last active coordinate.")
    c2 = {g: categories[g] for g in s2}
    return s2, _compact_category_map(problem, s2, c2)


def _insert_coordinate(
    problem: FixedDynamicsProblem,
    support: Sequence[Coordinate] | frozenset[Coordinate],
    categories: Mapping[Coordinate, CategoryLabel],
    coordinate: Coordinate,
    category: int,
) -> tuple[frozenset[Coordinate], dict[Coordinate, int]]:
    s = frozenset(support)
    if coordinate in s:
        raise ValueError("repair coordinate is already active.")
    c = _compact_category_map(problem, s, categories)
    K = 0 if not c else max(c.values()) + 1
    if int(category) < 0 or int(category) > K:
        raise ValueError("repair category must be existing or exactly one new category.")
    s2 = frozenset(set(s) | {coordinate})
    c2: dict[Coordinate, int] = dict(c)
    c2[coordinate] = int(category)
    return s2, _compact_category_map(problem, s2, c2)


def support_key(
    problem: FixedDynamicsProblem,
    support: Sequence[Coordinate] | frozenset[Coordinate],
) -> tuple[int, ...]:
    order = {g: i for i, g in enumerate(problem.coordinates)}
    unknown = [g for g in support if g not in order]
    if unknown:
        raise ValueError(f"unknown structural coordinates: {unknown!r}")
    return tuple(sorted(order[g] for g in set(support)))


def support_distance(
    problem: FixedDynamicsProblem,
    a: Sequence[Coordinate] | frozenset[Coordinate],
    b: Sequence[Coordinate] | frozenset[Coordinate],
) -> int:
    ka, kb = set(support_key(problem, a)), set(support_key(problem, b))
    return int(len(ka.symmetric_difference(kb)))


# ---------------------------------------------------------------------------
# Fast fixed-Theta category geometry
# ---------------------------------------------------------------------------


class _FixedThetaTopologyGeometry:
    """Gram-form exact fixed-Theta profiling under category constraints.

    For a fixed parent Theta this reproduces the same least-squares objective as
    explicit category-block aggregation, but makes thousands of destroy/repair
    scans practical for the small Step-3 shell search.
    """

    def __init__(
        self,
        problem: FixedDynamicsProblem,
        theta: Sequence[float],
        *,
        rcond: float = 1.0e-11,
    ) -> None:
        self.problem = problem
        self.theta = np.asarray(theta, dtype=float).reshape(-1)
        if self.theta.size != problem.n_library_atoms:
            raise ValueError("theta has the wrong library dimension.")
        self.rcond = float(rcond)
        self.coordinates = tuple(problem.coordinates)
        self.index = {g: i for i, g in enumerate(self.coordinates)}
        blocks = tuple(problem.structural_blocks[g] for g in self.coordinates)
        C = np.asarray(problem.linear_cache.matrix(blocks, self.theta), dtype=float)
        self.G = np.asarray(C.T @ C, dtype=float)
        self.b = np.asarray(C.T @ problem.y, dtype=float)
        self.y2 = float(problem.y @ problem.y)
        self.y_norm = max(float(np.linalg.norm(problem.y)), np.finfo(float).tiny)

    def rho(
        self,
        support: Sequence[Coordinate] | frozenset[Coordinate],
        categories: Mapping[Coordinate, CategoryLabel],
    ) -> float:
        ordered = _ordered_support(self.problem, support)
        cats = _compact_category_map(self.problem, ordered, categories)
        labels = np.asarray([cats[g] for g in ordered], dtype=int)
        K = int(labels.max()) + 1 if labels.size else 0
        if K == 0:
            return 1.0
        inds = np.asarray([self.index[g] for g in ordered], dtype=int)
        H = np.zeros((len(ordered), K), dtype=float)
        H[np.arange(len(ordered)), labels] = 1.0
        Gs = self.G[np.ix_(inds, inds)]
        bs = self.b[inds]
        A = H.T @ Gs @ H
        d = H.T @ bs
        try:
            # A = D^T D is normally positive definite for the small category
            # systems encountered here.  Direct solve is exactly the LS normal-
            # equation solution in the full-rank case and avoids thousands of
            # repeated SVDs during repair scans.
            z = np.linalg.solve(A, d)
        except np.linalg.LinAlgError:
            z = np.linalg.lstsq(A, d, rcond=self.rcond)[0]
        rss = max(self.y2 - float(d @ z), 0.0)
        return float(np.sqrt(rss) / self.y_norm)


# ---------------------------------------------------------------------------
# Proposal sampling
# ---------------------------------------------------------------------------


def _sample_ranked(
    records: Sequence[tuple[Any, ...]],
    rng: np.random.Generator,
    *,
    rank_temperature: float,
    global_tail_probability: float,
) -> tuple[tuple[Any, ...], int, str]:
    if not records:
        raise ValueError("cannot sample from an empty ranked list.")
    if float(rng.random()) < float(global_tail_probability):
        idx = int(rng.integers(len(records)))
        return tuple(records[idx]), idx + 1, "global_tail"
    ranks = np.arange(len(records), dtype=float)
    weights = np.exp(-ranks / float(rank_temperature))
    weights /= weights.sum()
    idx = int(rng.choice(len(records), p=weights))
    return tuple(records[idx]), idx + 1, "ranked"


class StructuralExplorationBackend(Protocol):
    def propose_two_exchange(
        self,
        state: ProfiledState,
        *,
        rng: np.random.Generator,
        config: In2Config,
        category_limit: int,
    ) -> RepairProposal | None:
        ...

    def build_replacement_seed(
        self,
        state: ProfiledState,
        proposal: RepairProposal,
    ) -> ProfiledState:
        ...

    def coverage_saturated(
        self,
        state: ProfiledState,
        history: Sequence[In2Move],
        visited_supports: Sequence[frozenset[Coordinate]],
    ) -> bool | CoverageAssessment:
        ...


CoverageAssessor = Callable[
    [ProfiledState, Sequence[In2Move], Sequence[frozenset[Coordinate]]],
    bool | CoverageAssessment,
]


class PhysicalStructuralComputation:
    """First-party category-aware destroy-first proposal engine."""

    def __init__(
        self,
        problem: FixedDynamicsProblem,
        *,
        active_atoms: Sequence[int],
        candidate_coordinates: Sequence[Coordinate] | None = None,
        category_backend: CategoryBackend | None = None,
        fixed_description_bits: float = 0.0,
        fixed_description_bits_fn: Callable[[frozenset[Coordinate]], float] | None = None,
        max_nfev: int = 120,
        coverage_assessor: CoverageAssessor | None = None,
    ) -> None:
        self.problem = problem
        self.active_atoms = tuple(sorted(set(int(a) for a in active_atoms)))
        if not self.active_atoms:
            raise ValueError("active_atoms must be non-empty.")
        self.candidate_coordinates = tuple(
            problem.coordinates if candidate_coordinates is None else candidate_coordinates
        )
        unknown = set(self.candidate_coordinates) - set(problem.coordinates)
        if unknown:
            raise ValueError(
                "candidate_coordinates contains unknown entries: "
                f"{sorted(map(repr, unknown))}"
            )
        self.max_nfev = int(max_nfev)
        if self.max_nfev < 1:
            raise ValueError("max_nfev must be >= 1.")
        self.category_backend = (
            category_backend
            if category_backend is not None
            else PhysicalCategoryComputation(
                problem,
                active_atoms=self.active_atoms,
                fixed_description_bits=float(fixed_description_bits),
                fixed_description_bits_fn=fixed_description_bits_fn,
                max_nfev=self.max_nfev,
            )
        )
        self.coverage_assessor = coverage_assessor

    def _scan_destroy(
        self,
        geometry: _FixedThetaTopologyGeometry,
        support: frozenset[Coordinate],
        categories: Mapping[Coordinate, CategoryLabel],
        *,
        exclude: Sequence[Coordinate] = (),
    ) -> list[tuple[float, Coordinate]]:
        excluded = set(exclude)
        out: list[tuple[float, Coordinate]] = []
        for g in _ordered_support(self.problem, support):
            if g in excluded:
                continue
            sd, cd = _delete_coordinate(self.problem, support, categories, g)
            out.append((geometry.rho(sd, cd), g))
        return sorted(out, key=lambda x: x[0])

    def _scan_repair(
        self,
        geometry: _FixedThetaTopologyGeometry,
        support: frozenset[Coordinate],
        categories: Mapping[Coordinate, CategoryLabel],
        *,
        category_limit: int,
    ) -> list[tuple[float, Coordinate, int, str]]:
        c = _compact_category_map(self.problem, support, categories)
        K = 0 if not c else max(c.values()) + 1
        choices = list(range(K))
        if K < int(category_limit):
            choices.append(K)

        active = set(support)
        out: list[tuple[float, Coordinate, int, str]] = []
        for g in self.candidate_coordinates:
            if g in active:
                continue
            for k in choices:
                sr, cr = _insert_coordinate(self.problem, support, c, g, int(k))
                rho = geometry.rho(sr, cr)
                kind = "existing_category" if int(k) < K else "new_category"
                out.append((float(rho), g, int(k), kind))
        return sorted(out, key=lambda x: x[0])

    def _one_leg(
        self,
        geometry: _FixedThetaTopologyGeometry,
        support: frozenset[Coordinate],
        categories: Mapping[Coordinate, CategoryLabel],
        *,
        rng: np.random.Generator,
        config: In2Config,
        category_limit: int,
        exclude_destroy: Sequence[Coordinate] = (),
    ) -> tuple[frozenset[Coordinate], dict[Coordinate, int], TopologyMoveLeg]:
        destroy_records = self._scan_destroy(
            geometry, support, categories, exclude=exclude_destroy
        )
        destroy, destroy_rank, destroy_source = _sample_ranked(
            destroy_records,
            rng,
            rank_temperature=config.destroy_rank_temperature,
            global_tail_probability=config.global_tail_probability,
        )
        destroy_rho, destroyed = destroy
        sd, cd = _delete_coordinate(self.problem, support, categories, destroyed)

        repair_records = self._scan_repair(
            geometry, sd, cd, category_limit=category_limit
        )
        repair, repair_rank, repair_source = _sample_ranked(
            repair_records,
            rng,
            rank_temperature=config.repair_rank_temperature,
            global_tail_probability=config.global_tail_probability,
        )
        repair_rho, repaired, repair_category, repair_kind = repair
        sr, cr = _insert_coordinate(
            self.problem, sd, cd, repaired, int(repair_category)
        )
        if _category_count(self.problem, sr, cr) > int(category_limit):
            raise RuntimeError("repair exceeded the category limit.")

        leg = TopologyMoveLeg(
            destroyed_coordinate=destroyed,
            destroy_rank=int(destroy_rank),
            destroy_source=str(destroy_source),
            destroyed_relative_residual=float(destroy_rho),
            repaired_coordinate=repaired,
            repair_category=int(repair_category),
            repair_kind=str(repair_kind),
            repair_rank=int(repair_rank),
            repair_source=str(repair_source),
            repaired_relative_residual=float(repair_rho),
        )
        return sr, cr, leg

    def propose_two_exchange(
        self,
        state: ProfiledState,
        *,
        rng: np.random.Generator,
        config: In2Config,
        category_limit: int,
    ) -> RepairProposal | None:
        support0 = frozenset(state.support)
        if len(support0) < 2:
            return None
        categories0 = _compact_category_map(self.problem, support0, state.categories)
        geometry = _FixedThetaTopologyGeometry(
            self.problem, state.theta, rcond=config.ls_rcond
        )
        try:
            s1, c1, leg1 = self._one_leg(
                geometry,
                support0,
                categories0,
                rng=rng,
                config=config,
                category_limit=category_limit,
            )
            # Historical validated kernel forbids immediately destroying the
            # group introduced/reintroduced by the first repair.
            s2, c2, leg2 = self._one_leg(
                geometry,
                s1,
                c1,
                rng=rng,
                config=config,
                category_limit=category_limit,
                exclude_destroy=(leg1.repaired_coordinate,),
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            return None

        # Pure category rearrangements are handled by in1.  The in2 leaf must
        # actually change the topology shell representative.
        if support_key(self.problem, s2) == support_key(self.problem, support0):
            return None
        rho = geometry.rho(s2, c2)
        eps = float(self.problem.uncertainty_floor)
        if rho > eps * (1.0 + 2.0e-10):
            return None

        path_score = float(
            math.log1p(leg1.destroy_rank)
            + math.log1p(leg1.repair_rank)
            + math.log1p(leg2.destroy_rank)
            + math.log1p(leg2.repair_rank)
        )
        return RepairProposal(
            support=frozenset(s2),
            categories=dict(c2),
            fixed_theta_relative_residual=float(rho),
            path_score=path_score,
            legs=(leg1, leg2),
            metadata={
                "category_aware_repair": True,
                "two_exchange": True,
                "global_singleton_release": False,
            },
        )

    # Compatibility spelling: now returns stochastic *two-exchange* leaves.
    def replacement_proposals(
        self,
        state: ProfiledState,
        *,
        rng: np.random.Generator,
        config: In2Config | None = None,
        category_limit: int | None = None,
    ) -> Iterable[RepairProposal]:
        cfg = In2Config() if config is None else config
        lim = (
            _category_count(self.problem, state.support, state.categories)
            + cfg.category_growth
            if category_limit is None
            else int(category_limit)
        )
        out: list[RepairProposal] = []
        for _ in range(int(cfg.proposals_per_parent)):
            p = self.propose_two_exchange(
                state, rng=rng, config=cfg, category_limit=lim
            )
            if p is not None:
                out.append(p)
        return tuple(out)

    def build_replacement_seed(
        self,
        state: ProfiledState,
        proposal: RepairProposal,
    ) -> ProfiledState:
        """Exact joint (z,Theta) profile preserving proposal categories.

        There is deliberately no global singleton release here.
        """

        profiled = self.category_backend.profile(
            frozenset(proposal.support),
            dict(proposal.categories),
            warm_start=state,
        )
        return ProfiledState(
            support=frozenset(profiled.support),
            categories=dict(profiled.categories),
            description_length=float(profiled.description_length),
            relative_residual=float(profiled.relative_residual),
            weights=profiled.weights,
            theta=np.asarray(profiled.theta, dtype=float).copy(),
            metadata={
                **dict(profiled.metadata),
                "in2_replacement_seed": True,
                "fixed_theta_leaf_rho": float(
                    proposal.fixed_theta_relative_residual
                ),
                "proposal_path_score": float(proposal.path_score),
                "proposal_legs": proposal.legs,
                "global_singleton_release": False,
            },
        )

    def coverage_saturated(
        self,
        state: ProfiledState,
        history: Sequence[In2Move],
        visited_supports: Sequence[frozenset[Coordinate]],
    ) -> bool | CoverageAssessment:
        if self.coverage_assessor is None:
            return CoverageAssessment(
                saturated=False,
                reason="no_explicit_coverage_assessor",
            )
        return self.coverage_assessor(state, history, visited_supports)


# ---------------------------------------------------------------------------
# Persistent archive exploration
# ---------------------------------------------------------------------------


In1RelaxFn = Callable[[ProfiledState], ProfiledState]


def _rho_ls(state: ProfiledState) -> float:
    return float(state.metadata.get("relative_residual_ls", state.relative_residual))


def _normalise_coverage_assessment(
    raw: bool | CoverageAssessment,
) -> CoverageAssessment:
    if isinstance(raw, CoverageAssessment):
        return raw
    if isinstance(raw, (bool, np.bool_)):
        return CoverageAssessment(saturated=bool(raw))
    raise TypeError("coverage_saturated() must return bool or CoverageAssessment.")


def _select_archive_parents(
    problem: FixedDynamicsProblem,
    archive: Mapping[tuple[int, ...], TopologyArchiveRecord],
    expanded: set[tuple[int, ...]],
    rng: np.random.Generator,
    config: In2Config,
) -> list[TopologyArchiveRecord]:
    available = [rec for key, rec in archive.items() if key not in expanded]
    if not available:
        return []

    selected: list[TopologyArchiveRecord] = []
    used: set[tuple[int, ...]] = set()

    def add(rec: TopologyArchiveRecord) -> None:
        key = support_key(problem, rec.state.support)
        if key not in used:
            selected.append(rec)
            used.add(key)

    for rec in sorted(available, key=lambda r: _rho_ls(r.state))[: config.residual_parents]:
        add(rec)
    target = int(config.residual_parents + config.path_parents)
    for rec in sorted(available, key=lambda r: (r.path_score, _rho_ls(r.state))):
        if len(selected) >= target:
            break
        add(rec)

    for _ in range(int(config.diversity_parents)):
        remaining = [
            r for r in available
            if support_key(problem, r.state.support) not in used
        ]
        if not remaining:
            break
        if selected:
            rec = max(
                remaining,
                key=lambda r: min(
                    support_distance(problem, r.state.support, s.state.support)
                    for s in selected
                ),
            )
        else:
            rec = remaining[0]
        add(rec)

    remaining = [
        r for r in available
        if support_key(problem, r.state.support) not in used
    ]
    for _ in range(min(int(config.random_tail_parents), len(remaining))):
        idx = int(rng.integers(len(remaining)))
        add(remaining.pop(idx))
    return selected


def _thin_archive(
    problem: FixedDynamicsProblem,
    archive: Mapping[tuple[int, ...], TopologyArchiveRecord],
    expanded: set[tuple[int, ...]],
    budget: int,
) -> tuple[dict[tuple[int, ...], TopologyArchiveRecord], set[tuple[int, ...]]]:
    vals = list(archive.values())
    keep: list[TopologyArchiveRecord] = []
    used: set[tuple[int, ...]] = set()

    def add(rec: TopologyArchiveRecord) -> None:
        key = support_key(problem, rec.state.support)
        if key not in used:
            keep.append(rec)
            used.add(key)

    for rec in sorted(vals, key=lambda r: _rho_ls(r.state))[: max(1, budget // 3)]:
        add(rec)
    for rec in sorted(vals, key=lambda r: (r.path_score, _rho_ls(r.state))):
        if len(keep) >= max(1, 2 * budget // 3):
            break
        add(rec)
    while len(keep) < budget:
        remaining = [
            r for r in vals
            if support_key(problem, r.state.support) not in used
        ]
        if not remaining:
            break
        rec = max(
            remaining,
            key=lambda r: min(
                support_distance(problem, r.state.support, s.state.support)
                for s in keep
            ) if keep else 0,
        )
        add(rec)
    new_archive = {support_key(problem, r.state.support): r for r in keep}
    new_expanded = set(expanded).intersection(new_archive.keys())
    return new_archive, new_expanded


def _representative_states(
    problem: FixedDynamicsProblem,
    archive: Mapping[tuple[int, ...], TopologyArchiveRecord],
    best: ProfiledState,
    *,
    max_representatives: int,
) -> tuple[ProfiledState, ...]:
    vals = list(archive.values())
    chosen: list[ProfiledState] = [best]
    used = {support_key(problem, best.support)}
    while len(chosen) < max_representatives:
        remaining = [r.state for r in vals if support_key(problem, r.state.support) not in used]
        if not remaining:
            break
        state = max(
            remaining,
            key=lambda r: min(
                support_distance(problem, r.support, s.support) for s in chosen
            ),
        )
        chosen.append(state)
        used.add(support_key(problem, state.support))
    return tuple(chosen)


def run_structural_support_exploration(
    initial: ProfiledState,
    backend: StructuralExplorationBackend,
    *,
    relax_in1: In1RelaxFn | None = None,
    config: In2Config = In2Config(),
    capacity_diagnostic: Any = None,
    build_birth_seed: Any = None,
) -> In2Result:
    """Run fixed-E category-aware reversible structural exploration.

    ``capacity_diagnostic`` and ``build_birth_seed`` are accepted only for
    source compatibility with the superseded implementation; the validated
    fixed-E reference kernel never invokes them.
    """

    del capacity_diagnostic, build_birth_seed
    if config.allow_birth:
        raise ValueError("birth is not part of the validated fixed-E in2 kernel.")
    if len(initial.support) < 2:
        raise ValueError("in2 requires at least two active structural coordinates.")
    if initial.relative_residual > backend.problem.uncertainty_floor * (1.0 + 2.0e-10):  # type: ignore[attr-defined]
        raise ValueError("initial in2 state must be hard-floor feasible.")

    problem: FixedDynamicsProblem = backend.problem  # type: ignore[attr-defined]
    E0 = len(initial.support)
    initial_categories = _compact_category_map(problem, initial.support, initial.categories)
    initial_state = ProfiledState(
        support=frozenset(initial.support),
        categories=initial_categories,
        description_length=float(initial.description_length),
        relative_residual=float(initial.relative_residual),
        weights=initial.weights,
        theta=np.asarray(initial.theta, dtype=float).copy(),
        metadata=dict(initial.metadata),
    )
    K0 = _category_count(problem, initial_state.support, initial_state.categories)
    category_limit = int(K0 + config.category_growth)
    rng = np.random.default_rng(config.random_seed)

    if relax_in1 is None:
        if not isinstance(backend, PhysicalStructuralComputation):
            raise ValueError("relax_in1 is required for a custom in2 backend.")

        def _default_relax(seed: ProfiledState) -> ProfiledState:
            return run_weight_category_distribution(
                seed,
                backend.category_backend,
                config=In1Config(
                    uncertainty_floor=problem.uncertainty_floor,
                    max_sweeps=config.in1_max_sweeps,
                    random_seed=config.in1_random_seed,
                ),
            ).state

        relax = _default_relax
    else:
        relax = relax_in1

    key0 = support_key(problem, initial_state.support)
    archive: dict[tuple[int, ...], TopologyArchiveRecord] = {
        key0: TopologyArchiveRecord(
            state=initial_state,
            path_score=0.0,
            parent_support=None,
            proposal_legs=tuple(),
            discovery_sweep=0,
        )
    }
    expanded: set[tuple[int, ...]] = set()
    parents = [archive[key0]]
    moves: list[In2Move] = []
    failures: list[In2CandidateFailure] = []
    visited: list[frozenset[Coordinate]] = [frozenset(initial_state.support)]
    checks: list[CoverageCheck] = []
    sweeps: list[In2Sweep] = []
    coverage_status = "budget_exhausted"

    for sweep in range(1, int(config.max_sweeps) + 1):
        parents_expanded = len(parents)
        generated = 0
        floor_feasible = 0
        exact_candidates = 0

        for parent in parents:
            pkey = support_key(problem, parent.state.support)
            expanded.add(pkey)
            local_seen: set[tuple[int, ...]] = set()

            for proposal_index in range(int(config.proposals_per_parent)):
                proposal = backend.propose_two_exchange(
                    parent.state,
                    rng=rng,
                    config=config,
                    category_limit=category_limit,
                )
                generated += 1
                if proposal is None:
                    continue
                floor_feasible += 1
                key = support_key(problem, proposal.support)
                if key in local_seen:
                    continue
                local_seen.add(key)

                try:
                    seed = backend.build_replacement_seed(parent.state, proposal)
                    if len(seed.support) != E0:
                        raise RuntimeError("in2 exact seed changed E.")
                    if frozenset(seed.support) != frozenset(proposal.support):
                        raise RuntimeError("exact seed does not match proposal support.")
                    # Crucial invariant: exact profiling preserves the proposal
                    # partition instead of resetting all groups to singleton.
                    if _compact_category_map(problem, seed.support, seed.categories) != _compact_category_map(
                        problem, proposal.support, proposal.categories
                    ):
                        raise RuntimeError("exact seed changed the proposal category partition.")
                    relaxed = relax(seed)
                except (In1InfeasibleError, ValueError, RuntimeError) as exc:
                    failures.append(In2CandidateFailure(
                        kind="exact_or_in1_infeasible",
                        support=frozenset(proposal.support),
                        metadata={
                            "reason": str(exc),
                            "legs": proposal.legs,
                            "fixed_theta_rho": proposal.fixed_theta_relative_residual,
                        },
                    ))
                    continue

                if len(relaxed.support) != E0:
                    raise RuntimeError("in1 changed E during in2.")
                if frozenset(relaxed.support) != frozenset(proposal.support):
                    raise RuntimeError("in1 changed support during in2.")
                exact_candidates += 1

                record = TopologyArchiveRecord(
                    state=relaxed,
                    path_score=float(proposal.path_score),
                    parent_support=frozenset(parent.state.support),
                    proposal_legs=proposal.legs,
                    discovery_sweep=sweep,
                )
                old = archive.get(key)
                # Navigation only: local path quality then LS residual.  MDL is
                # intentionally absent from this archive replacement rule.
                if old is None or (
                    record.path_score,
                    _rho_ls(relaxed),
                ) < (
                    old.path_score,
                    _rho_ls(old.state),
                ):
                    archive[key] = record
                    visited.append(frozenset(relaxed.support))
                    moves.append(In2Move(
                        kind="two_exchange",
                        before_support=frozenset(parent.state.support),
                        after_support=frozenset(relaxed.support),
                        metadata={
                            "path_score": float(proposal.path_score),
                            "fixed_theta_rho": float(
                                proposal.fixed_theta_relative_residual
                            ),
                            "legs": proposal.legs,
                            "n_categories_after_in1": _category_count(
                                problem, relaxed.support, relaxed.categories
                            ),
                        },
                    ))

        if config.max_archive_size is not None and len(archive) > config.max_archive_size:
            archive, expanded = _thin_archive(
                problem, archive, expanded, int(config.max_archive_size)
            )

        parents = _select_archive_parents(problem, archive, expanded, rng, config)
        vals = list(archive.values())
        best_residual = min(vals, key=lambda r: _rho_ls(r.state)).state
        best_checkpoint = min(vals, key=lambda r: r.state.description_length).state
        sweeps.append(In2Sweep(
            sweep=sweep,
            parents_expanded=int(parents_expanded),
            proposals_generated=int(generated),
            floor_feasible_leaves=int(floor_feasible),
            exact_candidates=int(exact_candidates),
            archive_size=len(archive),
            next_parent_count=len(parents),
            best_relative_residual_ls=float(_rho_ls(best_residual)),
            best_checkpoint_description_length=float(
                best_checkpoint.description_length
            ),
        ))

        if sweep % int(config.coverage_check_interval) == 0:
            assessment = _normalise_coverage_assessment(
                backend.coverage_saturated(
                    best_checkpoint, tuple(moves), tuple(visited)
                )
            )
            checks.append(CoverageCheck(
                sweep=sweep,
                trigger="periodic",
                support=frozenset(best_checkpoint.support),
                assessment=assessment,
            ))
            if assessment.saturated:
                coverage_status = "coverage_saturated"
                break

        if not parents:
            coverage_status = "archive_exhausted"
            break

    vals = list(archive.values())
    best_checkpoint = min(vals, key=lambda r: r.state.description_length).state
    best_residual = min(vals, key=lambda r: _rho_ls(r.state)).state
    representatives = _representative_states(
        problem,
        archive,
        best_checkpoint,
        max_representatives=config.max_representatives,
    )
    return In2Result(
        state=best_checkpoint,
        history=tuple(moves),
        visited_supports=tuple(visited),
        representative_states=representatives,
        candidate_failures=tuple(failures),
        coverage_status=coverage_status,
        coverage_checks=tuple(checks),
        archive=tuple(vals),
        sweep_history=tuple(sweeps),
        best_residual_state=best_residual,
        expanded_supports=len(expanded),
    )


run_in2 = run_structural_support_exploration


__all__ = [
    "Coordinate",
    "TopologyMoveLeg",
    "RepairProposal",
    "BirthProposal",
    "In2Config",
    "In2Move",
    "In2CandidateFailure",
    "CoverageAssessment",
    "CoverageCheck",
    "TopologyArchiveRecord",
    "In2Sweep",
    "In2Result",
    "StructuralExplorationBackend",
    "CoverageAssessor",
    "PhysicalStructuralComputation",
    "In1RelaxFn",
    "support_key",
    "support_distance",
    "run_structural_support_exploration",
    "run_in2",
]
