"""Reference outer search for TIDES conditional-MDL models.

The purpose of this module is to keep *model-space navigation* separate from
fixed-model numerical profiling (``mdl_precision.py``) and description-length
bookkeeping (``mdl_description_length.py``).

This first implementation is intentionally a transparent reference / diagnostic
search, not yet the final scalable production optimizer.  It supports a linear
"group-local polynomial" adapter in which every active structural group owns a
subset of columns from a common polynomial library.  That adapter is useful for
exercising add/drop/swap and expression moves on the current Step-2 projected
B-space design.  A later hypothesis-specific adapter can replace it (for
example a shared-law factorization) without changing the search/evaluator API.

The current precision profiler provides feasible q-bit *witnesses* but does not
certify minimal q_star.  Accordingly the scores returned here are marked as
upper bounds unless a certified precision profiler is supplied in the future.
The search can rank moves in this mode, but it does not claim a globally or even
locally certified MDL optimum.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Hashable, Mapping, Optional, Sequence

import numpy as np

from mdl_description_length import (
    MDLScore,
    PolynomialLibrary,
    PolynomialObject,
    TemporalSupportBlock,
    compute_conditional_mdl,
    elias_delta_integer_code_length_nats,
)
from mdl_precision import (
    ContinuousProfile,
    PrecisionProfile,
    UniformDyadicQuantizer,
    exact_add_residual_sq,
    profile_continuous,
    profile_precision_witness,
    FixedDynamicsProfile,
    FixedDynamicsPrecisionProfile,
    profile_fixed_dynamics,
    profile_fixed_dynamics_precision_witness,
    FixedDynamicsVarProWorkspace,
    build_fixed_dynamics_varpro_workspace,
    profile_fixed_dynamics_varpro,
)


FloatArray = np.ndarray
GroupLabel = Hashable


@dataclass(frozen=True)
class LinearGroupMDLProblem:
    """Linear Step-2 adapter used by the reference search.

    Parameters
    ----------
    y
        Profiled observation vector (e.g. Step-2 ``y_perp``).
    group_blocks
        Mapping from structural group label to its full polynomial block with
        shape ``(n_rows, library.n_atoms)``.
    temporal_block_of_group
        Mapping label -> encoded temporal-support block index.
    temporal_candidate_counts
        Candidate count M_t for each temporal block.
    library
        Canonical polynomial library used by every group block.
    uncertainty_floor
        Relative residual floor.

    Notes
    -----
    In this adapter each active group is an independent PolynomialObject.  This
    is the current unrestricted projected-B representation and is *not* a claim
    that a shared-law hypothesis should ultimately be encoded this way.
    """

    y: FloatArray
    group_blocks: Mapping[GroupLabel, FloatArray]
    temporal_block_of_group: Mapping[GroupLabel, int]
    temporal_candidate_counts: tuple[int, ...]
    library: PolynomialLibrary
    uncertainty_floor: float

    def __post_init__(self) -> None:
        y = np.asarray(self.y, dtype=float).reshape(-1)
        if y.size == 0:
            raise ValueError("y must be non-empty.")
        if not self.group_blocks:
            raise ValueError("group_blocks must be non-empty.")
        L = self.library.n_atoms
        blocks = {}
        for g, B in self.group_blocks.items():
            B = np.asarray(B, dtype=float)
            if B.shape != (y.size, L):
                raise ValueError(
                    f"group {g!r} has shape {B.shape}; expected {(y.size, L)}."
                )
            blocks[g] = B
            if g not in self.temporal_block_of_group:
                raise ValueError(f"group {g!r} lacks a temporal block assignment.")
            t = int(self.temporal_block_of_group[g])
            if t < 0 or t >= len(self.temporal_candidate_counts):
                raise ValueError(f"invalid temporal block index {t} for group {g!r}.")
        counts = tuple(int(x) for x in self.temporal_candidate_counts)
        if any(x < 0 for x in counts):
            raise ValueError("temporal candidate counts must be non-negative.")
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "group_blocks", blocks)
        object.__setattr__(self, "temporal_candidate_counts", counts)

    @property
    def groups(self) -> tuple[GroupLabel, ...]:
        return tuple(self.group_blocks.keys())


@dataclass(frozen=True)
class MDLMove:
    kind: str
    group_out: Optional[GroupLabel] = None
    group_in: Optional[GroupLabel] = None
    atom_out: Optional[int] = None
    atom_in: Optional[int] = None
    proposal_score: Optional[float] = None

    def label(self) -> str:
        if self.kind == "drop_group":
            return f"drop_group({self.group_out!r})"
        if self.kind == "add_group":
            return f"add_group({self.group_in!r})"
        if self.kind == "swap_group":
            return f"swap_group({self.group_out!r}->{self.group_in!r})"
        if self.kind == "drop_atom":
            if self.group_out is None:
                return f"drop_atom(shared_theta, {self.atom_out})"
            return f"drop_atom({self.group_out!r}, {self.atom_out})"
        if self.kind == "add_atom":
            if self.group_in is None:
                return f"add_atom(shared_theta, {self.atom_in})"
            return f"add_atom({self.group_in!r}, {self.atom_in})"
        if self.kind == "swap_atom":
            if self.group_out is None and self.group_in is None:
                return f"swap_atom(shared_theta, {self.atom_out}->{self.atom_in})"
            return (
                f"swap_atom({self.group_out!r}, "
                f"{self.atom_out}->{self.atom_in})"
            )
        return self.kind


@dataclass(frozen=True)
class MDLSearchState:
    active_atoms: tuple[tuple[GroupLabel, tuple[int, ...]], ...]
    design: FloatArray
    coefficient_slices: Mapping[GroupLabel, slice]
    continuous: ContinuousProfile
    precision: PrecisionProfile
    mdl_upper: MDLScore
    mdl_lower: float

    @property
    def active_groups(self) -> tuple[GroupLabel, ...]:
        return tuple(g for g, atoms in self.active_atoms if atoms)

    def atom_map(self) -> dict[GroupLabel, tuple[int, ...]]:
        return {g: tuple(a) for g, a in self.active_atoms}


@dataclass(frozen=True)
class MoveEvaluation:
    move: MDLMove
    status: str
    continuous_relative_residual: Optional[float]
    precision_q_upper: Optional[int]
    mdl_lower: float
    mdl_upper: float
    delta_upper_vs_current_upper: Optional[float]
    certified_reject: bool
    certified_improve: bool
    new_state: Optional[MDLSearchState]


def _normalise_atom_map(
    problem: LinearGroupMDLProblem,
    active_atoms: Mapping[GroupLabel, Sequence[int]],
) -> tuple[tuple[GroupLabel, tuple[int, ...]], ...]:
    order = {g: i for i, g in enumerate(problem.groups)}
    out = []
    for g, atoms0 in active_atoms.items():
        if g not in problem.group_blocks:
            raise ValueError(f"unknown group {g!r}.")
        atoms = tuple(sorted(set(int(a) for a in atoms0)))
        if any(a < 0 or a >= problem.library.n_atoms for a in atoms):
            raise ValueError(f"invalid atom index for group {g!r}.")
        if atoms:
            out.append((g, atoms))
    out.sort(key=lambda ga: order[ga[0]])
    return tuple(out)


def _assemble_design(
    problem: LinearGroupMDLProblem,
    active_atoms: tuple[tuple[GroupLabel, tuple[int, ...]], ...],
) -> tuple[FloatArray, dict[GroupLabel, slice]]:
    blocks = []
    slices: dict[GroupLabel, slice] = {}
    start = 0
    for g, atoms in active_atoms:
        B = problem.group_blocks[g][:, np.asarray(atoms, dtype=int)]
        stop = start + len(atoms)
        slices[g] = slice(start, stop)
        blocks.append(B)
        start = stop
    if blocks:
        A = np.hstack(blocks)
    else:
        A = np.zeros((problem.y.size, 0), dtype=float)
    return np.asarray(A, dtype=float), slices


def _temporal_blocks(
    problem: LinearGroupMDLProblem,
    active_atoms: tuple[tuple[GroupLabel, tuple[int, ...]], ...],
) -> tuple[TemporalSupportBlock, ...]:
    E = [0 for _ in problem.temporal_candidate_counts]
    for g, atoms in active_atoms:
        if atoms:
            E[int(problem.temporal_block_of_group[g])] += 1
    return tuple(
        TemporalSupportBlock(M, e, label=("transition", t))
        for t, (M, e) in enumerate(zip(problem.temporal_candidate_counts, E))
    )


def _polynomial_objects(
    active_atoms: tuple[tuple[GroupLabel, tuple[int, ...]], ...],
) -> tuple[PolynomialObject, ...]:
    return tuple(
        PolynomialObject(tuple(atoms), label=g)
        for g, atoms in active_atoms
        if atoms
    )


def _mdl_lower_bound(
    problem: LinearGroupMDLProblem,
    active_atoms: tuple[tuple[GroupLabel, tuple[int, ...]], ...],
    *,
    q_min: int,
) -> float:
    # This deliberately asks only for the shortest *possible* code under the
    # current discrete state.  It does not assert q_min is numerically feasible.
    objects = _polynomial_objects(active_atoms)
    Q = sum(len(o.active_atoms) for o in objects)
    q_for_accounting = None if Q == 0 else int(q_min)
    score = compute_conditional_mdl(
        temporal_blocks=_temporal_blocks(problem, active_atoms),
        polynomial_objects=objects,
        library=problem.library,
        relative_residual=0.0,
        uncertainty_floor=problem.uncertainty_floor,
        q_star=q_for_accounting,
    )
    return float(score.total)


def build_search_state(
    problem: LinearGroupMDLProblem,
    active_atoms: Mapping[GroupLabel, Sequence[int]],
    *,
    quantizer: UniformDyadicQuantizer,
    q_min: int = 1,
    q_max: int = 64,
    warm_q: Optional[int] = None,
    warm_coefficients: Optional[FloatArray] = None,
    max_coordinate_passes: int = 8,
) -> MDLSearchState:
    """Fully profile one state for reference/scoring purposes."""

    active = _normalise_atom_map(problem, active_atoms)
    A, slices = _assemble_design(problem, active)
    continuous = profile_continuous(
        A,
        problem.y,
        uncertainty_floor=problem.uncertainty_floor,
        compute_basis=True,
    )
    if not continuous.feasible:
        raise ValueError(
            "The supplied initial state is outside the uncertainty floor: "
            f"rho={continuous.relative_residual:.6e} > "
            f"eps={problem.uncertainty_floor:.6e}."
        )

    precision = profile_precision_witness(
        continuous,
        quantizer=quantizer,
        q_min=q_min,
        q_max=q_max,
        warm_q=warm_q,
        warm_coefficients=warm_coefficients,
        max_coordinate_passes=max_coordinate_passes,
    )
    if precision.q_upper is None or precision.witness is None:
        raise ValueError(
            "No feasible quantized witness was found up to q_max. "
            "Increase q_max or change the explicit reference quantizer."
        )

    objects = _polynomial_objects(active)
    Q = sum(len(o.active_atoms) for o in objects)
    mdl_upper = compute_conditional_mdl(
        temporal_blocks=_temporal_blocks(problem, active),
        polynomial_objects=objects,
        library=problem.library,
        relative_residual=precision.witness.relative_residual,
        uncertainty_floor=problem.uncertainty_floor,
        q_star=None if Q == 0 else precision.q_upper,
    )
    lower = _mdl_lower_bound(problem, active, q_min=q_min)

    return MDLSearchState(
        active_atoms=active,
        design=A,
        coefficient_slices=slices,
        continuous=continuous,
        precision=precision,
        mdl_upper=mdl_upper,
        mdl_lower=lower,
    )


def _apply_move_to_atom_map(
    problem: LinearGroupMDLProblem,
    state: MDLSearchState,
    move: MDLMove,
) -> dict[GroupLabel, tuple[int, ...]]:
    amap = state.atom_map()
    full = tuple(range(problem.library.n_atoms))

    if move.kind == "drop_group":
        if move.group_out not in amap:
            raise ValueError("drop_group targets an inactive group.")
        amap.pop(move.group_out)
    elif move.kind == "add_group":
        if move.group_in in amap:
            raise ValueError("add_group targets an active group.")
        amap[move.group_in] = full
    elif move.kind == "swap_group":
        if move.group_out not in amap or move.group_in in amap:
            raise ValueError("invalid swap_group endpoints.")
        amap.pop(move.group_out)
        amap[move.group_in] = full
    elif move.kind == "drop_atom":
        g = move.group_out
        if g not in amap or move.atom_out not in amap[g]:
            raise ValueError("drop_atom targets an inactive atom.")
        atoms = tuple(a for a in amap[g] if a != move.atom_out)
        if atoms:
            amap[g] = atoms
        else:
            amap.pop(g)
    elif move.kind == "add_atom":
        g = move.group_in
        if g not in amap or move.atom_in in amap[g]:
            raise ValueError("add_atom requires an active group and inactive atom.")
        amap[g] = tuple(sorted(amap[g] + (int(move.atom_in),)))
    elif move.kind == "swap_atom":
        g = move.group_out
        if g not in amap or move.atom_out not in amap[g] or move.atom_in in amap[g]:
            raise ValueError("invalid swap_atom.")
        amap[g] = tuple(
            sorted((set(amap[g]) - {int(move.atom_out)}) | {int(move.atom_in)})
        )
    else:
        raise ValueError(f"unknown move kind {move.kind!r}.")
    return amap


def rank_group_adds(
    problem: LinearGroupMDLProblem,
    state: MDLSearchState,
    *,
    top_k: int = 8,
) -> list[MDLMove]:
    """Rank inactive groups by exact continuous conditional gain."""

    active = set(state.active_groups)
    scored = []
    for g in problem.groups:
        if g in active:
            continue
        new_rss, gain = exact_add_residual_sq(state.continuous, problem.group_blocks[g])
        scored.append((float(gain), g))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        MDLMove("add_group", group_in=g, proposal_score=gain)
        for gain, g in scored[: max(0, int(top_k))]
    ]


def rank_group_drops(
    problem: LinearGroupMDLProblem,
    state: MDLSearchState,
    *,
    top_k: int = 8,
) -> list[MDLMove]:
    """Rank active groups by a cheap deletion-witness damage score.

    The score is only a proposal heuristic.  It sets the group's current
    coefficients to zero without refitting the remaining groups; the exact
    continuous deletion profile is computed only for shortlisted moves.
    """

    scored = []
    r = state.continuous.residual
    c = state.continuous.coefficients
    for g in state.active_groups:
        sl = state.coefficient_slices[g]
        B = state.design[:, sl]
        contribution = B @ c[sl]
        witness_r = r + contribution
        damage = float(witness_r @ witness_r - state.continuous.residual_sq)
        scored.append((damage, g))
    scored.sort(key=lambda x: x[0])
    return [
        MDLMove("drop_group", group_out=g, proposal_score=damage)
        for damage, g in scored[: max(0, int(top_k))]
    ]


def rank_atom_drops(
    problem: LinearGroupMDLProblem,
    state: MDLSearchState,
    *,
    top_k: int = 8,
) -> list[MDLMove]:
    """Rank atom deletions by cheap zero-without-refit damage."""

    scored = []
    r = state.continuous.residual
    c = state.continuous.coefficients
    for g, atoms in state.active_atoms:
        sl = state.coefficient_slices[g]
        for local_j, atom in enumerate(atoms):
            col = state.design[:, sl.start + local_j]
            contribution = col * c[sl.start + local_j]
            witness_r = r + contribution
            damage = float(witness_r @ witness_r - state.continuous.residual_sq)
            # Tie-break toward higher degrees for compression proposals.
            degree = problem.library.atom_degrees[atom]
            scored.append((damage, -degree, g, atom))
    scored.sort(key=lambda x: (x[0], x[1]))
    return [
        MDLMove("drop_atom", group_out=g, atom_out=atom, proposal_score=damage)
        for damage, _, g, atom in scored[: max(0, int(top_k))]
    ]


def make_group_swaps(
    drops: Sequence[MDLMove],
    adds: Sequence[MDLMove],
    *,
    top_k: int = 8,
) -> list[MDLMove]:
    pairs = []
    for d in drops:
        for a in adds:
            score = float((d.proposal_score or 0.0) - (a.proposal_score or 0.0))
            pairs.append((score, d.group_out, a.group_in))
    pairs.sort(key=lambda x: x[0])
    return [
        MDLMove(
            "swap_group",
            group_out=gout,
            group_in=gin,
            proposal_score=score,
        )
        for score, gout, gin in pairs[: max(0, int(top_k))]
    ]


def evaluate_move(
    problem: LinearGroupMDLProblem,
    state: MDLSearchState,
    move: MDLMove,
    *,
    quantizer: UniformDyadicQuantizer,
    q_min: int = 1,
    q_max: int = 64,
    max_coordinate_passes: int = 8,
) -> MoveEvaluation:
    """Evaluate one shortlisted move through progressively more expensive tests."""

    new_map = _apply_move_to_atom_map(problem, state, move)
    active = _normalise_atom_map(problem, new_map)
    lb = _mdl_lower_bound(problem, active, q_min=q_min)

    # A strong safe rejection: even the candidate's impossible-best code is no
    # shorter than a known feasible code for the current model.
    if lb >= state.mdl_upper.total:
        return MoveEvaluation(
            move=move,
            status="rejected_by_discrete_lower_bound",
            continuous_relative_residual=None,
            precision_q_upper=None,
            mdl_lower=lb,
            mdl_upper=np.inf,
            delta_upper_vs_current_upper=None,
            certified_reject=True,
            certified_improve=False,
            new_state=None,
        )

    A, slices = _assemble_design(problem, active)
    continuous = profile_continuous(
        A,
        problem.y,
        uncertainty_floor=problem.uncertainty_floor,
        compute_basis=True,
    )
    if not continuous.feasible:
        return MoveEvaluation(
            move=move,
            status="rejected_by_continuous_floor",
            continuous_relative_residual=continuous.relative_residual,
            precision_q_upper=None,
            mdl_lower=lb,
            mdl_upper=np.inf,
            delta_upper_vs_current_upper=None,
            certified_reject=True,
            certified_improve=False,
            new_state=None,
        )

    # Warm q is useful even if dimensions changed; warm coefficients are only
    # passed when the dimensions match.
    warm_coeff = None
    if continuous.n_parameters == state.continuous.n_parameters:
        warm_coeff = state.precision.witness.coefficients if state.precision.witness else None
    precision = profile_precision_witness(
        continuous,
        quantizer=quantizer,
        q_min=q_min,
        q_max=q_max,
        warm_q=state.precision.q_upper,
        warm_coefficients=warm_coeff,
        max_coordinate_passes=max_coordinate_passes,
    )
    if precision.q_upper is None or precision.witness is None:
        return MoveEvaluation(
            move=move,
            status="no_quantized_witness_within_q_max",
            continuous_relative_residual=continuous.relative_residual,
            precision_q_upper=None,
            mdl_lower=lb,
            mdl_upper=np.inf,
            delta_upper_vs_current_upper=None,
            certified_reject=False,
            certified_improve=False,
            new_state=None,
        )

    objects = _polynomial_objects(active)
    Q = sum(len(o.active_atoms) for o in objects)
    mdl_upper = compute_conditional_mdl(
        temporal_blocks=_temporal_blocks(problem, active),
        polynomial_objects=objects,
        library=problem.library,
        relative_residual=precision.witness.relative_residual,
        uncertainty_floor=problem.uncertainty_floor,
        q_star=None if Q == 0 else precision.q_upper,
    )

    new_state = MDLSearchState(
        active_atoms=active,
        design=A,
        coefficient_slices=slices,
        continuous=continuous,
        precision=precision,
        mdl_upper=mdl_upper,
        mdl_lower=lb,
    )

    # Because both current and candidate q values are only witness upper bounds,
    # candidate_upper < current_upper is a ranking result, not a proof.  A truly
    # certified improvement would require candidate_upper < current_lower.
    certified_improve = bool(mdl_upper.total < state.mdl_lower)
    delta_upper = float(mdl_upper.total - state.mdl_upper.total)
    status = "certified_improvement" if certified_improve else "profiled_uncertified"

    return MoveEvaluation(
        move=move,
        status=status,
        continuous_relative_residual=continuous.relative_residual,
        precision_q_upper=precision.q_upper,
        mdl_lower=lb,
        mdl_upper=float(mdl_upper.total),
        delta_upper_vs_current_upper=delta_upper,
        certified_reject=False,
        certified_improve=certified_improve,
        new_state=new_state,
    )


def run_one_reference_sweep(
    problem: LinearGroupMDLProblem,
    state: MDLSearchState,
    *,
    quantizer: UniformDyadicQuantizer,
    top_group_drops: int = 6,
    top_group_adds: int = 6,
    top_group_swaps: int = 6,
    top_atom_drops: int = 6,
    q_min: int = 1,
    q_max: int = 64,
    max_coordinate_passes: int = 8,
) -> tuple[list[MDLMove], list[MoveEvaluation]]:
    """Generate a small proposal set and profile it once.

    This is the intended first smoke-test entry point.  It does not mutate the
    state.  Callers can inspect which proposals were pruned, which crossed the
    floor, and which have the best witness-MDL upper bounds.
    """

    drops = rank_group_drops(problem, state, top_k=top_group_drops)
    adds = rank_group_adds(problem, state, top_k=top_group_adds)
    swaps = make_group_swaps(drops, adds, top_k=top_group_swaps)
    atom_drops = rank_atom_drops(problem, state, top_k=top_atom_drops)
    proposals = list(drops) + list(adds) + list(swaps) + list(atom_drops)

    evaluations = [
        evaluate_move(
            problem,
            state,
            move,
            quantizer=quantizer,
            q_min=q_min,
            q_max=q_max,
            max_coordinate_passes=max_coordinate_passes,
        )
        for move in proposals
    ]
    evaluations.sort(
        key=lambda e: (
            np.inf if e.delta_upper_vs_current_upper is None else e.delta_upper_vs_current_upper,
            e.move.label(),
        )
    )
    return proposals, evaluations


# -----------------------------------------------------------------------------
# Hypothesis-specific branch: H = fixed dynamics / varying structure
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class FixedDynamicsHypothesis:
    """Side-information declaration used by the conditional MDL search.

    The current branch means

        B_a = w_a theta^T,

    for every stationary edge block and every active temporal-change block.
    Thus the polynomial law ``theta`` is shared globally while scalar structural
    amplitudes may vary.  The stationary anchor is retained as a dense nuisance
    amplitude vector, matching the current Step-2 philosophy that temporal
    support rather than baseline topology is the object being selected.

    Gauge: the lowest-index active polynomial atom is fixed to +1.  Only the
    remaining law coefficients and all scalar amplitudes are charged by the
    precision code.
    """

    name: str = "fixed_dynamics"
    varying_structure: bool = True
    shared_polynomial_law: bool = True
    stationary_anchor_mode: str = "dense_free_amplitudes"
    gauge: str = "first_active_atom_unit"

    def __post_init__(self) -> None:
        if self.name != "fixed_dynamics":
            raise ValueError("FixedDynamicsHypothesis.name must be 'fixed_dynamics'.")
        if self.stationary_anchor_mode != "dense_free_amplitudes":
            raise ValueError(
                "The reference fixed-dynamics branch currently supports only "
                "stationary_anchor_mode='dense_free_amplitudes'."
            )
        if self.gauge != "first_active_atom_unit":
            raise ValueError(
                "The reference fixed-dynamics branch currently supports only "
                "gauge='first_active_atom_unit'."
            )


@dataclass(frozen=True)
class FixedDynamicsMDLProblem:
    """Unprojected Step-2 problem under H = fixed dynamics.

    ``baseline_blocks`` contains one raw node-space polynomial block per
    candidate edge for the stationary anchor.  ``change_blocks`` contains one
    raw cumulative temporal block for every candidate (transition, edge) group.
    All blocks must have shape ``(n_rows, library.n_atoms)``.

    Unlike the unrestricted Step-2 B-space adapter, this problem uses raw
    observations rather than ``y_perp`` because projecting out an unrestricted
    baseline would destroy the global shared-law constraint that defines H.
    """

    y: FloatArray
    baseline_blocks: tuple[FloatArray, ...]
    change_blocks: Mapping[GroupLabel, FloatArray]
    temporal_block_of_group: Mapping[GroupLabel, int]
    temporal_candidate_counts: tuple[int, ...]
    library: PolynomialLibrary
    uncertainty_floor: float
    hypothesis: FixedDynamicsHypothesis = FixedDynamicsHypothesis()

    def __post_init__(self) -> None:
        y = np.asarray(self.y, dtype=float).reshape(-1)
        if y.size == 0:
            raise ValueError("y must be non-empty.")
        L = self.library.n_atoms
        baseline = tuple(np.asarray(B, dtype=float) for B in self.baseline_blocks)
        if not baseline:
            raise ValueError("baseline_blocks must be non-empty.")
        for i, B in enumerate(baseline):
            if B.shape != (y.size, L):
                raise ValueError(
                    f"baseline block {i} has shape {B.shape}; expected {(y.size, L)}."
                )
        changes = {}
        for g, B in self.change_blocks.items():
            B = np.asarray(B, dtype=float)
            if B.shape != (y.size, L):
                raise ValueError(
                    f"change group {g!r} has shape {B.shape}; expected {(y.size, L)}."
                )
            if g not in self.temporal_block_of_group:
                raise ValueError(f"change group {g!r} lacks a temporal block assignment.")
            changes[g] = B
        counts = tuple(int(x) for x in self.temporal_candidate_counts)
        if any(x < 0 for x in counts):
            raise ValueError("temporal candidate counts must be non-negative.")
        for g in changes:
            t = int(self.temporal_block_of_group[g])
            if t < 0 or t >= len(counts):
                raise ValueError(f"invalid temporal block index {t} for group {g!r}.")
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "baseline_blocks", baseline)
        object.__setattr__(self, "change_blocks", changes)
        object.__setattr__(self, "temporal_candidate_counts", counts)

    @property
    def groups(self) -> tuple[GroupLabel, ...]:
        return tuple(self.change_blocks.keys())


@dataclass(frozen=True)
class FixedDynamicsSearchState:
    active_groups: tuple[GroupLabel, ...]
    active_atoms: tuple[int, ...]
    continuous: FixedDynamicsProfile
    precision: FixedDynamicsPrecisionProfile
    mdl_upper: MDLScore
    mdl_lower: float


@dataclass(frozen=True)
class FixedDynamicsMoveEvaluation:
    move: MDLMove
    status: str
    continuous_relative_residual: Optional[float]
    relaxed_relative_residual: Optional[float]
    precision_q_upper: Optional[int]
    mdl_lower: float
    mdl_upper: float
    delta_upper_vs_current_upper: Optional[float]
    certified_reject: bool
    certified_improve: bool
    new_state: Optional[FixedDynamicsSearchState]


def _fd_normalise_groups(
    problem: FixedDynamicsMDLProblem,
    groups: Sequence[GroupLabel],
) -> tuple[GroupLabel, ...]:
    order = {g: i for i, g in enumerate(problem.groups)}
    gs = tuple(dict.fromkeys(groups))
    for g in gs:
        if g not in problem.change_blocks:
            raise ValueError(f"unknown change group {g!r}.")
    return tuple(sorted(gs, key=lambda g: order[g]))


def _fd_normalise_atoms(
    problem: FixedDynamicsMDLProblem,
    atoms: Sequence[int],
) -> tuple[int, ...]:
    out = tuple(sorted(set(int(a) for a in atoms)))
    if not out:
        raise ValueError("fixed_dynamics requires at least one active polynomial atom.")
    if any(a < 0 or a >= problem.library.n_atoms for a in out):
        raise ValueError("active atom index out of range.")
    return out


def _fd_change_blocks(
    problem: FixedDynamicsMDLProblem,
    groups: Sequence[GroupLabel],
) -> tuple[FloatArray, ...]:
    return tuple(problem.change_blocks[g] for g in groups)


def _fd_temporal_blocks(
    problem: FixedDynamicsMDLProblem,
    groups: Sequence[GroupLabel],
) -> tuple[TemporalSupportBlock, ...]:
    E = [0 for _ in problem.temporal_candidate_counts]
    for g in groups:
        E[int(problem.temporal_block_of_group[g])] += 1
    return tuple(
        TemporalSupportBlock(M, e, label=("transition", t))
        for t, (M, e) in enumerate(zip(problem.temporal_candidate_counts, E))
    )


def _fd_polynomial_objects(atoms: Sequence[int]) -> tuple[PolynomialObject, ...]:
    # H declares exactly one independent interaction law.
    return (PolynomialObject(tuple(atoms), label="shared_dynamics"),)


def _fd_Q(
    problem: FixedDynamicsMDLProblem,
    groups: Sequence[GroupLabel],
    atoms: Sequence[int],
) -> int:
    # One scalar for every dense stationary-anchor edge, one scalar for every
    # active temporal change, and |J|-1 free shared-law coefficients after gauge.
    return int(len(problem.baseline_blocks) + len(tuple(groups)) + len(tuple(atoms)) - 1)


def _fd_mdl_lower_bound(
    problem: FixedDynamicsMDLProblem,
    groups: Sequence[GroupLabel],
    atoms: Sequence[int],
    *,
    q_min: int,
) -> float:
    Q = _fd_Q(problem, groups, atoms)
    score = compute_conditional_mdl(
        temporal_blocks=_fd_temporal_blocks(problem, groups),
        polynomial_objects=_fd_polynomial_objects(atoms),
        library=problem.library,
        relative_residual=0.0,
        uncertainty_floor=problem.uncertainty_floor,
        q_star=None if Q == 0 else int(q_min),
        precision_parameter_count=Q,
    )
    return float(score.total)


def profile_fixed_dynamics_candidate(
    problem: FixedDynamicsMDLProblem,
    active_groups: Sequence[GroupLabel],
    active_atoms: Sequence[int],
    *,
    warm_theta: Optional[FloatArray] = None,
    max_nfev: int = 250,
    compute_linear_relaxation: bool = True,
    include_default_start: bool = True,
) -> FixedDynamicsProfile:
    groups = _fd_normalise_groups(problem, active_groups)
    atoms = _fd_normalise_atoms(problem, active_atoms)
    return profile_fixed_dynamics(
        problem.y,
        problem.baseline_blocks,
        _fd_change_blocks(problem, groups),
        active_atoms=atoms,
        uncertainty_floor=problem.uncertainty_floor,
        warm_theta=warm_theta,
        max_nfev=max_nfev,
        compute_linear_relaxation=compute_linear_relaxation,
        include_default_start=include_default_start,
    )


def build_fixed_dynamics_state(
    problem: FixedDynamicsMDLProblem,
    active_groups: Sequence[GroupLabel],
    active_atoms: Sequence[int],
    *,
    quantizer: UniformDyadicQuantizer,
    q_min: int = 1,
    q_max: int = 64,
    warm_theta: Optional[FloatArray] = None,
    warm_q: Optional[int] = None,
    max_nfev: int = 250,
    max_coordinate_passes: int = 8,
) -> FixedDynamicsSearchState:
    """Build a *feasible* MDL state under the fixed-dynamics hypothesis."""

    groups = _fd_normalise_groups(problem, active_groups)
    atoms = _fd_normalise_atoms(problem, active_atoms)
    continuous = profile_fixed_dynamics_candidate(
        problem,
        groups,
        atoms,
        warm_theta=warm_theta,
        max_nfev=max_nfev,
        compute_linear_relaxation=True,
    )
    if not continuous.feasible_witness:
        why = (
            "linear relaxation proves infeasible"
            if continuous.relaxation_proves_infeasible
            else "no feasible shared-law witness was found"
        )
        raise ValueError(
            f"Initial fixed-dynamics state is not usable ({why}): "
            f"rho_shared={continuous.relative_residual:.6e}, "
            f"rho_relaxed={continuous.relaxed_relative_residual:.6e}, "
            f"eps={problem.uncertainty_floor:.6e}."
        )

    changes = _fd_change_blocks(problem, groups)
    precision = profile_fixed_dynamics_precision_witness(
        continuous,
        problem.baseline_blocks,
        changes,
        quantizer=quantizer,
        q_min=q_min,
        q_max=q_max,
        warm_q=warm_q,
        max_coordinate_passes=max_coordinate_passes,
    )
    if precision.q_upper is None or precision.witness is None:
        raise ValueError("No feasible fixed-dynamics quantized witness found up to q_max.")

    Q = _fd_Q(problem, groups, atoms)
    mdl_upper = compute_conditional_mdl(
        temporal_blocks=_fd_temporal_blocks(problem, groups),
        polynomial_objects=_fd_polynomial_objects(atoms),
        library=problem.library,
        relative_residual=precision.witness.relative_residual,
        uncertainty_floor=problem.uncertainty_floor,
        q_star=precision.q_upper,
        precision_parameter_count=Q,
    )
    lower = _fd_mdl_lower_bound(problem, groups, atoms, q_min=q_min)
    return FixedDynamicsSearchState(
        active_groups=groups,
        active_atoms=atoms,
        continuous=continuous,
        precision=precision,
        mdl_upper=mdl_upper,
        mdl_lower=lower,
    )


def _fd_apply_move(
    problem: FixedDynamicsMDLProblem,
    state: FixedDynamicsSearchState,
    move: MDLMove,
) -> tuple[tuple[GroupLabel, ...], tuple[int, ...]]:
    groups = list(state.active_groups)
    atoms = list(state.active_atoms)
    if move.kind == "drop_group":
        if move.group_out not in groups:
            raise ValueError("drop_group targets an inactive group.")
        groups.remove(move.group_out)
    elif move.kind == "add_group":
        if move.group_in in groups:
            raise ValueError("add_group targets an active group.")
        groups.append(move.group_in)
    elif move.kind == "swap_group":
        if move.group_out not in groups or move.group_in in groups:
            raise ValueError("invalid swap_group endpoints.")
        groups.remove(move.group_out)
        groups.append(move.group_in)
    elif move.kind == "drop_atom":
        if move.atom_out not in atoms or len(atoms) <= 1:
            raise ValueError("drop_atom targets an inactive atom or would empty J.")
        atoms.remove(int(move.atom_out))
    elif move.kind == "add_atom":
        if move.atom_in in atoms:
            raise ValueError("add_atom targets an active atom.")
        atoms.append(int(move.atom_in))
    elif move.kind == "swap_atom":
        if move.atom_out not in atoms or move.atom_in in atoms:
            raise ValueError("invalid swap_atom.")
        atoms.remove(int(move.atom_out))
        atoms.append(int(move.atom_in))
    else:
        raise ValueError(f"unknown move kind {move.kind!r}.")
    return _fd_normalise_groups(problem, groups), _fd_normalise_atoms(problem, atoms)


def rank_fixed_dynamics_group_adds(
    problem: FixedDynamicsMDLProblem,
    state: FixedDynamicsSearchState,
    *,
    top_k: int = 8,
) -> list[MDLMove]:
    """Proposal ranking by conditional gain with the current shared law fixed."""

    theta = state.continuous.theta
    current_blocks = problem.baseline_blocks + _fd_change_blocks(problem, state.active_groups)
    C = np.column_stack([B @ theta for B in current_blocks])
    # Rank-revealing orthonormal basis for the current amplitude span.
    U, s, _ = np.linalg.svd(C, full_matrices=False)
    if s.size:
        tol = max(C.shape) * np.finfo(float).eps * float(s[0])
        Qb = U[:, s > tol]
    else:
        Qb = np.zeros((problem.y.size, 0), dtype=float)
    r = state.continuous.residual
    active = set(state.active_groups)
    scored = []
    for g in problem.groups:
        if g in active:
            continue
        v = problem.change_blocks[g] @ theta
        z = v - Qb @ (Qb.T @ v) if Qb.shape[1] else v
        zz = float(z @ z)
        gain = 0.0 if zz <= np.finfo(float).tiny else float((r @ z) ** 2 / zz)
        scored.append((gain, g))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        MDLMove("add_group", group_in=g, proposal_score=float(gain))
        for gain, g in scored[: max(0, int(top_k))]
    ]


def rank_fixed_dynamics_group_drops(
    problem: FixedDynamicsMDLProblem,
    state: FixedDynamicsSearchState,
    *,
    top_k: int = 8,
) -> list[MDLMove]:
    """Proposal ranking by zero-without-refit damage at the current shared law."""

    n0 = len(problem.baseline_blocks)
    theta = state.continuous.theta
    r = state.continuous.residual
    scored = []
    for j, g in enumerate(state.active_groups):
        a = float(state.continuous.amplitudes[n0 + j])
        contribution = a * (problem.change_blocks[g] @ theta)
        rw = r + contribution
        damage = float(rw @ rw - state.continuous.residual_sq)
        scored.append((damage, g))
    scored.sort(key=lambda x: x[0])
    return [
        MDLMove("drop_group", group_out=g, proposal_score=float(damage))
        for damage, g in scored[: max(0, int(top_k))]
    ]


def rank_fixed_dynamics_atom_drops(
    problem: FixedDynamicsMDLProblem,
    state: FixedDynamicsSearchState,
    *,
    top_k: int = 8,
) -> list[MDLMove]:
    """Prefer small shared-law coefficients, breaking ties toward high degree."""

    if len(state.active_atoms) <= 1:
        return []
    scored = []
    for atom in state.active_atoms:
        # Keep the current gauge pivot out of routine compression proposals.
        if atom == state.continuous.pivot_atom:
            continue
        mag = abs(float(state.continuous.theta[atom]))
        degree = int(problem.library.atom_degrees[atom])
        scored.append((mag, -degree, atom))
    scored.sort(key=lambda x: (x[0], x[1]))
    return [
        MDLMove("drop_atom", atom_out=atom, proposal_score=float(mag))
        for mag, _, atom in scored[: max(0, int(top_k))]
    ]


def rank_fixed_dynamics_atom_adds(
    problem: FixedDynamicsMDLProblem,
    state: FixedDynamicsSearchState,
    *,
    top_k: int = 8,
) -> list[MDLMove]:
    """Hierarchy-biased proposal: lower absent degrees are offered first."""

    active = set(state.active_atoms)
    candidates = [a for a in range(problem.library.n_atoms) if a not in active]
    candidates.sort(key=lambda a: (problem.library.atom_degrees[a], a))
    return [
        MDLMove(
            "add_atom",
            atom_in=a,
            proposal_score=float(problem.library.atom_degrees[a]),
        )
        for a in candidates[: max(0, int(top_k))]
    ]


def evaluate_fixed_dynamics_move(
    problem: FixedDynamicsMDLProblem,
    state: FixedDynamicsSearchState,
    move: MDLMove,
    *,
    quantizer: UniformDyadicQuantizer,
    q_min: int = 1,
    q_max: int = 64,
    max_nfev: int = 250,
    max_coordinate_passes: int = 8,
) -> FixedDynamicsMoveEvaluation:
    """Staged evaluator for one H=fixed_dynamics proposal."""

    groups, atoms = _fd_apply_move(problem, state, move)
    lb = _fd_mdl_lower_bound(problem, groups, atoms, q_min=q_min)
    if lb >= state.mdl_upper.total:
        return FixedDynamicsMoveEvaluation(
            move, "rejected_by_discrete_lower_bound", None, None, None,
            lb, np.inf, None, True, False, None,
        )

    continuous = profile_fixed_dynamics_candidate(
        problem,
        groups,
        atoms,
        warm_theta=state.continuous.theta,
        max_nfev=max_nfev,
        compute_linear_relaxation=True,
    )
    if continuous.relaxation_proves_infeasible:
        return FixedDynamicsMoveEvaluation(
            move,
            "rejected_by_linear_relaxation_floor",
            continuous.relative_residual,
            continuous.relaxed_relative_residual,
            None,
            lb,
            np.inf,
            None,
            True,
            False,
            None,
        )
    if not continuous.feasible_witness:
        # Nonconvex shared-law profiling did not find a floor-feasible point.
        # This is deliberately NOT marked as a certified rejection.
        return FixedDynamicsMoveEvaluation(
            move,
            "no_shared_law_feasible_witness_found",
            continuous.relative_residual,
            continuous.relaxed_relative_residual,
            None,
            lb,
            np.inf,
            None,
            False,
            False,
            None,
        )

    changes = _fd_change_blocks(problem, groups)
    precision = profile_fixed_dynamics_precision_witness(
        continuous,
        problem.baseline_blocks,
        changes,
        quantizer=quantizer,
        q_min=q_min,
        q_max=q_max,
        warm_q=state.precision.q_upper,
        max_coordinate_passes=max_coordinate_passes,
    )
    if precision.q_upper is None or precision.witness is None:
        return FixedDynamicsMoveEvaluation(
            move,
            "no_quantized_witness_within_q_max",
            continuous.relative_residual,
            continuous.relaxed_relative_residual,
            None,
            lb,
            np.inf,
            None,
            False,
            False,
            None,
        )

    Q = _fd_Q(problem, groups, atoms)
    mdl_upper = compute_conditional_mdl(
        temporal_blocks=_fd_temporal_blocks(problem, groups),
        polynomial_objects=_fd_polynomial_objects(atoms),
        library=problem.library,
        relative_residual=precision.witness.relative_residual,
        uncertainty_floor=problem.uncertainty_floor,
        q_star=precision.q_upper,
        precision_parameter_count=Q,
    )
    new_state = FixedDynamicsSearchState(
        active_groups=groups,
        active_atoms=atoms,
        continuous=continuous,
        precision=precision,
        mdl_upper=mdl_upper,
        mdl_lower=lb,
    )
    certified_improve = bool(mdl_upper.total < state.mdl_lower)
    delta_upper = float(mdl_upper.total - state.mdl_upper.total)
    return FixedDynamicsMoveEvaluation(
        move=move,
        status=("certified_improvement" if certified_improve else "profiled_uncertified"),
        continuous_relative_residual=continuous.relative_residual,
        relaxed_relative_residual=continuous.relaxed_relative_residual,
        precision_q_upper=precision.q_upper,
        mdl_lower=lb,
        mdl_upper=float(mdl_upper.total),
        delta_upper_vs_current_upper=delta_upper,
        certified_reject=False,
        certified_improve=certified_improve,
        new_state=new_state,
    )



@dataclass(frozen=True)
class SharedLawBasinHopStep:
    """One basin-to-basin proposal in the profiled shared-law landscape."""

    iteration: int
    block_size: int
    sigma: float
    perturbed_atoms: tuple[int, ...]
    raw_theta: FloatArray
    relaxed_theta: FloatArray
    proposed_relative_residual: float
    accepted: bool
    best_relative_residual: float
    optimizer_nfev: int


@dataclass(frozen=True)
class SharedLawBasinSearchResult:
    """Outcome of monotone multiscale block basin hopping."""

    initial_profile: FixedDynamicsProfile
    best_profile: FixedDynamicsProfile
    history: tuple[SharedLawBasinHopStep, ...]
    reached_floor: bool
    seed: Optional[int]

    @property
    def n_hops(self) -> int:
        return len(self.history)

    @property
    def n_accepted(self) -> int:
        return sum(int(h.accepted) for h in self.history)


def discover_fixed_dynamics_shared_law(
    problem: FixedDynamicsMDLProblem,
    *,
    active_groups: Optional[Sequence[GroupLabel]] = None,
    active_atoms: Optional[Sequence[int]] = None,
    n_hops: int = 80,
    sigma_min: float = 5e-3,
    sigma_max: float = 1.5e-1,
    min_block_size: int = 1,
    max_block_size: Optional[int] = None,
    small_block_probability: float = 0.60,
    low_degree_bias_power: float = 1.0,
    local_max_nfev: int = 50,
    seed: Optional[int] = None,
    initial_theta: Optional[FloatArray] = None,
    improvement_rtol: float = 1e-12,
) -> SharedLawBasinSearchResult:
    """Discover a floor-feasible shared law with one simple global idea.

    The structural support and polynomial library are held fixed during this
    phase.  At each iteration we (i) draw one random block of free shared-law
    coefficients (small and low-degree blocks are proposed more often), (ii)
    perturb that block with one log-uniform step scale, (iii) locally relax the
    raw proposal by variable projection, and (iv) keep
    the resulting basin only when its profiled residual is lower.

    There is no annealing, parallel tempering, HMC, or population method here.
    Escape from a metastable basin is provided only by the nonlocal raw jump.
    The analytic variable-projection Jacobian is a numerical acceleration, not
    an additional optimization principle.
    """

    groups = _fd_normalise_groups(
        problem, problem.groups if active_groups is None else active_groups
    )
    atoms = _fd_normalise_atoms(
        problem,
        tuple(range(problem.library.n_atoms))
        if active_atoms is None
        else active_atoms,
    )
    free_atoms = tuple(a for a in atoms if a != atoms[0])
    if n_hops < 0:
        raise ValueError("n_hops must be non-negative.")
    if not (0.0 < sigma_min <= sigma_max):
        raise ValueError("Require 0 < sigma_min <= sigma_max.")

    workspace = build_fixed_dynamics_varpro_workspace(
        problem.y,
        problem.baseline_blocks,
        _fd_change_blocks(problem, groups),
        active_atoms=atoms,
        uncertainty_floor=problem.uncertainty_floor,
    )

    initial = profile_fixed_dynamics_varpro(
        workspace,
        warm_theta=initial_theta,
        max_nfev=local_max_nfev,
        include_default_start=True,
    )
    if initial.feasible_witness or not free_atoms:
        return SharedLawBasinSearchResult(
            initial, initial, tuple(), bool(initial.feasible_witness), seed
        )

    d = len(free_atoms)
    kmin = max(1, int(min_block_size))
    kmax = d if max_block_size is None else min(d, int(max_block_size))
    if kmin > kmax:
        raise ValueError("Invalid block-size range.")
    p_small = float(small_block_probability)
    if not (0.0 < p_small <= 1.0):
        raise ValueError("small_block_probability must lie in (0,1].")
    degree_power = float(low_degree_bias_power)
    if degree_power < 0.0 or not np.isfinite(degree_power):
        raise ValueError("low_degree_bias_power must be finite and non-negative.")

    rng = np.random.default_rng(seed)
    best = initial
    history: list[SharedLawBasinHopStep] = []
    log_lo = np.log(float(sigma_min))
    log_hi = np.log(float(sigma_max))
    free_arr = np.asarray(free_atoms, dtype=int)
    degrees = np.asarray(
        [problem.library.atom_degrees[a] for a in free_atoms], dtype=float
    )
    atom_prob = degrees ** (-degree_power)
    atom_prob = atom_prob / atom_prob.sum()

    for it in range(int(n_hops)):
        # Geometric block size: single/small groups dominate, while the tail
        # still gives nonzero probability to genuinely nonlocal group jumps.
        k = kmin - 1 + int(rng.geometric(p_small))
        k = min(max(k, kmin), kmax)
        chosen = tuple(
            int(x)
            for x in rng.choice(
                free_arr, size=k, replace=False, p=atom_prob
            )
        )
        sigma = float(np.exp(rng.uniform(log_lo, log_hi)))
        raw = np.asarray(best.theta, dtype=float).copy()
        raw[np.asarray(chosen, dtype=int)] += sigma * rng.normal(size=k)
        raw[best.pivot_atom] = 1.0

        candidate = profile_fixed_dynamics_varpro(
            workspace,
            warm_theta=raw,
            max_nfev=local_max_nfev,
            include_default_start=False,
        )
        threshold = best.relative_residual * (1.0 - float(improvement_rtol))
        accepted = bool(candidate.relative_residual < threshold)
        if accepted:
            best = candidate

        history.append(
            SharedLawBasinHopStep(
                iteration=it,
                block_size=k,
                sigma=sigma,
                perturbed_atoms=tuple(sorted(chosen)),
                raw_theta=np.asarray(raw, dtype=float),
                relaxed_theta=np.asarray(candidate.theta, dtype=float),
                proposed_relative_residual=float(candidate.relative_residual),
                accepted=accepted,
                best_relative_residual=float(best.relative_residual),
                optimizer_nfev=int(candidate.optimizer_nfev),
            )
        )
        if best.feasible_witness:
            break

    return SharedLawBasinSearchResult(
        initial_profile=initial,
        best_profile=best,
        history=tuple(history),
        reached_floor=bool(best.feasible_witness),
        seed=seed,
    )

def run_one_fixed_dynamics_sweep(
    problem: FixedDynamicsMDLProblem,
    state: FixedDynamicsSearchState,
    *,
    quantizer: UniformDyadicQuantizer,
    top_group_drops: int = 4,
    top_group_adds: int = 4,
    top_group_swaps: int = 4,
    top_atom_drops: int = 4,
    top_atom_adds: int = 2,
    q_min: int = 1,
    q_max: int = 64,
    max_nfev: int = 250,
    max_coordinate_passes: int = 8,
) -> tuple[list[MDLMove], list[FixedDynamicsMoveEvaluation]]:
    drops = rank_fixed_dynamics_group_drops(problem, state, top_k=top_group_drops)
    adds = rank_fixed_dynamics_group_adds(problem, state, top_k=top_group_adds)
    swaps = make_group_swaps(drops, adds, top_k=top_group_swaps)
    atom_drops = rank_fixed_dynamics_atom_drops(problem, state, top_k=top_atom_drops)
    atom_adds = rank_fixed_dynamics_atom_adds(problem, state, top_k=top_atom_adds)
    proposals = list(drops) + list(adds) + list(swaps) + list(atom_drops) + list(atom_adds)
    evaluations = [
        evaluate_fixed_dynamics_move(
            problem,
            state,
            move,
            quantizer=quantizer,
            q_min=q_min,
            q_max=q_max,
            max_nfev=max_nfev,
            max_coordinate_passes=max_coordinate_passes,
        )
        for move in proposals
    ]
    return proposals, evaluations


__all__ = [
    "LinearGroupMDLProblem",
    "MDLMove",
    "MDLSearchState",
    "MoveEvaluation",
    "build_search_state",
    "rank_group_adds",
    "rank_group_drops",
    "rank_atom_drops",
    "make_group_swaps",
    "evaluate_move",
    "run_one_reference_sweep",
    "FixedDynamicsHypothesis",
    "FixedDynamicsMDLProblem",
    "FixedDynamicsSearchState",
    "FixedDynamicsMoveEvaluation",
    "profile_fixed_dynamics_candidate",
    "build_fixed_dynamics_state",
    "rank_fixed_dynamics_group_adds",
    "rank_fixed_dynamics_group_drops",
    "rank_fixed_dynamics_atom_drops",
    "rank_fixed_dynamics_atom_adds",
    "evaluate_fixed_dynamics_move",
    "run_one_fixed_dynamics_sweep",
    "SharedLawBasinHopStep",
    "SharedLawBasinSearchResult",
    "discover_fixed_dynamics_shared_law",
]
