"""Hypothesis-specific MDL search for TIDES Step 3.

This module navigates physical representation space after Step 2 has already
constructed the observational family B_epsilon.  The implemented branch is
H_FD (fixed dynamics, varying structure):

    B^(r) = W^(r) Theta^T.

For coding/search, W_(1:K) is represented through the invertible physical
coordinates W^(1), Delta W^(1),...,Delta W^(K-1).  Structural sparsity is not
an admissibility assumption; active supports are selected only because they may
shorten the final MDL code.

Search uses a dynamics-first, structure-guided joint-refinement strategy:
1. recover a floor-feasible shared-law anchor Theta by variable projection;
2. with Theta temporarily fixed, use the resulting linear network problem to
   screen large structural-support reductions cheaply;
3. jointly re-profile Theta and structural amplitudes on shortlisted supports,
   and accept only true MDL improvements;
4. polish structure locally, then compress the interaction expression and run
   a final joint profile.

Theta is therefore an anchor / warm start, never a permanently frozen law.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Hashable, Mapping, Optional, Sequence

import numpy as np

try:  # package form
    from .mdl_description_length import (
        MDLScore,
        PolynomialLibrary,
        InteractionLibraryCode,
        CandidateLibrary,
        PolynomialObject,
        StructuralSupportBlock,
        compute_conditional_mdl,
    )
    from .mdl_precision import (
        FixedDynamicsProfile,
        FixedDynamicsPrecisionProfile,
        FixedDynamicsVarProWorkspace,
        UniformDyadicQuantizer,
        build_fixed_dynamics_varpro_workspace,
        profile_fixed_dynamics,
        profile_fixed_dynamics_precision_witness,
        profile_fixed_dynamics_varpro,
    )
except ImportError:  # flat-file form
    from mdl_description_length import (
        MDLScore,
        PolynomialLibrary,
        InteractionLibraryCode,
        CandidateLibrary,
        PolynomialObject,
        StructuralSupportBlock,
        compute_conditional_mdl,
    )
    from mdl_precision import (
        FixedDynamicsProfile,
        FixedDynamicsPrecisionProfile,
        FixedDynamicsVarProWorkspace,
        UniformDyadicQuantizer,
        build_fixed_dynamics_varpro_workspace,
        profile_fixed_dynamics,
        profile_fixed_dynamics_precision_witness,
        profile_fixed_dynamics_varpro,
    )

FloatArray = np.ndarray
GroupLabel = Hashable


@dataclass(frozen=True)
class FixedDynamicsHypothesis:
    name: str = "fixed_dynamics"
    gauge: str = "first_active_atom_unit"

    def __post_init__(self) -> None:
        if self.name != "fixed_dynamics":
            raise ValueError("name must be 'fixed_dynamics'.")
        if self.gauge != "first_active_atom_unit":
            raise ValueError("Only gauge='first_active_atom_unit' is implemented.")


@dataclass(frozen=True)
class FixedDynamicsMDLProblem:
    """Step-3 H_FD problem in physical structural coordinates.

    Every entry of ``structural_blocks`` is an observation-space matrix with
    shape (n_rows, L).  Multiplying it by the shared law Theta gives the response
    direction associated with one scalar structural amplitude.

    ``structural_block_of_group`` assigns each scalar group to one encoded
    support block.  The canonical TIDES adapter uses block 0 for W^(1) and
    blocks 1,...,K-1 for Delta W^(1),...,Delta W^(K-1).
    """

    y: FloatArray
    structural_blocks: Mapping[GroupLabel, FloatArray]
    structural_block_of_group: Mapping[GroupLabel, int]
    structural_candidate_counts: tuple[int, ...]
    structural_block_labels: tuple[Hashable, ...]
    library: InteractionLibraryCode
    uncertainty_floor: float
    hypothesis: FixedDynamicsHypothesis = FixedDynamicsHypothesis()

    def __post_init__(self) -> None:
        y = np.asarray(self.y, dtype=float).reshape(-1)
        if y.size == 0 or not np.all(np.isfinite(y)):
            raise ValueError("y must be finite and non-empty.")
        if not self.structural_blocks:
            raise ValueError("structural_blocks must be non-empty.")
        L = self.library.n_atoms
        blocks = {}
        for g, B0 in self.structural_blocks.items():
            B = np.asarray(B0, dtype=float)
            if B.shape != (y.size, L):
                raise ValueError(f"group {g!r} has shape {B.shape}; expected {(y.size, L)}.")
            if not np.all(np.isfinite(B)):
                raise ValueError(f"group {g!r} contains non-finite values.")
            if g not in self.structural_block_of_group:
                raise ValueError(f"group {g!r} lacks a structural block assignment.")
            blocks[g] = B
        counts = tuple(int(x) for x in self.structural_candidate_counts)
        labels = tuple(self.structural_block_labels)
        if len(counts) != len(labels):
            raise ValueError("structural_candidate_counts and labels must have equal length.")
        if any(x < 0 for x in counts):
            raise ValueError("candidate counts must be non-negative.")
        for g in blocks:
            b = int(self.structural_block_of_group[g])
            if b < 0 or b >= len(counts):
                raise ValueError(f"group {g!r} has invalid structural block index {b}.")
        eps = float(self.uncertainty_floor)
        if not np.isfinite(eps) or eps < 0.0:
            raise ValueError("uncertainty_floor must be finite and non-negative.")
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "structural_blocks", blocks)
        object.__setattr__(self, "structural_candidate_counts", counts)
        object.__setattr__(self, "structural_block_labels", labels)
        object.__setattr__(self, "uncertainty_floor", eps)

    @property
    def groups(self) -> tuple[GroupLabel, ...]:
        return tuple(self.structural_blocks.keys())


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
            return f"drop_atom({self.atom_out})"
        if self.kind == "add_atom":
            return f"add_atom({self.atom_in})"
        if self.kind == "swap_atom":
            return f"swap_atom({self.atom_out}->{self.atom_in})"
        if self.kind == "drop_batch":
            try:
                n = len(self.group_out)  # type: ignore[arg-type]
            except Exception:
                n = "?"
            return f"drop_batch({n})"
        return self.kind


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


@dataclass(frozen=True)
class SharedLawBasinHopStep:
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


@dataclass(frozen=True)
class FixedThetaSupportScreen:
    """One support proposal screened with the current shared law held fixed."""

    active_groups: tuple[GroupLabel, ...]
    dropped_groups: tuple[GroupLabel, ...]
    drop_count: int
    fixed_theta_relative_residual: float
    fixed_theta_feasible: bool
    provisional_mdl_upper: float


@dataclass(frozen=True)
class FixedDynamicsCompressionStep:
    sweep: int
    move: MDLMove
    old_mdl_upper: float
    new_mdl_upper: float
    certified_improvement: bool
    phase: str = "joint"
    screened_relative_residual: Optional[float] = None
    theta_relative_change: Optional[float] = None
    n_exact_refits: int = 0


@dataclass(frozen=True)
class CandidateBranchSummary:
    """Auditable outcome of a pure-law seed; scores are in nats, as elsewhere."""

    seed_atom: int
    initial_relative_residual: float
    status: str
    initial_mdl: Optional[float] = None
    final_mdl: Optional[float] = None
    final_atoms: tuple[int, ...] = ()
    final_groups: tuple[GroupLabel, ...] = ()


@dataclass(frozen=True)
class FixedDynamicsCompressionResult:
    initial_state: FixedDynamicsSearchState
    best_state: FixedDynamicsSearchState
    basin_search: SharedLawBasinSearchResult
    history: tuple[FixedDynamicsCompressionStep, ...]
    sweeps_completed: int
    theta_anchor: Optional[FloatArray] = None
    structural_history: tuple[FixedDynamicsCompressionStep, ...] = ()
    expression_history: tuple[FixedDynamicsCompressionStep, ...] = ()
    outer_iterations: int = 0
    search_strategy: str = "legacy"
    candidate_branches: tuple[CandidateBranchSummary, ...] = ()


def _normalise_groups(problem: FixedDynamicsMDLProblem, groups: Sequence[GroupLabel]) -> tuple[GroupLabel, ...]:
    order = {g: i for i, g in enumerate(problem.groups)}
    out = tuple(dict.fromkeys(groups))
    for g in out:
        if g not in problem.structural_blocks:
            raise ValueError(f"unknown structural group {g!r}.")
    return tuple(sorted(out, key=lambda g: order[g]))


def _normalise_atoms(problem: FixedDynamicsMDLProblem, atoms: Sequence[int]) -> tuple[int, ...]:
    out = tuple(sorted(set(int(a) for a in atoms)))
    if not out:
        raise ValueError("H_FD requires at least one active interaction atom.")
    if any(a < 0 or a >= problem.library.n_atoms for a in out):
        raise ValueError("active atom index out of range.")
    return out


def _selected_blocks(problem: FixedDynamicsMDLProblem, groups: Sequence[GroupLabel]) -> tuple[FloatArray, ...]:
    return tuple(problem.structural_blocks[g] for g in groups)


def _structural_support_blocks(problem: FixedDynamicsMDLProblem, groups: Sequence[GroupLabel]) -> tuple[StructuralSupportBlock, ...]:
    E = [0 for _ in problem.structural_candidate_counts]
    for g in groups:
        E[int(problem.structural_block_of_group[g])] += 1
    return tuple(
        StructuralSupportBlock(M, e, label=label)
        for M, e, label in zip(problem.structural_candidate_counts, E, problem.structural_block_labels)
    )


def _polynomial_objects(atoms: Sequence[int]) -> tuple[PolynomialObject, ...]:
    return (PolynomialObject(tuple(atoms), label="shared_dynamics"),)


def _free_parameter_count(groups: Sequence[GroupLabel], atoms: Sequence[int]) -> int:
    return int(len(tuple(groups)) + len(tuple(atoms)) - 1)


def _mdl_lower_bound(problem: FixedDynamicsMDLProblem, groups: Sequence[GroupLabel], atoms: Sequence[int], *, q_min: int) -> float:
    Q = _free_parameter_count(groups, atoms)
    score = compute_conditional_mdl(
        structural_blocks=_structural_support_blocks(problem, groups),
        polynomial_objects=_polynomial_objects(atoms),
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
    groups = _normalise_groups(problem, active_groups)
    atoms = _normalise_atoms(problem, active_atoms)
    if not groups:
        # A zero-structure model cannot use the shared-law profiler.  Return a
        # synthetic profile carrying its exact residual.
        y = problem.y
        y_norm = float(np.linalg.norm(y))
        rss = float(y @ y)
        eps = problem.uncertainty_floor
        return FixedDynamicsProfile(
            active_atoms=atoms,
            pivot_atom=atoms[0],
            theta=np.eye(1, problem.library.n_atoms, atoms[0], dtype=float).reshape(-1),
            amplitudes=np.zeros(0),
            residual=y.copy(),
            residual_sq=rss,
            relative_residual=1.0,
            feasible_witness=bool(1.0 <= eps),
            relaxed_relative_residual=1.0,
            relaxation_proves_infeasible=bool(1.0 > eps),
            y_norm=y_norm,
            uncertainty_floor=eps,
            threshold_sq=float((eps * y_norm) ** 2),
            optimizer_nfev=0,
            optimizer_success=True,
        )
    return profile_fixed_dynamics(
        problem.y,
        _selected_blocks(problem, groups),
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
    groups = _normalise_groups(problem, active_groups)
    atoms = _normalise_atoms(problem, active_atoms)
    if not groups:
        raise ValueError("A feasible H_FD state currently requires at least one structural group.")
    continuous = profile_fixed_dynamics_candidate(
        problem,
        groups,
        atoms,
        warm_theta=warm_theta,
        max_nfev=max_nfev,
        compute_linear_relaxation=True,
    )
    if not continuous.feasible_witness:
        why = "linear relaxation proves infeasible" if continuous.relaxation_proves_infeasible else "no shared-law feasible witness found"
        raise ValueError(
            f"Initial H_FD state is not feasible ({why}): rho={continuous.relative_residual:.6e}, "
            f"rho_relaxed={continuous.relaxed_relative_residual:.6e}, eps={problem.uncertainty_floor:.6e}."
        )
    selected = _selected_blocks(problem, groups)
    precision = profile_fixed_dynamics_precision_witness(
        continuous,
        selected,
        quantizer=quantizer,
        q_min=q_min,
        q_max=q_max,
        warm_q=warm_q,
        max_coordinate_passes=max_coordinate_passes,
    )
    if precision.q_upper is None or precision.witness is None:
        raise ValueError("No feasible quantized H_FD witness found up to q_max.")
    Q = _free_parameter_count(groups, atoms)
    mdl = compute_conditional_mdl(
        structural_blocks=_structural_support_blocks(problem, groups),
        polynomial_objects=_polynomial_objects(atoms),
        library=problem.library,
        relative_residual=precision.witness.relative_residual,
        uncertainty_floor=problem.uncertainty_floor,
        q_star=precision.q_upper,
        precision_parameter_count=Q,
    )
    return FixedDynamicsSearchState(
        active_groups=groups,
        active_atoms=atoms,
        continuous=continuous,
        precision=precision,
        mdl_upper=mdl,
        mdl_lower=_mdl_lower_bound(problem, groups, atoms, q_min=q_min),
    )


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
    groups = _normalise_groups(problem, problem.groups if active_groups is None else active_groups)
    atoms = _normalise_atoms(problem, tuple(range(problem.library.n_atoms)) if active_atoms is None else active_atoms)
    if not groups:
        raise ValueError("Basin discovery requires at least one structural group.")
    free_atoms = tuple(a for a in atoms if a != atoms[0])
    if n_hops < 0 or not (0.0 < sigma_min <= sigma_max):
        raise ValueError("Invalid basin-hopping controls.")

    workspace = build_fixed_dynamics_varpro_workspace(
        problem.y,
        _selected_blocks(problem, groups),
        active_atoms=atoms,
        uncertainty_floor=problem.uncertainty_floor,
        compute_linear_relaxation=True,
    )
    initial = profile_fixed_dynamics_varpro(
        workspace,
        warm_theta=initial_theta,
        max_nfev=local_max_nfev,
        include_default_start=True,
    )
    if initial.feasible_witness or not free_atoms:
        return SharedLawBasinSearchResult(initial, initial, tuple(), bool(initial.feasible_witness), seed)

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
    log_lo, log_hi = np.log(float(sigma_min)), np.log(float(sigma_max))
    free_arr = np.asarray(free_atoms, dtype=int)
    degrees = np.asarray([problem.library.search_complexities[a] for a in free_atoms], dtype=float)
    probs = degrees ** (-degree_power)
    probs /= probs.sum()

    for it in range(int(n_hops)):
        k = kmin - 1 + int(rng.geometric(p_small))
        k = min(max(k, kmin), kmax)
        chosen = tuple(int(x) for x in rng.choice(free_arr, size=k, replace=False, p=probs))
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
                raw_theta=raw,
                relaxed_theta=np.asarray(candidate.theta, dtype=float),
                proposed_relative_residual=float(candidate.relative_residual),
                accepted=accepted,
                best_relative_residual=float(best.relative_residual),
                optimizer_nfev=int(candidate.optimizer_nfev),
            )
        )
        if best.feasible_witness:
            break

    return SharedLawBasinSearchResult(initial, best, tuple(history), bool(best.feasible_witness), seed)


def _apply_move(problem: FixedDynamicsMDLProblem, state: FixedDynamicsSearchState, move: MDLMove) -> tuple[tuple[GroupLabel, ...], tuple[int, ...]]:
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
            raise ValueError("drop_atom invalid or would empty the library support.")
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
    return _normalise_groups(problem, groups), _normalise_atoms(problem, atoms)


def rank_fixed_dynamics_group_drops(problem: FixedDynamicsMDLProblem, state: FixedDynamicsSearchState, *, top_k: int = 8) -> list[MDLMove]:
    r = state.continuous.residual
    theta = state.continuous.theta
    scored = []
    for j, g in enumerate(state.active_groups):
        a = float(state.continuous.amplitudes[j])
        contribution = a * (problem.structural_blocks[g] @ theta)
        rw = r + contribution
        damage = float(rw @ rw - state.continuous.residual_sq)
        scored.append((damage, g))
    scored.sort(key=lambda x: x[0])
    return [MDLMove("drop_group", group_out=g, proposal_score=float(d)) for d, g in scored[:max(0, int(top_k))]]


def rank_fixed_dynamics_group_adds(problem: FixedDynamicsMDLProblem, state: FixedDynamicsSearchState, *, top_k: int = 8) -> list[MDLMove]:
    theta = state.continuous.theta
    active = set(state.active_groups)
    selected = _selected_blocks(problem, state.active_groups)
    if selected:
        C = np.column_stack([B @ theta for B in selected])
        U, s, _ = np.linalg.svd(C, full_matrices=False)
        if s.size:
            tol = max(C.shape) * np.finfo(float).eps * float(s[0])
            Qb = U[:, s > tol]
        else:
            Qb = np.zeros((problem.y.size, 0))
    else:
        Qb = np.zeros((problem.y.size, 0))
    r = state.continuous.residual
    scored = []
    for g in problem.groups:
        if g in active:
            continue
        v = problem.structural_blocks[g] @ theta
        z = v - Qb @ (Qb.T @ v) if Qb.shape[1] else v
        zz = float(z @ z)
        gain = 0.0 if zz <= np.finfo(float).tiny else float((r @ z) ** 2 / zz)
        scored.append((gain, g))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [MDLMove("add_group", group_in=g, proposal_score=float(gain)) for gain, g in scored[:max(0, int(top_k))]]


def rank_fixed_dynamics_atom_drops(problem: FixedDynamicsMDLProblem, state: FixedDynamicsSearchState, *, top_k: int = 8) -> list[MDLMove]:
    if len(state.active_atoms) <= 1:
        return []
    scored = []
    for atom in state.active_atoms:
        if atom == state.continuous.pivot_atom and not isinstance(problem.library, CandidateLibrary):
            continue
        scored.append((abs(float(state.continuous.theta[atom])), -problem.library.search_complexities[atom], atom))
    scored.sort(key=lambda x: (x[0], x[1]))
    return [MDLMove("drop_atom", atom_out=a, proposal_score=float(mag)) for mag, _, a in scored[:max(0, int(top_k))]]


def rank_fixed_dynamics_atom_adds(problem: FixedDynamicsMDLProblem, state: FixedDynamicsSearchState, *, top_k: int = 4) -> list[MDLMove]:
    active = set(state.active_atoms)
    candidates = [a for a in range(problem.library.n_atoms) if a not in active]
    candidates.sort(key=lambda a: (problem.library.search_complexities[a], a))
    return [MDLMove("add_atom", atom_in=a, proposal_score=float(problem.library.search_complexities[a])) for a in candidates[:max(0, int(top_k))]]


def make_group_swaps(drops: Sequence[MDLMove], adds: Sequence[MDLMove], *, top_k: int = 8) -> list[MDLMove]:
    pairs = []
    for d in drops:
        for a in adds:
            score = float((d.proposal_score or 0.0) - (a.proposal_score or 0.0))
            pairs.append((score, d.group_out, a.group_in))
    pairs.sort(key=lambda x: x[0])
    return [MDLMove("swap_group", group_out=go, group_in=gi, proposal_score=s) for s, go, gi in pairs[:max(0, int(top_k))]]


def make_atom_swaps(drops: Sequence[MDLMove], adds: Sequence[MDLMove], *, top_k: int = 6) -> list[MDLMove]:
    pairs = []
    for d in drops:
        for a in adds:
            score = float((d.proposal_score or 0.0) + (a.proposal_score or 0.0))
            pairs.append((score, d.atom_out, a.atom_in))
    pairs.sort(key=lambda x: x[0])
    return [MDLMove("swap_atom", atom_out=ao, atom_in=ai, proposal_score=s) for s, ao, ai in pairs[:max(0, int(top_k))]]


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
    groups, atoms = _apply_move(problem, state, move)
    if not groups:
        return FixedDynamicsMoveEvaluation(move, "rejected_empty_structure", 1.0, 1.0, None, np.inf, np.inf, None, True, False, None)
    lb = _mdl_lower_bound(problem, groups, atoms, q_min=q_min)
    if lb >= state.mdl_upper.total:
        return FixedDynamicsMoveEvaluation(move, "rejected_by_discrete_lower_bound", None, None, None, lb, np.inf, None, True, False, None)

    continuous = profile_fixed_dynamics_candidate(
        problem,
        groups,
        atoms,
        warm_theta=state.continuous.theta,
        max_nfev=max_nfev,
        compute_linear_relaxation=True,
    )
    if continuous.relaxation_proves_infeasible:
        return FixedDynamicsMoveEvaluation(move, "rejected_by_linear_relaxation_floor", continuous.relative_residual, continuous.relaxed_relative_residual, None, lb, np.inf, None, True, False, None)
    if not continuous.feasible_witness:
        return FixedDynamicsMoveEvaluation(move, "no_shared_law_feasible_witness_found", continuous.relative_residual, continuous.relaxed_relative_residual, None, lb, np.inf, None, False, False, None)

    selected = _selected_blocks(problem, groups)
    precision = profile_fixed_dynamics_precision_witness(
        continuous,
        selected,
        quantizer=quantizer,
        q_min=q_min,
        q_max=q_max,
        warm_q=state.precision.q_upper,
        max_coordinate_passes=max_coordinate_passes,
    )
    if precision.q_upper is None or precision.witness is None:
        return FixedDynamicsMoveEvaluation(move, "no_quantized_witness_within_q_max", continuous.relative_residual, continuous.relaxed_relative_residual, None, lb, np.inf, None, False, False, None)

    Q = _free_parameter_count(groups, atoms)
    mdl = compute_conditional_mdl(
        structural_blocks=_structural_support_blocks(problem, groups),
        polynomial_objects=_polynomial_objects(atoms),
        library=problem.library,
        relative_residual=precision.witness.relative_residual,
        uncertainty_floor=problem.uncertainty_floor,
        q_star=precision.q_upper,
        precision_parameter_count=Q,
    )
    new_state = FixedDynamicsSearchState(groups, atoms, continuous, precision, mdl, lb)
    certified_improve = bool(mdl.total < state.mdl_lower)
    return FixedDynamicsMoveEvaluation(
        move=move,
        status="certified_improvement" if certified_improve else "profiled_feasible",
        continuous_relative_residual=continuous.relative_residual,
        relaxed_relative_residual=continuous.relaxed_relative_residual,
        precision_q_upper=precision.q_upper,
        mdl_lower=lb,
        mdl_upper=float(mdl.total),
        delta_upper_vs_current_upper=float(mdl.total - state.mdl_upper.total),
        certified_reject=False,
        certified_improve=certified_improve,
        new_state=new_state,
    )


def run_one_fixed_dynamics_sweep(
    problem: FixedDynamicsMDLProblem,
    state: FixedDynamicsSearchState,
    *,
    quantizer: UniformDyadicQuantizer,
    top_group_drops: int = 6,
    top_group_adds: int = 4,
    top_group_swaps: int = 4,
    top_atom_drops: int = 4,
    top_atom_adds: int = 2,
    top_atom_swaps: int = 2,
    q_min: int = 1,
    q_max: int = 64,
    max_nfev: int = 250,
    max_coordinate_passes: int = 8,
) -> tuple[list[MDLMove], list[FixedDynamicsMoveEvaluation]]:
    gd = rank_fixed_dynamics_group_drops(problem, state, top_k=top_group_drops)
    ga = rank_fixed_dynamics_group_adds(problem, state, top_k=top_group_adds)
    gs = make_group_swaps(gd, ga, top_k=top_group_swaps)
    ad = rank_fixed_dynamics_atom_drops(problem, state, top_k=top_atom_drops)
    aa = rank_fixed_dynamics_atom_adds(problem, state, top_k=top_atom_adds)
    ass = make_atom_swaps(ad, aa, top_k=top_atom_swaps)
    proposals = list(gd) + list(ga) + list(gs) + list(ad) + list(aa) + list(ass)
    evaluations = [
        evaluate_fixed_dynamics_move(
            problem,
            state,
            m,
            quantizer=quantizer,
            q_min=q_min,
            q_max=q_max,
            max_nfev=max_nfev,
            max_coordinate_passes=max_coordinate_passes,
        )
        for m in proposals
    ]
    evaluations.sort(key=lambda e: (e.mdl_upper, e.move.label()))
    return proposals, evaluations


def _run_candidate_branches(
    problem: FixedDynamicsMDLProblem,
    optimizer: Callable[..., FixedDynamicsCompressionResult],
    options: Mapping,
) -> FixedDynamicsCompressionResult:
    """Screen only by continuous feasibility, then finish every surviving branch.

    Explicit initial_atoms bypasses this dispatcher in the recursive calls.
    Each branch inherits all user search/precision budgets. Within-branch MDL
    compression is unchanged; cross-branch ranking uses completed total MDL.
    """
    kwargs = dict(options)
    kwargs.pop("problem")
    groups = problem.groups if kwargs["initial_groups"] is None else tuple(kwargs["initial_groups"])
    summaries = []
    best = None
    for atom in range(problem.library.n_atoms):
        profile = profile_fixed_dynamics_candidate(
            problem, groups, (atom,), max_nfev=kwargs["max_nfev"],
            compute_linear_relaxation=False,
        )
        if not profile.feasible_witness:
            summaries.append(CandidateBranchSummary(
                atom, profile.relative_residual, "outside_continuous_floor",
            ))
            continue
        branch_kwargs = {**kwargs, "initial_atoms": (atom,), "basin_initial_theta": profile.theta}
        try:
            result = optimizer(problem, **branch_kwargs)
        except ValueError as exc:
            # A finite search failure is recorded, not interpreted as proof that
            # the law is impossible. Unexpected errors must still propagate.
            if str(exc) != "No feasible quantized H_FD witness found up to q_max.":
                raise
            summaries.append(CandidateBranchSummary(
                atom, profile.relative_residual, "no_quantized_initial_witness",
            ))
            continue
        state = result.best_state
        summaries.append(CandidateBranchSummary(
            atom, profile.relative_residual, "completed",
            result.initial_state.mdl_upper.total, state.mdl_upper.total,
            state.active_atoms, state.active_groups,
        ))
        if best is None or state.mdl_upper.total < best.best_state.mdl_upper.total:
            best = result
    if best is None:
        # Retain the existing mixed-library fallback; explicit support prevents
        # dispatch recursion. No MDL threshold was used to discard a pure law.
        best = optimizer(problem, **{**kwargs, "initial_atoms": tuple(range(problem.library.n_atoms))})
    return replace(best, candidate_branches=tuple(summaries))


def optimize_fixed_dynamics_mdl(
    problem: FixedDynamicsMDLProblem,
    *,
    quantizer: UniformDyadicQuantizer,
    initial_groups: Optional[Sequence[GroupLabel]] = None,
    initial_atoms: Optional[Sequence[int]] = None,
    n_basin_hops: int = 80,
    basin_seed: Optional[int] = None,
    basin_sigma_min: float = 5e-3,
    basin_sigma_max: float = 1.5e-1,
    basin_local_max_nfev: int = 50,
    basin_initial_theta: Optional[FloatArray] = None,
    max_sweeps: int = 50,
    mdl_improvement_tol: float = 1e-10,
    q_min: int = 1,
    q_max: int = 64,
    max_nfev: int = 250,
    max_coordinate_passes: int = 8,
    top_group_drops: int = 6,
    top_group_adds: int = 4,
    top_group_swaps: int = 4,
    top_atom_drops: int = 4,
    top_atom_adds: int = 2,
    top_atom_swaps: int = 2,
) -> FixedDynamicsCompressionResult:
    """Find a feasible H_FD basin, then greedily shorten the witness MDL code."""

    if isinstance(problem.library, CandidateLibrary) and initial_atoms is None:
        return _run_candidate_branches(problem, optimize_fixed_dynamics_mdl, locals())

    groups0 = problem.groups if initial_groups is None else tuple(initial_groups)
    atoms0 = tuple(range(problem.library.n_atoms)) if initial_atoms is None else tuple(initial_atoms)
    basin = discover_fixed_dynamics_shared_law(
        problem,
        active_groups=groups0,
        active_atoms=atoms0,
        n_hops=n_basin_hops,
        sigma_min=basin_sigma_min,
        sigma_max=basin_sigma_max,
        local_max_nfev=basin_local_max_nfev,
        seed=basin_seed,
        initial_theta=basin_initial_theta,
    )
    if not basin.reached_floor:
        raise RuntimeError(
            "Fixed-dynamics basin search did not enter the Step-2 uncertainty ball: "
            f"best rho={basin.best_profile.relative_residual:.6e}, eps={problem.uncertainty_floor:.6e}."
        )

    state = build_fixed_dynamics_state(
        problem,
        groups0,
        atoms0,
        quantizer=quantizer,
        q_min=q_min,
        q_max=q_max,
        warm_theta=basin.best_profile.theta,
        max_nfev=max_nfev,
        max_coordinate_passes=max_coordinate_passes,
    )
    initial_state = state
    history: list[FixedDynamicsCompressionStep] = []
    sweeps = 0

    for sweep in range(max(0, int(max_sweeps))):
        sweeps = sweep + 1
        _, evals = run_one_fixed_dynamics_sweep(
            problem,
            state,
            quantizer=quantizer,
            top_group_drops=top_group_drops,
            top_group_adds=top_group_adds,
            top_group_swaps=top_group_swaps,
            top_atom_drops=top_atom_drops,
            top_atom_adds=top_atom_adds,
            top_atom_swaps=top_atom_swaps,
            q_min=q_min,
            q_max=q_max,
            max_nfev=max_nfev,
            max_coordinate_passes=max_coordinate_passes,
        )
        feasible = [e for e in evals if e.new_state is not None and np.isfinite(e.mdl_upper)]
        if not feasible:
            break
        best_eval = min(feasible, key=lambda e: e.mdl_upper)
        if best_eval.mdl_upper >= state.mdl_upper.total - float(mdl_improvement_tol):
            break
        old = state
        state = best_eval.new_state
        assert state is not None
        history.append(
            FixedDynamicsCompressionStep(
                sweep=sweep,
                move=best_eval.move,
                old_mdl_upper=float(old.mdl_upper.total),
                new_mdl_upper=float(state.mdl_upper.total),
                certified_improvement=bool(best_eval.certified_improve),
            )
        )

    return FixedDynamicsCompressionResult(
        initial_state=initial_state,
        best_state=state,
        basin_search=basin,
        history=tuple(history),
        sweeps_completed=int(sweeps),
    )


# -----------------------------------------------------------------------------
# Dynamics-first structural screening + joint refinement
# -----------------------------------------------------------------------------


def _theta_relative_change(old: FloatArray, new: FloatArray) -> float:
    old = np.asarray(old, dtype=float).reshape(-1)
    new = np.asarray(new, dtype=float).reshape(-1)
    denom = max(float(np.linalg.norm(old)), np.finfo(float).tiny)
    return float(np.linalg.norm(new - old) / denom)


def _fixed_theta_linear_profile(
    problem: FixedDynamicsMDLProblem,
    groups: Sequence[GroupLabel],
    theta: FloatArray,
) -> tuple[FloatArray, FloatArray, float, float, int, FloatArray, FloatArray, FloatArray]:
    """Solve the structural amplitudes exactly for one temporarily fixed Theta.

    Returns
    -------
    amplitudes, residual, rss, rho, rank, C, C_scaled, beta_scaled

    The solve uses RMS column scaling.  This is only a structural-screening
    coordinate; shortlisted supports are subsequently re-profiled jointly in
    (W, Theta).
    """

    gs = _normalise_groups(problem, groups)
    theta = np.asarray(theta, dtype=float).reshape(-1)
    if theta.size != problem.library.n_atoms:
        raise ValueError("theta has the wrong library dimension.")
    if not gs:
        r = np.asarray(problem.y, dtype=float).copy()
        rss = float(r @ r)
        yn = float(np.linalg.norm(problem.y))
        return (
            np.zeros(0, dtype=float), r, rss,
            float(np.sqrt(max(rss, 0.0)) / yn), 0,
            np.zeros((problem.y.size, 0), dtype=float),
            np.zeros((problem.y.size, 0), dtype=float),
            np.zeros(0, dtype=float),
        )

    C = np.column_stack([problem.structural_blocks[g] @ theta for g in gs])
    scales = np.sqrt(np.mean(C * C, axis=0))
    bad = ~np.isfinite(scales) | (scales <= np.finfo(float).tiny)
    scales[bad] = 1.0
    Cs = C / scales[None, :]
    beta_s, _, rank, _ = np.linalg.lstsq(Cs, problem.y, rcond=None)
    beta = np.asarray(beta_s / scales, dtype=float)
    r = np.asarray(problem.y - C @ beta, dtype=float)
    rss = float(r @ r)
    yn = float(np.linalg.norm(problem.y))
    rho = float(np.sqrt(max(rss, 0.0)) / yn)
    return beta, r, rss, rho, int(rank), C, Cs, np.asarray(beta_s, dtype=float)


def _fixed_theta_drop_order(
    problem: FixedDynamicsMDLProblem,
    state: FixedDynamicsSearchState,
) -> tuple[GroupLabel, ...]:
    """Rank active structural groups by fixed-Theta deletion damage.

    For a full-column-rank structural design, the single-deletion RSS increase
    is exact:

        Delta RSS_j = beta_j^2 / [(C^T C)^(-1)]_jj

    evaluated in column-scaled coordinates.  If the current structural design
    is numerically rank-deficient, a conservative zero-without-refit damage is
    used only as a proposal ordering heuristic.
    """

    groups = tuple(state.active_groups)
    if len(groups) <= 1:
        return groups

    beta, _, _, _, rank, C, Cs, beta_s = _fixed_theta_linear_profile(
        problem, groups, state.continuous.theta
    )
    p = len(groups)

    if rank == p:
        G = np.asarray(Cs.T @ Cs, dtype=float)
        try:
            Ginv = np.linalg.inv(G)
        except np.linalg.LinAlgError:
            Ginv = np.linalg.pinv(G, rcond=1e-12)
        d = np.diag(Ginv)
        damage = np.full(p, np.inf, dtype=float)
        good = np.isfinite(d) & (d > np.finfo(float).tiny)
        damage[good] = (beta_s[good] ** 2) / d[good]
    else:
        # Proposal ranking only.  Exact shortlisted candidates are always
        # re-fitted and jointly re-profiled before they can be accepted.
        damage = (beta ** 2) * np.sum(C * C, axis=0)

    order = np.argsort(damage, kind="stable")
    return tuple(groups[int(j)] for j in order)


def _provisional_fixed_theta_mdl(
    problem: FixedDynamicsMDLProblem,
    groups: Sequence[GroupLabel],
    atoms: Sequence[int],
    *,
    relative_residual: float,
    q_reference: int,
) -> float:
    """Heuristic MDL used only to rank fixed-Theta support proposals."""

    if relative_residual > problem.uncertainty_floor * (1.0 + 1e-10):
        return np.inf
    Q = _free_parameter_count(groups, atoms)
    score = compute_conditional_mdl(
        structural_blocks=_structural_support_blocks(problem, groups),
        polynomial_objects=_polynomial_objects(atoms),
        library=problem.library,
        relative_residual=relative_residual,
        uncertainty_floor=problem.uncertainty_floor,
        q_star=int(q_reference),
        precision_parameter_count=Q,
    )
    return float(score.total)


def screen_fixed_theta_support_path(
    problem: FixedDynamicsMDLProblem,
    state: FixedDynamicsSearchState,
    *,
    max_candidates: int = 4,
    include_beyond_boundary: int = 1,
) -> tuple[FixedThetaSupportScreen, ...]:
    """Construct a cheap nested backward-elimination path at fixed Theta.

    The path is not the final inference.  It only proposes structural supports.
    The largest fixed-Theta-feasible pruning level is located by binary search
    along a deletion ordering, then several supports around that boundary are
    ranked with a provisional MDL that keeps the current q as a reference.

    A small number of supports immediately beyond the fixed-Theta feasibility
    boundary are retained deliberately: joint re-profiling of Theta may rescue
    them.
    """

    groups0 = tuple(state.active_groups)
    p = len(groups0)
    if p <= 1 or max_candidates <= 0:
        return tuple()

    order = _fixed_theta_drop_order(problem, state)
    order_pos = {g: i for i, g in enumerate(order)}
    # Keep original problem ordering in every candidate support.
    problem_order = {g: i for i, g in enumerate(problem.groups)}

    cache: dict[int, FixedThetaSupportScreen] = {}

    def screen_count(b: int) -> FixedThetaSupportScreen:
        b = int(b)
        if b in cache:
            return cache[b]
        if b < 0 or b >= p:
            raise ValueError("drop_count must satisfy 0 <= b < n_active_groups.")
        dropped = tuple(order[:b])
        drop_set = set(dropped)
        groups = tuple(sorted(
            (g for g in groups0 if g not in drop_set),
            key=lambda g: problem_order[g],
        ))
        _, _, _, rho, _, _, _, _ = _fixed_theta_linear_profile(
            problem, groups, state.continuous.theta
        )
        qref = int(state.precision.q_upper or 1)
        provisional = _provisional_fixed_theta_mdl(
            problem,
            groups,
            state.active_atoms,
            relative_residual=rho,
            q_reference=qref,
        )
        out = FixedThetaSupportScreen(
            active_groups=groups,
            dropped_groups=dropped,
            drop_count=b,
            fixed_theta_relative_residual=float(rho),
            fixed_theta_feasible=bool(
                rho <= problem.uncertainty_floor * (1.0 + 1e-10)
            ),
            provisional_mdl_upper=float(provisional),
        )
        cache[b] = out
        return out

    # Monotonicity: on a nested support path the best fixed-Theta residual
    # cannot decrease when more columns are removed.
    lo, hi = 0, p - 1
    if screen_count(hi).fixed_theta_feasible:
        bmax = hi
    else:
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if screen_count(mid).fixed_theta_feasible:
                lo = mid
            else:
                hi = mid
        bmax = lo

    counts = set()
    if bmax > 0:
        counts.add(bmax)
        # A sparse geometric ladder gives proposals far from and near the
        # boundary without evaluating every nested support.
        b = 1
        while b < bmax:
            counts.add(b)
            b *= 2
        for d in (-8, -4, -2, -1, 1, 2, 4):
            q = bmax + d
            if 1 <= q < p:
                counts.add(q)
        counts.add(max(1, bmax // 2))
        counts.add(max(1, (3 * bmax) // 4))
    else:
        counts.add(1)

    feasible = []
    infeasible = []
    for b in sorted(counts):
        s = screen_count(b)
        if s.fixed_theta_feasible:
            feasible.append(s)
        else:
            infeasible.append(s)

    feasible.sort(
        key=lambda s: (
            s.provisional_mdl_upper,
            s.fixed_theta_relative_residual,
            -s.drop_count,
        )
    )
    infeasible.sort(
        key=lambda s: (
            s.fixed_theta_relative_residual,
            -s.drop_count,
        )
    )

    chosen: list[FixedThetaSupportScreen] = []
    chosen.extend(feasible[: max(0, int(max_candidates))])

    # Always expose the actual fixed-Theta boundary even if its provisional MDL
    # is not among the first few; joint Theta motion can alter the ranking.
    if bmax > 0:
        boundary = screen_count(bmax)
        if boundary not in chosen:
            chosen.append(boundary)

    n_beyond = max(0, int(include_beyond_boundary))
    if n_beyond and bmax + 1 < p:
        for b in range(bmax + 1, min(p, bmax + 1 + n_beyond)):
            s = screen_count(b)
            if s not in chosen:
                chosen.append(s)

    # Deduplicate supports and keep the shortlist compact.
    uniq = {}
    for s in chosen:
        uniq[s.active_groups] = s
    out = list(uniq.values())
    out.sort(
        key=lambda s: (
            0 if s.fixed_theta_feasible else 1,
            s.provisional_mdl_upper if np.isfinite(s.provisional_mdl_upper) else np.inf,
            s.fixed_theta_relative_residual,
        )
    )
    cap = max(1, int(max_candidates) + max(0, int(include_beyond_boundary)) + 1)
    return tuple(out[:cap])


def _build_joint_refined_state_fast(
    problem: FixedDynamicsMDLProblem,
    groups: Sequence[GroupLabel],
    atoms: Sequence[int],
    *,
    quantizer: UniformDyadicQuantizer,
    warm_theta: FloatArray,
    warm_q: Optional[int],
    q_min: int,
    q_max: int,
    max_nfev: int,
    max_coordinate_passes: int,
) -> Optional[FixedDynamicsSearchState]:
    """Jointly re-profile a shortlisted support without a costly relaxation.

    The linear relaxation is unnecessary for proposal candidates that already
    have a fixed-Theta feasible witness.  For beyond-boundary candidates,
    failure of the nonlinear warm-start profile is treated as a failed proposal,
    not as a mathematical infeasibility certificate.
    """

    gs = _normalise_groups(problem, groups)
    ats = _normalise_atoms(problem, atoms)
    if not gs:
        return None

    continuous = profile_fixed_dynamics_candidate(
        problem,
        gs,
        ats,
        warm_theta=warm_theta,
        max_nfev=max_nfev,
        compute_linear_relaxation=False,
        include_default_start=abs(float(warm_theta[ats[0]])) <= 1e-14,
    )
    if not continuous.feasible_witness:
        return None

    selected = _selected_blocks(problem, gs)
    precision = profile_fixed_dynamics_precision_witness(
        continuous,
        selected,
        quantizer=quantizer,
        q_min=q_min,
        q_max=q_max,
        warm_q=warm_q,
        max_coordinate_passes=max_coordinate_passes,
    )
    if precision.q_upper is None or precision.witness is None:
        return None

    Q = _free_parameter_count(gs, ats)
    mdl = compute_conditional_mdl(
        structural_blocks=_structural_support_blocks(problem, gs),
        polynomial_objects=_polynomial_objects(ats),
        library=problem.library,
        relative_residual=precision.witness.relative_residual,
        uncertainty_floor=problem.uncertainty_floor,
        q_star=precision.q_upper,
        precision_parameter_count=Q,
    )
    return FixedDynamicsSearchState(
        active_groups=gs,
        active_atoms=ats,
        continuous=continuous,
        precision=precision,
        mdl_upper=mdl,
        mdl_lower=_mdl_lower_bound(problem, gs, ats, q_min=q_min),
    )



def evaluate_fixed_dynamics_move_fast(
    problem: FixedDynamicsMDLProblem,
    state: FixedDynamicsSearchState,
    move: MDLMove,
    *,
    quantizer: UniformDyadicQuantizer,
    q_min: int = 1,
    q_max: int = 64,
    max_nfev: int = 120,
    max_coordinate_passes: int = 4,
) -> FixedDynamicsMoveEvaluation:
    """Jointly refine one shortlisted move without the relaxed-model certificate."""

    groups, atoms = _apply_move(problem, state, move)
    if not groups:
        return FixedDynamicsMoveEvaluation(
            move, "rejected_empty_structure", 1.0, None, None,
            np.inf, np.inf, None, True, False, None,
        )
    lb = _mdl_lower_bound(problem, groups, atoms, q_min=q_min)
    if lb >= state.mdl_upper.total:
        return FixedDynamicsMoveEvaluation(
            move, "rejected_by_discrete_lower_bound", None, None, None,
            lb, np.inf, None, True, False, None,
        )

    new_state = _build_joint_refined_state_fast(
        problem,
        groups,
        atoms,
        quantizer=quantizer,
        warm_theta=state.continuous.theta,
        warm_q=state.precision.q_upper,
        q_min=q_min,
        q_max=q_max,
        max_nfev=max_nfev,
        max_coordinate_passes=max_coordinate_passes,
    )
    if new_state is None:
        return FixedDynamicsMoveEvaluation(
            move,
            "no_joint_feasible_witness_found",
            None,
            None,
            None,
            lb,
            np.inf,
            None,
            False,
            False,
            None,
        )

    mdl = new_state.mdl_upper
    return FixedDynamicsMoveEvaluation(
        move=move,
        status="profiled_feasible_fast",
        continuous_relative_residual=float(new_state.continuous.relative_residual),
        relaxed_relative_residual=None,
        precision_q_upper=new_state.precision.q_upper,
        mdl_lower=lb,
        mdl_upper=float(mdl.total),
        delta_upper_vs_current_upper=float(mdl.total - state.mdl_upper.total),
        certified_reject=False,
        certified_improve=bool(mdl.total < state.mdl_lower),
        new_state=new_state,
    )


def run_one_fixed_dynamics_fast_sweep(
    problem: FixedDynamicsMDLProblem,
    state: FixedDynamicsSearchState,
    *,
    quantizer: UniformDyadicQuantizer,
    top_group_drops: int = 2,
    top_group_adds: int = 2,
    top_group_swaps: int = 2,
    top_atom_drops: int = 2,
    top_atom_adds: int = 1,
    top_atom_swaps: int = 1,
    q_min: int = 1,
    q_max: int = 64,
    max_nfev: int = 120,
    max_coordinate_passes: int = 4,
) -> tuple[list[MDLMove], list[FixedDynamicsMoveEvaluation]]:
    """Small shortlisted sweep used after the batch structural screen."""

    gd = rank_fixed_dynamics_group_drops(problem, state, top_k=top_group_drops)
    ga = rank_fixed_dynamics_group_adds(problem, state, top_k=top_group_adds)
    gs = make_group_swaps(gd, ga, top_k=top_group_swaps)
    ad = rank_fixed_dynamics_atom_drops(problem, state, top_k=top_atom_drops)
    aa = rank_fixed_dynamics_atom_adds(problem, state, top_k=top_atom_adds)
    ass = make_atom_swaps(ad, aa, top_k=top_atom_swaps)
    proposals = list(gd) + list(ga) + list(gs) + list(ad) + list(aa) + list(ass)
    evaluations = [
        evaluate_fixed_dynamics_move_fast(
            problem,
            state,
            m,
            quantizer=quantizer,
            q_min=q_min,
            q_max=q_max,
            max_nfev=max_nfev,
            max_coordinate_passes=max_coordinate_passes,
        )
        for m in proposals
    ]
    evaluations.sort(key=lambda e: (e.mdl_upper, e.move.label()))
    return proposals, evaluations


def optimize_fixed_dynamics_mdl_alternating(
    problem: FixedDynamicsMDLProblem,
    *,
    quantizer: UniformDyadicQuantizer,
    initial_groups: Optional[Sequence[GroupLabel]] = None,
    initial_atoms: Optional[Sequence[int]] = None,
    n_basin_hops: int = 80,
    basin_seed: Optional[int] = None,
    basin_sigma_min: float = 5e-3,
    basin_sigma_max: float = 1.5e-1,
    basin_local_max_nfev: int = 50,
    basin_initial_theta: Optional[FloatArray] = None,
    max_structural_outer: int = 8,
    structural_joint_candidates: int = 3,
    structural_beyond_boundary: int = 1,
    structural_polish_sweeps: int = 4,
    max_expression_sweeps: int = 12,
    mdl_improvement_tol: float = 1e-10,
    theta_stability_tol: float = 1e-8,
    q_min: int = 1,
    q_max: int = 64,
    max_nfev: int = 120,
    final_max_nfev: int = 300,
    max_coordinate_passes: int = 4,
    polish_group_drops: int = 2,
    polish_group_adds: int = 2,
    polish_group_swaps: int = 2,
    top_atom_drops: int = 2,
    top_atom_adds: int = 1,
    top_atom_swaps: int = 1,
) -> FixedDynamicsCompressionResult:
    """Dynamics-first structural inference with joint refinement.

    Outer iteration
    ---------------
    1. The current Theta is an anchor obtained from exact variable projection.
    2. Holding that Theta fixed *temporarily*, the structural problem is linear.
       A nested backward-elimination path proposes large support reductions.
    3. Only a small shortlist is re-profiled jointly in (W, Theta), quantized,
       and scored with the full MDL objective.
    4. The best true MDL improvement is accepted, which also updates Theta.

    This is not a strict two-step estimator.  Theta can move after every
    accepted structural change.  The initial shared law serves only to place the
    search in the correct dynamical basin.
    """

    if isinstance(problem.library, CandidateLibrary) and initial_atoms is None:
        return _run_candidate_branches(problem, optimize_fixed_dynamics_mdl_alternating, locals())

    groups0 = problem.groups if initial_groups is None else tuple(initial_groups)
    atoms0 = (
        tuple(range(problem.library.n_atoms))
        if initial_atoms is None
        else tuple(initial_atoms)
    )

    basin = discover_fixed_dynamics_shared_law(
        problem,
        active_groups=groups0,
        active_atoms=atoms0,
        n_hops=n_basin_hops,
        sigma_min=basin_sigma_min,
        sigma_max=basin_sigma_max,
        local_max_nfev=basin_local_max_nfev,
        seed=basin_seed,
        initial_theta=basin_initial_theta,
    )
    if not basin.reached_floor:
        raise RuntimeError(
            "Fixed-dynamics basin search did not enter the Step-2 uncertainty ball: "
            f"best rho={basin.best_profile.relative_residual:.6e}, "
            f"eps={problem.uncertainty_floor:.6e}."
        )

    state = build_fixed_dynamics_state(
        problem,
        groups0,
        atoms0,
        quantizer=quantizer,
        q_min=q_min,
        q_max=q_max,
        warm_theta=basin.best_profile.theta,
        max_nfev=max_nfev,
        max_coordinate_passes=max_coordinate_passes,
    )
    initial_state = state
    theta_anchor = np.asarray(state.continuous.theta, dtype=float).copy()

    history: list[FixedDynamicsCompressionStep] = []
    structural_history: list[FixedDynamicsCompressionStep] = []
    expression_history: list[FixedDynamicsCompressionStep] = []
    outer_done = 0
    sweep_counter = 0

    # Phase B: batch structural compression.  Atom support is held fixed, but
    # Theta's numerical coefficients are re-profiled after every accepted batch.
    for outer in range(max(0, int(max_structural_outer))):
        outer_done = outer + 1
        screens = screen_fixed_theta_support_path(
            problem,
            state,
            max_candidates=max(1, int(structural_joint_candidates)),
            include_beyond_boundary=max(0, int(structural_beyond_boundary)),
        )
        if not screens:
            break

        refined: list[tuple[FixedThetaSupportScreen, FixedDynamicsSearchState]] = []
        for screen in screens:
            cand = _build_joint_refined_state_fast(
                problem,
                screen.active_groups,
                state.active_atoms,
                quantizer=quantizer,
                warm_theta=state.continuous.theta,
                warm_q=state.precision.q_upper,
                q_min=q_min,
                q_max=q_max,
                max_nfev=max_nfev,
                max_coordinate_passes=max_coordinate_passes,
            )
            if cand is not None:
                refined.append((screen, cand))

        if not refined:
            break
        screen_best, cand_best = min(refined, key=lambda sc: sc[1].mdl_upper.total)
        if cand_best.mdl_upper.total >= state.mdl_upper.total - float(mdl_improvement_tol):
            break

        old = state
        state = cand_best
        theta_drift = _theta_relative_change(old.continuous.theta, state.continuous.theta)
        step = FixedDynamicsCompressionStep(
            sweep=sweep_counter,
            move=MDLMove(
                "drop_batch",
                group_out=screen_best.dropped_groups,
                proposal_score=float(screen_best.drop_count),
            ),
            old_mdl_upper=float(old.mdl_upper.total),
            new_mdl_upper=float(state.mdl_upper.total),
            certified_improvement=bool(state.mdl_upper.total < old.mdl_lower),
            phase="structural_batch",
            screened_relative_residual=float(screen_best.fixed_theta_relative_residual),
            theta_relative_change=float(theta_drift),
            n_exact_refits=len(refined),
        )
        sweep_counter += 1
        history.append(step)
        structural_history.append(step)

        # Once the law barely moves, a new structural screen is still useful;
        # convergence is declared only when no MDL-improving support is found.
        _ = theta_stability_tol  # retained as an exposed diagnostic tolerance

    # Local single-move structural polish / rescue.  This can re-add a group or
    # swap one if the coarse nested path made a suboptimal deletion.
    for _ in range(max(0, int(structural_polish_sweeps))):
        _, evals = run_one_fixed_dynamics_fast_sweep(
            problem,
            state,
            quantizer=quantizer,
            top_group_drops=polish_group_drops,
            top_group_adds=polish_group_adds,
            top_group_swaps=polish_group_swaps,
            top_atom_drops=0,
            top_atom_adds=0,
            top_atom_swaps=0,
            q_min=q_min,
            q_max=q_max,
            max_nfev=max_nfev,
            max_coordinate_passes=max_coordinate_passes,
        )
        feasible = [
            e for e in evals
            if e.new_state is not None and np.isfinite(e.mdl_upper)
        ]
        if not feasible:
            break
        best = min(feasible, key=lambda e: e.mdl_upper)
        if best.mdl_upper >= state.mdl_upper.total - float(mdl_improvement_tol):
            break
        old = state
        state = best.new_state
        assert state is not None
        step = FixedDynamicsCompressionStep(
            sweep=sweep_counter,
            move=best.move,
            old_mdl_upper=float(old.mdl_upper.total),
            new_mdl_upper=float(state.mdl_upper.total),
            certified_improvement=bool(best.certified_improve),
            phase="structural_polish",
            screened_relative_residual=best.continuous_relative_residual,
            theta_relative_change=_theta_relative_change(
                old.continuous.theta, state.continuous.theta
            ),
            n_exact_refits=len(evals),
        )
        sweep_counter += 1
        history.append(step)
        structural_history.append(step)

    # Phase C: with the structural sector selected, compress the shared
    # interaction expression.  Every accepted atom move still re-profiles W and
    # all remaining Theta coefficients jointly.
    for _ in range(max(0, int(max_expression_sweeps))):
        _, evals = run_one_fixed_dynamics_fast_sweep(
            problem,
            state,
            quantizer=quantizer,
            top_group_drops=0,
            top_group_adds=0,
            top_group_swaps=0,
            top_atom_drops=top_atom_drops,
            top_atom_adds=top_atom_adds,
            top_atom_swaps=top_atom_swaps,
            q_min=q_min,
            q_max=q_max,
            max_nfev=max_nfev,
            max_coordinate_passes=max_coordinate_passes,
        )
        feasible = [
            e for e in evals
            if e.new_state is not None and np.isfinite(e.mdl_upper)
        ]
        if not feasible:
            break
        best = min(feasible, key=lambda e: e.mdl_upper)
        if best.mdl_upper >= state.mdl_upper.total - float(mdl_improvement_tol):
            break
        old = state
        state = best.new_state
        assert state is not None
        step = FixedDynamicsCompressionStep(
            sweep=sweep_counter,
            move=best.move,
            old_mdl_upper=float(old.mdl_upper.total),
            new_mdl_upper=float(state.mdl_upper.total),
            certified_improvement=bool(best.certified_improve),
            phase="expression",
            screened_relative_residual=best.continuous_relative_residual,
            theta_relative_change=_theta_relative_change(
                old.continuous.theta, state.continuous.theta
            ),
            n_exact_refits=len(evals),
        )
        sweep_counter += 1
        history.append(step)
        expression_history.append(step)

    # Phase D: final joint polish on the selected discrete model.
    polished = _build_joint_refined_state_fast(
        problem,
        state.active_groups,
        state.active_atoms,
        quantizer=quantizer,
        warm_theta=state.continuous.theta,
        warm_q=state.precision.q_upper,
        q_min=q_min,
        q_max=q_max,
        max_nfev=max(final_max_nfev, max_nfev),
        max_coordinate_passes=max_coordinate_passes,
    )
    if polished is not None and polished.mdl_upper.total <= state.mdl_upper.total + float(mdl_improvement_tol):
        state = polished

    return FixedDynamicsCompressionResult(
        initial_state=initial_state,
        best_state=state,
        basin_search=basin,
        history=tuple(history),
        sweeps_completed=int(sweep_counter),
        theta_anchor=theta_anchor,
        structural_history=tuple(structural_history),
        expression_history=tuple(expression_history),
        outer_iterations=int(outer_done),
        search_strategy="alternating_profiled",
    )


__all__ = [
    "FixedDynamicsHypothesis",
    "FixedDynamicsMDLProblem",
    "MDLMove",
    "FixedDynamicsSearchState",
    "FixedDynamicsMoveEvaluation",
    "SharedLawBasinHopStep",
    "SharedLawBasinSearchResult",
    "FixedThetaSupportScreen",
    "FixedDynamicsCompressionStep",
    "FixedDynamicsCompressionResult",
    "CandidateBranchSummary",
    "profile_fixed_dynamics_candidate",
    "build_fixed_dynamics_state",
    "discover_fixed_dynamics_shared_law",
    "rank_fixed_dynamics_group_drops",
    "rank_fixed_dynamics_group_adds",
    "rank_fixed_dynamics_atom_drops",
    "rank_fixed_dynamics_atom_adds",
    "make_group_swaps",
    "make_atom_swaps",
    "evaluate_fixed_dynamics_move",
    "run_one_fixed_dynamics_sweep",
    "screen_fixed_theta_support_path",
    "evaluate_fixed_dynamics_move_fast",
    "run_one_fixed_dynamics_fast_sweep",
    "optimize_fixed_dynamics_mdl",
    "optimize_fixed_dynamics_mdl_alternating",
]
