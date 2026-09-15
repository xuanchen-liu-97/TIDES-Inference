"""Conditional MDL accounting for TIDES Step 3 physical compression.

For a physical hypothesis H and interaction library Psi supplied as side
information, Step 3 scores an observationally feasible physical representation
through

    L_MDL = L_struct + L_expr + L_prec.

Observational admissibility is enforced by the Step-2 uncertainty ball.  For
convenience this module keeps an observation term with the hard convention

    L_obs = 0      if rho <= epsilon,
            +inf   otherwise.

The public accounting routines are hypothesis-agnostic.  A hypothesis-specific
search supplies exactly the structural support blocks, independent interaction
objects, free numerical parameter count, residual, and precision depth implied
by that representation.

All code lengths are returned in nats.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, lgamma, log, log1p
from typing import Callable, Hashable, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class StructuralSupportBlock:
    """One structural support block encoded under physical hypothesis H.

    Examples include the baseline edge support W^(1) and each temporal-change
    support supp(Delta W^(k)) under H_FD.
    """

    candidate_count: int
    active_count: int
    label: Optional[Hashable] = None

    def __post_init__(self) -> None:
        M = int(self.candidate_count)
        E = int(self.active_count)
        if M < 0:
            raise ValueError("candidate_count must be non-negative.")
        if E < 0 or E > M:
            raise ValueError("Require 0 <= active_count <= candidate_count.")
        object.__setattr__(self, "candidate_count", M)
        object.__setattr__(self, "active_count", E)


# Backward-compatible name used by earlier notebooks.
TemporalSupportBlock = StructuralSupportBlock


@dataclass(frozen=True)
class PolynomialLibrary:
    """Side-information specification for a polynomial interaction library."""

    atom_degrees: tuple[int, ...]
    max_degree: int

    def __post_init__(self) -> None:
        degrees = tuple(int(d) for d in self.atom_degrees)
        P = int(self.max_degree)
        if P < 1:
            raise ValueError("max_degree must be >= 1.")
        if not degrees:
            raise ValueError("PolynomialLibrary must contain at least one atom.")
        if any(d < 1 or d > P for d in degrees):
            raise ValueError("Each atom degree must lie in 1,...,max_degree.")
        object.__setattr__(self, "atom_degrees", degrees)
        object.__setattr__(self, "max_degree", P)

    @property
    def n_atoms(self) -> int:
        return len(self.atom_degrees)

    def count_up_to_degree(self, p: int) -> int:
        return sum(d <= int(p) for d in self.atom_degrees)


@dataclass(frozen=True)
class PolynomialObject:
    """One genuinely independent polynomial interaction object under H."""

    active_atoms: tuple[int, ...]
    label: Optional[Hashable] = None

    def __post_init__(self) -> None:
        atoms = tuple(sorted(set(int(i) for i in self.active_atoms)))
        if not atoms:
            raise ValueError("PolynomialObject requires at least one active atom.")
        if any(i < 0 for i in atoms):
            raise ValueError("active_atoms must be non-negative indices.")
        object.__setattr__(self, "active_atoms", atoms)


@dataclass(frozen=True)
class StructuralBlockUpdate:
    old: StructuralSupportBlock
    new: StructuralSupportBlock


TemporalBlockUpdate = StructuralBlockUpdate


@dataclass(frozen=True)
class ExpressionObjectUpdate:
    old: Optional[PolynomialObject]
    new: Optional[PolynomialObject]

    def __post_init__(self) -> None:
        if self.old is None and self.new is None:
            raise ValueError("At least one of old/new must be present.")


@dataclass(frozen=True)
class MDLScore:
    total: float
    observation: float
    structural: float
    expression: float
    precision: float

    Q: int
    q_star: Optional[int]
    relative_residual: float
    uncertainty_floor: float
    feasible: bool

    structural_components: tuple[float, ...]
    expression_components: tuple[float, ...]

    @property
    def total_bits(self) -> float:
        return float(self.total / log(2.0))

    @property
    def observation_bits(self) -> float:
        return float(self.observation / log(2.0))

    @property
    def structural_bits(self) -> float:
        return float(self.structural / log(2.0))

    @property
    def expression_bits(self) -> float:
        return float(self.expression / log(2.0))

    @property
    def precision_bits(self) -> float:
        return float(self.precision / log(2.0))

    # Compatibility aliases for the earlier implementation.
    @property
    def temporal(self) -> float:
        return self.structural

    @property
    def temporal_components(self) -> tuple[float, ...]:
        return self.structural_components

    @property
    def temporal_bits(self) -> float:
        return self.structural_bits


@dataclass(frozen=True)
class MDLDelta:
    delta_total: Optional[float]
    delta_observation: Optional[float]
    delta_structural: float
    delta_expression: float
    delta_precision: Optional[float]

    new_Q: int
    new_q_star: Optional[int]
    new_relative_residual: Optional[float]
    new_feasible: Optional[bool]

    exact: bool
    requires_continuous_refit: bool
    requires_precision_reprofile: bool
    provisional_delta_if_q_unchanged: Optional[float]

    @property
    def delta_total_bits(self) -> Optional[float]:
        return None if self.delta_total is None else float(self.delta_total / log(2.0))

    @property
    def delta_temporal(self) -> float:
        return self.delta_structural


def _log_choose(n: int, k: int) -> float:
    n, k = int(n), int(k)
    if n < 0 or k < 0 or k > n:
        return -np.inf
    if k == 0 or k == n:
        return 0.0
    k = min(k, n - k)
    return float(lgamma(n + 1.0) - lgamma(k + 1.0) - lgamma(n - k + 1.0))


def _logdiffexp(log_a: float, log_b: float) -> float:
    if np.isneginf(log_b):
        return float(log_a)
    if not np.isfinite(log_a) or log_b >= log_a:
        raise ValueError("log-difference requires finite log_a > log_b.")
    return float(log_a + log1p(-np.exp(log_b - log_a)))


def elias_delta_integer_code_length_nats(n: int) -> float:
    n = int(n)
    if n < 1:
        raise ValueError("Universal integer code is defined for n >= 1.")
    ell = floor(np.log2(n)) + 1
    bits = floor(np.log2(n)) + 2 * floor(np.log2(ell)) + 1
    return float(bits * log(2.0))


def _structural_block_code(block: StructuralSupportBlock) -> float:
    M, E = block.candidate_count, block.active_count
    return float(log(M + 1.0) + _log_choose(M, E))


def _polynomial_object_code(obj: PolynomialObject, library: PolynomialLibrary) -> float:
    atoms = obj.active_atoms
    if max(atoms) >= library.n_atoms:
        raise ValueError("PolynomialObject references an atom outside the library.")
    s = len(atoms)
    p = max(library.atom_degrees[i] for i in atoms)
    Lp = library.count_up_to_degree(p)
    Lprev = library.count_up_to_degree(p - 1)
    log_all = _log_choose(Lp, s)
    log_prev = _log_choose(Lprev, s)
    if np.isneginf(log_all):
        raise ValueError("Invalid polynomial support count.")
    log_support = log_all if np.isneginf(log_prev) else _logdiffexp(log_all, log_prev)
    return float(log(float(library.max_degree)) + log(float(Lp)) + log_support)


def _precision_code(Q: int, q_star: Optional[int], integer_code_length: Callable[[int], float]) -> float:
    Q = int(Q)
    if Q < 0:
        raise ValueError("Q must be non-negative.")
    if Q == 0:
        if q_star not in (None, 0):
            raise ValueError("q_star must be None or 0 when Q == 0.")
        return 0.0
    if q_star is None or int(q_star) < 1:
        raise ValueError("A feasible model with Q > 0 requires q_star >= 1.")
    q = int(q_star)
    return float(integer_code_length(q) + Q * q * log(2.0))


def _observation_code(relative_residual: float, uncertainty_floor: float) -> tuple[float, bool]:
    rho, eps = float(relative_residual), float(uncertainty_floor)
    if not np.isfinite(rho) or rho < 0.0:
        raise ValueError("relative_residual must be finite and non-negative.")
    if not np.isfinite(eps) or eps < 0.0:
        raise ValueError("uncertainty_floor must be finite and non-negative.")
    feasible = bool(rho <= eps * (1.0 + 100.0 * np.finfo(float).eps))
    return (0.0 if feasible else np.inf), feasible


def compute_conditional_mdl(
    *,
    structural_blocks: Optional[Sequence[StructuralSupportBlock]] = None,
    polynomial_objects: Sequence[PolynomialObject],
    library: PolynomialLibrary,
    relative_residual: float,
    uncertainty_floor: float,
    q_star: Optional[int],
    precision_parameter_count: Optional[int] = None,
    integer_code_length: Callable[[int], float] = elias_delta_integer_code_length_nats,
    temporal_blocks: Optional[Sequence[StructuralSupportBlock]] = None,
) -> MDLScore:
    """Compute conditional MDL for one completely profiled Step-3 candidate."""

    if structural_blocks is not None and temporal_blocks is not None:
        raise ValueError("Pass structural_blocks or temporal_blocks, not both.")
    blocks = tuple(structural_blocks if structural_blocks is not None else (temporal_blocks or ()))
    objects = tuple(polynomial_objects)

    structural_components = tuple(_structural_block_code(b) for b in blocks)
    expression_components = tuple(_polynomial_object_code(o, library) for o in objects)
    L_struct = float(sum(structural_components))
    L_expr = float(sum(expression_components))

    expression_Q = int(sum(len(o.active_atoms) for o in objects))
    Q = expression_Q if precision_parameter_count is None else int(precision_parameter_count)
    if Q < 0:
        raise ValueError("precision_parameter_count must be non-negative.")

    L_obs, feasible = _observation_code(relative_residual, uncertainty_floor)
    if feasible:
        L_prec = _precision_code(Q, q_star, integer_code_length)
        total = float(L_struct + L_expr + L_prec)
    else:
        L_prec = np.nan if q_star is None else _precision_code(Q, q_star, integer_code_length)
        total = np.inf

    return MDLScore(
        total=float(total),
        observation=float(L_obs),
        structural=L_struct,
        expression=L_expr,
        precision=float(L_prec),
        Q=Q,
        q_star=None if q_star is None else int(q_star),
        relative_residual=float(relative_residual),
        uncertainty_floor=float(uncertainty_floor),
        feasible=bool(feasible),
        structural_components=structural_components,
        expression_components=expression_components,
    )


def compute_mdl_delta(
    old_score: MDLScore,
    *,
    library: PolynomialLibrary,
    structural_updates: Sequence[StructuralBlockUpdate] = (),
    expression_updates: Sequence[ExpressionObjectUpdate] = (),
    new_relative_residual: Optional[float] = None,
    new_q_star: Optional[int] = None,
    new_precision_parameter_count: Optional[int] = None,
    integer_code_length: Callable[[int], float] = elias_delta_integer_code_length_nats,
    temporal_updates: Optional[Sequence[StructuralBlockUpdate]] = None,
) -> MDLDelta:
    """Incrementally account for one proposed structural/expression move."""

    if not isinstance(old_score, MDLScore) or not old_score.feasible or not np.isfinite(old_score.total):
        raise ValueError("old_score must be a finite feasible MDLScore.")
    if temporal_updates is not None:
        if structural_updates:
            raise ValueError("Pass structural_updates or temporal_updates, not both.")
        structural_updates = temporal_updates

    delta_struct = 0.0
    for u in structural_updates:
        delta_struct += _structural_block_code(u.new) - _structural_block_code(u.old)

    delta_expr = 0.0
    delta_Q_expr = 0
    for u in expression_updates:
        if u.old is not None:
            delta_expr -= _polynomial_object_code(u.old, library)
            delta_Q_expr -= len(u.old.active_atoms)
        if u.new is not None:
            delta_expr += _polynomial_object_code(u.new, library)
            delta_Q_expr += len(u.new.active_atoms)

    new_Q = int(old_score.Q + delta_Q_expr) if new_precision_parameter_count is None else int(new_precision_parameter_count)
    if new_Q < 0:
        raise ValueError("Updates imply a negative new_Q.")

    if new_relative_residual is None:
        delta_obs = None
        new_feasible = None
        need_refit = True
    else:
        new_obs, new_feasible = _observation_code(new_relative_residual, old_score.uncertainty_floor)
        delta_obs = float(new_obs)
        need_refit = False

    if new_Q == 0:
        resolved_q = None
        delta_prec = float(-old_score.precision)
        need_precision = False
    elif new_q_star is not None:
        resolved_q = int(new_q_star)
        delta_prec = float(_precision_code(new_Q, resolved_q, integer_code_length) - old_score.precision)
        need_precision = False
    else:
        resolved_q = None
        delta_prec = None
        need_precision = True

    exact = delta_obs is not None and delta_prec is not None
    if exact:
        delta_total = np.inf if new_feasible is False else float(delta_obs + delta_struct + delta_expr + delta_prec)
    else:
        delta_total = None

    provisional = None
    if old_score.q_star is not None:
        new_prec_same_q = 0.0 if new_Q == 0 else _precision_code(new_Q, old_score.q_star, integer_code_length)
        dp = new_prec_same_q - old_score.precision
        if delta_obs is None:
            provisional = float(delta_struct + delta_expr + dp)
        elif new_feasible is False:
            provisional = np.inf
        else:
            provisional = float(delta_obs + delta_struct + delta_expr + dp)

    return MDLDelta(
        delta_total=delta_total,
        delta_observation=delta_obs,
        delta_structural=float(delta_struct),
        delta_expression=float(delta_expr),
        delta_precision=delta_prec,
        new_Q=new_Q,
        new_q_star=resolved_q,
        new_relative_residual=None if new_relative_residual is None else float(new_relative_residual),
        new_feasible=new_feasible,
        exact=bool(exact),
        requires_continuous_refit=bool(need_refit),
        requires_precision_reprofile=bool(need_precision),
        provisional_delta_if_q_unchanged=provisional,
    )


__all__ = [
    "StructuralSupportBlock",
    "TemporalSupportBlock",
    "PolynomialLibrary",
    "PolynomialObject",
    "StructuralBlockUpdate",
    "TemporalBlockUpdate",
    "ExpressionObjectUpdate",
    "MDLScore",
    "MDLDelta",
    "elias_delta_integer_code_length_nats",
    "compute_conditional_mdl",
    "compute_mdl_delta",
]
