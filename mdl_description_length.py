"""Conditional MDL accounting utilities for TIDES.

This module implements the *description-length bookkeeping* layer only.  It does
not solve for continuous coefficients, does not search supports, and does not
compute the minimum precision ``q_star``.  Those tasks belong to an inner
optimization layer.

The intended Step-2 conditional MDL is

    Sigma_2(B | H, L) = L_obs + L_temp + L_expr + L_prec,

where the research hypothesis ``H`` (for example fixed dynamics / varying
structure) and the interaction library ``L`` are side information supplied by
the caller and therefore are not re-encoded here.

Current clean-benchmark semantics
---------------------------------
``L_obs`` is a hard uncertainty-ball code:

    L_obs = 0      if relative_residual <= uncertainty_floor,
            +inf   otherwise.

``L_temp`` encodes only support blocks that are *not* fixed by ``H``.  The
caller therefore passes exactly the temporal/structural blocks that must be
encoded under the chosen hypothesis.

``L_expr`` uses the normalized hierarchical polynomial support code discussed
for the TIDES polynomial branch.  For one independent polynomial object with
atom support J, support size s, and highest active degree p,

    L_expr = log(P) + log(L_p)
             + log[ C(L_p, s) - C(L_{p-1}, s) ],

where P is the user-declared maximum polynomial degree / library resolution and
L_p is the number of library atoms with degree <= p.

``L_prec`` uses a finite-precision code

    L_prec = L_N(q_star) + Q * q_star * log(2),

where Q is the number of genuinely free scalar coefficients and q_star is the
minimum bit depth required by the inner optimizer to remain inside the
uncertainty ball.  The default ``L_N`` is an Elias-delta prefix code for a
positive integer, converted to nats.  A different universal integer code can be
supplied explicitly; this choice should be frozen before final paper results.

All returned description lengths are in *nats*.  Divide by ``log(2)`` to obtain
bits.

Two public entry points are intended for the optimizer:

``compute_conditional_mdl``
    Full/reference accounting from a completely profiled model state.

``compute_mdl_delta``
    Incremental accounting for a proposed move.  It updates structural and
    expression codes locally and, when the new residual and q_star are supplied,
    returns the exact total delta.  If either inner quantity is unavailable it
    returns the exact known components plus flags indicating which re-profiling
    steps are still required.

Design principle
----------------
Declared equality/sharing is represented by passing one ``PolynomialObject``
for each genuinely independent polynomial object under H.  Two numerically
identical rows that are *not* declared to share dynamics must therefore be
passed as two objects and are encoded twice.  Accidental equality never earns a
compression bonus in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, lgamma, log, log1p
from typing import Callable, Hashable, Optional, Sequence

import numpy as np


# -----------------------------------------------------------------------------
# Basic code specifications
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class TemporalSupportBlock:
    """One support block that must be encoded under the chosen hypothesis H.

    Parameters
    ----------
    candidate_count
        Number ``M_t`` of candidate rows in this block.
    active_count
        Number ``E_t`` of active / nonzero rows.
    label
        Optional diagnostic label, e.g. ``"baseline"`` or ``("transition", 2)``.
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
            raise ValueError(
                "active_count must satisfy 0 <= active_count <= candidate_count."
            )
        object.__setattr__(self, "candidate_count", M)
        object.__setattr__(self, "active_count", E)


@dataclass(frozen=True)
class PolynomialLibrary:
    """Side-information specification for a polynomial interaction library.

    ``atom_degrees[ell]`` is the polynomial degree of library atom ``ell``.
    ``max_degree`` is the user-declared library resolution ``P`` and is side
    information; it is not itself inferred by MDL.
    """

    atom_degrees: tuple[int, ...]
    max_degree: int

    def __post_init__(self) -> None:
        degrees = tuple(int(d) for d in self.atom_degrees)
        P = int(self.max_degree)

        if P < 1:
            raise ValueError("max_degree must be >= 1.")
        if not degrees:
            raise ValueError("PolynomialLibrary must contain at least one atom.")
        if any(d < 1 for d in degrees):
            raise ValueError("Polynomial atom degrees must be positive integers.")
        if any(d > P for d in degrees):
            raise ValueError("Every atom degree must be <= max_degree.")

        object.__setattr__(self, "atom_degrees", degrees)
        object.__setattr__(self, "max_degree", P)

    @property
    def n_atoms(self) -> int:
        return len(self.atom_degrees)

    def count_up_to_degree(self, p: int) -> int:
        p = int(p)
        return sum(d <= p for d in self.atom_degrees)


@dataclass(frozen=True)
class PolynomialObject:
    """One genuinely independent polynomial object under hypothesis H.

    ``active_atoms`` contains library indices.  A declared shared law should be
    represented by one object; separate undeclared rows must be represented by
    separate objects even when their fitted coefficient values happen to match.
    """

    active_atoms: tuple[int, ...]
    label: Optional[Hashable] = None

    def __post_init__(self) -> None:
        atoms = tuple(int(i) for i in self.active_atoms)
        if not atoms:
            raise ValueError(
                "PolynomialObject must contain at least one active atom. "
                "An absent object should be omitted from the object list instead."
            )
        if len(set(atoms)) != len(atoms):
            raise ValueError("active_atoms must not contain duplicates.")
        if any(i < 0 for i in atoms):
            raise ValueError("active_atoms must contain non-negative indices.")
        object.__setattr__(self, "active_atoms", tuple(sorted(atoms)))


@dataclass(frozen=True)
class TemporalBlockUpdate:
    """Incremental replacement of one encoded temporal support block."""

    old: TemporalSupportBlock
    new: TemporalSupportBlock


@dataclass(frozen=True)
class ExpressionObjectUpdate:
    """Incremental replacement/addition/removal of one polynomial object.

    ``old=None`` represents adding a new independent object.
    ``new=None`` represents removing an independent object.
    """

    old: Optional[PolynomialObject]
    new: Optional[PolynomialObject]

    def __post_init__(self) -> None:
        if self.old is None and self.new is None:
            raise ValueError("At least one of old/new must be present.")


# -----------------------------------------------------------------------------
# Returned accounting objects
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class MDLScore:
    """Exact conditional-MDL accounting for one completely profiled model."""

    total: float
    observation: float
    temporal: float
    expression: float
    precision: float

    Q: int
    q_star: Optional[int]
    relative_residual: float
    uncertainty_floor: float
    feasible: bool

    temporal_components: tuple[float, ...]
    expression_components: tuple[float, ...]

    @property
    def total_bits(self) -> float:
        return float(self.total / log(2.0))

    @property
    def observation_bits(self) -> float:
        return float(self.observation / log(2.0))

    @property
    def temporal_bits(self) -> float:
        return float(self.temporal / log(2.0))

    @property
    def expression_bits(self) -> float:
        return float(self.expression / log(2.0))

    @property
    def precision_bits(self) -> float:
        return float(self.precision / log(2.0))


@dataclass(frozen=True)
class MDLDelta:
    """Incremental description-length change for a proposed move.

    ``delta_total`` is exact only when ``exact`` is True.  Structural and
    expression deltas are always exact because they are discrete bookkeeping.

    If ``new_q_star`` is unavailable, ``provisional_delta_if_q_unchanged`` gives
    the total delta that *would* result if the old q_star remained valid.  This is
    a diagnostic/proposal heuristic only, not an exact MDL change.
    """

    delta_total: Optional[float]
    delta_observation: Optional[float]
    delta_temporal: float
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
        if self.delta_total is None:
            return None
        return float(self.delta_total / log(2.0))


# -----------------------------------------------------------------------------
# Low-level code lengths
# -----------------------------------------------------------------------------


def _log_choose(n: int, k: int) -> float:
    """Natural log of C(n,k), evaluated stably through log-gamma."""

    n = int(n)
    k = int(k)
    if n < 0 or k < 0 or k > n:
        return -np.inf
    if k == 0 or k == n:
        return 0.0
    k = min(k, n - k)
    return float(lgamma(n + 1.0) - lgamma(k + 1.0) - lgamma(n - k + 1.0))


def _logdiffexp(log_a: float, log_b: float) -> float:
    """Return log(exp(log_a) - exp(log_b)) for log_a > log_b."""

    if np.isneginf(log_b):
        return float(log_a)
    if not np.isfinite(log_a):
        raise ValueError("log_a must be finite for log-difference evaluation.")
    if log_b >= log_a:
        raise ValueError("log-difference requires log_a > log_b.")
    return float(log_a + log1p(-np.exp(log_b - log_a)))


def elias_delta_integer_code_length_nats(n: int) -> float:
    """Prefix-code length for a positive integer using Elias-delta coding.

    This is a concrete default for ``L_N(n)``.  It is intentionally injectable:
    if the manuscript later fixes another universal integer code, pass that code
    length function to the public accounting routines instead of changing the
    rest of the MDL implementation.
    """

    n = int(n)
    if n < 1:
        raise ValueError("Universal integer code is defined here for n >= 1.")

    ell = floor(np.log2(n)) + 1
    bits = floor(np.log2(n)) + 2 * floor(np.log2(ell)) + 1
    return float(bits * log(2.0))


def _temporal_block_code(block: TemporalSupportBlock) -> float:
    M = block.candidate_count
    E = block.active_count
    return float(log(M + 1.0) + _log_choose(M, E))


def _polynomial_object_code(
    obj: PolynomialObject,
    library: PolynomialLibrary,
) -> float:
    atoms = obj.active_atoms
    if max(atoms) >= library.n_atoms:
        raise ValueError(
            f"Polynomial object {obj.label!r} references atom index {max(atoms)}, "
            f"but the library has only {library.n_atoms} atoms."
        )

    s = len(atoms)
    p = max(library.atom_degrees[i] for i in atoms)
    Lp = library.count_up_to_degree(p)
    Lprev = library.count_up_to_degree(p - 1)

    log_all = _log_choose(Lp, s)
    log_without_top_degree = _log_choose(Lprev, s)

    if np.isneginf(log_all):
        raise ValueError("Invalid polynomial support count for the supplied library.")

    if np.isneginf(log_without_top_degree):
        log_support_count = log_all
    else:
        if log_without_top_degree >= log_all:
            raise ValueError(
                "Polynomial support has no admissible realization with highest "
                f"degree exactly p={p}; check the library and support."
            )
        log_support_count = _logdiffexp(log_all, log_without_top_degree)

    return float(
        log(float(library.max_degree))
        + log(float(Lp))
        + log_support_count
    )


def _precision_code(
    Q: int,
    q_star: Optional[int],
    integer_code_length: Callable[[int], float],
) -> float:
    Q = int(Q)
    if Q < 0:
        raise ValueError("Q must be non-negative.")

    if Q == 0:
        if q_star not in (None, 0):
            raise ValueError("q_star must be None or 0 when Q == 0.")
        return 0.0

    if q_star is None:
        raise ValueError("A feasible model with Q > 0 requires q_star.")

    q = int(q_star)
    if q < 1:
        raise ValueError("q_star must be >= 1 when Q > 0.")

    return float(integer_code_length(q) + Q * q * log(2.0))


def _observation_code(relative_residual: float, uncertainty_floor: float) -> tuple[float, bool]:
    rho = float(relative_residual)
    eps = float(uncertainty_floor)

    if not np.isfinite(rho) or rho < 0.0:
        raise ValueError("relative_residual must be finite and non-negative.")
    if not np.isfinite(eps) or eps < 0.0:
        raise ValueError("uncertainty_floor must be finite and non-negative.")

    feasible = bool(rho <= eps)
    return (0.0 if feasible else np.inf), feasible


# -----------------------------------------------------------------------------
# Public reference accounting
# -----------------------------------------------------------------------------


def compute_conditional_mdl(
    *,
    temporal_blocks: Sequence[TemporalSupportBlock],
    polynomial_objects: Sequence[PolynomialObject],
    library: PolynomialLibrary,
    relative_residual: float,
    uncertainty_floor: float,
    q_star: Optional[int],
    integer_code_length: Callable[[int], float] = elias_delta_integer_code_length_nats,
) -> MDLScore:
    """Compute the full conditional MDL for one completely profiled model.

    Parameters
    ----------
    temporal_blocks
        Exactly the support blocks that are *not* fixed by hypothesis H.
    polynomial_objects
        Exactly the genuinely independent polynomial objects under H.  Declared
        sharing is represented by passing one object; accidental numerical
        equality does not alter this list.
    library
        User-declared polynomial library side information.
    relative_residual, uncertainty_floor
        Clean-benchmark observation code inputs.
    q_star
        Minimum coefficient bit depth found by the inner precision optimizer.
        Required for a feasible model whenever Q > 0.
    integer_code_length
        Code-length function for ``L_N(q_star)``, returning nats.

    Notes
    -----
    This function performs *accounting only*.  It does not fit coefficients and
    does not search for q_star.
    """

    temporal_blocks = tuple(temporal_blocks)
    polynomial_objects = tuple(polynomial_objects)

    temporal_components = tuple(_temporal_block_code(b) for b in temporal_blocks)
    expression_components = tuple(
        _polynomial_object_code(obj, library) for obj in polynomial_objects
    )

    L_temp = float(sum(temporal_components))
    L_expr = float(sum(expression_components))
    Q = int(sum(len(obj.active_atoms) for obj in polynomial_objects))

    L_obs, feasible = _observation_code(relative_residual, uncertainty_floor)

    if feasible:
        L_prec = _precision_code(Q, q_star, integer_code_length)
        total = float(L_temp + L_expr + L_prec)
    else:
        # Keep the discrete components visible for diagnostics.  q_star is not
        # needed because the model is observationally inadmissible.
        L_prec = np.nan if q_star is None else _precision_code(Q, q_star, integer_code_length)
        total = np.inf

    return MDLScore(
        total=float(total),
        observation=float(L_obs),
        temporal=L_temp,
        expression=L_expr,
        precision=float(L_prec),
        Q=Q,
        q_star=None if q_star is None else int(q_star),
        relative_residual=float(relative_residual),
        uncertainty_floor=float(uncertainty_floor),
        feasible=bool(feasible),
        temporal_components=temporal_components,
        expression_components=expression_components,
    )


# -----------------------------------------------------------------------------
# Public incremental accounting
# -----------------------------------------------------------------------------


def compute_mdl_delta(
    old_score: MDLScore,
    *,
    library: PolynomialLibrary,
    temporal_updates: Sequence[TemporalBlockUpdate] = (),
    expression_updates: Sequence[ExpressionObjectUpdate] = (),
    new_relative_residual: Optional[float] = None,
    new_q_star: Optional[int] = None,
    integer_code_length: Callable[[int], float] = elias_delta_integer_code_length_nats,
) -> MDLDelta:
    """Incrementally evaluate the MDL change of a proposed model move.

    The accepted/current state is assumed to be observationally feasible.  This
    matches the intended optimizer, which walks among admissible models and may
    propose an inadmissible model that is then rejected.

    Structural and expression changes are exact immediately.  The total change
    is exact only after the proposal has been re-profiled sufficiently to supply
    ``new_relative_residual`` and, for ``new_Q > 0``, ``new_q_star``.

    When ``new_q_star`` is not yet known, the returned
    ``provisional_delta_if_q_unchanged`` answers a useful but explicitly
    non-binding question: what would the total delta be if the old precision
    level remained valid?
    """

    if not isinstance(old_score, MDLScore):
        raise TypeError("old_score must be an MDLScore.")
    if not old_score.feasible or not np.isfinite(old_score.total):
        raise ValueError(
            "compute_mdl_delta expects the current/old state to be feasible with "
            "finite total description length."
        )

    temporal_updates = tuple(temporal_updates)
    expression_updates = tuple(expression_updates)

    # Exact local structural update.
    delta_temp = 0.0
    for update in temporal_updates:
        if not isinstance(update, TemporalBlockUpdate):
            raise TypeError("temporal_updates must contain TemporalBlockUpdate objects.")
        delta_temp += _temporal_block_code(update.new) - _temporal_block_code(update.old)

    # Exact local expression update and implied change in Q.
    delta_expr = 0.0
    delta_Q = 0
    for update in expression_updates:
        if not isinstance(update, ExpressionObjectUpdate):
            raise TypeError(
                "expression_updates must contain ExpressionObjectUpdate objects."
            )

        if update.old is not None:
            delta_expr -= _polynomial_object_code(update.old, library)
            delta_Q -= len(update.old.active_atoms)
        if update.new is not None:
            delta_expr += _polynomial_object_code(update.new, library)
            delta_Q += len(update.new.active_atoms)

    new_Q = int(old_score.Q + delta_Q)
    if new_Q < 0:
        raise ValueError("Expression updates imply a negative new_Q.")

    # Observation code is known exactly only after a new residual is supplied.
    if new_relative_residual is None:
        delta_obs: Optional[float] = None
        new_feasible: Optional[bool] = None
        requires_continuous_refit = True
    else:
        new_obs, feasible = _observation_code(
            float(new_relative_residual), old_score.uncertainty_floor
        )
        # old observation code is exactly zero for an accepted feasible state.
        delta_obs = float(new_obs)
        new_feasible = bool(feasible)
        requires_continuous_refit = False

    # Precision code is exact only after q_star has been profiled, unless Q=0.
    if new_Q == 0:
        resolved_new_q: Optional[int] = None
        new_prec = 0.0
        delta_prec: Optional[float] = float(new_prec - old_score.precision)
        requires_precision_reprofile = False
    elif new_q_star is not None:
        resolved_new_q = int(new_q_star)
        new_prec = _precision_code(new_Q, resolved_new_q, integer_code_length)
        delta_prec = float(new_prec - old_score.precision)
        requires_precision_reprofile = False
    else:
        resolved_new_q = None
        delta_prec = None
        requires_precision_reprofile = True

    # Exact total delta requires both observation feasibility and precision.
    exact = bool(delta_obs is not None and delta_prec is not None)

    if exact:
        if new_feasible is False:
            delta_total: Optional[float] = np.inf
        else:
            delta_total = float(delta_obs + delta_temp + delta_expr + delta_prec)
    else:
        delta_total = None

    # Cheap diagnostic under the explicit counterfactual q_new == q_old.
    provisional: Optional[float]
    if old_score.q_star is None:
        provisional = None
    else:
        if new_Q == 0:
            provisional_delta_prec = -old_score.precision
        else:
            provisional_new_prec = _precision_code(
                new_Q,
                old_score.q_star,
                integer_code_length,
            )
            provisional_delta_prec = provisional_new_prec - old_score.precision

        if delta_obs is None:
            # This deliberately ignores the unknown observation term; callers can
            # still use it as a proposal-ordering heuristic, never as acceptance.
            provisional = float(delta_temp + delta_expr + provisional_delta_prec)
        elif new_feasible is False:
            provisional = np.inf
        else:
            provisional = float(
                delta_obs + delta_temp + delta_expr + provisional_delta_prec
            )

    return MDLDelta(
        delta_total=delta_total,
        delta_observation=delta_obs,
        delta_temporal=float(delta_temp),
        delta_expression=float(delta_expr),
        delta_precision=delta_prec,
        new_Q=new_Q,
        new_q_star=resolved_new_q,
        new_relative_residual=(
            None if new_relative_residual is None else float(new_relative_residual)
        ),
        new_feasible=new_feasible,
        exact=exact,
        requires_continuous_refit=requires_continuous_refit,
        requires_precision_reprofile=requires_precision_reprofile,
        provisional_delta_if_q_unchanged=provisional,
    )


__all__ = [
    "TemporalSupportBlock",
    "PolynomialLibrary",
    "PolynomialObject",
    "TemporalBlockUpdate",
    "ExpressionObjectUpdate",
    "MDLScore",
    "MDLDelta",
    "elias_delta_integer_code_length_nats",
    "compute_conditional_mdl",
    "compute_mdl_delta",
]
