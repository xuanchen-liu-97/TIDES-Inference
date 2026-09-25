"""
TIDES Step 3 inner stage 1: Peixoto-style weight-category relaxation.

Role
----
For fixed dynamics support J and fixed structural support S, infer the latent
weight-category representation c while profiling the shared category values z
and interaction law Theta.  This stage never changes S.

The observation model is the TIDES hard uncertainty floor: all representations
with rho <= epsilon are observationally admissible.  Inside that set, in1
minimizes the conditional representation code

    L = L_fixed + L_K + L_assign + L_z,

where L_fixed contains terms constant during in1 (e.g. structural-support and
dynamics-support codes), and the variable category terms are the Peixoto-style
positive-composition assignment code plus an optimized quantized-Laplace code
for category values.

The latent relaxation schedule is

    reassign -> merge -> split -> merge-split,

with cheap fixed-Theta screening and exact joint (z, Theta) reprofiling only for
shortlisted candidates.  This restores the semantics of the historically
successful TIDES Step-3 category solver while retaining the formal in1 API and
shared numerical primitives from ``step3func_physical_profiling``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, lgamma, log, log1p
from typing import Any, Callable, Hashable, Iterable, Mapping, Protocol, Sequence
import math
import random

import numpy as np
from scipy.optimize import minimize_scalar

try:
    from .step3func_physical_profiling import (
        FixedDynamicsProblem,
        CategoryProfile,
        aggregate_blocks_by_category,
        profile_category_weights_and_dynamics,
    )
except ImportError:
    from step3func_physical_profiling import (
        FixedDynamicsProblem,
        CategoryProfile,
        aggregate_blocks_by_category,
        profile_category_weights_and_dynamics,
    )

Coordinate = Hashable
CategoryLabel = Hashable


class In1InfeasibleError(RuntimeError):
    """One fixed-(J,S,c) category representation lies outside B_epsilon."""


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


@dataclass(frozen=True)
class PeixotoCategoryConfig:
    """Coding and proposal controls for the first-party in1 backend."""

    value_resolution: float = 1.0e-8
    floor_rtol: float = 2.0e-10
    shrink_passes: int = 50
    ls_rcond: float = 1.0e-11

    reassignment_exact_top: int = 6
    merge_exact_top: int = 8
    split_exact_top: int = 6
    merge_split_exact_top: int = 8

    split_random_seeds: int = 4
    merge_split_pair_top: int = 14
    merge_split_random_seeds: int = 3
    local_two_way_sweeps: int = 6

    def __post_init__(self) -> None:
        if not math.isfinite(self.value_resolution) or self.value_resolution <= 0.0:
            raise ValueError("value_resolution must be finite and > 0.")
        if self.floor_rtol < 0.0:
            raise ValueError("floor_rtol must be nonnegative.")
        if self.shrink_passes < 0 or self.local_two_way_sweeps < 0:
            raise ValueError("iteration budgets must be nonnegative.")
        for name in (
            "reassignment_exact_top",
            "merge_exact_top",
            "split_exact_top",
            "merge_split_exact_top",
            "split_random_seeds",
            "merge_split_pair_top",
            "merge_split_random_seeds",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be nonnegative.")


class CategoryBackend(Protocol):
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
        rng: Any,
    ) -> Iterable[CategoryMove]:
        ...


# ---------------------------------------------------------------------------
# Peixoto category coding helpers
# ---------------------------------------------------------------------------


def _log2_choose(n: int, k: int) -> float:
    n, k = int(n), int(k)
    if n < 0 or k < 0 or k > n:
        return -math.inf
    if k == 0 or k == n:
        return 0.0
    return float(
        (lgamma(n + 1.0) - lgamma(k + 1.0) - lgamma(n - k + 1.0))
        / log(2.0)
    )


def compact_labels(labels: Sequence[int]) -> np.ndarray:
    """Canonicalize arbitrary integer labels by first appearance."""

    mapping: dict[int, int] = {}
    out: list[int] = []
    for raw in np.asarray(labels, dtype=int).reshape(-1):
        x = int(raw)
        if x not in mapping:
            mapping[x] = len(mapping)
        out.append(mapping[x])
    return np.asarray(out, dtype=np.int64)


def _ordered_support(
    problem: FixedDynamicsProblem,
    support: Sequence[Coordinate] | frozenset[Coordinate],
) -> tuple[Coordinate, ...]:
    out = problem.normalise_support(support)
    if not out:
        raise ValueError("in1 requires a nonempty structural support.")
    return out


def _labels_from_categories(
    problem: FixedDynamicsProblem,
    support: Sequence[Coordinate] | frozenset[Coordinate],
    categories: Mapping[Coordinate, CategoryLabel],
) -> tuple[tuple[Coordinate, ...], np.ndarray]:
    ordered = _ordered_support(problem, support)
    if set(categories) != set(ordered):
        missing = set(ordered) - set(categories)
        extra = set(categories) - set(ordered)
        raise ValueError(
            "Category assignment must cover exactly the active support. "
            f"missing={sorted(map(repr, missing))}, "
            f"extra={sorted(map(repr, extra))}"
        )
    raw = [categories[g] for g in ordered]
    mapping: dict[CategoryLabel, int] = {}
    labels: list[int] = []
    for label in raw:
        if label not in mapping:
            mapping[label] = len(mapping)
        labels.append(mapping[label])
    return ordered, np.asarray(labels, dtype=np.int64)


def _categories_from_labels(
    ordered_support: Sequence[Coordinate], labels: Sequence[int]
) -> dict[Coordinate, int]:
    lab = compact_labels(labels)
    if lab.size != len(ordered_support):
        raise ValueError("labels must match support length.")
    return {g: int(k) for g, k in zip(ordered_support, lab)}


def category_partition_code_bits(
    labels: Sequence[int],
) -> tuple[float, float, tuple[int, ...]]:
    """Return Peixoto (L_assign, L_K, counts) in bits.

    L_assign = log2(E!/prod m_k!) + log2 C(E-1,K-1),
    L_K      = log2(E),

    with strictly positive category occupancies.
    """

    lab = compact_labels(labels)
    if lab.size == 0:
        return 0.0, 0.0, tuple()
    K = int(lab.max()) + 1
    counts = np.bincount(lab, minlength=K).astype(int)
    E = int(lab.size)
    L_assign = float(
        (lgamma(E + 1.0) - sum(lgamma(int(m) + 1.0) for m in counts))
        / log(2.0)
    )
    if K > 1:
        L_assign += _log2_choose(E - 1, K - 1)
    L_K = float(np.log2(E))
    return L_assign, L_K, tuple(int(x) for x in counts)


def quantized_laplace_code_bits(
    values: Sequence[float], *, resolution: float = 1.0e-8
) -> tuple[float, float]:
    """Optimized quantized-Laplace category-value code in bits."""

    z = np.asarray(values, dtype=float).reshape(-1)
    K = int(z.size)
    if K == 0:
        return 0.0, 1.0
    s = float(np.sum(np.abs(z)))
    if not np.isfinite(s) or s <= 0.0:
        return math.inf, 1.0
    delta = float(resolution)

    def objective(log_lambda: float) -> float:
        lam = exp(float(log_lambda))
        x = lam * delta
        log_expm1 = x + log1p(-exp(-x)) if x > 50.0 else log(np.expm1(x))
        nats = lam * s - K * (log_expm1 - log(2.0))
        return float(nats / log(2.0))

    guess = log(max(1.0e-10, K / s))
    sol = minimize_scalar(
        objective,
        bounds=(guess - 12.0, guess + 12.0),
        method="bounded",
        options={"xatol": 1.0e-9},
    )
    return float(sol.fun), float(exp(sol.x))


def shrink_values_to_floor(
    y: Sequence[float],
    design: np.ndarray,
    values: Sequence[float],
    *,
    uncertainty_floor: float,
    max_passes: int = 50,
) -> tuple[np.ndarray, float, int]:
    """Shrink |z_k| coordinate-wise while remaining inside B_epsilon."""

    y = np.asarray(y, dtype=float).reshape(-1)
    D = np.asarray(design, dtype=float)
    z = np.asarray(values, dtype=float).reshape(-1).copy()
    if D.shape != (y.size, z.size):
        raise ValueError("design/value shape mismatch.")
    y_norm = max(float(np.linalg.norm(y)), np.finfo(float).tiny)
    tau2 = float((float(uncertainty_floor) * y_norm) ** 2)
    r = y - D @ z
    if float(r @ r) > tau2 * (1.0 + 1.0e-9):
        return z, float(np.linalg.norm(r) / y_norm), 0

    moves = 0
    for _ in range(max(0, int(max_passes))):
        changed = False
        for k in range(z.size):
            d = D[:, k]
            a = float(d @ d)
            if a <= 1.0e-20:
                continue
            residual_without_k = r + d * z[k]
            b = float(d @ residual_without_k)
            c = float(residual_without_k @ residual_without_k)
            disc = b * b - a * (c - tau2)
            if disc < -1.0e-9 * max(1.0, b * b):
                continue
            root = float(np.sqrt(max(0.0, disc)))
            lo = (b - root) / a
            hi = (b + root) / a
            if lo <= 0.0 <= hi:
                candidate = 0.0
            elif hi < 0.0:
                candidate = hi
            else:
                candidate = lo
            if abs(candidate) + 1.0e-14 < abs(z[k]):
                r = residual_without_k - d * candidate
                z[k] = candidate
                moves += 1
                changed = True
        if not changed:
            break
    return z, float(np.linalg.norm(r) / y_norm), int(moves)


# Legacy/custom scorer hook retained for compatibility.  If supplied, it
# replaces the first-party Peixoto category code for the exact state.
DescriptionLengthScorer = Callable[
    [frozenset[Coordinate], Mapping[Coordinate, CategoryLabel], CategoryProfile],
    float,
]
CategoryProposalFn = Callable[
    [ProfiledState, str, Any], Iterable[CategoryMove]
]
FixedDescriptionBitsFn = Callable[[frozenset[Coordinate]], float]


class _FixedThetaCategoryEvaluator:
    """Fast Peixoto category scoring for many assignments at one Theta."""

    def __init__(
        self,
        backend: "PhysicalCategoryComputation",
        state: ProfiledState,
    ) -> None:
        self.backend = backend
        self.problem = backend.problem
        self.theta = np.asarray(state.theta, dtype=float).reshape(-1)
        self.support, self.current_labels = _labels_from_categories(
            self.problem, state.support, state.categories
        )
        blocks = tuple(self.problem.structural_blocks[g] for g in self.support)
        # Primitive group columns C_g(Theta).  Candidate category designs are
        # cheap H-aggregations of these columns.
        self.C = np.asarray(
            self.problem.linear_cache.matrix(blocks, self.theta), dtype=float
        )
        self.G = np.asarray(self.C.T @ self.C, dtype=float)
        self.b = np.asarray(self.C.T @ self.problem.y, dtype=float)
        self.y2 = float(self.problem.y @ self.problem.y)
        self.y_norm = max(float(np.linalg.norm(self.problem.y)), np.finfo(float).tiny)

    def profile(self, labels: Sequence[int]) -> tuple[float, np.ndarray, np.ndarray]:
        lab = compact_labels(labels)
        if lab.size != len(self.support):
            raise ValueError("labels must match support length.")
        K = int(lab.max()) + 1 if lab.size else 0
        H = np.zeros((lab.size, K), dtype=float)
        if lab.size:
            H[np.arange(lab.size), lab] = 1.0
        GD = H.T @ self.G @ H
        bd = H.T @ self.b
        if K:
            z, _, _, _ = np.linalg.lstsq(
                GD, bd, rcond=self.backend.peixoto_config.ls_rcond
            )
        else:
            z = np.zeros(0, dtype=float)
        rss = max(self.y2 - float(bd @ z), 0.0)
        rho = float(np.sqrt(rss) / self.y_norm)
        D = self.C @ H
        return rho, np.asarray(z, dtype=float), np.asarray(D, dtype=float)

    def score(self, labels: Sequence[int]) -> dict[str, Any] | None:
        lab = compact_labels(labels)
        rho_ls, z_ls, D = self.profile(lab)
        cfg = self.backend.peixoto_config
        eps = float(self.problem.uncertainty_floor)
        if rho_ls > eps * (1.0 + cfg.floor_rtol):
            return None
        z, rho, n_shrink = shrink_values_to_floor(
            self.problem.y,
            D,
            z_ls,
            uncertainty_floor=eps,
            max_passes=min(35, cfg.shrink_passes),
        )
        if rho > eps * (1.0 + cfg.floor_rtol):
            return None
        L_assign, L_K, counts = category_partition_code_bits(lab)
        L_z, lam = quantized_laplace_code_bits(
            z, resolution=cfg.value_resolution
        )
        return {
            "labels": lab,
            "rho_ls": float(rho_ls),
            "rho": float(rho),
            "z_ls": z_ls,
            "z": z,
            "n_categories": (int(lab.max()) + 1 if lab.size else 0),
            "counts": counts,
            "L_assign_bits": float(L_assign),
            "L_K_bits": float(L_K),
            "L_z_bits": float(L_z),
            "total_bits": float(
                self.backend._fixed_bits(self.support) + L_K + L_assign + L_z
            ),
            "laplace_lambda": float(lam),
            "shrink_moves": int(n_shrink),
        }


class PhysicalCategoryComputation:
    """First-party fixed-(J,S) category solver using shared physical profiling.

    By default this class implements the Peixoto-style category objective and
    proposal kernel used by the successful historical TIDES inner solver.

    ``fixed_description_bits`` contains code terms constant within in1 (for
    example structural-support and dynamics-support terms).  It can be zero for
    conditional in1 comparisons.  Supplying ``description_length_scorer`` keeps
    compatibility with older callers that intentionally use a custom exact
    score, but proposal screening still follows the first-party category code.
    """

    def __init__(
        self,
        problem: FixedDynamicsProblem,
        *,
        active_atoms: Iterable[int],
        description_length_scorer: DescriptionLengthScorer | None = None,
        proposal_fn: CategoryProposalFn | None = None,
        fixed_description_bits: float = 0.0,
        fixed_description_bits_fn: FixedDescriptionBitsFn | None = None,
        peixoto_config: PeixotoCategoryConfig = PeixotoCategoryConfig(),
        max_nfev: int = 250,
        compute_linear_relaxation: bool = False,
    ) -> None:
        self.problem = problem
        self.active_atoms = tuple(sorted(set(int(a) for a in active_atoms)))
        if not self.active_atoms:
            raise ValueError("active_atoms must be non-empty.")
        self.description_length_scorer = description_length_scorer
        self.proposal_fn = proposal_fn
        self.fixed_description_bits = float(fixed_description_bits)
        if not math.isfinite(self.fixed_description_bits):
            raise ValueError("fixed_description_bits must be finite.")
        self.fixed_description_bits_fn = fixed_description_bits_fn
        self.peixoto_config = peixoto_config
        self.max_nfev = int(max_nfev)
        if self.max_nfev < 1:
            raise ValueError("max_nfev must be >= 1.")
        self.compute_linear_relaxation = bool(compute_linear_relaxation)

    def _fixed_bits(self, support: Sequence[Coordinate] | frozenset[Coordinate]) -> float:
        if self.fixed_description_bits_fn is None:
            return float(self.fixed_description_bits)
        value = float(self.fixed_description_bits_fn(frozenset(support)))
        if not math.isfinite(value):
            raise ValueError("fixed_description_bits_fn must return a finite value.")
        return value

    def _exact_peixoto_state(
        self,
        profiled: CategoryProfile,
    ) -> tuple[float, float, dict[Coordinate, float], dict[str, Any]]:
        fit = profiled.fit
        cfg = self.peixoto_config
        order = tuple(profiled.category_order)
        blocks = tuple(
            aggregate_blocks_by_category(
                self.problem, profiled.support, profiled.categories
            )[2]
        )
        D = np.column_stack(
            [np.asarray(block @ fit.theta, dtype=float).reshape(-1) for block in blocks]
        )
        z_ls = np.asarray([profiled.category_values[k] for k in order], dtype=float)
        z, rho, n_shrink = shrink_values_to_floor(
            self.problem.y,
            D,
            z_ls,
            uncertainty_floor=self.problem.uncertainty_floor,
            max_passes=cfg.shrink_passes,
        )
        if rho > self.problem.uncertainty_floor * (1.0 + cfg.floor_rtol):
            raise In1InfeasibleError(
                "Finite category-value coding moved representation outside "
                f"B_epsilon: rho={rho:.6e}."
            )

        _, labels = _labels_from_categories(
            self.problem, profiled.support, profiled.categories
        )
        L_assign, L_K, counts = category_partition_code_bits(labels)
        L_z, lam = quantized_laplace_code_bits(
            z, resolution=cfg.value_resolution
        )
        fixed_bits = self._fixed_bits(profiled.support)
        dl = float(fixed_bits + L_K + L_assign + L_z)
        values = {k: float(v) for k, v in zip(order, z)}
        weights = {
            g: values[profiled.categories[g]] for g in profiled.support
        }
        meta = {
            "n_categories": len(order),
            "category_order": order,
            "category_counts": counts,
            "category_values_ls": {
                k: float(v) for k, v in zip(order, z_ls)
            },
            "category_values": values,
            "relative_residual_ls": float(fit.relative_residual),
            "L_fixed_bits": float(fixed_bits),
            "L_K_bits": float(L_K),
            "L_assign_bits": float(L_assign),
            "L_z_bits": float(L_z),
            "laplace_lambda": float(lam),
            "shrink_moves": int(n_shrink),
        }
        return dl, float(rho), weights, meta

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
        cfg = self.peixoto_config
        if fit.relative_residual > self.problem.uncertainty_floor * (1.0 + cfg.floor_rtol):
            raise In1InfeasibleError(
                "Fixed-(J,S,c) physical profile lies outside B_epsilon: "
                f"rho={fit.relative_residual:.6e}, "
                f"eps={fit.uncertainty_floor:.6e}."
            )

        peixoto_dl, rho, weights, meta = self._exact_peixoto_state(profiled)
        if self.description_length_scorer is None:
            dl = peixoto_dl
        else:
            dl = float(
                self.description_length_scorer(
                    frozenset(profiled.support),
                    dict(profiled.categories),
                    profiled,
                )
            )
            if not math.isfinite(dl):
                raise ValueError(
                    "description_length_scorer must return a finite value."
                )

        return ProfiledState(
            support=frozenset(profiled.support),
            categories=dict(profiled.categories),
            description_length=float(dl),
            relative_residual=float(rho),
            weights=weights,
            theta=fit.theta.copy(),
            metadata={
                **({} if warm_start is None else dict(warm_start.metadata)),
                "physical_profiling": True,
                **meta,
                "optimizer_nfev": int(fit.optimizer_nfev),
                "optimizer_success": bool(fit.optimizer_success),
                "relaxed_relative_residual": float(fit.relaxed_relative_residual),
            },
        )

    # ---- proposal helpers -------------------------------------------------

    def _local_two_way(
        self,
        evaluator: _FixedThetaCategoryEvaluator,
        labels: np.ndarray,
        members: np.ndarray,
        a: int,
        b: int,
    ) -> np.ndarray | None:
        lab = np.asarray(labels, dtype=int).copy()
        current = evaluator.score(lab)
        if current is None:
            return None
        for _ in range(self.peixoto_config.local_two_way_sweeps):
            best = None
            for i in members:
                old = int(lab[i])
                if old not in (a, b):
                    continue
                vals = lab[members]
                if int(np.sum(vals == old)) <= 1:
                    continue
                new = b if old == a else a
                cand = lab.copy()
                cand[i] = new
                sc = evaluator.score(cand)
                if (
                    sc is not None
                    and float(sc["total_bits"])
                    < float(current["total_bits"]) - 1.0e-10
                ):
                    if best is None or float(sc["total_bits"]) < best[0]:
                        best = (float(sc["total_bits"]), cand, sc)
            if best is None:
                break
            lab = best[1]
            current = best[2]
        return compact_labels(lab)

    def _reassignments(
        self, state: ProfiledState, evaluator: _FixedThetaCategoryEvaluator
    ) -> list[tuple[float, float, np.ndarray, dict[str, Any]]]:
        _, lab = _labels_from_categories(self.problem, state.support, state.categories)
        K = int(lab.max()) + 1
        counts = np.bincount(lab, minlength=K)
        out = []
        for i, old in enumerate(lab):
            if counts[old] <= 1:
                continue
            for new in range(K):
                if new == old:
                    continue
                cand = lab.copy()
                cand[i] = new
                sc = evaluator.score(cand)
                if sc is not None:
                    out.append((
                        float(sc["total_bits"]),
                        float(sc["rho_ls"]),
                        compact_labels(cand),
                        {"index": int(i), "old": int(old), "new": int(new)},
                    ))
        return sorted(out, key=lambda x: (x[0], x[1]))

    def _merges(
        self, state: ProfiledState, evaluator: _FixedThetaCategoryEvaluator
    ) -> list[tuple[float, float, np.ndarray, dict[str, Any]]]:
        _, lab = _labels_from_categories(self.problem, state.support, state.categories)
        K = int(lab.max()) + 1
        out = []
        for a in range(K):
            for b in range(a + 1, K):
                cand = lab.copy()
                cand[cand == b] = a
                cand = compact_labels(cand)
                sc = evaluator.score(cand)
                if sc is not None:
                    out.append((
                        float(sc["total_bits"]),
                        float(sc["rho_ls"]),
                        cand,
                        {"a": int(a), "b": int(b)},
                    ))
        return sorted(out, key=lambda x: (x[0], x[1]))

    def _splits(
        self,
        state: ProfiledState,
        evaluator: _FixedThetaCategoryEvaluator,
        rng: Any,
    ) -> list[tuple[float, float, np.ndarray, dict[str, Any]]]:
        _, lab = _labels_from_categories(self.problem, state.support, state.categories)
        K = int(lab.max()) + 1
        out = []
        for k in range(K):
            members = np.flatnonzero(lab == k)
            if members.size < 2:
                continue
            seeds: list[np.ndarray] = []
            alt = lab.copy()
            alt[members[1::2]] = K
            seeds.append(alt)
            for _ in range(self.peixoto_config.split_random_seeds):
                if hasattr(rng, "integers"):
                    mask = np.asarray(rng.random(members.size) < 0.5, dtype=bool)
                    pick = lambda: int(rng.integers(members.size))
                else:
                    mask = np.asarray(
                        [rng.random() < 0.5 for _ in range(members.size)], dtype=bool
                    )
                    pick = lambda: int(rng.randrange(members.size))
                if not mask.any():
                    mask[pick()] = True
                if mask.all():
                    mask[pick()] = False
                cand = lab.copy()
                cand[members[mask]] = K
                seeds.append(cand)
            for seed in seeds:
                relaxed = self._local_two_way(evaluator, seed, members, k, K)
                if relaxed is None or int(relaxed.max()) + 1 != K + 1:
                    continue
                sc = evaluator.score(relaxed)
                if sc is not None:
                    out.append((
                        float(sc["total_bits"]),
                        float(sc["rho_ls"]),
                        relaxed,
                        {"source": int(k)},
                    ))
        return sorted(out, key=lambda x: (x[0], x[1]))

    def _merge_splits(
        self,
        state: ProfiledState,
        evaluator: _FixedThetaCategoryEvaluator,
        rng: Any,
    ) -> list[tuple[float, float, np.ndarray, dict[str, Any]]]:
        _, lab = _labels_from_categories(self.problem, state.support, state.categories)
        K = int(lab.max()) + 1
        pair_scores = []
        for a in range(K):
            for b in range(a + 1, K):
                merged = lab.copy()
                merged[merged == b] = a
                merged = compact_labels(merged)
                rho, _, _ = evaluator.profile(merged)
                pair_scores.append((float(rho), a, b))
        pair_scores.sort()

        out = []
        for _, a, b in pair_scores[: self.peixoto_config.merge_split_pair_top]:
            members = np.flatnonzero((lab == a) | (lab == b))
            seeds: list[np.ndarray] = [lab.copy()]
            alt = lab.copy()
            alt[members[::2]] = a
            alt[members[1::2]] = b
            seeds.append(alt)
            for _ in range(self.peixoto_config.merge_split_random_seeds):
                if hasattr(rng, "integers"):
                    mask = np.asarray(rng.random(members.size) < 0.5, dtype=bool)
                    pick = lambda: int(rng.integers(members.size))
                else:
                    mask = np.asarray(
                        [rng.random() < 0.5 for _ in range(members.size)], dtype=bool
                    )
                    pick = lambda: int(rng.randrange(members.size))
                if not mask.any():
                    mask[pick()] = True
                if mask.all():
                    mask[pick()] = False
                cand = lab.copy()
                cand[members[mask]] = a
                cand[members[~mask]] = b
                seeds.append(cand)
            for seed in seeds:
                relaxed = self._local_two_way(evaluator, seed, members, a, b)
                if relaxed is None or int(relaxed.max()) + 1 != K:
                    continue
                sc = evaluator.score(relaxed)
                if sc is not None:
                    out.append((
                        float(sc["total_bits"]),
                        float(sc["rho_ls"]),
                        relaxed,
                        {"a": int(a), "b": int(b)},
                    ))
        return sorted(out, key=lambda x: (x[0], x[1]))

    def propose(
        self,
        state: ProfiledState,
        move_kind: str,
        *,
        rng: Any,
    ) -> Iterable[CategoryMove]:
        if self.proposal_fn is not None:
            return self.proposal_fn(state, move_kind, rng)

        evaluator = _FixedThetaCategoryEvaluator(self, state)
        if move_kind == "reassign":
            candidates = self._reassignments(state, evaluator)
            exact_top = self.peixoto_config.reassignment_exact_top
        elif move_kind == "merge":
            candidates = self._merges(state, evaluator)
            exact_top = self.peixoto_config.merge_exact_top
        elif move_kind == "split":
            candidates = self._splits(state, evaluator, rng)
            exact_top = self.peixoto_config.split_exact_top
        elif move_kind == "merge_split":
            candidates = self._merge_splits(state, evaluator, rng)
            exact_top = self.peixoto_config.merge_split_exact_top
        else:
            raise ValueError(f"Unknown in1 move kind {move_kind!r}.")

        ordered = _ordered_support(self.problem, state.support)
        moves: list[CategoryMove] = []
        for cheap_bits, cheap_rho, labels, meta in candidates[: int(exact_top)]:
            moves.append(CategoryMove(
                kind=move_kind,
                categories=_categories_from_labels(ordered, labels),
                metadata={
                    **meta,
                    "fixed_theta_screen_bits": float(cheap_bits),
                    "fixed_theta_screen_rho_ls": float(cheap_rho),
                },
            ))
        return tuple(moves)


# ---------------------------------------------------------------------------
# Generic in1 driver
# ---------------------------------------------------------------------------


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
        and state.relative_residual <= eps * (1.0 + 2.0e-10)
        and math.isfinite(state.description_length)
    )


def run_weight_category_distribution(
    initial: ProfiledState,
    backend: CategoryBackend,
    *,
    config: In1Config,
) -> In1Result:
    """Relax the category representation at fixed (J,S)."""

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

    rng = np.random.default_rng(config.random_seed)
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
                    proposal_failures.append(CategoryProposalFailure(
                        sweep=sweep,
                        move_kind=move_kind,
                        categories=dict(move.categories),
                        failure_kind="profile_infeasible",
                        reason=str(exc),
                        metadata=dict(move.metadata),
                    ))
                    continue

                if frozenset(candidate.support) != support0:
                    raise RuntimeError(
                        "in1 proposal changed S; category moves must not alter support."
                    )
                _validate_partition(support0, candidate.categories)

                if not _is_feasible(candidate, config.uncertainty_floor):
                    proposal_failures.append(CategoryProposalFailure(
                        sweep=sweep,
                        move_kind=move_kind,
                        categories=dict(move.categories),
                        failure_kind="outside_uncertainty_floor",
                        reason=(
                            "Profiled category proposal is not a feasible finite-"
                            "description representation inside B_epsilon."
                        ),
                        relative_residual=float(candidate.relative_residual),
                        description_length=float(candidate.description_length),
                        metadata=dict(move.metadata),
                    ))
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
    "In1InfeasibleError",
    "ProfiledState",
    "CategoryMove",
    "CategoryProposalFailure",
    "In1Config",
    "In1Result",
    "PeixotoCategoryConfig",
    "CategoryBackend",
    "FixedDescriptionBitsFn",
    "PhysicalCategoryComputation",
    "compact_labels",
    "category_partition_code_bits",
    "quantized_laplace_code_bits",
    "shrink_values_to_floor",
    "run_weight_category_distribution",
    "run_in1",
]
