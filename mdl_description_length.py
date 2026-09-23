"""Discrete representation description length for TIDES Step 3.

The active discrete representation is (J, S, c): dynamics support J,
structural support S, and the unlabeled weight-category partition c.
For a physically feasible representation,

    L_rep(J,S,c) = L_J(J) + L_S(S|J) + L_c(c|S).

Continuous W and Theta are not assigned a q-bit precision penalty here.
Feasibility is handled by physical profiling; continuous robustness is handled
separately by basin geometry.  All code lengths are returned in nats.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import lgamma, log
from typing import Any, Callable, Hashable, Mapping, Sequence

Coordinate = Hashable
CategoryLabel = Hashable


def log_choose(n: int, k: int) -> float:
    """log binomial(n,k), in nats."""
    n, k = int(n), int(k)
    if n < 0:
        raise ValueError("n must be non-negative.")
    if k < 0 or k > n:
        raise ValueError("Require 0 <= k <= n.")
    if k == 0 or k == n:
        return 0.0
    k = min(k, n-k)
    return float(lgamma(n+1.0) - lgamma(k+1.0) - lgamma(n-k+1.0))


def uniform_model_index_code_nats(n_models: int) -> float:
    """Uniform code for one member of a predeclared finite J-family."""
    n_models = int(n_models)
    if n_models < 1:
        raise ValueError("n_models must be >= 1.")
    return float(log(n_models))


def subset_support_code_nats(candidate_count: int, active_count: int) -> float:
    """Two-part support code: log(M+1) + log binomial(M,E)."""
    M, E = int(candidate_count), int(active_count)
    if M < 0:
        raise ValueError("candidate_count must be non-negative.")
    if E < 0 or E > M:
        raise ValueError("Require 0 <= active_count <= candidate_count.")
    return float(log(M+1.0) + log_choose(M, E))


@dataclass(frozen=True)
class StructuralSupportBlock:
    """One independently encoded structural support block."""
    candidate_count: int
    active_count: int
    label: Hashable | None = None

    def __post_init__(self) -> None:
        M, E = int(self.candidate_count), int(self.active_count)
        if M < 0:
            raise ValueError("candidate_count must be non-negative.")
        if E < 0 or E > M:
            raise ValueError("Require 0 <= active_count <= candidate_count.")
        object.__setattr__(self, "candidate_count", M)
        object.__setattr__(self, "active_count", E)


TemporalSupportBlock = StructuralSupportBlock


def structural_support_code_nats(
    blocks: Sequence[StructuralSupportBlock],
) -> tuple[float, tuple[float, ...]]:
    components = tuple(
        subset_support_code_nats(b.candidate_count, b.active_count)
        for b in blocks
    )
    return float(sum(components)), components


def _partition_sizes(
    support: Sequence[Coordinate] | frozenset[Coordinate],
    categories: Mapping[Coordinate, CategoryLabel],
) -> tuple[int, tuple[int, ...]]:
    support = frozenset(support)
    keys = frozenset(categories)
    if keys != support:
        missing, extra = support-keys, keys-support
        raise ValueError(
            "Category assignment must cover exactly the active support. "
            f"missing={sorted(map(repr, missing))}, extra={sorted(map(repr, extra))}"
        )
    E = len(support)
    if E == 0:
        return 0, ()
    counts: dict[CategoryLabel, int] = {}
    for g in support:
        label = categories[g]
        counts[label] = counts.get(label, 0) + 1
    sizes = tuple(sorted(counts.values(), reverse=True))
    return E, sizes


def category_partition_code_nats(
    support: Sequence[Coordinate] | frozenset[Coordinate],
    categories: Mapping[Coordinate, CategoryLabel],
) -> float:
    """Exchangeable code for the unlabeled category partition.

    TIDES fixes the unit-concentration CRP/Ewens partition law

        P(c|S) = prod_k (n_k-1)! / E!,

    hence

        L_c = log(E!) - sum_k log((n_k-1)!).

    The code is invariant to category-label renaming.  The concentration is
    fixed at one by the model definition and is not fitted to the trajectory.
    """
    E, sizes = _partition_sizes(support, categories)
    if E == 0:
        return 0.0
    return float(lgamma(E+1.0) - sum(lgamma(n) for n in sizes))


def category_partition_summary(
    support: Sequence[Coordinate] | frozenset[Coordinate],
    categories: Mapping[Coordinate, CategoryLabel],
) -> tuple[int, tuple[int, ...]]:
    _, sizes = _partition_sizes(support, categories)
    return len(sizes), sizes


@dataclass(frozen=True)
class RepresentationDLScore:
    """Discrete description length of one Step-3 representation."""
    total: float
    dynamics: float
    structure: float
    categories: float
    structural_components: tuple[float, ...]
    n_structural_coordinates: int
    n_categories: int
    category_sizes: tuple[int, ...]

    @property
    def total_bits(self) -> float:
        return float(self.total / log(2.0))

    @property
    def dynamics_bits(self) -> float:
        return float(self.dynamics / log(2.0))

    @property
    def structure_bits(self) -> float:
        return float(self.structure / log(2.0))

    @property
    def categories_bits(self) -> float:
        return float(self.categories / log(2.0))


def compute_representation_description_length(
    *,
    dynamics_code_length: float,
    structural_blocks: Sequence[StructuralSupportBlock],
    support: Sequence[Coordinate] | frozenset[Coordinate],
    categories: Mapping[Coordinate, CategoryLabel],
) -> RepresentationDLScore:
    """Compute L_rep = L_J + L_S + L_c for an already feasible model."""
    L_J = float(dynamics_code_length)
    if L_J < 0.0:
        raise ValueError("dynamics_code_length must be non-negative.")
    blocks = tuple(structural_blocks)
    L_S, components = structural_support_code_nats(blocks)
    support = frozenset(support)
    implied_E = sum(b.active_count for b in blocks)
    if implied_E != len(support):
        raise ValueError(
            "Structural block active counts must equal support size: "
            f"blocks imply {implied_E}, support has {len(support)}."
        )
    L_c = category_partition_code_nats(support, categories)
    K, sizes = category_partition_summary(support, categories)
    return RepresentationDLScore(
        total=float(L_J + L_S + L_c),
        dynamics=L_J,
        structure=L_S,
        categories=L_c,
        structural_components=components,
        n_structural_coordinates=len(support),
        n_categories=K,
        category_sizes=sizes,
    )


StructuralBlockBuilder = Callable[
    [frozenset[Coordinate]], Sequence[StructuralSupportBlock]
]


def make_in1_description_length_scorer(
    *,
    dynamics_code_length: float,
    structural_block_builder: StructuralBlockBuilder,
) -> Callable[
    [frozenset[Coordinate], Mapping[Coordinate, CategoryLabel], Any], float
]:
    """Adapter with the signature expected by PhysicalCategoryComputation.

    The physical profile argument is deliberately ignored: residual depth,
    W, and Theta are not part of the discrete representation code.
    """
    L_J = float(dynamics_code_length)
    if L_J < 0.0:
        raise ValueError("dynamics_code_length must be non-negative.")

    def scorer(support, categories, profile) -> float:
        del profile
        return compute_representation_description_length(
            dynamics_code_length=L_J,
            structural_blocks=tuple(structural_block_builder(frozenset(support))),
            support=frozenset(support),
            categories=categories,
        ).total
    return scorer


def fixed_candidate_blocks_from_labels(
    support: Sequence[Coordinate] | frozenset[Coordinate],
    *,
    candidate_count_by_block: Mapping[Hashable, int],
    block_of_coordinate: Callable[[Coordinate], Hashable],
) -> tuple[StructuralSupportBlock, ...]:
    """Build support blocks without hard-coding TIDES coordinate labels."""
    support = frozenset(support)
    counts = {label: 0 for label in candidate_count_by_block}
    for g in support:
        label = block_of_coordinate(g)
        if label not in counts:
            raise ValueError(f"Coordinate {g!r} maps to unknown block {label!r}.")
        counts[label] += 1
    return tuple(
        StructuralSupportBlock(
            candidate_count=int(candidate_count_by_block[label]),
            active_count=int(counts[label]),
            label=label,
        )
        for label in candidate_count_by_block
    )


__all__ = [
    "Coordinate", "CategoryLabel", "StructuralSupportBlock",
    "TemporalSupportBlock", "RepresentationDLScore", "StructuralBlockBuilder",
    "log_choose", "uniform_model_index_code_nats", "subset_support_code_nats",
    "structural_support_code_nats", "category_partition_code_nats",
    "category_partition_summary", "compute_representation_description_length",
    "make_in1_description_length_scorer", "fixed_candidate_blocks_from_labels",
]
