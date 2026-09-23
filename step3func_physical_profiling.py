"""
Shared physical-profiling utilities for TIDES Step 3.

Purpose
-------
This module contains the numerical calculations reused inside Step 3.  It is
not an inference stage and it contains no topology-search or MDL-search logic.

For the fixed-dynamics physical hypothesis,

    B^(r) = W^(r) Theta^T,

the independent structural coordinates are the scalar entries of

    W^(1), Delta W^(1), ..., Delta W^(K-1).

The functions below provide four numerical operations needed by the new Step 3:

1. joint variable projection:
       fixed (J, S[, c]) -> profile W and Theta;

2. fixed-Theta structural profiling:
       fixed (J, S, Theta[, c]) -> profile W exactly by least squares;

3. conditional-LS repair geometry for in2;

4. feasible shared-law initialization for a fixed J.

Weight-category sharing is handled as a representation constraint.  If several
active structural coordinates share one category value, their observation-space
blocks are summed logically, without changing the underlying structural support.

Deliberately NOT included here
------------------------------
* MDL / description-length definitions;
* q-bit quantization or finite-precision coding;
* category proposal moves;
* structural add/delete/swap search;
* basin-retention compression;
* cross-J model comparison.

Those belong to separate Step-3 layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Hashable, Mapping, Optional, Sequence

import numpy as np

try:  # package form
    from .structural_linear_cache import StructuralLinearCache
    from .edge_space_operators import (
        TemporalEdgeBlock,
        coerce_block,
        dense_block,
        full_support_relaxation,
    )
except ImportError:  # flat-file form
    from structural_linear_cache import StructuralLinearCache
    from edge_space_operators import (
        TemporalEdgeBlock,
        coerce_block,
        dense_block,
        full_support_relaxation,
    )


FloatArray = np.ndarray
Coordinate = Hashable
CategoryLabel = Hashable


# ---------------------------------------------------------------------------
# Compact block algebra
# ---------------------------------------------------------------------------


class SummedStructuralBlock:
    """Logical sum of structural blocks sharing one scalar category amplitude.

    The object keeps category aggregation representation-level: individual
    structural coordinates remain identifiable, while the profiled scalar
    amplitude is shared.  Matrix-vector products are evaluated as sums of the
    constituent block responses, so compact TemporalEdgeBlock representations
    need not be materialized as full dense n_rows x L arrays.
    """

    ndim = 2

    def __init__(self, blocks: Sequence[Any]):
        parts = tuple(_coerce_structural_block(b) for b in blocks)
        if not parts:
            raise ValueError("SummedStructuralBlock requires at least one block.")
        shape = tuple(parts[0].shape)
        if len(shape) != 2:
            raise ValueError("Structural blocks must be two-dimensional.")
        if any(tuple(b.shape) != shape for b in parts):
            raise ValueError("All blocks in a category must have identical shape.")
        self.blocks = parts
        self.shape = shape

    def __matmul__(self, theta: FloatArray) -> FloatArray:
        theta = np.asarray(theta, dtype=float).reshape(-1)
        if theta.size != self.shape[1]:
            raise ValueError("Coefficient vector has incompatible length.")
        out = np.zeros(self.shape[0], dtype=float)
        for block in self.blocks:
            out += np.asarray(block @ theta, dtype=float)
        return out

    def __getitem__(self, key):
        rows, columns = key
        if not isinstance(rows, slice) or rows != slice(None):
            raise IndexError("Only full row slices are supported.")
        return SummedStructuralBlock(tuple(block[:, columns] for block in self.blocks))

    def to_dense(self) -> FloatArray:
        out = np.zeros(self.shape, dtype=float)
        for block in self.blocks:
            out += np.asarray(dense_block(block), dtype=float)
        return out


def _coerce_structural_block(block: Any) -> Any:
    if isinstance(block, SummedStructuralBlock):
        return block
    return coerce_block(block)


def _iter_primitive_blocks(block: Any):
    if isinstance(block, SummedStructuralBlock):
        for child in block.blocks:
            yield from _iter_primitive_blocks(child)
    else:
        yield block


def _dense_selected_columns(block: Any, atoms: Sequence[int]) -> FloatArray:
    atoms = tuple(int(a) for a in atoms)
    if isinstance(block, SummedStructuralBlock):
        out = np.zeros((block.shape[0], len(atoms)), dtype=float)
        for primitive in _iter_primitive_blocks(block):
            out += _dense_selected_columns(primitive, atoms)
        return out
    if isinstance(block, TemporalEdgeBlock):
        return np.asarray(dense_block(block[:, np.asarray(atoms, dtype=int)]), dtype=float)
    return np.asarray(block[:, np.asarray(atoms, dtype=int)], dtype=float)


def _derivative_products(
    blocks: Sequence[Any],
    amplitudes: FloatArray,
    residual: FloatArray,
    atoms: Sequence[int],
) -> tuple[FloatArray, FloatArray]:
    """Return [C_j a] and [C_j^T r] without an L x n x p tensor.

    This is the old VarPro derivative calculation generalized to logical sums
    of blocks used by weight categories.
    """

    atoms = tuple(int(a) for a in atoms)
    residual = np.asarray(residual, dtype=float).reshape(-1)
    amplitudes = np.asarray(amplitudes, dtype=float).reshape(-1)
    if len(blocks) != amplitudes.size:
        raise ValueError("amplitudes must match blocks.")

    v = np.zeros((residual.size, len(atoms)), dtype=float)
    b = np.zeros((len(blocks), len(atoms)), dtype=float)

    for i, (block, amplitude) in enumerate(zip(blocks, amplitudes)):
        for primitive in _iter_primitive_blocks(block):
            if isinstance(primitive, TemporalEdgeBlock):
                values = primitive.edge.values[
                    primitive.start:, np.asarray(atoms, dtype=int)
                ]
                v[primitive.rows] += float(amplitude) * values
                b[i] += values.T @ residual[primitive.rows]
            else:
                values = np.asarray(primitive[:, atoms], dtype=float)
                v += float(amplitude) * values
                b[i] += values.T @ residual

    return v, b


# ---------------------------------------------------------------------------
# Fixed-dynamics problem representation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FixedDynamicsProblem:
    """One fixed-J Step-3 physical profiling problem.

    ``structural_blocks[g]`` is the observation-space linear map associated
    with one independent structural coordinate g.  No MDL metadata is stored
    here: this object describes only the numerical physical-fitting problem.
    """

    y: FloatArray
    structural_blocks: Mapping[Coordinate, Any]
    uncertainty_floor: float
    linear_cache: StructuralLinearCache = field(
        default_factory=StructuralLinearCache,
        compare=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        y = np.asarray(self.y, dtype=float).reshape(-1)
        if y.size == 0 or not np.all(np.isfinite(y)):
            raise ValueError("y must be finite and non-empty.")
        if not self.structural_blocks:
            raise ValueError("structural_blocks must be non-empty.")

        blocks: dict[Coordinate, Any] = {}
        first_shape = None
        for coordinate, raw in self.structural_blocks.items():
            block = _coerce_structural_block(raw)
            if getattr(block, "ndim", None) != 2:
                raise ValueError(f"Structural block {coordinate!r} must be 2D.")
            shape = tuple(block.shape)
            if shape[0] != y.size:
                raise ValueError(
                    f"Structural block {coordinate!r} has {shape[0]} rows; "
                    f"expected {y.size}."
                )
            if first_shape is None:
                first_shape = shape
            elif shape != first_shape:
                raise ValueError(
                    "All structural blocks must have the same "
                    "(n_observations, n_library_atoms) shape."
                )
            if not isinstance(block, (TemporalEdgeBlock, SummedStructuralBlock)):
                if not np.all(np.isfinite(block)):
                    raise ValueError(
                        f"Structural block {coordinate!r} contains non-finite values."
                    )
            blocks[coordinate] = block

        eps = float(self.uncertainty_floor)
        if not np.isfinite(eps) or eps < 0.0:
            raise ValueError("uncertainty_floor must be finite and non-negative.")

        object.__setattr__(self, "y", y)
        object.__setattr__(self, "structural_blocks", blocks)
        object.__setattr__(self, "uncertainty_floor", eps)

    @property
    def coordinates(self) -> tuple[Coordinate, ...]:
        return tuple(self.structural_blocks.keys())

    @property
    def n_library_atoms(self) -> int:
        first = next(iter(self.structural_blocks.values()))
        return int(first.shape[1])

    def normalise_support(
        self,
        support: Sequence[Coordinate] | frozenset[Coordinate],
    ) -> tuple[Coordinate, ...]:
        requested = set(support)
        unknown = requested - set(self.structural_blocks)
        if unknown:
            raise ValueError(
                "Unknown structural coordinates: "
                f"{sorted(map(repr, unknown))}"
            )
        return tuple(g for g in self.coordinates if g in requested)

    def selected_blocks(
        self,
        support: Sequence[Coordinate] | frozenset[Coordinate],
    ) -> tuple[Any, ...]:
        return tuple(self.structural_blocks[g] for g in self.normalise_support(support))


def normalise_active_atoms(
    problem: FixedDynamicsProblem,
    active_atoms: Sequence[int],
) -> tuple[int, ...]:
    atoms = tuple(sorted(set(int(a) for a in active_atoms)))
    if not atoms:
        raise ValueError("active_atoms must be non-empty.")
    if any(a < 0 or a >= problem.n_library_atoms for a in atoms):
        raise ValueError("active_atoms contains an out-of-range index.")
    return atoms


# ---------------------------------------------------------------------------
# Core variable-projection geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FixedDynamicsProfile:
    """Joint continuous profile of structural amplitudes and shared dynamics."""

    active_atoms: tuple[int, ...]
    pivot_atom: int
    theta: FloatArray
    amplitudes: FloatArray
    residual: FloatArray
    residual_sq: float
    relative_residual: float
    feasible_witness: bool

    relaxed_relative_residual: float
    relaxation_proves_infeasible: bool

    y_norm: float
    uncertainty_floor: float
    threshold_sq: float

    optimizer_nfev: int
    optimizer_success: bool

    @property
    def n_structural_amplitudes(self) -> int:
        return int(self.amplitudes.size)

    @property
    def n_free_dynamics_coefficients(self) -> int:
        return max(0, len(self.active_atoms) - 1)

    @property
    def n_free_parameters(self) -> int:
        return self.n_structural_amplitudes + self.n_free_dynamics_coefficients


@dataclass(frozen=True)
class FixedDynamicsVarProWorkspace:
    """Precomputed fixed-support geometry for variable projection."""

    y: FloatArray
    structural_blocks: tuple[Any, ...]
    linear_cache: StructuralLinearCache
    active_atoms: tuple[int, ...]
    pivot_atom: int
    y_norm: float
    uncertainty_floor: float
    threshold_sq: float
    full_library_size: int
    relaxed_relative_residual: float
    relaxation_proves_infeasible: bool


def _validate_fixed_dynamics_blocks(
    y: FloatArray,
    structural_blocks: Sequence[Any],
) -> tuple[FloatArray, tuple[Any, ...], int]:
    y = np.asarray(y, dtype=float).reshape(-1)
    if y.size == 0:
        raise ValueError("y must be non-empty.")

    blocks = tuple(_coerce_structural_block(b) for b in structural_blocks)
    if not blocks:
        raise ValueError("At least one active structural block is required.")

    L = int(blocks[0].shape[1])
    if L < 1:
        raise ValueError("Structural blocks must contain at least one library atom.")

    for i, block in enumerate(blocks):
        if tuple(block.shape) != (y.size, L):
            raise ValueError(
                f"structural block {i} has shape {block.shape}; "
                f"expected {(y.size, L)}."
            )
        if not isinstance(block, (TemporalEdgeBlock, SummedStructuralBlock)):
            if not np.all(np.isfinite(block)):
                raise ValueError(f"structural block {i} contains non-finite values.")

    return y, blocks, L


def build_fixed_dynamics_varpro_workspace(
    y: FloatArray,
    structural_blocks: Sequence[Any],
    *,
    active_atoms: Sequence[int],
    uncertainty_floor: float,
    compute_linear_relaxation: bool = True,
    linear_cache: Optional[StructuralLinearCache] = None,
) -> FixedDynamicsVarProWorkspace:
    """Precompute one fixed-support variable-projection workspace."""

    y, blocks, L = _validate_fixed_dynamics_blocks(y, structural_blocks)
    atoms = tuple(sorted(set(int(a) for a in active_atoms)))
    if not atoms:
        raise ValueError("active_atoms must be non-empty.")
    if any(a < 0 or a >= L for a in atoms):
        raise ValueError("active_atoms contains an out-of-range atom index.")

    y_norm = float(np.linalg.norm(y))
    if y_norm <= np.finfo(float).tiny:
        raise ValueError("y must have nonzero norm.")

    eps = float(uncertainty_floor)
    if not np.isfinite(eps) or eps < 0.0:
        raise ValueError("uncertainty_floor must be finite and non-negative.")
    threshold_sq = float((eps * y_norm) ** 2)

    relaxed_rho = 0.0
    relaxation_infeasible = False

    if compute_linear_relaxation:
        # The specialized shortcut is exact only for the original complete
        # TemporalEdgeBlock support.  Logical category sums deliberately fall
        # through to the generic linear relaxation.
        shortcut = None
        if not any(isinstance(b, SummedStructuralBlock) for b in blocks):
            shortcut = full_support_relaxation(blocks, atoms)

        if shortcut is not None:
            relaxed_rho = float(shortcut)
        else:
            A_relaxed = np.hstack(
                [_dense_selected_columns(block, atoms) for block in blocks]
            )
            beta, _, _, _ = np.linalg.lstsq(A_relaxed, y, rcond=None)
            r_relaxed = np.asarray(y - A_relaxed @ beta, dtype=float)
            relaxed_rho = float(np.linalg.norm(r_relaxed) / y_norm)

        relaxation_infeasible = bool(
            relaxed_rho > eps * (1.0 + 1e-10)
        )

    return FixedDynamicsVarProWorkspace(
        y=y,
        structural_blocks=blocks,
        linear_cache=StructuralLinearCache() if linear_cache is None else linear_cache,
        active_atoms=atoms,
        pivot_atom=int(atoms[0]),
        y_norm=y_norm,
        uncertainty_floor=eps,
        threshold_sq=threshold_sq,
        full_library_size=L,
        relaxed_relative_residual=float(relaxed_rho),
        relaxation_proves_infeasible=bool(relaxation_infeasible),
    )


def profile_fixed_dynamics_varpro(
    workspace: FixedDynamicsVarProWorkspace,
    *,
    warm_theta: Optional[FloatArray] = None,
    max_nfev: int = 80,
    include_default_start: bool = True,
) -> FixedDynamicsProfile:
    """Locally minimize E(Theta)=min_W rho using exact variable projection.

    Gauge convention: the first active dynamics atom is fixed to coefficient 1.
    Only the remaining |J|-1 dynamics coefficients are optimized nonlinearly.
    Structural amplitudes are solved by SVD at every Theta.
    """

    from scipy.optimize import least_squares

    atoms = workspace.active_atoms
    d = len(atoms)
    y = workspace.y
    blocks = workspace.structural_blocks
    scale = max(
        workspace.uncertainty_floor * workspace.y_norm,
        np.finfo(float).tiny,
    )

    def theta_active_from_x(x: FloatArray) -> FloatArray:
        th = np.zeros(d, dtype=float)
        th[0] = 1.0
        if d > 1:
            th[1:] = np.asarray(x, dtype=float)
        return th

    def compute_state(x: FloatArray, need_jac: bool):
        theta_active = theta_active_from_x(x)
        theta_full = np.zeros(workspace.full_library_size, dtype=float)
        theta_full[list(atoms)] = theta_active

        key = workspace.linear_cache.key("svd", blocks, theta_full, y)
        fit = workspace.linear_cache.get(key)

        if fit is None:
            C = workspace.linear_cache.matrix(
                blocks,
                theta_full,
                cache_columns=(d == 1),
            )
            U, singular_values, Vt = np.linalg.svd(C, full_matrices=False)

            if singular_values.size:
                tol = (
                    np.finfo(float).eps
                    * max(C.shape)
                    * float(singular_values[0])
                )
                keep = singular_values > tol
            else:
                keep = np.zeros(0, dtype=bool)

            Ur = U[:, keep]
            sr = singular_values[keep]
            Vr = Vt[keep, :].T

            if sr.size:
                uy = Ur.T @ y
                amplitudes = Vr @ (uy / sr)
                residual = np.asarray(y - Ur @ uy, dtype=float)
            else:
                amplitudes = np.zeros(C.shape[1], dtype=float)
                residual = y.copy()

            fit = workspace.linear_cache.put(
                key,
                (Ur, sr, Vr, amplitudes, residual),
            )

        Ur, sr, Vr, amplitudes, residual = fit

        if not need_jac or d <= 1:
            return theta_active, amplitudes, residual, None

        vdir, b = _derivative_products(
            blocks,
            amplitudes,
            residual,
            atoms[1:],
        )

        if sr.size:
            term1 = -(vdir - Ur @ (Ur.T @ vdir))
            term2 = -Ur @ ((Vr.T @ b) / sr[:, None])
            jacobian = term1 + term2
        else:
            jacobian = -vdir

        return theta_active, amplitudes, residual, np.asarray(jacobian, dtype=float)

    starts: list[FloatArray] = []

    if include_default_start:
        starts.append(np.zeros(max(0, d - 1), dtype=float))

    if warm_theta is not None:
        warm = np.asarray(warm_theta, dtype=float).reshape(-1)
        if warm.size != workspace.full_library_size:
            raise ValueError("warm_theta has the wrong library dimension.")
        pivot = float(warm[workspace.pivot_atom])
        if abs(pivot) > 1e-14:
            warm = warm / pivot
            starts.append(
                np.asarray([warm[a] for a in atoms[1:]], dtype=float)
            )

    if d > 1 and not starts:
        raise ValueError("A local dynamics start is required.")

    unique_starts: list[FloatArray] = []
    for start in starts:
        if not any(
            np.allclose(start, existing, rtol=0.0, atol=1e-14)
            for existing in unique_starts
        ):
            unique_starts.append(start)

    best = None
    total_nfev = 0
    any_success = False

    if d == 1:
        theta_active, amplitudes, residual, _ = compute_state(
            np.zeros(0, dtype=float),
            False,
        )
        best = (
            float(residual @ residual),
            theta_active,
            amplitudes,
            residual,
        )
        any_success = True
    else:
        for x0 in unique_starts:
            cache_x = None
            cache_state = None

            def cached(x: FloatArray):
                nonlocal cache_x, cache_state
                xx = np.asarray(x, dtype=float)
                if cache_x is None or not np.array_equal(xx, cache_x):
                    cache_x = xx.copy()
                    cache_state = compute_state(xx, True)
                return cache_state

            solution = least_squares(
                lambda x: cached(x)[2] / scale,
                x0,
                jac=lambda x: cached(x)[3] / scale,
                method="trf",
                x_scale="jac",
                xtol=1e-12,
                ftol=1e-12,
                gtol=1e-12,
                max_nfev=max(1, int(max_nfev)),
            )

            total_nfev += int(solution.nfev)
            any_success = any_success or bool(solution.success)

            theta_active, amplitudes, residual, _ = compute_state(
                solution.x,
                False,
            )
            rss = float(residual @ residual)

            if best is None or rss < best[0]:
                best = (rss, theta_active, amplitudes, residual)

    assert best is not None

    rss, theta_active, amplitudes, residual = best
    theta = np.zeros(workspace.full_library_size, dtype=float)
    theta[np.asarray(atoms, dtype=int)] = theta_active

    rho = float(np.sqrt(max(rss, 0.0)) / workspace.y_norm)
    feasible = bool(
        rss
        <= workspace.threshold_sq
        * (1.0 + 100.0 * np.finfo(float).eps)
    )

    return FixedDynamicsProfile(
        active_atoms=atoms,
        pivot_atom=workspace.pivot_atom,
        theta=theta,
        amplitudes=np.asarray(amplitudes, dtype=float),
        residual=np.asarray(residual, dtype=float),
        residual_sq=float(rss),
        relative_residual=rho,
        feasible_witness=feasible,
        relaxed_relative_residual=workspace.relaxed_relative_residual,
        relaxation_proves_infeasible=workspace.relaxation_proves_infeasible,
        y_norm=workspace.y_norm,
        uncertainty_floor=workspace.uncertainty_floor,
        threshold_sq=workspace.threshold_sq,
        optimizer_nfev=int(total_nfev),
        optimizer_success=bool(any_success),
    )


def profile_fixed_dynamics(
    y: FloatArray,
    structural_blocks: Sequence[Any],
    *,
    active_atoms: Sequence[int],
    uncertainty_floor: float,
    warm_theta: Optional[FloatArray] = None,
    max_nfev: int = 250,
    compute_linear_relaxation: bool = True,
    include_default_start: bool = True,
    linear_cache: Optional[StructuralLinearCache] = None,
) -> FixedDynamicsProfile:
    """Convenience wrapper around the reusable VarPro workspace."""

    workspace = build_fixed_dynamics_varpro_workspace(
        y,
        structural_blocks,
        active_atoms=active_atoms,
        uncertainty_floor=uncertainty_floor,
        compute_linear_relaxation=compute_linear_relaxation,
        linear_cache=linear_cache,
    )
    return profile_fixed_dynamics_varpro(
        workspace,
        warm_theta=warm_theta,
        max_nfev=max_nfev,
        include_default_start=include_default_start,
    )


# ---------------------------------------------------------------------------
# Support-level and category-level profiling
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FreeWeightProfile:
    support: tuple[Coordinate, ...]
    fit: FixedDynamicsProfile
    weights: Mapping[Coordinate, float]


@dataclass(frozen=True)
class CategoryProfile:
    support: tuple[Coordinate, ...]
    categories: Mapping[Coordinate, CategoryLabel]
    category_order: tuple[CategoryLabel, ...]
    category_members: Mapping[CategoryLabel, tuple[Coordinate, ...]]
    fit: FixedDynamicsProfile
    category_values: Mapping[CategoryLabel, float]
    weights: Mapping[Coordinate, float]


def profile_free_weights_and_dynamics(
    problem: FixedDynamicsProblem,
    support: Sequence[Coordinate] | frozenset[Coordinate],
    *,
    active_atoms: Sequence[int],
    warm_theta: Optional[FloatArray] = None,
    max_nfev: int = 250,
    compute_linear_relaxation: bool = True,
    include_default_start: bool = True,
) -> FreeWeightProfile:
    """Profile independent W coordinates and the shared law Theta."""

    ordered_support = problem.normalise_support(support)
    if not ordered_support:
        raise ValueError("At least one structural coordinate must be active.")
    atoms = normalise_active_atoms(problem, active_atoms)

    fit = profile_fixed_dynamics(
        problem.y,
        tuple(problem.structural_blocks[g] for g in ordered_support),
        active_atoms=atoms,
        uncertainty_floor=problem.uncertainty_floor,
        warm_theta=warm_theta,
        max_nfev=max_nfev,
        compute_linear_relaxation=compute_linear_relaxation,
        include_default_start=include_default_start,
        linear_cache=problem.linear_cache,
    )

    weights = {
        g: float(value)
        for g, value in zip(ordered_support, fit.amplitudes)
    }
    return FreeWeightProfile(ordered_support, fit, weights)


def aggregate_blocks_by_category(
    problem: FixedDynamicsProblem,
    support: Sequence[Coordinate] | frozenset[Coordinate],
    categories: Mapping[Coordinate, CategoryLabel],
) -> tuple[
    tuple[CategoryLabel, ...],
    dict[CategoryLabel, tuple[Coordinate, ...]],
    tuple[SummedStructuralBlock, ...],
]:
    """Construct one logical observation block per shared weight category."""

    ordered_support = problem.normalise_support(support)
    if set(categories) != set(ordered_support):
        missing = set(ordered_support) - set(categories)
        extra = set(categories) - set(ordered_support)
        raise ValueError(
            "Category assignment must cover exactly the active support. "
            f"missing={sorted(map(repr, missing))}, "
            f"extra={sorted(map(repr, extra))}"
        )

    order: list[CategoryLabel] = []
    members: dict[CategoryLabel, list[Coordinate]] = {}

    for coordinate in ordered_support:
        label = categories[coordinate]
        if label not in members:
            order.append(label)
            members[label] = []
        members[label].append(coordinate)

    frozen_members = {
        label: tuple(coords)
        for label, coords in members.items()
    }

    blocks = tuple(
        SummedStructuralBlock(
            tuple(problem.structural_blocks[g] for g in frozen_members[label])
        )
        for label in order
    )

    return tuple(order), frozen_members, blocks


def profile_category_weights_and_dynamics(
    problem: FixedDynamicsProblem,
    support: Sequence[Coordinate] | frozenset[Coordinate],
    categories: Mapping[Coordinate, CategoryLabel],
    *,
    active_atoms: Sequence[int],
    warm_theta: Optional[FloatArray] = None,
    max_nfev: int = 250,
    compute_linear_relaxation: bool = False,
    include_default_start: bool = True,
) -> CategoryProfile:
    """Profile category-shared structural values together with Theta.

    If c_g = c_h, then w_g = w_h is enforced exactly by replacing the separate
    observation blocks with one logical summed block.
    """

    ordered_support = problem.normalise_support(support)
    atoms = normalise_active_atoms(problem, active_atoms)
    category_order, members, category_blocks = aggregate_blocks_by_category(
        problem,
        ordered_support,
        categories,
    )

    fit = profile_fixed_dynamics(
        problem.y,
        category_blocks,
        active_atoms=atoms,
        uncertainty_floor=problem.uncertainty_floor,
        warm_theta=warm_theta,
        max_nfev=max_nfev,
        compute_linear_relaxation=compute_linear_relaxation,
        include_default_start=include_default_start,
        linear_cache=problem.linear_cache,
    )

    category_values = {
        label: float(value)
        for label, value in zip(category_order, fit.amplitudes)
    }

    weights = {
        coordinate: category_values[categories[coordinate]]
        for coordinate in ordered_support
    }

    return CategoryProfile(
        support=ordered_support,
        categories=dict(categories),
        category_order=category_order,
        category_members=members,
        fit=fit,
        category_values=category_values,
        weights=weights,
    )


# ---------------------------------------------------------------------------
# Fixed-Theta profiling and in2 proposal geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FixedThetaProfile:
    theta: FloatArray
    amplitudes: FloatArray
    residual: FloatArray
    residual_sq: float
    relative_residual: float
    rank: int
    singular_values: FloatArray
    feasible: bool


def profile_structural_amplitudes_at_theta(
    y: FloatArray,
    structural_blocks: Sequence[Any],
    theta: FloatArray,
    *,
    uncertainty_floor: float,
    linear_cache: Optional[StructuralLinearCache] = None,
) -> FixedThetaProfile:
    """Solve the structural amplitudes exactly for one fixed shared law."""

    y, blocks, L = _validate_fixed_dynamics_blocks(y, structural_blocks)
    theta = np.asarray(theta, dtype=float).reshape(-1)
    if theta.size != L:
        raise ValueError("theta has the wrong library dimension.")

    cache = StructuralLinearCache() if linear_cache is None else linear_cache
    C = cache.matrix(blocks, theta)

    scales = np.sqrt(np.mean(C * C, axis=0))
    scales = np.asarray(scales, dtype=float)
    bad = ~np.isfinite(scales) | (scales <= np.finfo(float).tiny)
    scales[bad] = 1.0

    Cs = C / scales[None, :]
    beta_scaled, _, rank, singular_values = np.linalg.lstsq(
        Cs,
        y,
        rcond=None,
    )
    amplitudes = np.asarray(beta_scaled / scales, dtype=float)
    residual = np.asarray(y - C @ amplitudes, dtype=float)

    rss = float(residual @ residual)
    y_norm = float(np.linalg.norm(y))
    rho = float(np.sqrt(max(rss, 0.0)) / max(y_norm, np.finfo(float).tiny))

    eps = float(uncertainty_floor)
    feasible = bool(
        rho <= eps * (1.0 + 100.0 * np.finfo(float).eps)
    )

    return FixedThetaProfile(
        theta=theta,
        amplitudes=amplitudes,
        residual=residual,
        residual_sq=rss,
        relative_residual=rho,
        rank=int(rank),
        singular_values=np.asarray(singular_values, dtype=float),
        feasible=feasible,
    )


def profile_free_weights_at_theta(
    problem: FixedDynamicsProblem,
    support: Sequence[Coordinate] | frozenset[Coordinate],
    theta: FloatArray,
) -> tuple[tuple[Coordinate, ...], FixedThetaProfile]:
    """Free-W profile used directly by the in3 basin geometry."""

    ordered_support = problem.normalise_support(support)
    if not ordered_support:
        raise ValueError("At least one structural coordinate must be active.")
    profile = profile_structural_amplitudes_at_theta(
        problem.y,
        tuple(problem.structural_blocks[g] for g in ordered_support),
        theta,
        uncertainty_floor=problem.uncertainty_floor,
        linear_cache=problem.linear_cache,
    )
    return ordered_support, profile


def profile_category_weights_at_theta(
    problem: FixedDynamicsProblem,
    support: Sequence[Coordinate] | frozenset[Coordinate],
    categories: Mapping[Coordinate, CategoryLabel],
    theta: FloatArray,
) -> tuple[
    tuple[CategoryLabel, ...],
    dict[CategoryLabel, tuple[Coordinate, ...]],
    FixedThetaProfile,
]:
    """Fixed-Theta structural profile under category-sharing constraints."""

    labels, members, blocks = aggregate_blocks_by_category(
        problem,
        support,
        categories,
    )
    profile = profile_structural_amplitudes_at_theta(
        problem.y,
        blocks,
        theta,
        uncertainty_floor=problem.uncertainty_floor,
        linear_cache=problem.linear_cache,
    )
    return labels, members, profile


@dataclass(frozen=True)
class RepairGain:
    coordinate: Coordinate
    gain: float
    residualized_norm_sq: float
    score_defined: bool


def conditional_ls_repair_gain(
    problem: FixedDynamicsProblem,
    current_support: Sequence[Coordinate] | frozenset[Coordinate],
    candidate: Coordinate,
    theta: FloatArray,
) -> RepairGain:
    """Conditional least-squares repair gain used as an in2 proposal score.

    With the current support profiled at fixed Theta,

        v_tilde = (I - P_S) v_candidate,

        G = (r^T v_tilde)^2 / ||v_tilde||^2.

    G is a proposal score only.  It is not structural truth evidence.
    """

    support = problem.normalise_support(current_support)
    if candidate not in problem.structural_blocks:
        raise ValueError(f"Unknown candidate coordinate {candidate!r}.")
    if candidate in support:
        raise ValueError("Repair candidate is already active.")

    theta = np.asarray(theta, dtype=float).reshape(-1)
    if theta.size != problem.n_library_atoms:
        raise ValueError("theta has the wrong library dimension.")

    if support:
        blocks = tuple(problem.structural_blocks[g] for g in support)
        current = profile_structural_amplitudes_at_theta(
            problem.y,
            blocks,
            theta,
            uncertainty_floor=problem.uncertainty_floor,
            linear_cache=problem.linear_cache,
        )
        C = problem.linear_cache.matrix(blocks, theta)
        U, singular_values, _ = np.linalg.svd(C, full_matrices=False)

        if singular_values.size:
            tol = (
                np.finfo(float).eps
                * max(C.shape)
                * float(singular_values[0])
            )
            Q = U[:, singular_values > tol]
        else:
            Q = np.zeros((problem.y.size, 0), dtype=float)

        residual = current.residual
    else:
        Q = np.zeros((problem.y.size, 0), dtype=float)
        residual = problem.y

    v = np.asarray(problem.structural_blocks[candidate] @ theta, dtype=float)
    v_tilde = v - Q @ (Q.T @ v) if Q.shape[1] else v
    norm_sq = float(v_tilde @ v_tilde)

    if norm_sq <= np.finfo(float).tiny:
        return RepairGain(candidate, 0.0, norm_sq, False)

    gain = float((residual @ v_tilde) ** 2 / norm_sq)
    return RepairGain(candidate, gain, norm_sq, True)


def conditional_ls_repair_gains(
    problem: FixedDynamicsProblem,
    current_support: Sequence[Coordinate] | frozenset[Coordinate],
    candidates: Sequence[Coordinate],
    theta: FloatArray,
) -> tuple[RepairGain, ...]:
    """Evaluate the in2 conditional-LS gain for multiple inactive candidates."""

    return tuple(
        conditional_ls_repair_gain(
            problem,
            current_support,
            candidate,
            theta,
        )
        for candidate in candidates
    )


# ---------------------------------------------------------------------------
# Fixed-J feasible shared-law initialization
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SharedLawInitializationStep:
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
class SharedLawInitializationResult:
    initial_profile: FixedDynamicsProfile
    best_profile: FixedDynamicsProfile
    history: tuple[SharedLawInitializationStep, ...]
    reached_floor: bool
    seed: int | None


def discover_feasible_shared_law(
    problem: FixedDynamicsProblem,
    support: Sequence[Coordinate] | frozenset[Coordinate],
    *,
    active_atoms: Sequence[int],
    n_hops: int = 80,
    sigma_min: float = 5e-3,
    sigma_max: float = 1.5e-1,
    min_block_size: int = 1,
    max_block_size: Optional[int] = None,
    small_block_probability: float = 0.60,
    atom_complexities: Optional[Sequence[float]] = None,
    low_complexity_bias_power: float = 1.0,
    local_max_nfev: int = 50,
    seed: Optional[int] = None,
    initial_theta: Optional[FloatArray] = None,
    improvement_rtol: float = 1e-12,
) -> SharedLawInitializationResult:
    """Search only for a floor-feasible shared law at fixed (J,S).

    This is initialization, not topology search and not MDL optimization.
    Structural support and active dynamics atoms are held fixed throughout.
    """

    ordered_support = problem.normalise_support(support)
    if not ordered_support:
        raise ValueError("Shared-law initialization requires nonempty support.")

    atoms = normalise_active_atoms(problem, active_atoms)
    free_atoms = tuple(a for a in atoms if a != atoms[0])

    if n_hops < 0 or not (0.0 < sigma_min <= sigma_max):
        raise ValueError("Invalid shared-law initialization controls.")

    blocks = tuple(problem.structural_blocks[g] for g in ordered_support)
    workspace = build_fixed_dynamics_varpro_workspace(
        problem.y,
        blocks,
        active_atoms=atoms,
        uncertainty_floor=problem.uncertainty_floor,
        compute_linear_relaxation=True,
        linear_cache=problem.linear_cache,
    )

    initial = profile_fixed_dynamics_varpro(
        workspace,
        warm_theta=initial_theta,
        max_nfev=local_max_nfev,
        include_default_start=True,
    )

    if initial.feasible_witness or not free_atoms:
        return SharedLawInitializationResult(
            initial_profile=initial,
            best_profile=initial,
            history=(),
            reached_floor=bool(initial.feasible_witness),
            seed=seed,
        )

    d = len(free_atoms)
    kmin = max(1, int(min_block_size))
    kmax = d if max_block_size is None else min(d, int(max_block_size))
    if kmin > kmax:
        raise ValueError("Invalid block-size range.")

    p_small = float(small_block_probability)
    if not (0.0 < p_small <= 1.0):
        raise ValueError("small_block_probability must lie in (0,1].")

    bias_power = float(low_complexity_bias_power)
    if bias_power < 0.0 or not np.isfinite(bias_power):
        raise ValueError("low_complexity_bias_power must be finite and non-negative.")

    if atom_complexities is None:
        complexities = np.ones(problem.n_library_atoms, dtype=float)
    else:
        complexities = np.asarray(atom_complexities, dtype=float).reshape(-1)
        if complexities.size != problem.n_library_atoms:
            raise ValueError("atom_complexities has the wrong length.")
        if np.any(~np.isfinite(complexities)) or np.any(complexities <= 0.0):
            raise ValueError("atom_complexities must be finite and strictly positive.")

    rng = np.random.default_rng(seed)
    best = initial
    history: list[SharedLawInitializationStep] = []

    log_lo = np.log(float(sigma_min))
    log_hi = np.log(float(sigma_max))
    free_arr = np.asarray(free_atoms, dtype=int)

    probabilities = complexities[free_arr] ** (-bias_power)
    probabilities /= probabilities.sum()

    for iteration in range(int(n_hops)):
        k = kmin - 1 + int(rng.geometric(p_small))
        k = min(max(k, kmin), kmax)

        chosen = tuple(
            int(x)
            for x in rng.choice(
                free_arr,
                size=k,
                replace=False,
                p=probabilities,
            )
        )

        sigma = float(np.exp(rng.uniform(log_lo, log_hi)))
        raw_theta = np.asarray(best.theta, dtype=float).copy()
        raw_theta[np.asarray(chosen, dtype=int)] += sigma * rng.normal(size=k)
        raw_theta[best.pivot_atom] = 1.0

        candidate = profile_fixed_dynamics_varpro(
            workspace,
            warm_theta=raw_theta,
            max_nfev=local_max_nfev,
            include_default_start=False,
        )

        threshold = best.relative_residual * (1.0 - float(improvement_rtol))
        accepted = bool(candidate.relative_residual < threshold)
        if accepted:
            best = candidate

        history.append(
            SharedLawInitializationStep(
                iteration=iteration,
                block_size=k,
                sigma=sigma,
                perturbed_atoms=tuple(sorted(chosen)),
                raw_theta=raw_theta,
                relaxed_theta=np.asarray(candidate.theta, dtype=float),
                proposed_relative_residual=float(candidate.relative_residual),
                accepted=accepted,
                best_relative_residual=float(best.relative_residual),
                optimizer_nfev=int(candidate.optimizer_nfev),
            )
        )

        if best.feasible_witness:
            break

    return SharedLawInitializationResult(
        initial_profile=initial,
        best_profile=best,
        history=tuple(history),
        reached_floor=bool(best.feasible_witness),
        seed=seed,
    )


__all__ = [
    "Coordinate",
    "CategoryLabel",
    "SummedStructuralBlock",
    "FixedDynamicsProblem",
    "FixedDynamicsProfile",
    "FixedDynamicsVarProWorkspace",
    "FixedThetaProfile",
    "FreeWeightProfile",
    "CategoryProfile",
    "RepairGain",
    "SharedLawInitializationStep",
    "SharedLawInitializationResult",
    "normalise_active_atoms",
    "build_fixed_dynamics_varpro_workspace",
    "profile_fixed_dynamics_varpro",
    "profile_fixed_dynamics",
    "profile_free_weights_and_dynamics",
    "aggregate_blocks_by_category",
    "profile_category_weights_and_dynamics",
    "profile_structural_amplitudes_at_theta",
    "profile_free_weights_at_theta",
    "profile_category_weights_at_theta",
    "conditional_ls_repair_gain",
    "conditional_ls_repair_gains",
    "discover_feasible_shared_law",
]
