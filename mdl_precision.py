"""Fixed-model numerical profiling for TIDES conditional-MDL search.

This module is intentionally separate from the outer combinatorial search.
Given a concrete linear design A for a fixed discrete model (S, J), it answers:

1. Can the model class enter the observational uncertainty ball at all?
2. Can we exhibit a q-bit coefficient vector that remains in the ball?

The second question is *not* equivalent to rounding one arbitrary least-squares
representative.  Rank-deficient TIDES designs possess large observational
fibres, and the minimum-MDL representative may live far along a null direction.
Accordingly the code distinguishes:

- an ``optimal`` continuous profile used only to certify the best achievable
  residual of the model class; and
- a ``seed`` continuous representative used to search for short quantized
  witnesses.  By default the seed is the canonical unscaled minimum-Euclidean-
  norm least-squares solution whenever it is already floor-feasible.  This is a
  much better starting point than the column-scaled numerical representative in
  strongly rank-deficient polynomial designs.

The exact manuscript quantizer Q_q is still an open design choice.  The
``UniformDyadicQuantizer`` below is therefore explicitly a reference/smoke-test
quantizer.  Its fixed range is caller-declared side information.

The current lattice search returns a feasible witness and hence an *upper bound*
on q_star.  It does not certify that no lower-q point exists.  Final scientific
MDL search must either certify this inner problem or use lower/upper bounds that
are sufficient for the acceptance decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

FloatArray = np.ndarray


@dataclass(frozen=True)
class ContinuousProfile:
    """Continuous geometry of a fixed linear model."""

    # Best numerical fit used for the continuous feasibility certificate.
    optimal_coefficients: FloatArray
    optimal_residual: FloatArray
    optimal_residual_sq: float
    relative_residual: float

    # Canonical continuous seed used by the quantized-witness search.
    seed_coefficients: FloatArray
    seed_residual: FloatArray
    seed_residual_sq: float
    seed_relative_residual: float
    seed_residual_cross: FloatArray

    feasible: bool
    rank: int
    singular_values: FloatArray
    gram: FloatArray
    basis: FloatArray
    y_norm: float
    uncertainty_floor: float
    threshold_sq: float

    @property
    def n_parameters(self) -> int:
        return int(self.seed_coefficients.size)

    @property
    def coefficients(self) -> FloatArray:
        """Backward-compatible alias for the quantization/search seed."""
        return self.seed_coefficients

    @property
    def residual(self) -> FloatArray:
        """Backward-compatible alias for the optimal residual."""
        return self.optimal_residual

    @property
    def residual_sq(self) -> float:
        """Backward-compatible alias for the optimal RSS."""
        return self.optimal_residual_sq

    @property
    def quantization_budget_sq(self) -> float:
        return float(self.threshold_sq - self.seed_residual_sq)


@dataclass(frozen=True)
class UniformDyadicQuantizer:
    """Nested q-bit codebook on a fixed canonical range [-bound, bound)."""

    bound: float = 1.0

    def __post_init__(self) -> None:
        b = float(self.bound)
        if not np.isfinite(b) or b <= 0.0:
            raise ValueError("bound must be finite and strictly positive.")
        object.__setattr__(self, "bound", b)

    def step(self, q: int) -> float:
        q = int(q)
        if q < 1:
            raise ValueError("q must be >= 1.")
        return float(np.ldexp(self.bound, -(q - 1)))

    def integer_limits(self, q: int) -> tuple[int, int]:
        q = int(q)
        if q < 1:
            raise ValueError("q must be >= 1.")
        half = 1 << (q - 1)
        return -half, half - 1

    def quantize(self, values: FloatArray, q: int) -> FloatArray:
        values = np.asarray(values, dtype=float)
        step = self.step(q)
        lo, hi = self.integer_limits(q)
        k = np.clip(np.rint(values / step), lo, hi)
        return np.asarray(k * step, dtype=float)


@dataclass(frozen=True)
class QuantizedWitness:
    q: int
    coefficients: FloatArray
    relative_residual: float
    residual_sq: float
    feasible: bool
    coordinate_passes: int


@dataclass(frozen=True)
class PrecisionProfile:
    """Smallest q at which the implemented search found a feasible witness."""

    q_upper: Optional[int]
    witness: Optional[QuantizedWitness]
    tested_q: tuple[int, ...]
    certified_exact: bool
    quantizer: UniformDyadicQuantizer

    @property
    def feasible(self) -> bool:
        return self.witness is not None and self.witness.feasible

    @property
    def q_star(self) -> Optional[int]:
        return self.q_upper if self.certified_exact else None


def _rms_scales(A: FloatArray) -> FloatArray:
    scales = np.sqrt(np.mean(A * A, axis=0))
    scales = np.asarray(scales, dtype=float)
    bad = ~np.isfinite(scales) | (scales <= np.finfo(float).tiny)
    scales[bad] = 1.0
    return scales


def profile_continuous(
    A: FloatArray,
    y: FloatArray,
    *,
    uncertainty_floor: float,
    rcond: Optional[float] = None,
    compute_basis: bool = True,
) -> ContinuousProfile:
    """Profile continuous feasibility and construct a canonical seed.

    ``optimal`` uses the same RMS-column-scaled numerical geometry as the
    canonical TIDES dense least-squares solver.  ``seed`` uses the unscaled
    minimum-Euclidean-norm solution when that solution is already inside the
    floor; otherwise it falls back to the optimal numerical representative.
    """

    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if A.ndim != 2 or A.shape[0] != y.size:
        raise ValueError("A must be 2D and row-compatible with y.")
    eps = float(uncertainty_floor)
    if not np.isfinite(eps) or eps < 0.0:
        raise ValueError("uncertainty_floor must be finite and non-negative.")
    y_norm = float(np.linalg.norm(y))
    if y_norm <= np.finfo(float).tiny:
        raise ValueError("y must have nonzero norm.")
    threshold_sq = float((eps * y_norm) ** 2)

    if A.shape[1] == 0:
        optimal_c = seed_c = np.zeros(0, dtype=float)
        optimal_r = seed_r = y.copy()
        svals = np.zeros(0, dtype=float)
        rank = 0
        gram = np.zeros((0, 0), dtype=float)
        basis = np.zeros((A.shape[0], 0), dtype=float)
    else:
        # Floor-optimal numerical profile: match TIDES column scaling.
        scales = _rms_scales(A)
        As = A / scales[None, :]
        beta_s, _, rank, svals = np.linalg.lstsq(As, y, rcond=rcond)
        optimal_c = np.asarray(beta_s / scales, dtype=float)
        optimal_r = np.asarray(y - A @ optimal_c, dtype=float)

        if compute_basis:
            U, _, _ = np.linalg.svd(As, full_matrices=False)
            basis = np.asarray(U[:, : int(rank)], dtype=float)
        else:
            basis = np.zeros((A.shape[0], 0), dtype=float)

        # Canonical seed: minimum Euclidean norm in the *unscaled* coefficient
        # coordinates.  This selects a much smaller representative on a large
        # observational fibre and is only a seed, not an MDL optimum claim.
        raw_c, _, _, _ = np.linalg.lstsq(A, y, rcond=rcond)
        raw_c = np.asarray(raw_c, dtype=float)
        raw_r = np.asarray(y - A @ raw_c, dtype=float)
        if float(raw_r @ raw_r) <= threshold_sq * (1.0 + 1e-12):
            seed_c, seed_r = raw_c, raw_r
        else:
            seed_c, seed_r = optimal_c.copy(), optimal_r.copy()

        gram = np.asarray(A.T @ A, dtype=float)

    optimal_rss = float(optimal_r @ optimal_r)
    optimal_rho = float(np.sqrt(max(optimal_rss, 0.0)) / y_norm)
    tol = 100.0 * np.finfo(float).eps * max(threshold_sq, optimal_rss, 1.0)
    feasible = bool(optimal_rss <= threshold_sq + tol)

    seed_rss = float(seed_r @ seed_r)
    seed_rho = float(np.sqrt(max(seed_rss, 0.0)) / y_norm)
    seed_cross = (
        np.zeros(0, dtype=float)
        if A.shape[1] == 0
        else np.asarray(A.T @ seed_r, dtype=float)
    )

    return ContinuousProfile(
        optimal_coefficients=np.asarray(optimal_c, dtype=float),
        optimal_residual=np.asarray(optimal_r, dtype=float),
        optimal_residual_sq=optimal_rss,
        relative_residual=optimal_rho,
        seed_coefficients=np.asarray(seed_c, dtype=float),
        seed_residual=np.asarray(seed_r, dtype=float),
        seed_residual_sq=seed_rss,
        seed_relative_residual=seed_rho,
        seed_residual_cross=seed_cross,
        feasible=feasible,
        rank=int(rank),
        singular_values=np.asarray(svals, dtype=float),
        gram=gram,
        basis=basis,
        y_norm=y_norm,
        uncertainty_floor=eps,
        threshold_sq=threshold_sq,
    )


def exact_add_residual_sq(
    current: ContinuousProfile,
    block: FloatArray,
) -> tuple[float, float]:
    """Exact conditional RSS reduction from opening one additional block."""

    B = np.asarray(block, dtype=float)
    if B.ndim == 1:
        B = B[:, None]
    if B.ndim != 2 or B.shape[0] != current.optimal_residual.size:
        raise ValueError("block must be row-compatible with the current model.")
    if current.basis.shape[1]:
        Z = B - current.basis @ (current.basis.T @ B)
    else:
        Z = B.copy()
    if np.linalg.norm(Z) <= np.finfo(float).tiny:
        return current.optimal_residual_sq, 0.0
    alpha, _, _, _ = np.linalg.lstsq(Z, current.optimal_residual, rcond=None)
    fitted = Z @ alpha
    gain = float(max(fitted @ fitted, 0.0))
    return float(max(current.optimal_residual_sq - gain, 0.0)), gain


def _coordinate_descent_quantized(
    profile: ContinuousProfile,
    quantizer: UniformDyadicQuantizer,
    q: int,
    *,
    initial: Optional[FloatArray] = None,
    max_passes: int = 8,
) -> tuple[FloatArray, float, int]:
    """Locally minimize exact RSS on the q-bit lattice around a seed."""

    c0 = profile.seed_coefficients
    Q = c0.size
    if Q == 0:
        return np.zeros(0, dtype=float), profile.seed_residual_sq, 0
    cq = quantizer.quantize(c0 if initial is None else np.asarray(initial, dtype=float), q)
    G = profile.gram
    h = profile.seed_residual_cross
    diag = np.diag(G)
    e = cq - c0
    Ge = G @ e
    rss = float(max(profile.seed_residual_sq - 2.0 * (e @ h) + e @ Ge, 0.0))
    step = quantizer.step(q)
    lo, hi = quantizer.integer_limits(q)

    passes_done = 0
    for p in range(max(0, int(max_passes))):
        improved = False
        for i in range(Q):
            gii = float(diag[i])
            if gii <= np.finfo(float).tiny:
                continue
            grad_half = float(Ge[i] - h[i])
            target = cq[i] - grad_half / gii
            k0 = int(np.rint(target / step))
            best_val = cq[i]
            best_delta = 0.0
            for k in (k0 - 1, k0, k0 + 1):
                if k < lo or k > hi:
                    continue
                val = float(k * step)
                d = val - cq[i]
                if d == 0.0:
                    continue
                delta = 2.0 * d * grad_half + d * d * gii
                if delta < best_delta:
                    best_delta = float(delta)
                    best_val = val
            if best_val != cq[i]:
                d = best_val - cq[i]
                cq[i] = best_val
                e[i] += d
                Ge = Ge + d * G[:, i]
                rss = float(max(rss + best_delta, 0.0))
                improved = True
        passes_done = p + 1
        if not improved:
            break

    e = cq - c0
    rss = float(max(profile.seed_residual_sq - 2.0 * (e @ h) + e @ (G @ e), 0.0))
    return np.asarray(cq, dtype=float), rss, passes_done


def quantized_witness(
    profile: ContinuousProfile,
    *,
    q: int,
    quantizer: UniformDyadicQuantizer,
    initial: Optional[FloatArray] = None,
    max_coordinate_passes: int = 8,
) -> QuantizedWitness:
    q = int(q)
    if q < 1:
        raise ValueError("q must be >= 1.")
    if not profile.feasible:
        return QuantizedWitness(q, np.zeros_like(profile.seed_coefficients), np.inf, np.inf, False, 0)
    cq, rss, passes = _coordinate_descent_quantized(
        profile,
        quantizer,
        q,
        initial=initial,
        max_passes=max_coordinate_passes,
    )
    rho = float(np.sqrt(max(rss, 0.0)) / profile.y_norm)
    feasible = bool(rss <= profile.threshold_sq * (1.0 + 100.0 * np.finfo(float).eps))
    return QuantizedWitness(q, cq, rho, rss, feasible, int(passes))


def profile_precision_witness(
    profile: ContinuousProfile,
    *,
    quantizer: UniformDyadicQuantizer,
    q_min: int = 1,
    q_max: int = 64,
    warm_q: Optional[int] = None,
    warm_coefficients: Optional[FloatArray] = None,
    max_coordinate_passes: int = 8,
) -> PrecisionProfile:
    """Return the smallest q at which this heuristic finds a feasible witness."""

    q_min, q_max = int(q_min), int(q_max)
    if q_min < 1 or q_max < q_min:
        raise ValueError("Require 1 <= q_min <= q_max.")
    if not profile.feasible:
        return PrecisionProfile(None, None, tuple(), False, quantizer)
    if profile.n_parameters == 0:
        w = QuantizedWitness(0, np.zeros(0), profile.relative_residual, profile.optimal_residual_sq, True, 0)
        return PrecisionProfile(0, w, (0,), True, quantizer)

    tested = []
    found = None
    for q in range(q_min, q_max + 1):
        w = quantized_witness(
            profile,
            q=q,
            quantizer=quantizer,
            initial=None,
            max_coordinate_passes=max_coordinate_passes,
        )
        if warm_q is not None and q == int(warm_q) and warm_coefficients is not None:
            if np.asarray(warm_coefficients).size == profile.n_parameters:
                ww = quantized_witness(
                    profile,
                    q=q,
                    quantizer=quantizer,
                    initial=warm_coefficients,
                    max_coordinate_passes=max_coordinate_passes,
                )
                if ww.residual_sq < w.residual_sq:
                    w = ww
        tested.append(q)
        if w.feasible:
            found = w
            break
    return PrecisionProfile(
        None if found is None else int(found.q),
        found,
        tuple(tested),
        False,
        quantizer,
    )


# -----------------------------------------------------------------------------
# Fixed-dynamics hypothesis: shared polynomial law, varying scalar structure
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class FixedDynamicsProfile:
    """Continuous profile under H = fixed dynamics / varying structure.

    The model is

        y = sum_a w_a B_a theta + r,

    where the stationary-anchor amplitudes and the active temporal-change
    amplitudes are free scalars, while every block shares one polynomial law
    ``theta``.  The scale gauge is fixed by setting the lowest-index active atom
    to exactly +1.  That pivot is side information implied by the deterministic
    gauge rule and is therefore not counted as a free precision parameter.

    ``relaxed_relative_residual`` is the residual of the unrestricted linear
    relaxation in which every block has its own polynomial coefficient vector.
    It is a rigorous lower bound on the fixed-dynamics residual and can certify
    infeasibility when it already exceeds the uncertainty floor.
    """

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
    n_baseline_amplitudes: int
    n_change_amplitudes: int
    optimizer_nfev: int
    optimizer_success: bool

    @property
    def n_amplitudes(self) -> int:
        return int(self.n_baseline_amplitudes + self.n_change_amplitudes)

    @property
    def n_free_dynamics_coefficients(self) -> int:
        return max(0, len(self.active_atoms) - 1)

    @property
    def n_free_parameters(self) -> int:
        return int(self.n_amplitudes + self.n_free_dynamics_coefficients)


@dataclass(frozen=True)
class FixedDynamicsQuantizedWitness:
    q: int
    theta: FloatArray
    amplitudes: FloatArray
    relative_residual: float
    residual_sq: float
    feasible: bool
    coordinate_passes: int


@dataclass(frozen=True)
class FixedDynamicsPrecisionProfile:
    q_upper: Optional[int]
    witness: Optional[FixedDynamicsQuantizedWitness]
    tested_q: tuple[int, ...]
    certified_exact: bool
    quantizer: UniformDyadicQuantizer

    @property
    def feasible(self) -> bool:
        return self.witness is not None and self.witness.feasible

    @property
    def q_star(self) -> Optional[int]:
        return self.q_upper if self.certified_exact else None


def _validate_fixed_dynamics_blocks(
    y: FloatArray,
    baseline_blocks: Sequence[FloatArray],
    change_blocks: Sequence[FloatArray],
) -> tuple[FloatArray, tuple[FloatArray, ...], tuple[FloatArray, ...], int]:
    y = np.asarray(y, dtype=float).reshape(-1)
    if y.size == 0:
        raise ValueError("y must be non-empty.")
    baseline = tuple(np.asarray(B, dtype=float) for B in baseline_blocks)
    changes = tuple(np.asarray(B, dtype=float) for B in change_blocks)
    all_blocks = baseline + changes
    if not all_blocks:
        raise ValueError("At least one baseline or change block is required.")
    L = int(all_blocks[0].shape[1])
    for i, B in enumerate(all_blocks):
        if B.ndim != 2 or B.shape != (y.size, L):
            raise ValueError(
                f"fixed-dynamics block {i} has shape {B.shape}; "
                f"expected {(y.size, L)}."
            )
    return y, baseline, changes, L


def _fixed_dynamics_prediction(
    blocks: Sequence[FloatArray],
    amplitudes: FloatArray,
    theta: FloatArray,
) -> FloatArray:
    amplitudes = np.asarray(amplitudes, dtype=float).reshape(-1)
    theta = np.asarray(theta, dtype=float).reshape(-1)
    if len(blocks) != amplitudes.size:
        raise ValueError("amplitudes must match the number of blocks.")
    if not blocks:
        return np.zeros(0, dtype=float)
    pred = np.zeros(blocks[0].shape[0], dtype=float)
    for a, B in zip(amplitudes, blocks):
        pred += float(a) * (B @ theta)
    return pred


def profile_fixed_dynamics(
    y: FloatArray,
    baseline_blocks: Sequence[FloatArray],
    change_blocks: Sequence[FloatArray],
    *,
    active_atoms: Sequence[int],
    uncertainty_floor: float,
    warm_theta: Optional[FloatArray] = None,
    max_nfev: int = 250,
    compute_linear_relaxation: bool = True,
    include_default_start: bool = True,
) -> FixedDynamicsProfile:
    """Profile a fixed-dynamics model by variable projection.

    For a fixed shared law theta, all scalar amplitudes are solved exactly by
    least squares.  Only the ``|J|-1`` free law coefficients are optimized
    nonlinearly.  Residuals are scaled by the uncertainty radius so that the
    optimizer remains numerically sensitive at the very small TIDES floor.

    The routine returns a *feasible witness* when it enters the floor.  Failure
    to enter the floor is not, by itself, a certificate of infeasibility because
    the rank-one/shared-law problem is nonconvex.  The unrestricted linear
    relaxation provides a rigorous infeasibility certificate when requested.
    """

    from scipy.optimize import least_squares

    y, baseline, changes, L = _validate_fixed_dynamics_blocks(
        y, baseline_blocks, change_blocks
    )
    atoms = tuple(sorted(set(int(a) for a in active_atoms)))
    if not atoms:
        raise ValueError("active_atoms must be non-empty under fixed dynamics.")
    if any(a < 0 or a >= L for a in atoms):
        raise ValueError("active_atoms contains an out-of-range atom index.")

    pivot = int(atoms[0])
    free_atoms = tuple(a for a in atoms if a != pivot)
    y_norm = float(np.linalg.norm(y))
    if y_norm <= np.finfo(float).tiny:
        raise ValueError("y must have nonzero norm.")
    eps = float(uncertainty_floor)
    if not np.isfinite(eps) or eps < 0.0:
        raise ValueError("uncertainty_floor must be finite and non-negative.")
    threshold_sq = float((eps * y_norm) ** 2)
    residual_scale = max(float(np.sqrt(threshold_sq)), np.finfo(float).tiny)

    blocks_full = baseline + changes
    blocks = tuple(B[:, atoms] for B in blocks_full)
    pivot_local = 0  # atoms are sorted; pivot is atoms[0]

    def theta_from_free(x: FloatArray) -> FloatArray:
        th = np.zeros(len(atoms), dtype=float)
        th[pivot_local] = 1.0
        if free_atoms:
            th[1:] = np.asarray(x, dtype=float)
        return th

    def solve_amplitudes(theta_active: FloatArray):
        C = np.column_stack([B @ theta_active for B in blocks])
        a, _, _, _ = np.linalg.lstsq(C, y, rcond=None)
        r = np.asarray(y - C @ a, dtype=float)
        return np.asarray(a, dtype=float), r

    # Local-start set. Basin-hopping proposals can disable the repeatedly used
    # pivot-only start and relax only from the raw jump.
    starts: list[FloatArray] = []
    if include_default_start:
        starts.append(np.zeros(len(free_atoms), dtype=float))
    if warm_theta is not None:
        wt = np.asarray(warm_theta, dtype=float).reshape(-1)
        if wt.size == L and abs(wt[pivot]) > 1e-14:
            wt = wt / wt[pivot]
            starts.append(np.asarray([wt[a] for a in free_atoms], dtype=float))
    if free_atoms and not starts:
        raise ValueError(
            "At least one fixed-dynamics local start is required. Supply "
            "warm_theta or set include_default_start=True."
        )

    unique_starts: list[FloatArray] = []
    for x in starts:
        if not any(np.allclose(x, z, rtol=0.0, atol=1e-14) for z in unique_starts):
            unique_starts.append(x)

    best = None
    total_nfev = 0
    any_success = False
    if not free_atoms:
        theta_active = theta_from_free(np.zeros(0, dtype=float))
        amplitudes, residual = solve_amplitudes(theta_active)
        rss = float(residual @ residual)
        best = (rss, theta_active, amplitudes, residual)
        any_success = True
    else:
        def residual_fun(x: FloatArray) -> FloatArray:
            th = theta_from_free(x)
            _, r = solve_amplitudes(th)
            return r / residual_scale

        for x0 in unique_starts:
            sol = least_squares(
                residual_fun,
                x0,
                method="trf",
                x_scale="jac",
                xtol=1e-12,
                ftol=1e-12,
                gtol=1e-12,
                max_nfev=max(1, int(max_nfev)),
            )
            total_nfev += int(sol.nfev)
            any_success = any_success or bool(sol.success)
            th = theta_from_free(sol.x)
            a, r = solve_amplitudes(th)
            rss = float(r @ r)
            if best is None or rss < best[0]:
                best = (rss, th, a, r)

    assert best is not None
    rss, theta_active, amplitudes, residual = best
    theta_full = np.zeros(L, dtype=float)
    theta_full[np.asarray(atoms, dtype=int)] = theta_active
    rho = float(np.sqrt(max(rss, 0.0)) / y_norm)
    feasible = bool(rss <= threshold_sq * (1.0 + 100.0 * np.finfo(float).eps))

    if compute_linear_relaxation:
        A_relaxed = np.hstack(blocks)
        beta, _, _, _ = np.linalg.lstsq(A_relaxed, y, rcond=None)
        r_relaxed = np.asarray(y - A_relaxed @ beta, dtype=float)
        relaxed_rho = float(np.linalg.norm(r_relaxed) / y_norm)
    else:
        relaxed_rho = 0.0
    relaxation_proves_infeasible = bool(
        compute_linear_relaxation and relaxed_rho > eps * (1.0 + 1e-10)
    )

    return FixedDynamicsProfile(
        active_atoms=atoms,
        pivot_atom=pivot,
        theta=theta_full,
        amplitudes=np.asarray(amplitudes, dtype=float),
        residual=np.asarray(residual, dtype=float),
        residual_sq=float(rss),
        relative_residual=rho,
        feasible_witness=feasible,
        relaxed_relative_residual=relaxed_rho,
        relaxation_proves_infeasible=relaxation_proves_infeasible,
        y_norm=y_norm,
        uncertainty_floor=eps,
        threshold_sq=threshold_sq,
        n_baseline_amplitudes=len(baseline),
        n_change_amplitudes=len(changes),
        optimizer_nfev=int(total_nfev),
        optimizer_success=bool(any_success),
    )


def _fixed_dynamics_coordinate_descent_quantized(
    profile: FixedDynamicsProfile,
    baseline_blocks: Sequence[FloatArray],
    change_blocks: Sequence[FloatArray],
    quantizer: UniformDyadicQuantizer,
    q: int,
    *,
    max_passes: int = 8,
) -> tuple[FloatArray, FloatArray, float, int]:
    """Local lattice optimization for the fixed-dynamics gauge coordinates."""

    y_dummy = np.zeros(profile.residual.size, dtype=float)
    _, baseline, changes, L = _validate_fixed_dynamics_blocks(
        y_dummy, baseline_blocks, change_blocks
    )
    blocks_full = baseline + changes
    atoms = profile.active_atoms
    blocks = tuple(B[:, atoms] for B in blocks_full)
    pivot_local = 0
    free_local = tuple(range(1, len(atoms)))

    # Reconstruct y from the continuous profile and its residual.  Since
    # residual = y - prediction, this avoids storing y inside the profile.
    theta_active0 = profile.theta[np.asarray(atoms, dtype=int)]
    y = _fixed_dynamics_prediction(blocks, profile.amplitudes, theta_active0) + profile.residual

    a = quantizer.quantize(profile.amplitudes, q)
    theta_active = theta_active0.copy()
    if free_local:
        theta_active[1:] = quantizer.quantize(theta_active0[1:], q)
    theta_active[pivot_local] = 1.0

    pred = _fixed_dynamics_prediction(blocks, a, theta_active)
    r = np.asarray(y - pred, dtype=float)
    step = quantizer.step(q)
    lo, hi = quantizer.integer_limits(q)

    passes_done = 0
    for p in range(max(0, int(max_passes))):
        improved = False

        # Scalar amplitudes: exact 1D quadratic conditional update.
        for i, B in enumerate(blocks):
            v = B @ theta_active
            vv = float(v @ v)
            if vv <= np.finfo(float).tiny:
                continue
            target = float(a[i] + (v @ r) / vv)
            k0 = int(np.rint(target / step))
            best_val = float(a[i])
            best_delta = 0.0
            rv = float(r @ v)
            for k in (k0 - 1, k0, k0 + 1):
                if k < lo or k > hi:
                    continue
                val = float(k * step)
                d = val - a[i]
                if d == 0.0:
                    continue
                delta = -2.0 * d * rv + d * d * vv
                if delta < best_delta:
                    best_delta = float(delta)
                    best_val = val
            if best_val != a[i]:
                d = best_val - a[i]
                a[i] = best_val
                r = r - d * v
                improved = True

        # Shared-law coefficients except the fixed gauge pivot.
        for j in free_local:
            v = np.zeros_like(r)
            for ai, B in zip(a, blocks):
                v += float(ai) * B[:, j]
            vv = float(v @ v)
            if vv <= np.finfo(float).tiny:
                continue
            target = float(theta_active[j] + (v @ r) / vv)
            k0 = int(np.rint(target / step))
            best_val = float(theta_active[j])
            best_delta = 0.0
            rv = float(r @ v)
            for k in (k0 - 1, k0, k0 + 1):
                if k < lo or k > hi:
                    continue
                val = float(k * step)
                d = val - theta_active[j]
                if d == 0.0:
                    continue
                delta = -2.0 * d * rv + d * d * vv
                if delta < best_delta:
                    best_delta = float(delta)
                    best_val = val
            if best_val != theta_active[j]:
                d = best_val - theta_active[j]
                theta_active[j] = best_val
                r = r - d * v
                improved = True

        passes_done = p + 1
        if not improved:
            break

    theta_full = np.zeros(L, dtype=float)
    theta_full[np.asarray(atoms, dtype=int)] = theta_active
    rss = float(r @ r)
    return theta_full, np.asarray(a, dtype=float), rss, passes_done


def fixed_dynamics_quantized_witness(
    profile: FixedDynamicsProfile,
    baseline_blocks: Sequence[FloatArray],
    change_blocks: Sequence[FloatArray],
    *,
    q: int,
    quantizer: UniformDyadicQuantizer,
    max_coordinate_passes: int = 8,
) -> FixedDynamicsQuantizedWitness:
    q = int(q)
    if q < 1:
        raise ValueError("q must be >= 1.")
    if not profile.feasible_witness:
        return FixedDynamicsQuantizedWitness(
            q=q,
            theta=profile.theta.copy(),
            amplitudes=profile.amplitudes.copy(),
            relative_residual=np.inf,
            residual_sq=np.inf,
            feasible=False,
            coordinate_passes=0,
        )
    theta, amplitudes, rss, passes = _fixed_dynamics_coordinate_descent_quantized(
        profile,
        baseline_blocks,
        change_blocks,
        quantizer,
        q,
        max_passes=max_coordinate_passes,
    )
    rho = float(np.sqrt(max(rss, 0.0)) / profile.y_norm)
    feasible = bool(rss <= profile.threshold_sq * (1.0 + 100.0 * np.finfo(float).eps))
    return FixedDynamicsQuantizedWitness(
        q=q,
        theta=theta,
        amplitudes=amplitudes,
        relative_residual=rho,
        residual_sq=float(rss),
        feasible=feasible,
        coordinate_passes=int(passes),
    )


def profile_fixed_dynamics_precision_witness(
    profile: FixedDynamicsProfile,
    baseline_blocks: Sequence[FloatArray],
    change_blocks: Sequence[FloatArray],
    *,
    quantizer: UniformDyadicQuantizer,
    q_min: int = 1,
    q_max: int = 64,
    warm_q: Optional[int] = None,
    max_coordinate_passes: int = 8,
) -> FixedDynamicsPrecisionProfile:
    """Find the first feasible q-bit witness for a fixed-dynamics profile.

    As in the linear reference profiler, this is an upper bound on q_star, not
    a minimality certificate.
    """

    q_min, q_max = int(q_min), int(q_max)
    if q_min < 1 or q_max < q_min:
        raise ValueError("Require 1 <= q_min <= q_max.")
    if not profile.feasible_witness:
        return FixedDynamicsPrecisionProfile(None, None, tuple(), False, quantizer)

    order = list(range(q_min, q_max + 1))
    # We still test all lower q before claiming the first witness, but trying the
    # warm q first can cheaply establish an upper bound used by later screening.
    warm_result = None
    if warm_q is not None and q_min <= int(warm_q) <= q_max:
        warm_result = fixed_dynamics_quantized_witness(
            profile,
            baseline_blocks,
            change_blocks,
            q=int(warm_q),
            quantizer=quantizer,
            max_coordinate_passes=max_coordinate_passes,
        )

    tested = []
    found = None
    upper_limit = q_max if warm_result is None or not warm_result.feasible else int(warm_q)
    for q in range(q_min, upper_limit + 1):
        if warm_result is not None and q == int(warm_q):
            w = warm_result
        else:
            w = fixed_dynamics_quantized_witness(
                profile,
                baseline_blocks,
                change_blocks,
                q=q,
                quantizer=quantizer,
                max_coordinate_passes=max_coordinate_passes,
            )
        tested.append(q)
        if w.feasible:
            found = w
            break

    return FixedDynamicsPrecisionProfile(
        q_upper=None if found is None else int(found.q),
        witness=found,
        tested_q=tuple(tested),
        certified_exact=False,
        quantizer=quantizer,
    )


@dataclass(frozen=True)
class FixedDynamicsVarProWorkspace:
    """Cached fixed-support geometry for repeated shared-law relaxation."""

    y: FloatArray
    derivative_blocks: FloatArray  # (n_active_atoms, n_rows, n_amplitudes)
    active_atoms: tuple[int, ...]
    pivot_atom: int
    y_norm: float
    uncertainty_floor: float
    threshold_sq: float
    n_baseline_amplitudes: int
    n_change_amplitudes: int
    full_library_size: int

    @property
    def n_amplitudes(self) -> int:
        return int(self.n_baseline_amplitudes + self.n_change_amplitudes)


def build_fixed_dynamics_varpro_workspace(
    y: FloatArray,
    baseline_blocks: Sequence[FloatArray],
    change_blocks: Sequence[FloatArray],
    *,
    active_atoms: Sequence[int],
    uncertainty_floor: float,
) -> FixedDynamicsVarProWorkspace:
    """Precompute the repeated block geometry used by variable projection."""

    y, baseline, changes, L = _validate_fixed_dynamics_blocks(
        y, baseline_blocks, change_blocks
    )
    atoms = tuple(sorted(set(int(a) for a in active_atoms)))
    if not atoms:
        raise ValueError("active_atoms must be non-empty.")
    if any(a < 0 or a >= L for a in atoms):
        raise ValueError("active_atoms contains an out-of-range atom index.")
    blocks = baseline + changes
    D = np.stack(
        [np.column_stack([B[:, atom] for B in blocks]) for atom in atoms],
        axis=0,
    )
    y_norm = float(np.linalg.norm(y))
    if y_norm <= np.finfo(float).tiny:
        raise ValueError("y must have nonzero norm.")
    eps = float(uncertainty_floor)
    threshold_sq = float((eps * y_norm) ** 2)
    return FixedDynamicsVarProWorkspace(
        y=np.asarray(y, dtype=float),
        derivative_blocks=np.asarray(D, dtype=float),
        active_atoms=atoms,
        pivot_atom=int(atoms[0]),
        y_norm=y_norm,
        uncertainty_floor=eps,
        threshold_sq=threshold_sq,
        n_baseline_amplitudes=len(baseline),
        n_change_amplitudes=len(changes),
        full_library_size=L,
    )


def profile_fixed_dynamics_varpro(
    workspace: FixedDynamicsVarProWorkspace,
    *,
    warm_theta: Optional[FloatArray] = None,
    max_nfev: int = 80,
    include_default_start: bool = True,
) -> FixedDynamicsProfile:
    """Locally relax a shared law using the exact variable-projection Jacobian.

    This changes only the numerical implementation of the same profiled
    least-squares objective.  Structural amplitudes are eliminated exactly at
    every law evaluation.
    """

    from scipy.optimize import least_squares

    atoms = workspace.active_atoms
    d = len(atoms)
    y = workspace.y
    D = workspace.derivative_blocks
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
        tha = theta_active_from_x(x)
        C = np.tensordot(tha, D, axes=(0, 0))
        U, sv, Vt = np.linalg.svd(C, full_matrices=False)
        if sv.size:
            tol = np.finfo(float).eps * max(C.shape) * float(sv[0])
            keep = sv > tol
        else:
            keep = np.zeros(0, dtype=bool)
        Ur = U[:, keep]
        sr = sv[keep]
        Vr = Vt[keep, :].T
        if sr.size:
            uy = Ur.T @ y
            amp = Vr @ (uy / sr)
            r = np.asarray(y - Ur @ uy, dtype=float)
        else:
            amp = np.zeros(C.shape[1], dtype=float)
            r = np.asarray(y, dtype=float).copy()
        if not need_jac or d <= 1:
            return tha, amp, r, None
        Df = D[1:, :, :]
        vdir = np.einsum('jra,a->rj', Df, amp, optimize=True)
        if sr.size:
            term1 = -(vdir - Ur @ (Ur.T @ vdir))
            b = np.einsum('jra,r->aj', Df, r, optimize=True)
            term2 = -Ur @ ((Vr.T @ b) / sr[:, None])
            jac = term1 + term2
        else:
            jac = -vdir
        return tha, amp, r, np.asarray(jac, dtype=float)

    starts: list[FloatArray] = []
    if include_default_start:
        starts.append(np.zeros(max(0, d - 1), dtype=float))
    if warm_theta is not None:
        wt = np.asarray(warm_theta, dtype=float).reshape(-1)
        if wt.size != workspace.full_library_size:
            raise ValueError("warm_theta has the wrong library dimension.")
        pv = float(wt[workspace.pivot_atom])
        if abs(pv) <= 1e-14:
            raise ValueError("warm_theta has zero gauge-pivot coefficient.")
        wt = wt / pv
        starts.append(np.asarray([wt[a] for a in atoms[1:]], dtype=float))
    if d > 1 and not starts:
        raise ValueError("A local start is required.")

    unique: list[FloatArray] = []
    for x in starts:
        if not any(np.allclose(x, z, rtol=0.0, atol=1e-14) for z in unique):
            unique.append(x)

    best = None
    total_nfev = 0
    any_success = False
    if d == 1:
        tha, amp, r, _ = compute_state(np.zeros(0), False)
        best = (float(r @ r), tha, amp, r)
        any_success = True
    else:
        for x0 in unique:
            cache_x = None
            cache_state = None
            def cached(x: FloatArray):
                nonlocal cache_x, cache_state
                xx = np.asarray(x, dtype=float)
                if cache_x is None or not np.array_equal(xx, cache_x):
                    cache_x = xx.copy()
                    cache_state = compute_state(xx, True)
                return cache_state
            def fun(x: FloatArray) -> FloatArray:
                return cached(x)[2] / scale
            def jac(x: FloatArray) -> FloatArray:
                return cached(x)[3] / scale
            sol = least_squares(
                fun, x0, jac=jac, method='trf', x_scale='jac',
                xtol=1e-12, ftol=1e-12, gtol=1e-12,
                max_nfev=max(1, int(max_nfev)),
            )
            total_nfev += int(sol.nfev)
            any_success = any_success or bool(sol.success)
            tha, amp, r, _ = compute_state(sol.x, False)
            rss = float(r @ r)
            if best is None or rss < best[0]:
                best = (rss, tha, amp, r)

    assert best is not None
    rss, tha, amp, r = best
    theta = np.zeros(workspace.full_library_size, dtype=float)
    theta[np.asarray(atoms, dtype=int)] = tha
    rho = float(np.sqrt(max(rss, 0.0)) / workspace.y_norm)
    feasible = bool(
        rss <= workspace.threshold_sq * (1.0 + 100.0 * np.finfo(float).eps)
    )
    return FixedDynamicsProfile(
        active_atoms=atoms,
        pivot_atom=workspace.pivot_atom,
        theta=theta,
        amplitudes=np.asarray(amp, dtype=float),
        residual=np.asarray(r, dtype=float),
        residual_sq=rss,
        relative_residual=rho,
        feasible_witness=feasible,
        relaxed_relative_residual=0.0,
        relaxation_proves_infeasible=False,
        y_norm=workspace.y_norm,
        uncertainty_floor=workspace.uncertainty_floor,
        threshold_sq=workspace.threshold_sq,
        n_baseline_amplitudes=workspace.n_baseline_amplitudes,
        n_change_amplitudes=workspace.n_change_amplitudes,
        optimizer_nfev=int(total_nfev),
        optimizer_success=bool(any_success),
    )


__all__ = [
    "ContinuousProfile",
    "UniformDyadicQuantizer",
    "QuantizedWitness",
    "PrecisionProfile",
    "profile_continuous",
    "exact_add_residual_sq",
    "quantized_witness",
    "profile_precision_witness",
    "FixedDynamicsProfile",
    "FixedDynamicsQuantizedWitness",
    "FixedDynamicsPrecisionProfile",
    "profile_fixed_dynamics",
    "FixedDynamicsVarProWorkspace",
    "build_fixed_dynamics_varpro_workspace",
    "profile_fixed_dynamics_varpro",
    "fixed_dynamics_quantized_witness",
    "profile_fixed_dynamics_precision_witness",
]
