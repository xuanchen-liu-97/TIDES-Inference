"""Numerical profiling for TIDES Step 3 conditional-MDL search.

This module separates continuous physical fitting from discrete model-space
search.  The generic linear tools remain useful for hypothesis-specific
adapters.  The primary implemented physical branch is H_FD,

    B^(r) = W^(r) Theta^T,

parameterized for coding/search by the invertible structural coordinates
W^(1), Delta W^(1),...,Delta W^(K-1).  These coordinates are introduced only
in Step 3; the Step-2 canonical object remains the absolute edge-space family
B_epsilon.

For fixed active structural groups and active interaction atoms, scalar
structural amplitudes are profiled exactly for every shared law Theta.  The
remaining nonlinear search is therefore only over |J|-1 gauge-fixed dynamics
coefficients.

The finite-precision routines return feasible q-bit witnesses.  Unless an exact
lattice certificate is supplied, q is an upper bound on the true q_star.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

try:
    from .structural_linear_cache import StructuralLinearCache
    from .edge_space_operators import (
        TemporalEdgeBlock, coerce_block, dense_block,
        derivative_products, full_support_relaxation,
    )
except ImportError:
    from structural_linear_cache import StructuralLinearCache
    from edge_space_operators import (
        TemporalEdgeBlock, coerce_block, dense_block,
        derivative_products, full_support_relaxation,
    )

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

    def floating_integer_limits(self, q: int) -> tuple[int, int]:
        """Legal grid indices representable in float64, including at q > 54.

        Casting 2**63-1 to float rounds up; without this adjustment q=64 can
        incorrectly produce +bound, which is outside the half-open codebook.
        At high q floating arithmetic explores a subset of the ideal lattice.
        """
        lo, hi = self.integer_limits(q)
        hi_float = float(hi)
        if int(hi_float) > hi:
            hi_float = np.nextafter(hi_float, -np.inf)
        return lo, int(hi_float)

    def quantize(self, values: FloatArray, q: int) -> FloatArray:
        values = np.asarray(values, dtype=float)
        step = self.step(q)
        lo, hi = self.floating_integer_limits(q)
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
    lo, hi = quantizer.floating_integer_limits(q)

    passes_done = 0
    for p in range(max(0, int(max_passes))):
        improved = False
        for i in range(Q):
            gii = float(diag[i])
            if gii <= np.finfo(float).tiny:
                continue
            grad_half = float(Ge[i] - h[i])
            target = cq[i] - grad_half / gii
            k0 = min(hi, max(lo, int(np.rint(target / step))))
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
# Fixed dynamics: shared interaction law, free physical structural amplitudes
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class FixedDynamicsProfile:
    """Continuous profile of one H_FD structural/expression model."""

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
        return int(self.n_structural_amplitudes + self.n_free_dynamics_coefficients)


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
    diagnostics: Optional[dict] = None

    @property
    def feasible(self) -> bool:
        return self.witness is not None and self.witness.feasible

    @property
    def q_star(self) -> Optional[int]:
        return self.q_upper if self.certified_exact else None


class FixedDynamicsQuantizationError(ValueError):
    """A finite witness search failed; this is not an infeasibility proof."""

    def __init__(self, precision: FixedDynamicsPrecisionProfile):
        self.precision = precision
        self.diagnostics = precision.diagnostics or {}
        d = self.diagnostics
        super().__init__(
            "No feasible quantized H_FD witness found within the search budget: "
            f"q_max={d.get('q_max')}, bound={precision.quantizer.bound:g}, "
            f"continuous_rho={d.get('continuous_relative_residual')}, "
            f"eps={d.get('uncertainty_floor')}, "
            f"outside_amplitudes={d.get('outside_amplitudes')}, "
            f"outside_free_theta={d.get('outside_free_theta')}, "
            f"best_quantized_rho={d.get('best_tested_relative_residual')}, "
            f"code_domain_seed={d.get('code_domain_seed_status')}."
        )


@dataclass(frozen=True)
class FixedDynamicsVarProWorkspace:
    """Shared block maps for variable projection, without a derivative tensor."""

    y: FloatArray
    structural_blocks: tuple  # compact maps or legacy dense arrays; never L x n x p
    linear_cache: StructuralLinearCache
    active_atoms: tuple[int, ...]
    pivot_atom: int
    y_norm: float
    uncertainty_floor: float
    threshold_sq: float
    full_library_size: int
    relaxed_relative_residual: float
    relaxation_proves_infeasible: bool

    @property
    def n_amplitudes(self) -> int:
        return len(self.structural_blocks)


def _validate_fixed_dynamics_blocks(
    y: FloatArray,
    structural_blocks: Sequence[FloatArray],
) -> tuple[FloatArray, tuple[FloatArray, ...], int]:
    y = np.asarray(y, dtype=float).reshape(-1)
    if y.size == 0:
        raise ValueError("y must be non-empty.")
    blocks = tuple(coerce_block(B) for B in structural_blocks)
    if not blocks:
        raise ValueError("At least one active structural block is required.")
    if blocks[0].ndim != 2:
        raise ValueError("Each structural block must be two-dimensional.")
    L = int(blocks[0].shape[1])
    if L < 1:
        raise ValueError("Structural blocks must contain at least one library atom.")
    for i, B in enumerate(blocks):
        if B.shape != (y.size, L):
            raise ValueError(
                f"structural block {i} has shape {B.shape}; expected {(y.size, L)}."
            )
        if not isinstance(B, TemporalEdgeBlock) and not np.all(np.isfinite(B)):
            raise ValueError(f"structural block {i} contains non-finite values.")
    return y, blocks, L


def _fixed_dynamics_prediction(
    blocks: Sequence[FloatArray],
    amplitudes: FloatArray,
    theta: FloatArray,
) -> FloatArray:
    amplitudes = np.asarray(amplitudes, dtype=float).reshape(-1)
    theta = np.asarray(theta, dtype=float).reshape(-1)
    if len(blocks) != amplitudes.size:
        raise ValueError("amplitudes must match structural_blocks.")
    pred = np.zeros(blocks[0].shape[0], dtype=float)
    for a, B in zip(amplitudes, blocks):
        pred += float(a) * (B @ theta)
    return pred


def build_fixed_dynamics_varpro_workspace(
    y: FloatArray,
    structural_blocks: Sequence[FloatArray],
    *,
    active_atoms: Sequence[int],
    uncertainty_floor: float,
    compute_linear_relaxation: bool = True,
    linear_cache: Optional[StructuralLinearCache] = None,
) -> FixedDynamicsVarProWorkspace:
    """Precompute one fixed-support H_FD variable-projection workspace."""

    y, blocks, L = _validate_fixed_dynamics_blocks(y, structural_blocks)
    atoms = tuple(sorted(set(int(a) for a in active_atoms)))
    if not atoms:
        raise ValueError("active_atoms must be non-empty under H_FD.")
    if any(a < 0 or a >= L for a in atoms):
        raise ValueError("active_atoms contains an out-of-range atom index.")

    y_norm = float(np.linalg.norm(y))
    if y_norm <= np.finfo(float).tiny:
        raise ValueError("y must have nonzero norm.")
    eps = float(uncertainty_floor)
    if not np.isfinite(eps) or eps < 0.0:
        raise ValueError("uncertainty_floor must be finite and non-negative.")
    threshold_sq = float((eps * y_norm) ** 2)

    if compute_linear_relaxation:
        relaxed_rho = full_support_relaxation(blocks, atoms)
        if relaxed_rho is None:
            A_relaxed = np.hstack([dense_block(B[:, np.asarray(atoms, dtype=int)]) for B in blocks])
            beta, _, _, _ = np.linalg.lstsq(A_relaxed, y, rcond=None)
            r_relaxed = np.asarray(y - A_relaxed @ beta, dtype=float)
            relaxed_rho = float(np.linalg.norm(r_relaxed) / y_norm)
        relaxation_infeasible = bool(relaxed_rho > eps * (1.0 + 1e-10))
    else:
        relaxed_rho = 0.0
        relaxation_infeasible = False

    return FixedDynamicsVarProWorkspace(
        y=np.asarray(y, dtype=float),
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
    """Locally minimize E(Theta)=min_W rho using exact variable projection."""

    from scipy.optimize import least_squares

    atoms = workspace.active_atoms
    d = len(atoms)
    y = workspace.y
    blocks = workspace.structural_blocks
    scale = max(workspace.uncertainty_floor * workspace.y_norm, np.finfo(float).tiny)

    def theta_active_from_x(x: FloatArray) -> FloatArray:
        th = np.zeros(d, dtype=float)
        th[0] = 1.0
        if d > 1:
            th[1:] = np.asarray(x, dtype=float)
        return th

    def compute_state(x: FloatArray, need_jac: bool):
        tha = theta_active_from_x(x)
        full_theta = np.zeros(workspace.full_library_size)
        full_theta[list(atoms)] = tha
        key = workspace.linear_cache.key('svd', blocks, full_theta, y)
        fit = workspace.linear_cache.get(key)
        if fit is None:
            # Mixed-law nonlinear steps usually visit each theta only once.
            # Do not fill the column cache with those transient designs; the
            # exact SVD-result cache still handles repeated theta/support pairs.
            C = workspace.linear_cache.matrix(blocks, full_theta, cache_columns=(d == 1))
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
                r = y.copy()
            fit = workspace.linear_cache.put(key, (Ur, sr, Vr, amp, r))
        Ur, sr, Vr, amp, r = fit
        if not need_jac or d <= 1:
            return tha, amp, r, None

        vdir, b = derivative_products(blocks, amp, r, atoms[1:])
        if sr.size:
            term1 = -(vdir - Ur @ (Ur.T @ vdir))
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
        if abs(pv) > 1e-14:
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

            sol = least_squares(
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
    feasible = bool(rss <= workspace.threshold_sq * (1.0 + 100.0 * np.finfo(float).eps))

    return FixedDynamicsProfile(
        active_atoms=atoms,
        pivot_atom=workspace.pivot_atom,
        theta=theta,
        amplitudes=np.asarray(amp, dtype=float),
        residual=np.asarray(r, dtype=float),
        residual_sq=rss,
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
    structural_blocks: Sequence[FloatArray],
    *,
    active_atoms: Sequence[int],
    uncertainty_floor: float,
    warm_theta: Optional[FloatArray] = None,
    max_nfev: int = 250,
    compute_linear_relaxation: bool = True,
    include_default_start: bool = True,
    linear_cache: Optional[StructuralLinearCache] = None,
) -> FixedDynamicsProfile:
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


class _FixedDynamicsQuantizationWorkspace:
    """Reuse exact, q-independent data within one precision search.

    Keep at most 16 MiB of atom columns and 16 MiB of current-theta response
    columns. Only a prefix that fits is retained, so oversized sweeps cannot
    thrash an LRU cache. Arithmetic and column layout match the uncached path.
    Nothing survives a change of profile/support or a new precision search.
    """

    def __init__(self, profile, structural_blocks, *, max_cache_bytes=32 * 1024**2):
        y_dummy = np.zeros(profile.residual.size, dtype=float)
        _, blocks, self.library_size = _validate_fixed_dynamics_blocks(y_dummy, structural_blocks)
        indices = np.asarray(profile.active_atoms, dtype=int)
        self.blocks = tuple(B[:, indices] for B in blocks)
        self.theta0 = profile.theta[indices]
        self.y = _fixed_dynamics_prediction(self.blocks, profile.amplitudes, self.theta0) + profile.residual
        # All cached vectors have the same length. Account for response norms
        # too; Python container overhead is not part of this array-data budget.
        column_bytes = max(1, self.y.nbytes)
        half_budget = max(0, int(max_cache_bytes)) // 2
        self._atom_limit = half_budget // column_bytes
        self._response_limit = half_budget // (column_bytes + 8)
        self._atom_columns = {}
        self._responses = {}
        self._theta_key = None

    def atom_column(self, i, j):
        key = (i, j)
        if key in self._atom_columns:
            return self._atom_columns[key]
        column = self.blocks[i][:, j]
        if len(self._atom_columns) < self._atom_limit:
            self._atom_columns[key] = column
        return column

    def response(self, i, theta):
        key = theta.tobytes()
        if key != self._theta_key:
            self._responses.clear()
            self._theta_key = key
        cached = self._responses.get(i)
        if cached is not None:
            return cached
        v = self.blocks[i] @ theta
        result = (v, float(v @ v))
        if len(self._responses) < self._response_limit:
            self._responses[i] = result
        return result


def _fixed_dynamics_coordinate_descent_quantized(
    profile: FixedDynamicsProfile,
    structural_blocks: Sequence[FloatArray],
    quantizer: UniformDyadicQuantizer,
    q: int,
    *,
    max_passes: int = 8,
    _workspace: Optional[_FixedDynamicsQuantizationWorkspace] = None,
    _seed: Optional[tuple[FloatArray, FloatArray]] = None,
) -> tuple[FloatArray, FloatArray, float, int]:
    workspace = _workspace
    if workspace is None:
        workspace = _FixedDynamicsQuantizationWorkspace(profile, structural_blocks)
    L = workspace.library_size
    atoms = profile.active_atoms
    local_blocks = workspace.blocks
    theta0 = workspace.theta0 if _seed is None else _seed[0]
    y = workspace.y

    a = quantizer.quantize(profile.amplitudes if _seed is None else _seed[1], q)
    theta = theta0.copy()
    if len(atoms) > 1:
        theta[1:] = quantizer.quantize(theta0[1:], q)
    theta[0] = 1.0

    pred = _fixed_dynamics_prediction(local_blocks, a, theta)
    r = np.asarray(y - pred, dtype=float)
    step = quantizer.step(q)
    lo, hi = quantizer.floating_integer_limits(q)

    passes_done = 0
    for p in range(max(0, int(max_passes))):
        improved = False
        for i, B in enumerate(local_blocks):
            v, vv = workspace.response(i, theta)
            if vv <= np.finfo(float).tiny:
                continue
            target = float(a[i] + (v @ r) / vv)
            k0 = min(hi, max(lo, int(np.rint(target / step))))
            best_val, best_delta = float(a[i]), 0.0
            rv = float(r @ v)
            for k in (k0 - 1, k0, k0 + 1):
                if lo <= k <= hi:
                    val = float(k * step)
                    dlt = val - a[i]
                    delta = -2.0 * dlt * rv + dlt * dlt * vv
                    if delta < best_delta:
                        best_val, best_delta = val, float(delta)
            if best_val != a[i]:
                dlt = best_val - a[i]
                a[i] = best_val
                r = r - dlt * v
                improved = True

        for j in range(1, len(atoms)):
            v = np.zeros_like(r)
            for i, ai in enumerate(a):
                v += float(ai) * workspace.atom_column(i, j)
            vv = float(v @ v)
            if vv <= np.finfo(float).tiny:
                continue
            target = float(theta[j] + (v @ r) / vv)
            k0 = min(hi, max(lo, int(np.rint(target / step))))
            best_val, best_delta = float(theta[j]), 0.0
            rv = float(r @ v)
            for k in (k0 - 1, k0, k0 + 1):
                if lo <= k <= hi:
                    val = float(k * step)
                    dlt = val - theta[j]
                    delta = -2.0 * dlt * rv + dlt * dlt * vv
                    if delta < best_delta:
                        best_val, best_delta = val, float(delta)
            if best_val != theta[j]:
                dlt = best_val - theta[j]
                theta[j] = best_val
                r = r - dlt * v
                improved = True

        passes_done = p + 1
        if not improved:
            break

    theta_full = np.zeros(L, dtype=float)
    theta_full[np.asarray(atoms, dtype=int)] = theta
    return theta_full, np.asarray(a, dtype=float), float(r @ r), passes_done


def fixed_dynamics_quantized_witness(
    profile: FixedDynamicsProfile,
    structural_blocks: Sequence[FloatArray],
    *,
    q: int,
    quantizer: UniformDyadicQuantizer,
    max_coordinate_passes: int = 8,
    _workspace: Optional[_FixedDynamicsQuantizationWorkspace] = None,
    _seed: Optional[tuple[FloatArray, FloatArray]] = None,
) -> FixedDynamicsQuantizedWitness:
    q = int(q)
    if q < 1:
        raise ValueError("q must be >= 1.")
    if not profile.feasible_witness:
        return FixedDynamicsQuantizedWitness(
            q, profile.theta.copy(), profile.amplitudes.copy(), np.inf, np.inf, False, 0
        )
    theta, amplitudes, rss, passes = _fixed_dynamics_coordinate_descent_quantized(
        profile,
        structural_blocks,
        quantizer,
        q,
        max_passes=max_coordinate_passes,
        _workspace=_workspace,
        _seed=_seed,
    )
    rho = float(np.sqrt(max(rss, 0.0)) / profile.y_norm)
    feasible = bool(rss <= profile.threshold_sq * (1.0 + 100.0 * np.finfo(float).eps))
    return FixedDynamicsQuantizedWitness(
        q, theta, amplitudes, rho, float(rss), feasible, int(passes)
    )


def _code_domain_seed(workspace, quantizer, diagnostics):
    """One additional seed inside the existing code range; never change y.

    Holding the (possibly range-clipped) active theta fixed makes this a convex
    bounded linear fit in structural amplitudes. BVLS uses explicit C and direct
    least-squares subproblems. Neither its termination flag nor failure to reach
    epsilon is treated as a certificate for the joint nonlinear model.
    """
    from scipy.optimize import lsq_linear

    upper = float(np.nextafter(quantizer.bound, -np.inf))
    theta = workspace.theta0.copy()
    theta[1:] = np.clip(theta[1:], -quantizer.bound, upper)
    C = np.column_stack([block @ theta for block in workspace.blocks])
    fit = lsq_linear(C, workspace.y, bounds=(-quantizer.bound, upper),
                     method="bvls", tol=1e-10, max_iter=300)
    diagnostics['code_domain_solver_success'] = bool(fit.success)
    diagnostics['code_domain_solver_iterations'] = int(fit.nit)
    if not np.all(np.isfinite(fit.x)):
        diagnostics['code_domain_seed_status'] = 'nonfinite_bounded_fit'
        return None
    amplitudes = np.clip(np.asarray(fit.x, dtype=float), -quantizer.bound, upper)
    residual = workspace.y - C @ amplitudes
    diagnostics['code_domain_seed_relative_residual'] = float(
        np.linalg.norm(residual) / diagnostics['y_norm'])
    diagnostics['code_domain_seed_status'] = 'prepared'
    return theta, amplitudes


def profile_fixed_dynamics_precision_witness(
    profile: FixedDynamicsProfile,
    structural_blocks: Sequence[FloatArray],
    *,
    quantizer: UniformDyadicQuantizer,
    q_min: int = 1,
    q_max: int = 64,
    warm_q: Optional[int] = None,
    max_coordinate_passes: int = 8,
    repair_code_domain: bool = True,
) -> FixedDynamicsPrecisionProfile:
    """Scan every q with the original seed and, if needed, a bounded seed.

    The additional seed does not replace the original continuous optimum or
    change the model's support, precision code, epsilon or coordinate budget.
    Lowest residual chooses between seeds at the same q; complete model MDL
    comparisons remain the caller's responsibility. No global optimality claim.
    """
    q_min, q_max = int(q_min), int(q_max)
    if q_min < 1 or q_max < q_min:
        raise ValueError("Require 1 <= q_min <= q_max.")
    free_theta = profile.theta[np.asarray(profile.active_atoms[1:], dtype=int)]
    outside = lambda a: int(np.count_nonzero((a < -quantizer.bound) | (a >= quantizer.bound)))
    diagnostics = dict(
        continuous_relative_residual=profile.relative_residual,
        uncertainty_floor=profile.uncertainty_floor, y_norm=profile.y_norm,
        q_min=q_min, q_max=q_max, max_coordinate_passes=int(max_coordinate_passes),
        outside_amplitudes=outside(profile.amplitudes),
        outside_free_theta=outside(free_theta),
        code_domain_seed_status='not_needed' if repair_code_domain else 'disabled',
        best_tested_relative_residual=None, best_tested_q=None,
        selected_seed=None,
    )
    if not profile.feasible_witness:
        diagnostics['status'] = 'continuous_profile_outside_floor'
        return FixedDynamicsPrecisionProfile(None, None, tuple(), False, quantizer, diagnostics)

    workspace = _FixedDynamicsQuantizationWorkspace(profile, structural_blocks)
    seed = None
    if repair_code_domain and (diagnostics['outside_amplitudes'] or diagnostics['outside_free_theta']):
        seed = _code_domain_seed(workspace, quantizer, diagnostics)

    def evaluate(q):
        w = fixed_dynamics_quantized_witness(
            profile, structural_blocks, q=q, quantizer=quantizer,
            max_coordinate_passes=max_coordinate_passes, _workspace=workspace,
        )
        label = 'continuous'
        if seed is not None:
            alternative = fixed_dynamics_quantized_witness(
                profile, structural_blocks, q=q, quantizer=quantizer,
                max_coordinate_passes=max_coordinate_passes, _workspace=workspace, _seed=seed,
            )
            if alternative.residual_sq < w.residual_sq:
                w, label = alternative, 'code_domain'
        best_rho = diagnostics['best_tested_relative_residual']
        if best_rho is None or w.relative_residual < best_rho:
            diagnostics['best_tested_relative_residual'] = w.relative_residual
            diagnostics['best_tested_q'] = q
        return w, label

    warm = None
    if warm_q is not None and q_min <= int(warm_q) <= q_max:
        warm = evaluate(int(warm_q))
    upper = q_max if warm is None or not warm[0].feasible else int(warm_q)
    tested: list[int] = []
    found = None
    for q in range(q_min, upper + 1):
        w, label = warm if warm is not None and q == int(warm_q) else evaluate(q)
        tested.append(q)
        if w.feasible:
            found = w
            diagnostics['selected_seed'] = label
            break
    diagnostics['status'] = 'feasible_witness' if found is not None else 'no_witness_within_budget'
    return FixedDynamicsPrecisionProfile(
        None if found is None else int(found.q),
        found,
        tuple(tested),
        False,
        quantizer,
        diagnostics,
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
    "FixedDynamicsQuantizationError",
    "FixedDynamicsVarProWorkspace",
    "build_fixed_dynamics_varpro_workspace",
    "profile_fixed_dynamics_varpro",
    "profile_fixed_dynamics",
    "fixed_dynamics_quantized_witness",
    "profile_fixed_dynamics_precision_witness",
]
