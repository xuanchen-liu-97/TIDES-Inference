"""TIF Step 4: decompose reconstructed vector fields into microscopic sources.

Step 4 is explicitly hypothesis-dependent.  Two dual rank-one hypotheses are
implemented:

1. varying_structure
       B^(r) = W^(r) theta^T
   with one shared interaction law theta.

2. varying_dynamics
       B^(r) = W (theta^(r))^T
   with one fixed topology/weight vector W.

Both factorizations have an unavoidable global scale/sign gauge.  The routines
below fix that gauge by unit-normalizing the shared factor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class VaryingStructureDecomposition:
    W_stages: FloatArray
    theta: FloatArray
    singular_values: FloatArray
    relative_residual: float
    rank1_energy_fraction: float
    hypothesis: str = "varying_structure"

    @property
    def delta_W(self) -> FloatArray:
        return np.diff(self.W_stages, axis=0)


@dataclass(frozen=True)
class VaryingDynamicsDecomposition:
    W: FloatArray
    theta_stages: FloatArray
    singular_values: FloatArray
    relative_residual: float
    rank1_energy_fraction: float
    hypothesis: str = "varying_dynamics"

    @property
    def delta_theta(self) -> FloatArray:
        return np.diff(self.theta_stages, axis=0)


def _validate_B(B_stages: ArrayLike) -> FloatArray:
    B = np.asarray(B_stages, dtype=float)
    if B.ndim != 3:
        raise ValueError("B_stages must have shape (n_stages, n_edges, n_functions).")
    if min(B.shape) == 0:
        raise ValueError("B_stages must be non-empty.")
    return B


def _fix_sign_shared_vector(v: FloatArray) -> FloatArray:
    """Choose a deterministic sign gauge: largest-magnitude component positive."""

    v = np.asarray(v, dtype=float).copy()
    if np.all(v == 0):
        return v
    j = int(np.argmax(np.abs(v)))
    if v[j] < 0:
        v *= -1.0
    return v


def _energy_fraction(s: FloatArray) -> float:
    denom = float(np.sum(s**2))
    return float(s[0] ** 2 / denom) if denom > 0 else np.nan


def decompose_varying_structure_shared_dynamics(
    B_stages: ArrayLike,
) -> VaryingStructureDecomposition:
    """Fit B^(r) = W^(r) theta^T jointly across all stages."""

    B = _validate_B(B_stages)
    R, M, L = B.shape
    stacked = B.reshape(R * M, L)

    U, s, Vt = np.linalg.svd(stacked, full_matrices=False)
    theta = _fix_sign_shared_vector(Vt[0])  # unit norm
    weights_flat = stacked @ theta
    W = weights_flat.reshape(R, M)

    reconstructed = W[:, :, None] * theta[None, None, :]
    denom = max(float(np.linalg.norm(B)), np.finfo(float).eps)
    rel = float(np.linalg.norm(B - reconstructed) / denom)

    return VaryingStructureDecomposition(
        W_stages=W,
        theta=theta,
        singular_values=s,
        relative_residual=rel,
        rank1_energy_fraction=_energy_fraction(s),
    )


def decompose_fixed_structure_varying_dynamics(
    B_stages: ArrayLike,
) -> VaryingDynamicsDecomposition:
    """Fit B^(r) = W (theta^(r))^T jointly across all stages."""

    B = _validate_B(B_stages)
    R, M, L = B.shape

    # Horizontal stacking exposes the common left factor W:
    # [B1 | B2 | ... | BR] = W [theta1^T | theta2^T | ... | thetaR^T].
    horizontal = np.concatenate([B[r] for r in range(R)], axis=1)  # M x (R L)
    U, s, Vt = np.linalg.svd(horizontal, full_matrices=False)

    W = _fix_sign_shared_vector(U[:, 0])  # unit norm
    # Re-estimate the concatenated right factor after deterministic sign fixing.
    theta_concat = W @ horizontal  # shape (R L,)
    theta_stages = theta_concat.reshape(R, L)

    reconstructed = W[None, :, None] * theta_stages[:, None, :]
    denom = max(float(np.linalg.norm(B)), np.finfo(float).eps)
    rel = float(np.linalg.norm(B - reconstructed) / denom)

    return VaryingDynamicsDecomposition(
        W=W,
        theta_stages=theta_stages,
        singular_values=s,
        relative_residual=rel,
        rank1_energy_fraction=_energy_fraction(s),
    )


def decompose_vector_field(
    B_stages: ArrayLike,
    *,
    hypothesis: Literal["varying_structure", "varying_dynamics"],
):
    """Unified public interface for TIF Step 4."""

    if hypothesis == "varying_structure":
        return decompose_varying_structure_shared_dynamics(B_stages)
    if hypothesis == "varying_dynamics":
        return decompose_fixed_structure_varying_dynamics(B_stages)
    raise ValueError(f"Unknown Step-4 hypothesis: {hypothesis!r}")
