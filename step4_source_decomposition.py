"""
step4_source_decomposition.py
=============================

TIDES Step 4: decompose reconstructed edge-space vector fields into
microscopic source amplitudes and a shared interaction law.

Current formal hypothesis
-------------------------
For every stage r,

    B^(r) = W^(r) theta^T,

where

    B^(r) : (M, L) edge-space vector-field coefficient matrix,
    W^(r) : (M,) microscopic edge amplitudes,
    theta : (L,) shared interaction-law direction.

Stacking all stage-edge rows gives

    B_stack in R^((R M) x L),

so the shared-law hypothesis is exactly a rank-one factorization problem.

Step 4 therefore uses the leading SVD component

    B_stack ~= s1 u1 v1^T

and identifies

    W_stack = s1 u1,
    theta_unit = v1,

followed only by a gauge normalization.  No Step-1/2/3 oracle information is
required.

Important
---------
The factorization is intrinsically invariant under

    W -> c W,
    theta -> theta / c.

The ``normalization`` argument fixes this scale/sign gauge.  For the current
TIDES benchmark, ``normalization='reference_component'`` with
``reference_component=0`` sets theta[0] = 1.  This is a coordinate convention,
not use of the true interaction law.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SourceDecompositionResult:
    """Output of the current TIDES Step-4 shared-law decomposition."""

    # Canonically normalized microscopic/source factors.
    W_stages: FloatArray
    theta: FloatArray

    # Unit-norm SVD direction before canonical scale normalization.
    theta_unit: FloatArray
    W_unit_gauge: FloatArray

    # Rank-one reconstruction in the original B coordinates.
    B_reconstructed: FloatArray
    residual_B: FloatArray
    residual_norm: float
    relative_residual: float
    relative_residual_by_stage: tuple[float, ...]

    # SVD / low-rank diagnostics.
    singular_values: FloatArray
    numerical_rank: int
    rank_tolerance: float
    rank1_energy_fraction: float
    second_to_first_singular_ratio: float
    spectral_gap: float

    # Gauge information.
    normalization: str
    gauge_component: int
    gauge_value_before_normalization: float
    gauge_scale_applied_to_W: float

    # Dimensions / bookkeeping.
    n_stages: int
    n_edges: int
    n_edge_features: int
    metadata: dict

    @property
    def interaction_law(self) -> FloatArray:
        """Alias for theta."""
        return self.theta

    @property
    def source_amplitudes(self) -> FloatArray:
        """Alias for W_stages."""
        return self.W_stages


# -----------------------------------------------------------------------------
# Input handling
# -----------------------------------------------------------------------------


def _extract_B_stages(B_stages_or_step3_result) -> FloatArray:
    """
    Accept either a Step-3 result object or an explicit B_stages array.
    """

    source = B_stages_or_step3_result

    if hasattr(source, "B_stages"):
        source = source.B_stages

    B = np.asarray(source, dtype=float)

    if B.ndim != 3:
        raise ValueError(
            "Step 4 requires B_stages with shape "
            "(n_stages, n_edges, n_edge_features)."
        )

    R, M, L = B.shape

    if R < 1 or M < 1 or L < 1:
        raise ValueError("B_stages must be non-empty in every dimension.")

    if not np.all(np.isfinite(B)):
        raise ValueError("B_stages must contain only finite values.")

    if float(np.linalg.norm(B)) <= np.finfo(float).tiny:
        raise ValueError(
            "B_stages is numerically zero; a shared interaction-law direction "
            "cannot be identified."
        )

    return B


def _resolve_rank_tolerance(
    singular_values: FloatArray,
    matrix_shape: tuple[int, int],
    rank_tolerance: Optional[float],
) -> float:

    if singular_values.size == 0:
        return 0.0

    s1 = float(singular_values[0])

    if rank_tolerance is None:
        return float(
            np.finfo(float).eps
            * max(matrix_shape)
            * s1
        )

    tol = float(rank_tolerance)

    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError("rank_tolerance must be finite and non-negative.")

    return tol


# -----------------------------------------------------------------------------
# Gauge normalization
# -----------------------------------------------------------------------------


def _choose_sign_pivot(
    theta_unit: FloatArray,
    preferred_component: int,
    *,
    stability_tol: float,
) -> int:

    L = theta_unit.size

    if preferred_component < 0 or preferred_component >= L:
        raise ValueError(
            f"reference_component={preferred_component} is invalid for "
            f"L={L} edge features."
        )

    max_abs = float(np.max(np.abs(theta_unit)))

    if max_abs <= np.finfo(float).tiny:
        raise RuntimeError("Leading right singular vector is numerically zero.")

    if abs(float(theta_unit[preferred_component])) > stability_tol * max_abs:
        return int(preferred_component)

    return int(np.argmax(np.abs(theta_unit)))


def _normalize_rank1_factors(
    W_unit: FloatArray,
    theta_unit: FloatArray,
    *,
    normalization: str,
    reference_component: int,
    normalization_tol: float,
):
    """
    Fix sign/scale ambiguity while preserving W theta^T exactly.
    """

    normalization = str(normalization).lower()

    if normalization not in {
        "reference_component",
        "max_component",
        "unit_norm",
    }:
        raise ValueError(
            "normalization must be one of "
            "{'reference_component', 'max_component', 'unit_norm'}."
        )

    if not np.isfinite(normalization_tol) or normalization_tol <= 0.0:
        raise ValueError("normalization_tol must be finite and positive.")

    theta = np.asarray(theta_unit, dtype=float).copy()
    W = np.asarray(W_unit, dtype=float).copy()

    L = theta.size
    if reference_component < 0 or reference_component >= L:
        raise ValueError(
            f"reference_component={reference_component} is invalid for L={L}."
        )

    # First fix the arbitrary SVD sign.
    sign_pivot = _choose_sign_pivot(
        theta,
        int(reference_component),
        stability_tol=float(normalization_tol),
    )

    if theta[sign_pivot] < 0.0:
        theta *= -1.0
        W *= -1.0

    if normalization == "unit_norm":
        # SVD already returns a unit-norm right singular vector.
        return (
            W,
            theta,
            int(sign_pivot),
            float(theta[sign_pivot]),
            1.0,
        )

    if normalization == "reference_component":
        pivot = int(reference_component)
    else:
        pivot = int(np.argmax(np.abs(theta)))

    max_abs = float(np.max(np.abs(theta)))
    pivot_value = float(theta[pivot])

    if abs(pivot_value) <= float(normalization_tol) * max_abs:
        if normalization == "reference_component":
            raise ValueError(
                "The requested reference component is too small for stable "
                "gauge normalization. Use normalization='max_component' or "
                "'unit_norm', or choose another reference_component."
            )
        raise RuntimeError("No stable component exists for gauge normalization.")

    # theta_new = theta / pivot_value
    # W_new     = W * pivot_value
    #
    # so W_new theta_new^T == W theta^T.
    theta = theta / pivot_value
    W = W * pivot_value

    return (
        W,
        theta,
        int(pivot),
        float(pivot_value),
        float(pivot_value),
    )


# -----------------------------------------------------------------------------
# Shared-law decomposition
# -----------------------------------------------------------------------------


def decompose_shared_interaction_law(
    B_stages_or_step3_result,
    *,
    normalization: Literal[
        "reference_component",
        "max_component",
        "unit_norm",
    ] = "reference_component",
    reference_component: int = 0,
    normalization_tol: float = 1e-8,
    rank_tolerance: Optional[float] = None,
    component_labels: Optional[Sequence[str]] = None,
) -> SourceDecompositionResult:
    """
    Decompose stage-wise edge fields under

        B^(r) = W^(r) theta^T.

    Parameters
    ----------
    B_stages_or_step3_result
        Either a Step-3 result exposing ``.B_stages`` or an explicit array
        with shape ``(R, M, L)``.

    normalization
        Gauge convention for the rank-one factors.

        ``'reference_component'``
            Set ``theta[reference_component] = 1``.  This is the preferred
            convention when a basis component has a natural reference scale.

        ``'max_component'``
            Set the largest-magnitude component of theta to +1.  This is the
            most numerically robust scale convention.

        ``'unit_norm'``
            Keep ||theta||_2 = 1 and fix only its sign.

    reference_component
        Preferred component used to orient the SVD sign, and the component
        normalized to 1 under ``'reference_component'``.

    normalization_tol
        Relative threshold used to reject an unstable reference component.

    rank_tolerance
        Absolute singular-value threshold used only to report numerical rank.
        If None, a standard machine-precision SVD threshold is used.

    component_labels
        Optional labels for the L edge-feature coefficients.  These are stored
        only in metadata and never alter the factorization.
    """

    B = _extract_B_stages(B_stages_or_step3_result)

    R, M, L = B.shape
    B_stack = B.reshape(R * M, L)

    U, singular_values, Vt = np.linalg.svd(
        B_stack,
        full_matrices=False,
    )

    singular_values = np.asarray(singular_values, dtype=float)

    s1 = float(singular_values[0])
    u1 = np.asarray(U[:, 0], dtype=float)
    theta_unit = np.asarray(Vt[0, :], dtype=float)

    W_unit = s1 * u1

    (
        W_canonical,
        theta_canonical,
        gauge_component,
        gauge_value,
        W_scale,
    ) = _normalize_rank1_factors(
        W_unit,
        theta_unit,
        normalization=normalization,
        reference_component=int(reference_component),
        normalization_tol=float(normalization_tol),
    )

    W_stages = W_canonical.reshape(R, M)

    B_rank1_stack = np.outer(
        W_canonical,
        theta_canonical,
    )
    B_reconstructed = B_rank1_stack.reshape(R, M, L)

    residual_B = B - B_reconstructed
    residual_norm = float(np.linalg.norm(residual_B))
    B_norm = float(np.linalg.norm(B))
    relative_residual = float(
        residual_norm / max(B_norm, np.finfo(float).tiny)
    )

    stage_rel = []
    for r in range(R):
        denom = max(
            float(np.linalg.norm(B[r])),
            np.finfo(float).tiny,
        )
        stage_rel.append(
            float(np.linalg.norm(residual_B[r]) / denom)
        )

    total_energy = float(np.sum(singular_values ** 2))
    rank1_energy_fraction = float(
        (s1 * s1) / max(total_energy, np.finfo(float).tiny)
    )

    if singular_values.size >= 2:
        s2 = float(singular_values[1])
        second_to_first = float(
            s2 / max(s1, np.finfo(float).tiny)
        )
        spectral_gap = float(
            s1 / max(s2, np.finfo(float).tiny)
        )
    else:
        s2 = 0.0
        second_to_first = 0.0
        spectral_gap = np.inf

    tol = _resolve_rank_tolerance(
        singular_values,
        B_stack.shape,
        rank_tolerance,
    )
    numerical_rank = int(np.count_nonzero(singular_values > tol))

    if component_labels is not None:
        labels = tuple(str(x) for x in component_labels)
        if len(labels) != L:
            raise ValueError(
                "component_labels must contain one label per edge feature."
            )
    else:
        labels = None

    metadata = {
        "hypothesis": "shared_interaction_law",
        "factorization": "global-stage-edge-rank-one-svd",
        "oracle_information_used": False,
        "n_stages": int(R),
        "n_edges": int(M),
        "n_edge_features": int(L),
        "stacked_rows": int(R * M),
        "normalization": str(normalization),
        "reference_component": int(reference_component),
        "gauge_component": int(gauge_component),
        "component_labels": labels,
        "relative_rank1_residual": float(relative_residual),
        "rank1_energy_fraction": float(rank1_energy_fraction),
        "second_to_first_singular_ratio": float(second_to_first),
        "spectral_gap": float(spectral_gap),
        "numerical_rank": int(numerical_rank),
        "rank_tolerance": float(tol),
    }

    return SourceDecompositionResult(
        W_stages=W_stages,
        theta=theta_canonical,
        theta_unit=theta_unit.copy(),
        W_unit_gauge=W_unit.reshape(R, M).copy(),
        B_reconstructed=B_reconstructed,
        residual_B=residual_B,
        residual_norm=residual_norm,
        relative_residual=relative_residual,
        relative_residual_by_stage=tuple(stage_rel),
        singular_values=singular_values,
        numerical_rank=numerical_rank,
        rank_tolerance=float(tol),
        rank1_energy_fraction=rank1_energy_fraction,
        second_to_first_singular_ratio=second_to_first,
        spectral_gap=spectral_gap,
        normalization=str(normalization),
        gauge_component=int(gauge_component),
        gauge_value_before_normalization=float(gauge_value),
        gauge_scale_applied_to_W=float(W_scale),
        n_stages=int(R),
        n_edges=int(M),
        n_edge_features=int(L),
        metadata=metadata,
    )


# -----------------------------------------------------------------------------
# Public Step-4 dispatch
# -----------------------------------------------------------------------------


def decompose_vector_field_sources(
    B_stages_or_step3_result,
    *,
    hypothesis: Literal["shared_interaction_law"] = "shared_interaction_law",
    **kwargs,
) -> SourceDecompositionResult:
    """
    Formal TIDES Step-4 dispatch.

    Additional microscopic-source hypotheses can be added here later without
    changing the current shared-law implementation.
    """

    if hypothesis == "shared_interaction_law":
        return decompose_shared_interaction_law(
            B_stages_or_step3_result,
            **kwargs,
        )

    raise ValueError(f"Unknown Step-4 source hypothesis: {hypothesis!r}")


# Compact alias for interactive / pipeline use.
decompose_sources = decompose_vector_field_sources


__all__ = [
    "SourceDecompositionResult",
    "decompose_shared_interaction_law",
    "decompose_vector_field_sources",
    "decompose_sources",
]
