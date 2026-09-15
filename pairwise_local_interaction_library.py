"""Pairwise local-interaction libraries for TIDES.

This module is deliberately upstream of Steps 2--4.  It converts node states
sampled at the vector-field evaluation times into a library of *edge-local*
interaction atoms,

    Gamma_l(x_i, x_j) = [g_i, g_j],

where the two entries are the contributions of candidate edge {i,j} to its two
endpoints.  Stacking the atoms gives

    endpoint_features.shape == (T, M, L, 2).

The TIDES inference core then estimates stage/edge coefficients B^(r)_{m,l}
in

    g_m^(r)(x_i,x_j) = sum_l B^(r)_{m,l} Gamma_l(x_i,x_j).

Two library modes are intentionally supported:

``polynomial``
    Dynamics-agnostic fallback.  Generates the identifiable, swap-equivariant
    pairwise polynomial sector up to a requested total degree.  Pure own-state
    one-body monomials are excluded because assigning them to individual
    incident edges creates an edge-self gauge.  Those terms belong in a
    separate node-local / stationary-nuisance sector.

``candidate``
    Physics-/mechanism-informed mode.  The caller supplies a finite list of
    candidate PairwiseLocalInteractionAtom objects (for example a sinusoidal
    coupling atom or an SIS infection atom).  Candidate atoms are evaluated on
    the physical, unnormalised endpoint states; no polynomial approximation is
    introduced.

The module contains no temporal-change model, sparsity penalty, solver, or
network-wide assembly logic.  Its only job is local state -> local interaction
library construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
AtomEvaluator = Callable[[FloatArray, FloatArray], Any]


# -----------------------------------------------------------------------------
# Public data structures
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class PairwiseLocalInteractionAtom:
    """One candidate edge-local interaction mechanism.

    ``evaluate(x_tail, x_head)`` receives arrays with shape ``(T,M)`` and must
    return either

        * one array with shape ``(T,M,2)``, or
        * a two-tuple ``(tail_contribution, head_contribution)``, each ``(T,M)``.

    The returned values are *local endpoint contributions*.  The atom must not
    multiply by an edge-specific strength; that coefficient is inferred later
    by TIDES as B[m,l].

    For an undirected candidate graph, the preferred convention is swap
    equivariance:

        Gamma(x_j,x_i) = swap(Gamma(x_i,x_j)).

    The built-in atoms in this file satisfy that convention.
    """

    name: str
    evaluate: AtomEvaluator
    description: str = ""
    swap_equivariant: Optional[bool] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("PairwiseLocalInteractionAtom.name must be non-empty.")
        if not callable(self.evaluate):
            raise TypeError("PairwiseLocalInteractionAtom.evaluate must be callable.")


@dataclass(frozen=True)
class PairwiseLocalInteractionLibrary:
    """Evaluated local-interaction library passed to TIDES Steps 2 and 3."""

    # Physical endpoint states and orthonormal common/difference coordinates.
    endpoint_states: FloatArray          # (T,M,2): oriented (tail, head)
    local_states: FloatArray             # (T,M,2): (c,d)

    # The common TIDES interface.
    endpoint_features: FloatArray        # (T,M,L,2)
    feature_labels: tuple[str, ...]
    component_labels: tuple[str, ...]

    # Provenance / interpretation.
    mode: Literal["polynomial", "candidate"]
    feature_representation: str
    metadata: Mapping[str, Any]

    @property
    def n_features(self) -> int:
        return int(self.endpoint_features.shape[2])

    @property
    def edge_features(self) -> FloatArray:
        """Alias matching the Step-2/3 argument name."""

        return self.endpoint_features


# -----------------------------------------------------------------------------
# Incidence / endpoint handling
# -----------------------------------------------------------------------------


def _validate_node_states_and_incidence(
    node_states: ArrayLike,
    D: ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    X = np.asarray(node_states, dtype=float)
    D_arr = np.asarray(D, dtype=float)

    if X.ndim != 2:
        raise ValueError("node_states must have shape (n_observations, n_nodes).")
    if D_arr.ndim != 2:
        raise ValueError("D must have shape (n_nodes, n_candidate_edges).")
    if X.shape[1] != D_arr.shape[0]:
        raise ValueError("D.shape[0] must equal node_states.shape[1].")
    if X.shape[0] == 0 or X.shape[1] == 0 or D_arr.shape[1] == 0:
        raise ValueError("node_states and D must be non-empty.")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(D_arr)):
        raise ValueError("node_states and D must contain only finite values.")

    nz = np.abs(D_arr) > 1.0e-12
    if not np.all(np.sum(nz, axis=0) == 2):
        raise ValueError(
            "Each candidate pair must have exactly two nonzero incidence entries."
        )
    for m in range(D_arr.shape[1]):
        vals = np.sort(D_arr[nz[:, m], m])
        if not np.allclose(vals, np.array([-1.0, 1.0]), atol=1.0e-12, rtol=0.0):
            raise ValueError(
                "Pairwise local libraries require standard oriented incidence "
                "columns {-1,+1}. Edge strengths belong in inferred coefficients, "
                "not in D."
            )

    return X, D_arr


def pair_endpoints_from_incidence(D: ArrayLike) -> tuple[IntArray, IntArray]:
    """Return tail/head node indices for standard columns ``-e_i + e_j``."""

    dummy_X = np.zeros((1, np.asarray(D).shape[0]), dtype=float)
    _, D_arr = _validate_node_states_and_incidence(dummy_X, D)
    M = D_arr.shape[1]
    tail = np.empty(M, dtype=np.int64)
    head = np.empty(M, dtype=np.int64)
    for m in range(M):
        tail[m] = int(np.flatnonzero(D_arr[:, m] < -0.5)[0])
        head[m] = int(np.flatnonzero(D_arr[:, m] > 0.5)[0])
    return tail, head


def _gather_endpoint_states(
    X: FloatArray,
    D: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, IntArray, IntArray]:
    tail = np.empty(D.shape[1], dtype=np.int64)
    head = np.empty(D.shape[1], dtype=np.int64)
    for m in range(D.shape[1]):
        tail[m] = int(np.flatnonzero(D[:, m] < -0.5)[0])
        head[m] = int(np.flatnonzero(D[:, m] > 0.5)[0])

    xs = X[:, tail]
    xr = X[:, head]
    endpoint_states = np.stack([xs, xr], axis=-1)

    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    c = inv_sqrt2 * (xs + xr)
    d = inv_sqrt2 * (xr - xs)
    local_states = np.stack([c, d], axis=-1)
    return xs, xr, endpoint_states, local_states, tail, head


# -----------------------------------------------------------------------------
# Polynomial mode
# -----------------------------------------------------------------------------


def _resolve_polynomial_scale(
    X: FloatArray,
    *,
    normalization: Literal["none", "rms", "maxabs"],
    state_scale: Optional[float],
) -> float:
    normalization = str(normalization).lower()
    if normalization not in {"none", "rms", "maxabs"}:
        raise ValueError("normalization must be 'none', 'rms', or 'maxabs'.")

    if state_scale is not None:
        scale = float(state_scale)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("state_scale must be finite and positive.")
        return scale

    if normalization == "none":
        return 1.0
    if normalization == "rms":
        scale = float(np.sqrt(np.mean(X * X)))
    else:
        scale = float(np.max(np.abs(X)))

    return 1.0 if scale <= 1.0e-14 else scale


def build_polynomial_pairwise_local_interaction_library(
    node_states: ArrayLike,
    D: ArrayLike,
    *,
    degree: int = 2,
    normalization: Literal["none", "rms", "maxabs"] = "rms",
    state_scale: Optional[float] = None,
) -> PairwiseLocalInteractionLibrary:
    """Build the default dynamics-agnostic pairwise polynomial library.

    For one endpoint the local law is expanded in monomials

        self**a * neighbour**b,

    with total degree ``1 <= a+b <= degree`` and ``b>=1``.  The head endpoint
    receives the swapped monomial.  Excluding ``b=0`` removes pure own-state
    terms from the pairwise sector and therefore avoids the structural
    edge-self gauge.

    The number of pairwise atoms is ``degree*(degree+1)/2``.
    """

    X, D_arr = _validate_node_states_and_incidence(node_states, D)
    degree = int(degree)
    if degree < 1:
        raise ValueError("degree must be >= 1.")

    xs, xr, endpoints, local, _, _ = _gather_endpoint_states(X, D_arr)
    scale = _resolve_polynomial_scale(
        X,
        normalization=normalization,
        state_scale=state_scale,
    )
    xsn = xs / scale
    xrn = xr / scale

    powers: list[tuple[int, int]] = []
    labels: list[str] = []
    for total in range(1, degree + 1):
        for a in range(0, total):
            b = total - a  # neighbour power; b>=1 by construction
            powers.append((int(a), int(b)))
            if a == 0:
                label = "neighbor" if b == 1 else f"neighbor^{b}"
            else:
                left = "self" if a == 1 else f"self^{a}"
                right = "neighbor" if b == 1 else f"neighbor^{b}"
                label = f"{left}*{right}"
            labels.append(label)

    E = np.empty((X.shape[0], D_arr.shape[1], len(powers), 2), dtype=float)
    for ell, (a, b) in enumerate(powers):
        tail_term = np.ones_like(xsn)
        head_term = np.ones_like(xrn)
        if a:
            tail_term *= np.power(xsn, a)
            head_term *= np.power(xrn, a)
        tail_term *= np.power(xrn, b)
        head_term *= np.power(xsn, b)
        E[:, :, ell, 0] = tail_term
        E[:, :, ell, 1] = head_term

    return PairwiseLocalInteractionLibrary(
        endpoint_states=np.asarray(endpoints, dtype=float),
        local_states=np.asarray(local, dtype=float),
        endpoint_features=np.asarray(E, dtype=float),
        feature_labels=tuple(labels),
        component_labels=tuple(f"poly:{label}" for label in labels),
        mode="polynomial",
        feature_representation="endpoint_pairwise_irreducible_polynomial",
        metadata={
            "degree": degree,
            "powers": tuple(powers),
            "normalization": str(normalization),
            "state_scale": float(scale),
            "one_body_sector_excluded": True,
            "n_features": len(labels),
        },
    )


# -----------------------------------------------------------------------------
# Candidate mode
# -----------------------------------------------------------------------------


def _coerce_atom_output(
    value: Any,
    *,
    expected_shape: tuple[int, int],
    atom_name: str,
) -> FloatArray:
    T, M = expected_shape

    if isinstance(value, (tuple, list)) and len(value) == 2:
        tail = np.asarray(value[0], dtype=float)
        head = np.asarray(value[1], dtype=float)
        if tail.shape != (T, M) or head.shape != (T, M):
            raise ValueError(
                f"Candidate atom {atom_name!r} returned tuple entries with shapes "
                f"{tail.shape} and {head.shape}; expected {(T, M)} for both."
            )
        out = np.stack([tail, head], axis=-1)
    else:
        out = np.asarray(value, dtype=float)
        if out.shape != (T, M, 2):
            raise ValueError(
                f"Candidate atom {atom_name!r} returned shape {out.shape}; "
                f"expected {(T, M, 2)} or a pair of {(T, M)} arrays."
            )

    if not np.all(np.isfinite(out)):
        raise ValueError(f"Candidate atom {atom_name!r} produced non-finite values.")
    return out


def build_candidate_pairwise_local_interaction_library(
    node_states: ArrayLike,
    D: ArrayLike,
    atoms: Sequence[PairwiseLocalInteractionAtom],
    *,
    check_sampled_swap_equivariance: bool = False,
    swap_rtol: float = 1.0e-9,
    swap_atol: float = 1.0e-11,
) -> PairwiseLocalInteractionLibrary:
    """Evaluate a caller-specified finite candidate interaction dictionary.

    Candidate atoms are evaluated on *physical, unnormalised* endpoint states.
    This is essential for mechanisms such as SIS or sin(x_j-x_i), whose
    functional meaning would otherwise change under an implicit rescaling.

    ``check_sampled_swap_equivariance=True`` optionally verifies on the supplied
    samples that atoms marked ``swap_equivariant=True`` obey

        Gamma(x_j,x_i) == swap(Gamma(x_i,x_j)).

    This is a diagnostic, not a proof over the full state domain.
    """

    X, D_arr = _validate_node_states_and_incidence(node_states, D)
    atoms = tuple(atoms)
    if not atoms:
        raise ValueError("candidate mode requires at least one interaction atom.")
    if not all(isinstance(a, PairwiseLocalInteractionAtom) for a in atoms):
        raise TypeError("Every candidate must be a PairwiseLocalInteractionAtom.")

    names = tuple(str(a.name) for a in atoms)
    if len(set(names)) != len(names):
        raise ValueError("Candidate atom names must be unique.")

    xs, xr, endpoints, local, _, _ = _gather_endpoint_states(X, D_arr)
    blocks: list[FloatArray] = []
    swap_checks: dict[str, bool] = {}

    for atom in atoms:
        block = _coerce_atom_output(
            atom.evaluate(xs, xr),
            expected_shape=xs.shape,
            atom_name=atom.name,
        )

        if check_sampled_swap_equivariance and atom.swap_equivariant is True:
            swapped_input = _coerce_atom_output(
                atom.evaluate(xr, xs),
                expected_shape=xs.shape,
                atom_name=atom.name,
            )
            expected = block[..., ::-1]
            ok = bool(np.allclose(swapped_input, expected, rtol=swap_rtol, atol=swap_atol))
            swap_checks[atom.name] = ok
            if not ok:
                raise ValueError(
                    f"Candidate atom {atom.name!r} is marked swap_equivariant=True "
                    "but failed the sampled orientation-swap check."
                )

        blocks.append(block)

    E = np.stack(blocks, axis=2)  # (T,M,L,2)

    return PairwiseLocalInteractionLibrary(
        endpoint_states=np.asarray(endpoints, dtype=float),
        local_states=np.asarray(local, dtype=float),
        endpoint_features=np.asarray(E, dtype=float),
        feature_labels=names,
        component_labels=tuple(f"candidate:{name}" for name in names),
        mode="candidate",
        feature_representation="endpoint_pairwise_candidate",
        metadata={
            "atom_descriptions": tuple(a.description for a in atoms),
            "atom_metadata": tuple(dict(a.metadata) for a in atoms),
            "swap_equivariant_flags": tuple(a.swap_equivariant for a in atoms),
            "sampled_swap_checks": dict(swap_checks),
            "one_body_sector_excluded_by_convention": True,
            "n_features": len(atoms),
        },
    )


# -----------------------------------------------------------------------------
# Unified dispatch
# -----------------------------------------------------------------------------


def build_pairwise_local_interaction_library(
    node_states: ArrayLike,
    D: ArrayLike,
    *,
    mode: Literal["polynomial", "candidate"] = "polynomial",
    degree: int = 2,
    normalization: Literal["none", "rms", "maxabs"] = "rms",
    state_scale: Optional[float] = None,
    atoms: Optional[Sequence[PairwiseLocalInteractionAtom]] = None,
    check_sampled_swap_equivariance: bool = False,
) -> PairwiseLocalInteractionLibrary:
    """Unified constructor for the two supported TIDES local-library modes."""

    mode = str(mode).lower()
    if mode == "polynomial":
        if atoms is not None:
            raise ValueError("atoms must be omitted when mode='polynomial'.")
        return build_polynomial_pairwise_local_interaction_library(
            node_states,
            D,
            degree=degree,
            normalization=normalization,
            state_scale=state_scale,
        )

    if mode == "candidate":
        if atoms is None:
            raise ValueError("atoms are required when mode='candidate'.")
        return build_candidate_pairwise_local_interaction_library(
            node_states,
            D,
            atoms,
            check_sampled_swap_equivariance=check_sampled_swap_equivariance,
        )

    raise ValueError("mode must be 'polynomial' or 'candidate'.")


# -----------------------------------------------------------------------------
# Small built-in candidate atoms / factories
# -----------------------------------------------------------------------------


def make_antisymmetric_difference_atom(
    name: str,
    function: Callable[[FloatArray], ArrayLike],
    *,
    description: str = "",
    metadata: Optional[Mapping[str, Any]] = None,
) -> PairwiseLocalInteractionAtom:
    """Wrap ``q=f(x_head-x_tail)`` as the endpoint law ``[q,-q]``.

    This helper is appropriate when ``f`` is odd (for example ``sin`` or the
    identity), in which case the resulting undirected law is swap-equivariant.
    """

    def evaluate(xs: FloatArray, xr: FloatArray) -> FloatArray:
        q = np.asarray(function(xr - xs), dtype=float)
        if q.shape != xs.shape:
            raise ValueError(
                f"Difference function for atom {name!r} returned shape {q.shape}; "
                f"expected {xs.shape}."
            )
        return np.stack([q, -q], axis=-1)

    return PairwiseLocalInteractionAtom(
        name=name,
        evaluate=evaluate,
        description=description,
        swap_equivariant=True,
        metadata={} if metadata is None else dict(metadata),
    )


def _sis_evaluate(xs: FloatArray, xr: FloatArray) -> FloatArray:
    tail = (1.0 - xs) * xr
    head = (1.0 - xr) * xs
    return np.stack([tail, head], axis=-1)


SIS_INTERACTION_ATOM = PairwiseLocalInteractionAtom(
    name="sis_infection",
    evaluate=_sis_evaluate,
    description=(
        "Undirected SIS pair-infection contribution: "
        "[(1-x_tail)x_head, (1-x_head)x_tail]."
    ),
    swap_equivariant=True,
    metadata={"family": "SIS"},
)

KURAMOTO_SIN_ATOM = make_antisymmetric_difference_atom(
    "sin_difference",
    np.sin,
    description="Antisymmetric sinusoidal coupling [sin(x_head-x_tail), -sin(x_head-x_tail)].",
    metadata={"family": "Kuramoto-like"},
)

LINEAR_DIFFUSIVE_ATOM = make_antisymmetric_difference_atom(
    "linear_difference",
    lambda z: z,
    description="Antisymmetric linear diffusive coupling [x_head-x_tail, x_tail-x_head].",
    metadata={"family": "diffusive"},
)


__all__ = [
    "PairwiseLocalInteractionAtom",
    "PairwiseLocalInteractionLibrary",
    "pair_endpoints_from_incidence",
    "build_polynomial_pairwise_local_interaction_library",
    "build_candidate_pairwise_local_interaction_library",
    "build_pairwise_local_interaction_library",
    "make_antisymmetric_difference_atom",
    "SIS_INTERACTION_ATOM",
    "KURAMOTO_SIN_ATOM",
    "LINEAR_DIFFUSIVE_ATOM",
]
