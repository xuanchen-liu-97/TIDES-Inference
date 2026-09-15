"""TIDES Step 3: physical factorization and MDL compression.

Step 3 receives the observational edge-space family B_epsilon from Step 2 and a
physical hypothesis H.  It searches the hypothesis-induced representation class
H^epsilon and selects the shortest feasible description.

Implemented branch
------------------
H_FD: fixed dynamics, varying structure

    B^(r) = W^(r) Theta^T.

The Step-2 object remains the absolute family B_(1:K).  Within Step 3 only, the
invertible coordinates W^(1), Delta W^(1),...,Delta W^(K-1) are used because
support coding and structural moves are naturally expressed there.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

try:  # package form
    from .step2_edge_space_reconstruction import EdgeSpaceFieldFamily
    from .mdl_description_length import MDLScore, PolynomialLibrary
    from .mdl_precision import UniformDyadicQuantizer
    from .mdl_search import (
        FixedDynamicsCompressionResult,
        FixedDynamicsMDLProblem,
        optimize_fixed_dynamics_mdl,
    )
except ImportError:  # flat-file form
    from step2_edge_space_reconstruction import EdgeSpaceFieldFamily
    from mdl_description_length import MDLScore, PolynomialLibrary
    from mdl_precision import UniformDyadicQuantizer
    from mdl_search import (
        FixedDynamicsCompressionResult,
        FixedDynamicsMDLProblem,
        optimize_fixed_dynamics_mdl,
    )

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class PhysicalCompressionResult:
    """Selected Step-3 physical representation."""

    hypothesis: str

    # Final finite-precision MDL witness.
    W_stages: FloatArray
    theta: FloatArray
    B_stages: FloatArray
    delta_W: FloatArray
    delta_B: FloatArray

    # Continuous representative of the same selected discrete model.
    continuous_W_stages: FloatArray
    continuous_theta: FloatArray
    continuous_B_stages: FloatArray

    relative_residual: float
    continuous_relative_residual: float
    uncertainty_floor: float
    q_upper: int
    precision_certified_exact: bool
    mdl_score: MDLScore

    baseline_support: IntArray
    change_supports: tuple[IntArray, ...]
    active_atoms: tuple[int, ...]

    search_result: FixedDynamicsCompressionResult
    metadata: Mapping[str, Any]

    @property
    def feasible(self) -> bool:
        return bool(self.relative_residual <= self.uncertainty_floor * (1.0 + 1e-10))



def _infer_polynomial_library(
    family: EdgeSpaceFieldFamily,
    mdl_library: Optional[PolynomialLibrary],
) -> PolynomialLibrary:
    if mdl_library is not None:
        if mdl_library.n_atoms != family.n_library_atoms:
            raise ValueError(
                "mdl_library.n_atoms must equal the Step-2 library dimension."
            )
        return mdl_library

    meta = dict(family.library_metadata)
    powers = meta.get("powers")
    max_degree = meta.get("degree")
    if powers is not None and max_degree is not None:
        degrees = tuple(int(a) + int(b) for a, b in powers)
    else:
        atom_meta = meta.get("atom_metadata")
        if atom_meta is not None and len(atom_meta) == family.n_library_atoms:
            extracted = []
            for item in atom_meta:
                if not isinstance(item, Mapping) or "power" not in item:
                    extracted = []
                    break
                extracted.append(int(item["power"]))
            degrees = tuple(extracted)
            max_degree = max(degrees) if degrees else None
        else:
            degrees = tuple()
    if not degrees or max_degree is None:
        raise ValueError(
            "Cannot infer the polynomial MDL code from this Step-2 library. "
            "Supply mdl_library=PolynomialLibrary(...)."
        )
    if len(degrees) != family.n_library_atoms:
        raise ValueError("Polynomial metadata does not match the Step-2 library size.")
    return PolynomialLibrary(atom_degrees=degrees, max_degree=int(max_degree))


def _edge_atom_block(design, edge_index: int) -> FloatArray:
    """Observation-space block for one edge and all L atoms in one stage."""

    m = int(edge_index)
    L = int(design.n_library_atoms)
    if m < 0 or m >= int(design.n_edges):
        raise ValueError("edge_index out of range.")
    A = design.matrix
    if isinstance(A, np.ndarray):
        return np.asarray(A[:, m * L : (m + 1) * L], dtype=float)

    # Implicit/operator fallback.  This is intentionally simple; Step 3 search
    # can later replace it with a block-aware operator backend for large systems.
    cols = np.empty((int(design.n_scalar_observations), L), dtype=float)
    p = int(design.n_parameters)
    for ell in range(L):
        e = np.zeros(p, dtype=float)
        e[m * L + ell] = 1.0
        cols[:, ell] = np.asarray(A @ e, dtype=float).reshape(-1)
    return cols


def build_fixed_dynamics_problem(
    field_family: EdgeSpaceFieldFamily,
    *,
    mdl_library: Optional[PolynomialLibrary] = None,
) -> FixedDynamicsMDLProblem:
    """Construct the H_FD Step-3 search problem from the Step-2 family.

    The physical structural coordinates are

        W^(1), Delta W^(1), ..., Delta W^(K-1).

    Every candidate edge is available in every block.  No support is inherited
    from Step 2.
    """

    if not isinstance(field_family, EdgeSpaceFieldFamily):
        raise TypeError("field_family must be an EdgeSpaceFieldFamily.")
    if field_family.is_empty:
        raise ValueError("Step-2 edge-space family is empty at the supplied floor.")

    library = _infer_polynomial_library(field_family, mdl_library)
    K, M, L = (
        int(field_family.n_stages),
        int(field_family.n_edges),
        int(field_family.n_library_atoms),
    )

    y = np.concatenate([np.asarray(d.y, dtype=float) for d in field_family.designs])
    stage_rows = [int(d.n_scalar_observations) for d in field_family.designs]
    stage_edge_blocks = [
        tuple(_edge_atom_block(d, m) for m in range(M))
        for d in field_family.designs
    ]

    blocks: dict[tuple, FloatArray] = {}
    block_of: dict[tuple, int] = {}

    # Baseline W^(1): contributes to every stage.
    for m in range(M):
        g = ("baseline", m)
        blocks[g] = np.vstack([stage_edge_blocks[r][m] for r in range(K)])
        block_of[g] = 0

    # Delta W^(k): contributes to all stages after transition k.
    for k in range(K - 1):
        for m in range(M):
            pieces = []
            for r in range(K):
                if r <= k:
                    pieces.append(np.zeros((stage_rows[r], L), dtype=float))
                else:
                    pieces.append(stage_edge_blocks[r][m])
            g = ("transition", k, m)
            blocks[g] = np.vstack(pieces)
            block_of[g] = k + 1

    labels = ("baseline",) + tuple(("transition", k + 1) for k in range(K - 1))
    return FixedDynamicsMDLProblem(
        y=y,
        structural_blocks=blocks,
        structural_block_of_group=block_of,
        structural_candidate_counts=tuple(M for _ in range(K)),
        structural_block_labels=labels,
        library=library,
        uncertainty_floor=float(field_family.uncertainty_floor),
    )


def _amplitudes_to_W(
    *,
    active_groups: Sequence[tuple],
    amplitudes: FloatArray,
    n_stages: int,
    n_edges: int,
) -> tuple[FloatArray, FloatArray]:
    a = np.asarray(amplitudes, dtype=float).reshape(-1)
    if len(active_groups) != a.size:
        raise ValueError("amplitudes do not match active_groups.")

    W0 = np.zeros(n_edges, dtype=float)
    dW = np.zeros((max(0, n_stages - 1), n_edges), dtype=float)
    for value, g in zip(a, active_groups):
        if not isinstance(g, tuple) or not g:
            raise ValueError(f"Unrecognized structural group label {g!r}.")
        if g[0] == "baseline":
            _, m = g
            W0[int(m)] = float(value)
        elif g[0] == "transition":
            _, k, m = g
            dW[int(k), int(m)] = float(value)
        else:
            raise ValueError(f"Unrecognized structural group label {g!r}.")

    W = np.empty((n_stages, n_edges), dtype=float)
    W[0] = W0
    for r in range(1, n_stages):
        W[r] = W[r - 1] + dW[r - 1]
    return W, dW


def _supports_from_groups(
    active_groups: Sequence[tuple],
    *,
    n_stages: int,
) -> tuple[IntArray, tuple[IntArray, ...]]:
    baseline = []
    changes = [[] for _ in range(max(0, n_stages - 1))]
    for g in active_groups:
        if g[0] == "baseline":
            baseline.append(int(g[1]))
        elif g[0] == "transition":
            changes[int(g[1])].append(int(g[2]))
    return (
        np.asarray(sorted(baseline), dtype=np.int64),
        tuple(np.asarray(sorted(s), dtype=np.int64) for s in changes),
    )


def compress_fixed_dynamics(
    field_family: EdgeSpaceFieldFamily,
    *,
    mdl_library: Optional[PolynomialLibrary] = None,
    quantizer: Optional[UniformDyadicQuantizer] = None,
    quantizer_bound: float = 1.0,
    initial_groups: Optional[Sequence[tuple]] = None,
    initial_atoms: Optional[Sequence[int]] = None,
    n_basin_hops: int = 80,
    basin_seed: Optional[int] = None,
    basin_sigma_min: float = 5e-3,
    basin_sigma_max: float = 1.5e-1,
    basin_local_max_nfev: int = 50,
    basin_initial_theta: Optional[FloatArray] = None,
    max_sweeps: int = 50,
    q_min: int = 1,
    q_max: int = 64,
    max_nfev: int = 250,
    max_coordinate_passes: int = 8,
    top_group_drops: int = 6,
    top_group_adds: int = 4,
    top_group_swaps: int = 4,
    top_atom_drops: int = 4,
    top_atom_adds: int = 2,
    top_atom_swaps: int = 2,
) -> PhysicalCompressionResult:
    """Run the complete H_FD factorization + conditional-MDL search."""

    problem = build_fixed_dynamics_problem(field_family, mdl_library=mdl_library)
    if quantizer is None:
        quantizer = UniformDyadicQuantizer(bound=float(quantizer_bound))

    search = optimize_fixed_dynamics_mdl(
        problem,
        quantizer=quantizer,
        initial_groups=initial_groups,
        initial_atoms=initial_atoms,
        n_basin_hops=n_basin_hops,
        basin_seed=basin_seed,
        basin_sigma_min=basin_sigma_min,
        basin_sigma_max=basin_sigma_max,
        basin_local_max_nfev=basin_local_max_nfev,
        basin_initial_theta=basin_initial_theta,
        max_sweeps=max_sweeps,
        q_min=q_min,
        q_max=q_max,
        max_nfev=max_nfev,
        max_coordinate_passes=max_coordinate_passes,
        top_group_drops=top_group_drops,
        top_group_adds=top_group_adds,
        top_group_swaps=top_group_swaps,
        top_atom_drops=top_atom_drops,
        top_atom_adds=top_atom_adds,
        top_atom_swaps=top_atom_swaps,
    )

    state = search.best_state
    witness = state.precision.witness
    if witness is None or state.precision.q_upper is None:
        raise RuntimeError("Internal error: selected state has no precision witness.")

    W, dW = _amplitudes_to_W(
        active_groups=state.active_groups,
        amplitudes=witness.amplitudes,
        n_stages=field_family.n_stages,
        n_edges=field_family.n_edges,
    )
    theta = np.asarray(witness.theta, dtype=float)
    B = W[:, :, None] * theta[None, None, :]

    Wc, dWc = _amplitudes_to_W(
        active_groups=state.active_groups,
        amplitudes=state.continuous.amplitudes,
        n_stages=field_family.n_stages,
        n_edges=field_family.n_edges,
    )
    thetac = np.asarray(state.continuous.theta, dtype=float)
    Bc = Wc[:, :, None] * thetac[None, None, :]

    rho = float(field_family.relative_residual(B))
    rho_c = float(field_family.relative_residual(Bc))
    # The block adapter and Step-2 feasibility oracle must agree numerically.
    if abs(rho - float(witness.relative_residual)) > 1e-8 * max(1.0, rho, witness.relative_residual):
        raise RuntimeError("Step-3 block model and Step-2 feasibility oracle disagree.")
    if not field_family.is_feasible(B, rtol=1e-8):
        raise RuntimeError("Selected finite-precision representation is outside B_epsilon.")

    baseline_support, change_supports = _supports_from_groups(
        state.active_groups,
        n_stages=field_family.n_stages,
    )

    return PhysicalCompressionResult(
        hypothesis="fixed_dynamics",
        W_stages=W,
        theta=theta,
        B_stages=B,
        delta_W=dW,
        delta_B=np.diff(B, axis=0),
        continuous_W_stages=Wc,
        continuous_theta=thetac,
        continuous_B_stages=Bc,
        relative_residual=rho,
        continuous_relative_residual=rho_c,
        uncertainty_floor=float(field_family.uncertainty_floor),
        q_upper=int(state.precision.q_upper),
        precision_certified_exact=bool(state.precision.certified_exact),
        mdl_score=state.mdl_upper,
        baseline_support=baseline_support,
        change_supports=change_supports,
        active_atoms=tuple(state.active_atoms),
        search_result=search,
        metadata={
            "step2_minimum_relative_residual": float(field_family.minimum_relative_residual),
            "n_active_structural_groups": int(len(state.active_groups)),
            "n_active_atoms": int(len(state.active_atoms)),
            "n_mdl_moves": int(len(search.history)),
            "quantizer_bound": float(quantizer.bound),
            "precision_status": "feasible_upper_bound",
        },
    )


def compress_physical_representation(
    field_family: EdgeSpaceFieldFamily,
    *,
    hypothesis: str = "fixed_dynamics",
    **kwargs,
) -> PhysicalCompressionResult:
    """Unified Step-3 dispatcher."""

    H = str(hypothesis).lower()
    if H in {"fixed_dynamics", "h_fd", "fd"}:
        return compress_fixed_dynamics(field_family, **kwargs)
    if H in {"fixed_topology", "h_ft", "ft"}:
        raise NotImplementedError(
            "The H_FT search backend will use the same Step-2 EdgeSpaceFieldFamily "
            "but requires its own factorization/search geometry."
        )
    raise ValueError(f"Unknown physical hypothesis {hypothesis!r}.")


# Short public alias.
compress = compress_physical_representation


__all__ = [
    "PhysicalCompressionResult",
    "build_fixed_dynamics_problem",
    "compress_fixed_dynamics",
    "compress_physical_representation",
    "compress",
]
