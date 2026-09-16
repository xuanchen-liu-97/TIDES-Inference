"""Compact temporal edge blocks for Step 3 (no node/time zero padding).

Only the current structural design C(theta) is materialized for the reference
SVD solver. Edge values are shared by baseline and all transition blocks.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class EdgeValues:
    rows: np.ndarray
    values: np.ndarray
    stage_offsets: tuple[int, ...]
    family: object = None
    _theta: bytes | None = None
    _response: np.ndarray | None = None

    def response(self, theta):
        theta = np.asarray(theta, dtype=float)
        key = theta.tobytes()
        if key != self._theta:
            self._response = self.values @ theta
            self._theta = key
        return self._response


class TemporalEdgeBlock:
    """An n_rows x L linear map backed by its nonzero endpoint rows."""
    ndim = 2

    def __init__(self, edge, first_stage, n_rows, atoms=None):
        self.edge = edge
        self.first_stage = int(first_stage)
        self.start = edge.stage_offsets[self.first_stage]
        self.atoms = tuple(range(edge.values.shape[1])) if atoms is None else tuple(atoms)
        self.shape = (int(n_rows), len(self.atoms))

    @property
    def rows(self):
        return self.edge.rows[self.start:]

    @property
    def values(self):
        values = self.edge.values[self.start:]
        if self.atoms == tuple(range(values.shape[1])):
            return values
        return values[:, self.atoms]

    def __matmul__(self, theta):
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (self.shape[1],):
            raise ValueError("TemporalEdgeBlock expects a coefficient vector.")
        full = np.zeros(self.edge.values.shape[1])
        full[list(self.atoms)] = theta
        out = np.zeros(self.shape[0])
        out[self.rows] = self.edge.response(full)[self.start:]
        return out

    def matvec(self, theta):
        return self @ theta

    def rmatvec(self, residual):
        return self.values.T @ np.asarray(residual)[self.rows]

    def __getitem__(self, key):
        rows, columns = key
        if not isinstance(rows, slice) or rows != slice(None):
            raise IndexError("Only full row slices are supported.")
        if np.isscalar(columns):
            out = np.zeros(self.shape[0])
            out[self.rows] = self.edge.values[self.start:, self.atoms[int(columns)]]
            return out
        indices = np.arange(self.shape[1])[columns]
        return TemporalEdgeBlock(self.edge, self.first_stage, self.shape[0],
                                 tuple(self.atoms[int(i)] for i in indices))

    def to_dense(self):
        out = np.zeros(self.shape)
        out[self.rows] = self.values
        return out


def coerce_block(block):
    return block if isinstance(block, TemporalEdgeBlock) else np.asarray(block, dtype=float)


def dense_block(block):
    return block.to_dense() if isinstance(block, TemporalEdgeBlock) else block


def derivative_products(blocks, amplitudes, residual, atoms):
    """Return [C_j a] and [C_j.T residual] without an L x n x p tensor."""
    v = np.zeros((residual.size, len(atoms)))
    b = np.empty((len(blocks), len(atoms)))
    for i, (block, amplitude) in enumerate(zip(blocks, amplitudes)):
        if isinstance(block, TemporalEdgeBlock):
            values = block.edge.values[block.start:, np.asarray(atoms)]
            v[block.rows] += amplitude * values
            b[i] = values.T @ residual[block.rows]
        else:
            values = block[:, atoms]
            v += amplitude * values
            b[i] = values.T @ residual
    return v, b


def temporal_blocks_from_family(family):
    """Use retained endpoint features; older family objects can use dense designs."""
    K, M, L = family.n_stages, family.n_edges, family.n_library_atoms
    n_rows = sum(d.n_scalar_observations for d in family.designs)
    edges = []
    for e in range(M):
        rows, values, offsets = [], [], [0]
        row_base = 0
        for design in family.designs:
            features = getattr(design, 'endpoint_features', None)
            if features is not None:
                endpoints = design.edge_endpoints[e]
                local_rows = (np.arange(design.n_stage_samples)[:, None] * design.n_nodes
                              + endpoints[None, :]).ravel()
                local_values = features[:, e].transpose(0, 2, 1).reshape(-1, L)
            elif isinstance(design.matrix, np.ndarray):
                block = design.matrix[:, e * L:(e + 1) * L]
                local_rows = np.flatnonzero(np.any(block != 0., axis=1))
                local_values = block[local_rows]
            else:
                raise ValueError("Operator Step-2 design must retain endpoint features for compact Step 3.")
            rows.append(row_base + local_rows)
            values.append(local_values)
            offsets.append(offsets[-1] + len(local_rows))
            row_base += design.n_scalar_observations
        edges.append(EdgeValues(np.concatenate(rows), np.concatenate(values), tuple(offsets), family))
    blocks = {}
    for first_stage in range(K):
        for e in range(M):
            group = ('baseline', e) if first_stage == 0 else ('transition', first_stage - 1, e)
            blocks[group] = TemporalEdgeBlock(edges[e], first_stage, n_rows)
    return blocks


def full_support_relaxation(blocks, atoms):
    """Full temporal support is a change of coordinates of independent stages.

    Return None for restricted support or an operator-only subset relaxation.
    This reuses only a residual bound, never changes the minimum-norm seed.
    """
    if not blocks or not all(isinstance(b, TemporalEdgeBlock) for b in blocks):
        return None
    family = blocks[0].edge.family
    if family is None or len(blocks) != family.n_stages * family.n_edges:
        return None
    if len({(id(b.edge), b.first_stage) for b in blocks}) != len(blocks):
        return None
    if tuple(atoms) == tuple(range(family.n_library_atoms)):
        return family.minimum_relative_residual
    columns = (np.arange(family.n_edges)[:, None] * family.n_library_atoms
               + np.asarray(atoms)[None, :]).ravel()
    rss = 0.
    for design in family.designs:
        if not isinstance(design.matrix, np.ndarray):
            return None
        A = design.matrix[:, columns]
        coef = np.linalg.lstsq(A, design.y, rcond=None)[0]
        residual = design.y - A @ coef
        rss += float(residual @ residual)
    return float(np.sqrt(rss) / family.target_norm)
