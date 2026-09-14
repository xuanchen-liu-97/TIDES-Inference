#!/usr/bin/env python3
"""Overnight scalability runner for TIDES Step 2.

Purpose
-------
This script is intentionally a *Step-2 algorithm benchmark*, not a full TIDES
scaling study.  It keeps the microscopic change complexity fixed while growing
ambient network size and records whether the scalable Step-2 backend continues
to recover the correct changed-edge supports without materialising the full
global change design.

Default experiment
------------------
- N = 16, 32, 64, 128
- six stages, five transitions
- true mean degree approximately 4 in every stage
- exactly 4 changed edges per transition (2 removed + 2 added)
- shared law phi(x) = x + 0.5 x^2
- complete candidate graph for inference
- Step 2 uses oracle stage boundaries by default to isolate the Step-2 solver
  (blind Step 1 is still run and logged as a diagnostic)
- dense/scalable cross-check for N <= 16 by default

The parent process launches each (N, replicate) as a separate subprocess.  This
means a crash, memory error, or timeout in one case does not destroy results from
previous cases.  A JSON result and text log are written per case, and a summary
CSV is refreshed after every case.

Typical Windows usage from the repository root:

    python -u run_step2_scaling.py --sizes 16 32 64 128 --resume

For a longer unattended run:

    python -u run_step2_scaling.py --sizes 16 32 64 128 256 \
        --timeout-minutes 240 --resume

The script expects the current repository modules (in particular
step2_change_structure.py and solvers_sparse_structure.py) to be importable from
its working directory / Python path.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from itertools import combinations
from typing import Iterable

import numpy as np


# -----------------------------------------------------------------------------
# Frozen benchmark constants
# -----------------------------------------------------------------------------

DT = 5.0e-4
STAGE_DURATION = 0.060
N_STAGES = 6
INTERVALS_PER_STAGE = int(round(STAGE_DURATION / DT))
PAIR_LAW_TRUE = np.array([1.0, 0.5], dtype=float)
N_CHANGES_PER_TRANSITION = 4
N_REMOVE = 2
N_ADD = 2
DEFAULT_AVG_DEGREE = 4.0

assert INTERVALS_PER_STAGE == 120
assert N_CHANGES_PER_TRANSITION == N_REMOVE + N_ADD


# -----------------------------------------------------------------------------
# Graph helpers
# -----------------------------------------------------------------------------


def _canon_edge(i: int, j: int) -> tuple[int, int]:
    if i == j:
        raise ValueError("Self edges are not allowed.")
    return (i, j) if i < j else (j, i)


def _is_connected(n: int, edges: Iterable[tuple[int, int]]) -> bool:
    adjacency = [[] for _ in range(n)]
    for i1, j1 in edges:
        i = i1 - 1
        j = j1 - 1
        adjacency[i].append(j)
        adjacency[j].append(i)

    seen = np.zeros(n, dtype=bool)
    stack = [0]
    seen[0] = True
    while stack:
        u = stack.pop()
        for v in adjacency[u]:
            if not seen[v]:
                seen[v] = True
                stack.append(v)
    return bool(np.all(seen))


def _is_forest(n: int, edges: Iterable[tuple[int, int]]) -> bool:
    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1
        return True

    for i1, j1 in edges:
        if not union(i1 - 1, j1 - 1):
            return False
    return True


def _sample_connected_graph(
    n: int,
    n_edges: int,
    all_edges: tuple[tuple[int, int], ...],
    rng: np.random.Generator,
) -> set[tuple[int, int]]:
    if n_edges < n - 1:
        raise ValueError("A connected graph needs at least N-1 edges.")
    if n_edges > len(all_edges):
        raise ValueError("Requested more active edges than candidate edges.")

    # Random spanning tree: each newly introduced node attaches to one earlier
    # node in a random ordering.
    order = rng.permutation(np.arange(1, n + 1))
    active: set[tuple[int, int]] = set()
    for pos in range(1, n):
        child = int(order[pos])
        parent = int(order[int(rng.integers(0, pos))])
        active.add(_canon_edge(child, parent))

    if len(active) < n_edges:
        remaining = [e for e in all_edges if e not in active]
        rng.shuffle(remaining)
        active.update(remaining[: n_edges - len(active)])

    if len(active) != n_edges or not _is_connected(n, active):
        raise RuntimeError("Failed to generate connected initial graph.")
    return active


def _next_snapshot(
    n: int,
    active: set[tuple[int, int]],
    all_edges: tuple[tuple[int, int], ...],
    rng: np.random.Generator,
    *,
    require_change_forest: bool = True,
    max_attempts: int = 20000,
) -> tuple[set[tuple[int, int]], tuple[tuple[int, int], ...]]:
    active_list = tuple(active)
    inactive = tuple(e for e in all_edges if e not in active)

    for _ in range(max_attempts):
        rem_idx = rng.choice(len(active_list), size=N_REMOVE, replace=False)
        removed = tuple(active_list[int(i)] for i in rem_idx)
        remaining = active.difference(removed)
        if not _is_connected(n, remaining):
            continue

        add_idx = rng.choice(len(inactive), size=N_ADD, replace=False)
        added = tuple(inactive[int(i)] for i in add_idx)
        changed = tuple(sorted((*removed, *added)))
        if len(set(changed)) != N_CHANGES_PER_TRANSITION:
            continue
        if require_change_forest and not _is_forest(n, changed):
            continue

        new_active = remaining.union(added)
        if len(new_active) != len(active):
            continue
        if not _is_connected(n, new_active):
            continue
        return new_active, changed

    raise RuntimeError("Could not generate a valid sparse structural transition.")


# -----------------------------------------------------------------------------
# Synthetic benchmark construction
# -----------------------------------------------------------------------------


def build_benchmark(
    n: int,
    seed: int,
    avg_degree: float = DEFAULT_AVG_DEGREE,
) -> dict:
    rng = np.random.default_rng(seed)
    candidate_edges = tuple(combinations(range(1, n + 1), 2))
    edge_index = {e: m for m, e in enumerate(candidate_edges)}
    m = len(candidate_edges)

    n_active = max(n - 1, int(round(avg_degree * n / 2.0)))
    n_active = min(n_active, m)

    # Fixed heterogeneous weight attached to every candidate edge.  Structural
    # switching changes only whether that edge is active.
    weight_values = rng.integers(800, 1201, size=m).astype(float) / 1000.0

    active = _sample_connected_graph(n, n_active, candidate_edges, rng)
    snapshots = [tuple(sorted(active))]
    changed_supports = []
    for _ in range(N_STAGES - 1):
        active, changed = _next_snapshot(n, active, candidate_edges, rng)
        snapshots.append(tuple(sorted(active)))
        changed_supports.append(changed)

    snapshots = tuple(snapshots)
    changed_supports = tuple(changed_supports)

    # Generic non-symmetric initial state; recenter to enforce the one conserved
    # sum exactly.  Keep amplitude comparable to the canonical N=8 benchmark.
    x0 = rng.uniform(-0.35, 0.35, size=n)
    x0 -= x0.mean()

    n_total_intervals = N_STAGES * INTERVALS_PER_STAGE
    t = np.arange(n_total_intervals + 1, dtype=float) * DT
    x = np.empty((n_total_intervals + 1, n), dtype=float)
    x[0] = x0

    stage_of_interval = np.repeat(
        np.arange(N_STAGES, dtype=int), INTERVALS_PER_STAGE
    )

    # Precompile each stage into endpoint/weight arrays for fast RK4 evaluation.
    compiled = []
    for snapshot in snapshots:
        idx = np.fromiter((edge_index[e] for e in snapshot), dtype=np.int64)
        ii = np.fromiter((e[0] - 1 for e in snapshot), dtype=np.int64)
        jj = np.fromiter((e[1] - 1 for e in snapshot), dtype=np.int64)
        ww = weight_values[idx]
        compiled.append((ii, jj, ww))

    def phi(v: np.ndarray) -> np.ndarray:
        return v + 0.5 * v * v

    def stage_field(state: np.ndarray, stage: int) -> np.ndarray:
        ii, jj, ww = compiled[stage]
        ph = phi(state)
        flux = ww * (ph[jj] - ph[ii])
        out = np.zeros(n, dtype=float)
        np.add.at(out, ii, flux)
        np.add.at(out, jj, -flux)
        return out

    def rk4_step(state: np.ndarray, stage: int) -> np.ndarray:
        k1 = stage_field(state, stage)
        k2 = stage_field(state + 0.5 * DT * k1, stage)
        k3 = stage_field(state + 0.5 * DT * k2, stage)
        k4 = stage_field(state + DT * k3, stage)
        return state + (DT / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    for q, stage in enumerate(stage_of_interval):
        x[q + 1] = rk4_step(x[q], int(stage))

    true_transition_indices = (
        np.arange(1, N_STAGES, dtype=int) * INTERVALS_PER_STAGE
    )
    true_transition_times = true_transition_indices.astype(float) * DT

    truth_groups = set()
    for k, edges in enumerate(changed_supports):
        for edge in edges:
            truth_groups.add((k, edge_index[edge]))

    return {
        "N": n,
        "seed": seed,
        "candidate_edges": candidate_edges,
        "edge_index": edge_index,
        "edge_weights": weight_values,
        "snapshots": snapshots,
        "changed_supports": changed_supports,
        "truth_groups": truth_groups,
        "n_active_edges": n_active,
        "x0": x0,
        "X": x,
        "t": t,
        "true_transition_indices": true_transition_indices,
        "true_transition_times": true_transition_times,
    }


# -----------------------------------------------------------------------------
# Step-1 diagnostic and preprocessing
# -----------------------------------------------------------------------------


def blind_step1_diagnostic(X: np.ndarray, t: np.ndarray) -> dict:
    """Run the same median+20 MAD secant detector used in the N=8 benchmark."""
    import tides

    v_sec = np.diff(X, axis=0) / DT
    jump = np.linalg.norm(v_sec[1:] - v_sec[:-1], axis=1)
    med = float(np.median(jump))
    mad = float(np.median(np.abs(jump - med)))
    threshold = med + 20.0 * mad

    result = tides.detect_changes(
        X,
        t,
        method="secant",
        threshold=threshold,
        min_separation=1,
    )
    return {
        "result": result,
        "threshold": threshold,
        "jump_median": med,
        "jump_mad": mad,
    }


def preprocess_midpoints(
    X: np.ndarray,
    t: np.ndarray,
    transition_indices: np.ndarray,
) -> dict:
    n_intervals = X.shape[0] - 1
    bounds = np.concatenate(([0], transition_indices, [n_intervals])).astype(int)
    stage_of_interval = np.empty(n_intervals, dtype=int)
    for stage in range(len(bounds) - 1):
        stage_of_interval[bounds[stage] : bounds[stage + 1]] = stage

    x_mid = []
    v_mid = []
    obs_stage = []
    t_mid = []

    for q in range(1, n_intervals - 1):
        if not (
            stage_of_interval[q - 1]
            == stage_of_interval[q]
            == stage_of_interval[q + 1]
        ):
            continue

        xq = (-X[q - 1] + 9.0 * X[q] + 9.0 * X[q + 1] - X[q + 2]) / 16.0
        vq = (
            X[q - 1] - 27.0 * X[q] + 27.0 * X[q + 1] - X[q + 2]
        ) / (24.0 * DT)
        x_mid.append(xq)
        v_mid.append(vq)
        obs_stage.append(stage_of_interval[q])
        t_mid.append(0.5 * (t[q] + t[q + 1]))

    return {
        "X_MID": np.asarray(x_mid, dtype=float),
        "V_MID": np.asarray(v_mid, dtype=float),
        "OBS_STAGE": np.asarray(obs_stage, dtype=int),
        "T_MID": np.asarray(t_mid, dtype=float),
        "stage_of_interval": stage_of_interval,
        "bounds": bounds,
    }


def finite_difference_weights(nodes: np.ndarray, derivative_order: int) -> np.ndarray:
    nodes = np.asarray(nodes, dtype=float)
    p = len(nodes)
    A = np.vstack([nodes**k for k in range(p)])
    b = np.zeros(p, dtype=float)
    b[derivative_order] = math.factorial(derivative_order)
    return np.linalg.solve(A, b)


def resolution_floor(
    X: np.ndarray,
    stage_of_interval: np.ndarray,
) -> tuple[float, dict]:
    z4 = np.array([-1.5, -0.5, 0.5, 1.5], dtype=float)
    z6 = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], dtype=float)
    w4_x = finite_difference_weights(z4, 0)
    w4_d = finite_difference_weights(z4, 1)
    w6_x = finite_difference_weights(z6, 0)
    w6_d = finite_difference_weights(z6, 1)

    rel_x = []
    rel_v = []
    n_intervals = X.shape[0] - 1
    for q in range(2, n_intervals - 2):
        stages = {stage_of_interval[q + j] for j in (-2, -1, 0, 1, 2)}
        if len(stages) != 1:
            continue
        p4 = X[q - 1 : q + 3]
        p6 = X[q - 2 : q + 4]
        x4 = (w4_x[:, None] * p4).sum(axis=0)
        x6 = (w6_x[:, None] * p6).sum(axis=0)
        v4 = (w4_d[:, None] * p4).sum(axis=0) / DT
        v6 = (w6_d[:, None] * p6).sum(axis=0) / DT
        rel_x.append(np.linalg.norm(x6 - x4) / max(np.linalg.norm(x6), 1e-15))
        rel_v.append(np.linalg.norm(v6 - v4) / max(np.linalg.norm(v6), 1e-15))

    rel_x_arr = np.asarray(rel_x, dtype=float)
    rel_v_arr = np.asarray(rel_v, dtype=float)
    if rel_x_arr.size == 0:
        raise RuntimeError("No valid six-point resolution-audit stencil.")
    resolution_max = max(
        float(rel_x_arr.max()), float(rel_v_arr.max()), 1.0e-14
    )
    floor = 20.0 * resolution_max
    return floor, {
        "resolution_samples": int(rel_x_arr.size),
        "max_relative_state_4v6": float(rel_x_arr.max()),
        "max_relative_velocity_4v6": float(rel_v_arr.max()),
        "median_relative_velocity_4v6": float(np.median(rel_v_arr)),
        "resolution_relative_max": float(resolution_max),
    }


# -----------------------------------------------------------------------------
# Step-2 input geometry
# -----------------------------------------------------------------------------


def build_step2_geometry(
    X_mid: np.ndarray,
    candidate_edges: tuple[tuple[int, int], ...],
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    m = len(candidate_edges)
    ii = np.fromiter((e[0] - 1 for e in candidate_edges), dtype=np.int64, count=m)
    jj = np.fromiter((e[1] - 1 for e in candidate_edges), dtype=np.int64, count=m)

    D = np.zeros((n, m), dtype=float)
    cols = np.arange(m, dtype=np.int64)
    D[ii, cols] = 1.0
    D[jj, cols] = -1.0

    edge_psi = np.empty((X_mid.shape[0], m, 2), dtype=float)
    xi = X_mid[:, ii]
    xj = X_mid[:, jj]
    edge_psi[:, :, 0] = xj - xi
    edge_psi[:, :, 1] = xj * xj - xi * xi
    return D, edge_psi


# -----------------------------------------------------------------------------
# Result helpers
# -----------------------------------------------------------------------------


def _jsonify(value):
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, set):
        return [_jsonify(v) for v in sorted(value)]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _support_metrics(selected: set[tuple[int, int]], truth: set[tuple[int, int]]) -> dict:
    tp = len(selected & truth)
    fp = len(selected - truth)
    fn = len(truth - selected)
    precision = tp / len(selected) if selected else (1.0 if not truth else 0.0)
    recall = tp / len(truth) if truth else 1.0
    return {
        "exact_support": selected == truth,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": float(precision),
        "recall": float(recall),
    }


def _compact_case_row(result: dict) -> dict:
    keys = [
        "status",
        "N",
        "replicate",
        "seed",
        "n_candidate_edges",
        "n_candidate_groups",
        "n_active_edges",
        "blind_step1_exact",
        "segmentation_used",
        "profile_floor",
        "scalable_runtime_seconds",
        "scalable_exact_support",
        "scalable_precision",
        "scalable_recall",
        "scalable_selected_groups",
        "scalable_selected_prefix",
        "scalable_final_relative_residual",
        "screening_group_count",
        "screening_ratio",
        "estimated_dense_bytes",
        "estimated_dense_gib",
        "global_change_design_materialized",
        "dense_ran",
        "dense_runtime_seconds",
        "dense_exact_support",
        "dense_scalable_same_support",
        "error_type",
        "error_message",
    ]
    return {k: result.get(k) for k in keys}


def _write_summary_csv(output_root: Path) -> None:
    results = []
    for p in sorted(output_root.glob("N*_rep*/result.json")):
        try:
            results.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    if not results:
        return
    rows = [_compact_case_row(r) for r in results]
    path = output_root / "summary.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# -----------------------------------------------------------------------------
# One benchmark case
# -----------------------------------------------------------------------------


def run_case(args: argparse.Namespace) -> dict:
    from step2_change_structure import infer_change_structure_from_observations

    n = int(args.N)
    rep = int(args.replicate)
    seed = int(args.seed)
    case_dir = Path(args.output).resolve() / f"N{n:04d}_rep{rep:03d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    result_path = case_dir / "result.json"

    result = {
        "status": "running",
        "N": n,
        "replicate": rep,
        "seed": seed,
        "started_at_epoch": time.time(),
    }

    try:
        print("=" * 88, flush=True)
        print(f"TIDES STEP-2 SCALING CASE: N={n}, replicate={rep}, seed={seed}", flush=True)
        print("=" * 88, flush=True)

        t0 = time.perf_counter()
        bench = build_benchmark(n, seed, avg_degree=float(args.avg_degree))
        result["benchmark_build_seconds"] = time.perf_counter() - t0
        X = bench["X"]
        t = bench["t"]
        candidate_edges = bench["candidate_edges"]
        truth_groups = set(bench["truth_groups"])
        m = len(candidate_edges)
        k = N_STAGES - 1

        result.update(
            {
                "n_candidate_edges": m,
                "n_candidate_groups": k * m,
                "n_active_edges": int(bench["n_active_edges"]),
                "true_changed_groups": len(truth_groups),
                "max_conservation_drift": float(
                    np.max(np.abs(X.sum(axis=1) - X[0].sum()))
                ),
                "max_abs_state": float(np.max(np.abs(X))),
                "state_displacement": float(np.linalg.norm(X[-1] - X[0])),
            }
        )

        # Blind Step 1 is a diagnostic.  The default Step-2 benchmark deliberately
        # uses oracle stage boundaries so Step-2 scalability is not confounded by
        # change-point detection.
        try:
            s1 = blind_step1_diagnostic(X, t)
            detected = np.asarray(s1["result"].transition_indices, dtype=int)
            blind_exact = np.array_equal(detected, bench["true_transition_indices"])
            result.update(
                {
                    "blind_step1_exact": bool(blind_exact),
                    "blind_step1_detected_count": int(len(detected)),
                    "blind_step1_detected_indices": detected.tolist(),
                    "blind_step1_threshold": float(s1["threshold"]),
                    "blind_step1_jump_median": float(s1["jump_median"]),
                    "blind_step1_jump_mad": float(s1["jump_mad"]),
                }
            )
        except Exception as exc:
            detected = np.array([], dtype=int)
            result.update(
                {
                    "blind_step1_exact": False,
                    "blind_step1_error": f"{type(exc).__name__}: {exc}",
                }
            )

        if args.segmentation == "blind":
            if not result.get("blind_step1_exact", False):
                raise RuntimeError(
                    "Blind Step 1 did not exactly recover the benchmark transitions; "
                    "cannot use --segmentation blind for this Step-2 case."
                )
            transition_indices = detected
            transition_times = t[transition_indices]
        else:
            transition_indices = bench["true_transition_indices"]
            transition_times = bench["true_transition_times"]
        result["segmentation_used"] = args.segmentation

        prep = preprocess_midpoints(X, t, np.asarray(transition_indices, dtype=int))
        X_mid = prep["X_MID"]
        V_mid = prep["V_MID"]
        obs_stage = prep["OBS_STAGE"]
        floor, floor_diag = resolution_floor(X, prep["stage_of_interval"])
        result["profile_floor"] = float(floor)
        result.update(floor_diag)
        result["n_preprocessed_observations"] = int(len(X_mid))

        tgeom = time.perf_counter()
        D, edge_psi = build_step2_geometry(X_mid, candidate_edges, n)
        result["geometry_build_seconds"] = time.perf_counter() - tgeom
        result["D_shape"] = list(D.shape)
        result["EDGE_PSI_shape"] = list(edge_psi.shape)

        print(
            f"candidate edges={m} | groups={k*m} | observations={len(X_mid)} | "
            f"floor={floor:.3e}",
            flush=True,
        )

        scalable_kwargs = {
            "verbose": True,
            "restricted_solver_kwargs": {
                "max_iter": int(args.restricted_max_iter),
                "tol": float(args.restricted_tol),
                "check_every": int(args.restricted_check_every),
            },
        }

        ts = time.perf_counter()
        scalable = infer_change_structure_from_observations(
            Y=V_mid,
            D=D,
            edge_features=edge_psi,
            stage_of_sample=obs_stage,
            hypothesis="varying_structure",
            transition_indices=transition_indices,
            transition_times=transition_times,
            edge_labels=candidate_edges,
            uncertainty_floor=floor,
            solver_method="group_bpdn_prefix",
            backend="scalable",
            solver_kwargs=scalable_kwargs,
            refit_method="dense_lstsq",
            require_solver_convergence=True,
            require_floor_reached=True,
            return_design=False,
        )
        scalable_runtime = time.perf_counter() - ts
        selected = set(tuple(map(int, g)) for g in scalable.selected_groups)
        met = _support_metrics(selected, truth_groups)
        md = scalable.metadata

        result.update(
            {
                "scalable_runtime_seconds": float(scalable_runtime),
                "scalable_exact_support": bool(met["exact_support"]),
                "scalable_precision": met["precision"],
                "scalable_recall": met["recall"],
                "scalable_true_positive": met["true_positive"],
                "scalable_false_positive": met["false_positive"],
                "scalable_false_negative": met["false_negative"],
                "scalable_selected_groups": int(len(selected)),
                "scalable_selected_prefix": int(
                    scalable.selection_result.selected_prefix_size
                ),
                "scalable_final_relative_residual": float(scalable.relative_residual),
                "scalable_convex_objective": float(md["convex_objective_value"]),
                "scalable_convex_iterations": int(md["convex_iterations"]),
                "scalable_convex_fixed_point_residual": float(
                    md["convex_fixed_point_residual"]
                ),
                "screening_group_count": int(md.get("screening_group_count", -1)),
                "screening_relative_residual": float(
                    md.get("screening_relative_residual", np.nan)
                ),
                "screening_ratio": float(
                    md.get("screening_group_count", np.nan) / (k * m)
                ),
                "estimated_dense_bytes": int(md["estimated_explicit_dense_bytes"]),
                "estimated_dense_gib": float(
                    md["estimated_explicit_dense_bytes"] / 1024.0**3
                ),
                "global_change_design_materialized": bool(
                    md["global_change_design_materialized"]
                ),
                "scalable_selected_group_labels": sorted([list(g) for g in selected]),
            }
        )

        # Optional dense cross-check while still cheap enough.
        result["dense_ran"] = False
        if n <= int(args.dense_up_to):
            print("\nStarting dense cross-backend reference...", flush=True)
            td = time.perf_counter()
            dense = infer_change_structure_from_observations(
                Y=V_mid,
                D=D,
                edge_features=edge_psi,
                stage_of_sample=obs_stage,
                hypothesis="varying_structure",
                transition_indices=transition_indices,
                transition_times=transition_times,
                edge_labels=candidate_edges,
                uncertainty_floor=floor,
                solver_method="group_bpdn_prefix",
                backend="dense",
                solver_kwargs={
                    "max_iter": int(args.dense_max_iter),
                    "tol": float(args.dense_tol),
                    "check_every": int(args.dense_check_every),
                    "verbose": True,
                },
                refit_method="dense_lstsq",
                require_solver_convergence=True,
                require_floor_reached=True,
                return_design=False,
            )
            dense_runtime = time.perf_counter() - td
            dense_selected = set(tuple(map(int, g)) for g in dense.selected_groups)
            dense_met = _support_metrics(dense_selected, truth_groups)
            result.update(
                {
                    "dense_ran": True,
                    "dense_runtime_seconds": float(dense_runtime),
                    "dense_exact_support": bool(dense_met["exact_support"]),
                    "dense_selected_groups": int(len(dense_selected)),
                    "dense_selected_prefix": int(
                        dense.selection_result.selected_prefix_size
                    ),
                    "dense_final_relative_residual": float(dense.relative_residual),
                    "dense_scalable_same_support": dense_selected == selected,
                    "dense_convex_objective": float(
                        dense.metadata["convex_objective_value"]
                    ),
                }
            )

        result["status"] = "success"
        result["finished_at_epoch"] = time.time()
        result["total_worker_seconds"] = (
            result["finished_at_epoch"] - result["started_at_epoch"]
        )

        print("\n" + "=" * 88, flush=True)
        print(
            f"CASE COMPLETE N={n}: exact_support={result['scalable_exact_support']} | "
            f"selected={result['scalable_selected_groups']} | "
            f"screened={result['screening_group_count']} | "
            f"runtime={scalable_runtime:.1f}s",
            flush=True,
        )
        print("=" * 88, flush=True)

    except BaseException as exc:
        result["status"] = "failed"
        result["finished_at_epoch"] = time.time()
        result["error_type"] = type(exc).__name__
        result["error_message"] = str(exc)
        result["traceback"] = traceback.format_exc()
        print("\nCASE FAILED", flush=True)
        traceback.print_exc()

    result_path.write_text(
        json.dumps(_jsonify(result), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result


# -----------------------------------------------------------------------------
# Parent orchestration
# -----------------------------------------------------------------------------


def parent_main(args: argparse.Namespace) -> int:
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    script = Path(__file__).resolve()
    tasks = []
    for rep in range(int(args.repeats)):
        for n in args.sizes:
            seed = int(args.base_seed) + rep * 100000 + int(n) * 1009
            tasks.append((int(n), rep, seed))

    print(f"Output directory: {output_root}")
    print(f"Cases: {len(tasks)}")
    print(f"Sizes: {list(map(int, args.sizes))}")
    print(f"Replicates: {args.repeats}")
    print(f"Per-case timeout: {args.timeout_minutes} minutes")
    print(f"Step-2 segmentation: {args.segmentation}")
    print(f"Dense cross-check through N={args.dense_up_to}")
    print()

    for case_index, (n, rep, seed) in enumerate(tasks, start=1):
        case_dir = output_root / f"N{n:04d}_rep{rep:03d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        result_path = case_dir / "result.json"
        log_path = case_dir / "run.log"

        if args.resume and result_path.exists():
            try:
                old = json.loads(result_path.read_text(encoding="utf-8"))
                if old.get("status") == "success":
                    print(
                        f"[{case_index}/{len(tasks)}] N={n} rep={rep}: "
                        "already successful, skipping (--resume)."
                    )
                    _write_summary_csv(output_root)
                    continue
            except Exception:
                pass

        print(f"[{case_index}/{len(tasks)}] Starting N={n} rep={rep} seed={seed}")
        cmd = [
            sys.executable,
            "-u",
            str(script),
            "--worker",
            "--N",
            str(n),
            "--replicate",
            str(rep),
            "--seed",
            str(seed),
            "--output",
            str(output_root),
            "--avg-degree",
            str(args.avg_degree),
            "--segmentation",
            str(args.segmentation),
            "--dense-up-to",
            str(args.dense_up_to),
            "--restricted-max-iter",
            str(args.restricted_max_iter),
            "--restricted-tol",
            str(args.restricted_tol),
            "--restricted-check-every",
            str(args.restricted_check_every),
            "--dense-max-iter",
            str(args.dense_max_iter),
            "--dense-tol",
            str(args.dense_tol),
            "--dense-check-every",
            str(args.dense_check_every),
        ]

        start = time.time()
        with log_path.open("w", encoding="utf-8") as log:
            try:
                proc = subprocess.run(
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=os.getcwd(),
                    timeout=float(args.timeout_minutes) * 60.0,
                    check=False,
                )
                elapsed = time.time() - start
                print(
                    f"    finished returncode={proc.returncode} in {elapsed/60.0:.1f} min; "
                    f"log={log_path.name}"
                )
            except subprocess.TimeoutExpired:
                elapsed = time.time() - start
                timeout_result = {
                    "status": "timeout",
                    "N": n,
                    "replicate": rep,
                    "seed": seed,
                    "elapsed_seconds": elapsed,
                    "error_type": "TimeoutExpired",
                    "error_message": (
                        f"Case exceeded {args.timeout_minutes} minute timeout."
                    ),
                }
                result_path.write_text(
                    json.dumps(timeout_result, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                print(f"    TIMEOUT after {elapsed/60.0:.1f} min; continuing.")

        _write_summary_csv(output_root)

    _write_summary_csv(output_root)
    print("\nAll requested cases processed.")
    print(f"Summary: {output_root / 'summary.csv'}")
    return 0


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Overnight TIDES Step-2 large-N scalability benchmark."
    )
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--N", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--replicate", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--seed", type=int, default=None, help=argparse.SUPPRESS)

    p.add_argument("--sizes", type=int, nargs="+", default=[16, 32, 64, 128])
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--base-seed", type=int, default=20260912)
    p.add_argument("--avg-degree", type=float, default=DEFAULT_AVG_DEGREE)
    p.add_argument(
        "--segmentation",
        choices=["oracle", "blind"],
        default="oracle",
        help=(
            "Stage boundaries used by Step 2. 'oracle' isolates Step-2 algorithm "
            "scaling; blind Step 1 is still logged as a diagnostic."
        ),
    )
    p.add_argument(
        "--dense-up-to",
        type=int,
        default=16,
        help="Also run dense reference for N <= this value; set 0 to disable.",
    )
    p.add_argument(
        "--timeout-minutes",
        type=float,
        default=180.0,
        help="Hard timeout per (N, replicate) subprocess.",
    )
    p.add_argument(
        "--output",
        type=str,
        default="step2_scaling_results",
        help="Result directory.",
    )
    p.add_argument("--resume", action="store_true")

    # Scalable restricted convex solve.
    p.add_argument("--restricted-max-iter", type=int, default=20000)
    p.add_argument("--restricted-tol", type=float, default=1.0e-7)
    p.add_argument("--restricted-check-every", type=int, default=100)

    # Dense reference solve.
    p.add_argument("--dense-max-iter", type=int, default=10000)
    p.add_argument("--dense-tol", type=float, default=1.0e-8)
    p.add_argument("--dense-check-every", type=int, default=100)
    return p


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()
    if args.worker:
        if args.N is None or args.seed is None:
            parser.error("worker mode requires --N and --seed")
        result = run_case(args)
        return 0 if result.get("status") == "success" else 2
    return parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
