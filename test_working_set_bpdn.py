"""Minimal regression test for the dual-certified working-set grouped BPDN solver."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SOLVER = HERE / "solvers_sparse_structure.py"

spec = importlib.util.spec_from_file_location("solvers_sparse_structure_test", SOLVER)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)


def main() -> None:
    rng = np.random.default_rng(4)
    n = 70
    n_groups = 20
    group_width = 2
    p = n_groups * group_width

    X = rng.normal(size=(n, p))
    # Mild coherence makes the initial marginal seed imperfect and exercises
    # omitted-group reactivation.
    X[:, 10:16] += 0.3 * X[:, 0:6]

    groups = {
        g: np.arange(group_width * g, group_width * (g + 1), dtype=int)
        for g in range(n_groups)
    }

    beta_true = np.zeros(p)
    for g in (2, 6, 11, 17):
        beta_true[groups[g]] = rng.normal(size=group_width)

    noise = 0.015 * rng.normal(size=n)
    y = X @ beta_true + noise
    radius = 1.05 * np.linalg.norm(noise)

    dense = module.solve_group_basis_pursuit_denoising(
        X,
        y,
        groups,
        residual_radius=radius,
        max_iter=25_000,
        tol=1e-9,
        check_every=100,
        verbose=False,
    )

    def group_block_provider(label):
        return X[:, groups[label]]

    def group_adjoint_norm_provider(vector, labels):
        adjoint = X.T @ vector
        return {
            label: float(np.linalg.norm(adjoint[groups[label]]))
            for label in labels
        }

    working = module.solve_working_set_group_basis_pursuit_denoising(
        y,
        groups,
        n_coefficients=p,
        group_block_provider=group_block_provider,
        group_adjoint_norm_provider=group_adjoint_norm_provider,
        residual_radius=radius,
        seed_batch_size=3,
        seed_growth_factor=1.6,
        max_seed_scans=10,
        kkt_tol=1e-7,
        restricted_solver_kwargs={
            "max_iter": 25_000,
            "tol": 1e-9,
            "check_every": 100,
            "verbose": False,
        },
        verbose=False,
    )

    coefficient_relative_error = np.linalg.norm(
        working.coefficients - dense.coefficients
    ) / max(np.linalg.norm(dense.coefficients), np.finfo(float).tiny)

    objective_relative_error = abs(
        working.objective_value - dense.objective_value
    ) / max(abs(dense.objective_value), np.finfo(float).tiny)

    assert dense.converged and dense.feasible
    assert working.converged and working.feasible
    assert working.global_dual_feasible
    assert working.max_global_dual_ratio <= 1.0 + 1e-7
    assert coefficient_relative_error < 1e-6
    assert objective_relative_error < 1e-7
    assert working.total_reactivations >= 1

    print("working-set BPDN regression passed")
    print(f"  dense objective      : {dense.objective_value:.12e}")
    print(f"  working objective    : {working.objective_value:.12e}")
    print(f"  coefficient rel diff : {coefficient_relative_error:.3e}")
    print(f"  objective rel diff   : {objective_relative_error:.3e}")
    print(f"  seed groups          : {len(working.seed_groups)}")
    print(f"  final working groups : {len(working.working_set)}")
    print(f"  reactivations        : {working.total_reactivations}")
    print(f"  KKT audits           : {working.kkt_audits}")
    print(f"  max global dual ratio: {working.max_global_dual_ratio:.12e}")
    print(f"  global duality gap   : {working.global_duality_gap:.3e}")


if __name__ == "__main__":
    main()
