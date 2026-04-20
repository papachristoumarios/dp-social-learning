"""
run_experiments.py
==================
Main entry point.  Runs experiments 1–6, GWAS-metrics vs ε / vs n centers, NYC exp 7,
and prints a summary table.  Also runs a numerical stability test suite first.
"""

import sys
import numpy as np
from pathlib import Path
import argparse
import os


# ---------------------------------------------------------------------------
# Numerical stability test suite
# ---------------------------------------------------------------------------

def run_stability_tests() -> bool:
    from dp_gwas_core import (
        log_belief_init, laplace_noise_log_belief,
        log_linear_update_all, make_adjacency, sensitivity_score_stat,
    )
    from scipy.special import logsumexp

    rng = np.random.default_rng(0)
    passed = 0; failed = 0

    def check(name, cond):
        nonlocal passed, failed
        if cond: print(f"  [PASS] {name}"); passed += 1
        else:    print(f"  [FAIL] {name}"); failed += 1

    print("\n=== Numerical stability tests ===")

    llr_tiny = np.full(100, 1e-12)
    lb = log_belief_init(llr_tiny)
    check("init: no NaN with tiny log-LR", not np.any(np.isnan(lb)))
    check("init: normalised (logsumexp ≈ 0)", np.allclose(logsumexp(lb, axis=1), 0, atol=1e-6))

    llr_huge = np.full(100, 1e6)
    lb = log_belief_init(llr_huge)
    check("init: no inf with huge log-LR", not np.any(np.isinf(lb)))
    check("init: normalised with huge log-LR", np.allclose(logsumexp(lb, axis=1), 0, atol=1e-6))

    llr = rng.normal(0, 1, size=200)
    lb = log_belief_init(llr)
    lb_noisy = laplace_noise_log_belief(lb, sensitivity=2.0, epsilon=0.01, n_states=2, K=20, rng=rng)
    check("laplace noise: no NaN at ε=0.01", not np.any(np.isnan(lb_noisy)))
    check("laplace noise: normalised", np.allclose(logsumexp(lb_noisy, axis=1), 0, atol=1e-6))

    L = np.stack([log_belief_init(rng.normal(0, 0.5, 50)) for _ in range(5)], axis=0)
    A = make_adjacency(5, "complete")
    for _ in range(200):
        L = log_linear_update_all(L, A)
    log_beliefs = [L[i] for i in range(5)]
    max_lse = max(np.max(np.abs(logsumexp(lb, axis=1))) for lb in log_beliefs)
    check(f"200 iterations: normalisation drift < 1e-5 (got {max_lse:.2e})", max_lse < 1e-5)
    max_lb = max(np.max(lb) for lb in log_beliefs)
    check(f"200 iterations: max log-belief ≤ 0 (got {max_lb:.4f})", max_lb <= 1e-6)

    lb_mat = np.stack(log_beliefs, axis=0)
    spread = lb_mat[:, :, 1].std(axis=0).max()
    check(f"consensus on complete graph: max std < 0.05 (got {spread:.4f})", spread < 0.05)

    delta = sensitivity_score_stat(500)
    check(f"sensitivity > 0 (got {delta:.4f})", delta > 0)

    for top in ["complete", "ring", "random"]:
        A = make_adjacency(5, topology=top, seed=0)
        check(f"adjacency {top}: doubly stochastic",
              np.allclose(A.sum(axis=1), 1, atol=1e-6) and
              np.allclose(A.sum(axis=0), 1, atol=1e-6))

    print(f"\nStability: {passed} passed, {failed} failed")
    return failed == 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    size = args.size
    os.makedirs(args.output_dir, exist_ok=True)
    ok = run_stability_tests()
    if not ok:
        print("\nAborting: stability tests failed.")
        sys.exit(1)

    print("\n=== Running experiments ===\n")

    from dp_gwas_experiments import (
        experiment1_privacy_utility, experiment2_three_way,
        experiment3_topology, experiment4_stratified,
        experiment5_scaling, experiment6_rizk_comparison,
        experiment_gwas_metrics_vs_epsilon,
        experiment_gwas_metrics_vs_n_centers,
        experiment_posterior_gm_am,
    )
    from dp_gwas_exp7_nyc import run_experiment7

    if size == 'medium':
        n_individuals = 25000
        n_snps = 600
        n_causal = 25
        n_centers_list = [5, 10, 20]
        n_centers = 5
        n_reps = 5
        epsilons = [0.2, 0.5, 1.0, 1.5]
        r1 = experiment1_privacy_utility(n_individuals=n_individuals, n_snps=n_snps, n_causal=n_causal, n_centers=n_centers, n_reps=n_reps, output_dir=args.output_dir)
        r2 = experiment2_three_way(n_individuals_list=[5000, 10000, 15000, 20000, 25000], n_snps=n_snps, n_causal=n_causal, n_centers=n_centers, n_reps=n_reps, output_dir=args.output_dir)
        r3 = experiment3_topology(n_individuals=n_individuals, n_snps=n_snps, n_causal=n_causal, n_centers=30, n_reps=n_reps, T=200, output_dir=args.output_dir)
        r4 = experiment4_stratified(n_individuals=n_individuals, n_snps=n_snps, n_causal=n_causal, n_reps=n_reps, output_dir=args.output_dir)
        r5 = experiment5_scaling(n_individuals_total=n_individuals, n_snps=n_snps, n_causal=n_causal, n_reps=n_reps, output_dir=args.output_dir)
        r6 = experiment6_rizk_comparison(n_individuals=n_individuals, n_snps=n_snps, n_causal=n_causal,
                                          n_centers_list=n_centers_list, epsilons=epsilons, n_reps=n_reps, output_dir=args.output_dir)
        r6b = experiment_gwas_metrics_vs_epsilon(
            n_individuals=n_individuals, n_snps=n_snps, n_causal=n_causal, n_centers=n_centers, n_reps=n_reps,
            output_dir=args.output_dir
        )
        r6c = experiment_gwas_metrics_vs_n_centers(
            n_individuals=n_individuals, n_snps=n_snps, n_causal=n_causal, n_reps=n_reps,
            epsilon=1.0,
            output_dir=args.output_dir
        )
        r9 = experiment_posterior_gm_am(
            n_individuals=n_individuals, n_snps=n_snps, n_causal=n_causal, n_centers=n_centers, epsilon=1.0,
            output_dir=args.output_dir
        )
        r7 = run_experiment7(size='medium', output_dir=args.output_dir)
    elif size == 'medium_nyc':
        r7 = run_experiment7(size='medium', output_dir=args.output_dir)
    elif size == 'large':
        n_individuals = 25000
        n_snps = 100000
        n_causal = 25
        n_reps = 5
        epsilons = [0.2, 0.5, 1.0, 1.5]
        n_centers_list = [5, 10, 20]
        n_centers = 5

        r1 = experiment1_privacy_utility(n_individuals=n_individuals, n_snps=n_snps, n_causal=n_causal, n_centers=n_centers, n_reps=n_reps, output_dir=args.output_dir)
        r2 = experiment2_three_way(n_individuals_list=[10000, 20000, 30000, 40000, 50000], n_snps=n_snps, n_causal=n_causal, n_centers=n_centers, n_reps=n_reps, output_dir=args.output_dir)
        r3 = experiment3_topology(n_individuals=n_individuals, n_snps=n_snps, n_causal=n_causal, n_centers=30, n_reps=n_reps, T=200, output_dir=args.output_dir)
        r4 = experiment4_stratified(n_individuals=n_individuals, n_snps=n_snps, n_causal=n_causal, n_reps=n_reps, output_dir=args.output_dir)
        r5 = experiment5_scaling(n_individuals_total=n_individuals, n_snps=n_snps, n_causal=n_causal, n_reps=n_reps, output_dir=args.output_dir)
        r6 = experiment6_rizk_comparison(n_individuals=n_individuals, n_snps=n_snps, n_causal=n_causal, n_centers_list=n_centers_list, epsilons=epsilons, n_reps=n_reps, output_dir=args.output_dir)
        r6b = experiment_gwas_metrics_vs_epsilon(n_individuals=n_individuals, n_snps=n_snps, n_causal=n_causal, n_centers=n_centers, n_reps=n_reps, epsilons=epsilons, output_dir=args.output_dir)
        r6c = experiment_gwas_metrics_vs_n_centers(
            n_individuals=n_individuals, n_snps=n_snps, n_causal=n_causal, n_reps=n_reps,
            epsilon=1.0,
            output_dir=args.output_dir
        )
        r9 = experiment_posterior_gm_am(
            n_individuals=n_individuals, n_snps=n_snps, n_causal=n_causal, n_centers=n_centers, epsilon=1.0,
            output_dir=args.output_dir
        )
        r7 = run_experiment7(size='large', output_dir=args.output_dir)
    elif size == 'large_nyc':
        r7 = run_experiment7(size='large', output_dir=args.output_dir)
    elif size == 'real_world_nyc':
        r7 = run_experiment7(size='real_world', output_dir=args.output_dir)
    
    
    print(f"\nAll figures saved to: {args.output_dir}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', choices=['small', 'medium', 'large', 'medium_nyc', 'large_nyc', 'real_world_nyc'], default='small')
    parser.add_argument('--output_dir', type=str, default='../figures')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
