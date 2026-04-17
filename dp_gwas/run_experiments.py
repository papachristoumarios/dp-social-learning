"""
run_experiments.py
==================
Main entry point.  Runs experiments 1–6, GWAS-metrics vs ε / vs n centers, NYC exp 7,
and prints a summary table.  Also runs a numerical stability test suite first.
"""

import sys
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Numerical stability test suite
# ---------------------------------------------------------------------------

def run_stability_tests() -> bool:
    from dp_gwas_core import (
        log_belief_init, laplace_noise_log_belief,
        log_linear_update, make_adjacency, sensitivity_score_stat,
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

    log_beliefs = [log_belief_init(rng.normal(0, 0.5, 50)) for _ in range(5)]
    A = make_adjacency(5, "complete")
    for _ in range(200):
        log_beliefs = [log_linear_update(log_beliefs, A, i) for i in range(5)]
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

def main(fast: bool = False) -> None:
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

    if fast:
        cfg = dict(n_individuals=800, n_snps=200, n_causal=10, n_reps=2)
        r1 = experiment1_privacy_utility(**cfg, epsilons=[0.1, 0.5, 1.0, 2.0])
        r2 = experiment2_three_way(n_individuals_list=[400, 800], n_snps=200, n_causal=10, n_reps=2)
        r3 = experiment3_topology(n_individuals=800, n_snps=200, n_causal=10, n_centers=4, n_reps=2, T=60)
        r4 = experiment4_stratified(**cfg)
        r5 = experiment5_scaling(n_individuals_total=800, n_snps=200, n_causal=10, n_reps=2)
        r6 = experiment6_rizk_comparison(n_individuals=800, n_snps=200, n_causal=10,
                                          n_centers_list=[3, 5], epsilons=[0.5, 1.0], n_reps=2)
        r6b = experiment_gwas_metrics_vs_epsilon(**cfg, epsilons=[0.1, 0.5, 1.0, 2.0])
        r6c = experiment_gwas_metrics_vs_n_centers(
            **cfg, epsilon=1.0, n_centers_list=[2, 3, 5, 8],
        )
        r9 = experiment_posterior_gm_am(
            n_individuals=cfg["n_individuals"],
            n_snps=cfg["n_snps"],
            n_causal=cfg["n_causal"],
            n_centers=4,
            epsilon=1.0,
        )
        r7 = run_experiment7(fast=True)
    else:
        r1 = experiment1_privacy_utility(n_individuals=25000, n_snps=600, n_causal=25, n_centers=5, n_reps=5)
        r2 = experiment2_three_way(n_individuals_list=[500, 1000, 2000, 4000], n_snps=600, n_causal=25, n_centers=5, n_reps=4)
        r3 = experiment3_topology(n_individuals=2500, n_snps=400, n_causal=20, n_centers=30, n_reps=3, T=200)
        r4 = experiment4_stratified(n_individuals=3000, n_snps=600, n_causal=25, n_reps=4)
        r5 = experiment5_scaling(n_individuals_total=3000, n_snps=400, n_causal=20, n_reps=3)
        r6 = experiment6_rizk_comparison(n_individuals=2000, n_snps=400, n_causal=20,
                                          n_centers_list=[5, 10, 20], epsilons=[0.2, 0.5, 1.0, 1.5], n_reps=3)
        r6b = experiment_gwas_metrics_vs_epsilon(
            n_individuals=2500, n_snps=600, n_causal=25, n_centers=5, n_reps=5,
        )
        r6c = experiment_gwas_metrics_vs_n_centers(
            n_individuals=2500, n_snps=600, n_causal=25, n_reps=5,
            epsilon=1.0,
        )
        r9 = experiment_posterior_gm_am(
            n_individuals=2500, n_snps=600, n_causal=25, n_centers=5, epsilon=1.0,
        )
        r7 = run_experiment7(fast=False)

    # ---- Summary table ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Exp 1 — ε*: {r1.get('eps_crit', 'N/A')}  "
          f"Oracle power: {r1['oracle_power']:.3f}  Single-site: {r1['single_power']:.3f}")

    if "df" in r2:
        import pandas as pd
        agg = r2["df"].groupby("n")[["oracle_power", "dp_power", "single_power"]].mean()
        row = agg.iloc[-1]
        print(f"Exp 2 — n={agg.index.max()}: Oracle={row['oracle_power']:.3f}  "
              f"DP={row['dp_power']:.3f}  Single={row['single_power']:.3f}")

    print(f"Exp 5 — centers {r5['n_centers_list']}: powers {[f'{p:.2f}' for p in r5['powers']]}")

    nc0 = list(r6.keys())[0]; r6v = r6[nc0]
    if 1.0 in r6v["epsilons"]:
        idx = r6v["epsilons"].index(1.0)
        print(f"Exp 6 — TVD@ε=1: GM={r6v['tvd_gm'][idx]:.4f}  "
              f"AM={r6v['tvd_am'][idx]:.4f}  Rizk={r6v['tvd_rizk'][idx]:.4f}")

    if 1.0 in r6b["epsilons"]:
        i = r6b["epsilons"].index(1.0)
        print(
            f"Exp 6b (metrics vs ε) — @ε=1: power_GM={r6b['power_gm_mean'][i]:.3f}  "
            f"F1_GM={r6b['f1_gm_mean'][i]:.3f}  FPR_GM={r6b['fpr_gm_mean'][i]:.4f}"
        )

    nc_mid = r6c["n_centers_list"][len(r6c["n_centers_list"]) // 2]
    i_n = r6c["n_centers_list"].index(nc_mid)
    print(
        f"Exp 6c (metrics vs n @ε={r6c['epsilon']}) — n={nc_mid}: "
        f"power_GM={r6c['power_gm_mean'][i_n]:.3f}  F1_GM={r6c['f1_gm_mean'][i_n]:.3f}"
    )

    r7b = r7["r7b"]
    print(f"\nExp 7 — NYC network ({len(r7['hospitals'])} hospitals):")
    print(f"  ε* ≈ {r7b['eps_crit']}  Oracle power: {r7b['oracle_power']:.3f}  "
          f"Best hospital: {r7b['best_hospital'][:40]}")
    for top_name, v in r7["r7a"].items():
        print(f"  Topology {top_name:15s}: power={np.mean(v['power']):.3f}  gap={v['sg']:.3f}")

    print(f"\nAll figures saved to: {Path('../figures').resolve()}")


if __name__ == "__main__":
    main(fast="--fast" in sys.argv)
