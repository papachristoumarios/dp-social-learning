"""
dp_gwas_experiments.py
======================
Experiments validating the DP distributed GWAS extension.
Each experiment returns a results dict and saves its own figure.

Experiments
-----------
1. Privacy-utility tradeoff   : power & FDR vs epsilon (mirrors Figs 1-2)
2. Three-way comparison        : single-site vs DP-distributed vs oracle
3. Network topology effects    : complete / ring / random on convergence
4. Heritability & MAF strata   : power stratified by h2 and minor allele freq
5. Scaling with n_centers      : communication cost & power vs n
6. Comparison with Rizk 2023   : TVD to ground truth vs epsilon (mirrors Fig 6)
7. GWAS metrics vs ε           : power, FDR, F1, FPR from evaluate_gwas vs budget
8. GWAS metrics vs n           : same metrics with fixed ε, varying number of centers
9. GM vs AM posteriors         : scatter and marginal histograms of $P(H_1)$ under both aggregators
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.special import expit
import time
import os
import seaborn as sns
import pandas as pd

from dp_gwas_core import (
    simulate_gwas_data,
    split_data_across_centers,
    run_dp_gwas_mle,
    centralized_gwas,
    single_center_gwas,
    run_rizk_baseline,
    evaluate_gwas,
    spectral_gap,
    make_adjacency,
)

# Shared style
plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "axes.prop_cycle": plt.cycler(
        color=["#1d9e75", "#534ab7", "#d85a30", "#888780", "#ba7517"]
    ),
})

ALPHA_GWAS = 5e-8     # genome-wide significance


NO_DP_DISTRIBUTED_LABEL = "No DP Distributed"
DP_DISTRIBUTED_GM_LABEL = "DP Distributed (GM)"
DP_DISTRIBUTED_AM_LABEL = "DP Distributed (AM)"

SINGLE_CENTER_LABEL = "Single Center"
ORACLE_LABEL = "Centralized"

STATISTICAL_POWER_LABEL = "Statistical Power (Recall of causal SNPs)"

# ---------------------------------------------------------------------------
# Experiment 1 : Privacy-utility tradeoff
# ---------------------------------------------------------------------------

def experiment1_privacy_utility(
    n_individuals: int = 6000,
    n_snps: int = 500,
    n_causal: int = 20,
    n_centers: int = 5,
    n_reps: int = 3,
    epsilons: list[float] | None = None,
    seed: int = 1,
    output_dir: str = "../figures",
) -> dict:
    """
    Sweep epsilon in [0.05, 5].  For each epsilon, run DP-GWAS (GM + AM)
    and record power, FDR, and F1.  Overlay the Centralized,
    single-center, and no-DP distributed baselines as horizontal lines.
    No-DP is :func:`run_dp_gwas_mle` with ``K=1`` and ``epsilon=np.inf``
    (distributed consensus, no privacy noise).

    This mirrors Figures 1-2 of the paper: statistical power as a function
    of privacy budget, with a critical epsilon above which distributed DP
    outperforms acting alone.
    """
    if epsilons is None:
        epsilons = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]

    print("Experiment 1: Privacy-utility tradeoff")

    results = {eps: {"power_gm": [], "fdr_gm": [], "f1_gm": [],
                     "power_am": [], "fdr_am": [], "f1_am": []}
               for eps in epsilons}


    # Oracle, single-site, and no-DP distributed benchmarks (computed once per rep)
    oracle_power, oracle_fdr, oracle_f1 = [], [], []
    single_power, single_fdr, single_f1 = [], [], []
    nodp_power_gm, nodp_power_am = [], []
    nodp_f1_gm, nodp_f1_am = [], []

    for rep in range(n_reps):
        data = simulate_gwas_data(n_individuals, n_snps, n_causal, seed=seed + rep)
        centers = split_data_across_centers(data, n_centers, seed=seed + rep)
        causal_idx = data["causal_idx"]

        oracle = centralized_gwas(centers, alpha=ALPHA_GWAS)
        oracle_power.append(oracle["power"])
        oracle_fdr.append(oracle["fdr"])
        oracle_f1.append(oracle["f1"])

        single = single_center_gwas(centers, alpha=ALPHA_GWAS)
        single_power.append(single["power"])
        single_fdr.append(single["fdr"])
        single_f1.append(single["f1"])

        nodp = run_dp_gwas_mle(
            centers,
            epsilon=np.inf,
            alpha=ALPHA_GWAS,
            K=1,
            topology="complete",
            seed=seed + rep * 100,
        )
        m_nodp_gm = evaluate_gwas(nodp.selected_gm, causal_idx, n_snps)
        m_nodp_am = evaluate_gwas(nodp.selected_am, causal_idx, n_snps)
        nodp_power_gm.append(m_nodp_gm["power"])
        nodp_power_am.append(m_nodp_am["power"])
        nodp_f1_gm.append(m_nodp_gm["f1"])
        nodp_f1_am.append(m_nodp_am["f1"])

        for eps in epsilons:
            res = run_dp_gwas_mle(
                centers, epsilon=eps, alpha=ALPHA_GWAS, K=15,
                topology="complete", seed=seed + rep * 100,
            )
            m_gm = evaluate_gwas(res.selected_gm, causal_idx, n_snps)
            m_am = evaluate_gwas(res.selected_am, causal_idx, n_snps)
            results[eps]["power_gm"].append(m_gm["power"])
            results[eps]["fdr_gm"].append(m_gm["fdr"])
            results[eps]["f1_gm"].append(m_gm["f1"])
            results[eps]["power_am"].append(m_am["power"])
            results[eps]["fdr_am"].append(m_am["fdr"])
            results[eps]["f1_am"].append(m_am["f1"])

    # Aggregate
    eps_arr = np.array(epsilons)
    mean = lambda lst: np.mean(lst)
    se   = lambda lst: np.std(lst) / np.sqrt(max(len(lst), 1))

    power_gm  = [mean(results[e]["power_gm"])  for e in epsilons]
    power_am  = [mean(results[e]["power_am"])  for e in epsilons]
    fdr_gm    = [mean(results[e]["fdr_gm"])    for e in epsilons]
    fdr_am    = [mean(results[e]["fdr_am"])    for e in epsilons]

    se_pgm    = [se(results[e]["power_gm"])    for e in epsilons]
    se_pam    = [se(results[e]["power_am"])    for e in epsilons]

    f1_gm     = [mean(results[e]["f1_gm"])     for e in epsilons]
    f1_am     = [mean(results[e]["f1_am"])     for e in epsilons]

    oracle_p  = mean(oracle_power)
    single_p  = mean(single_power)
    nodp_p_gm = mean(nodp_power_gm)
    nodp_p_am = mean(nodp_power_am)
    oracle_f1_m = mean(oracle_f1)
    single_f1_m = mean(single_f1)
    nodp_f1_gm_m = mean(nodp_f1_gm)
    nodp_f1_am_m = mean(nodp_f1_am)

    # Find critical epsilon (first eps where power_gm > single_p)
    eps_crit = None
    eps_crit_f1 = None
    for i, eps in enumerate(epsilons):
        if power_gm[i] > single_p:
            eps_crit = eps
            break
        if f1_gm[i] > single_p:
            eps_crit_f1 = eps
            break

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.errorbar(eps_arr, power_gm, yerr=se_pgm, marker="o", label=DP_DISTRIBUTED_GM_LABEL,
                capsize=3, linewidth=1.5)
    ax.errorbar(eps_arr, power_am, yerr=se_pam, marker="s", label=DP_DISTRIBUTED_AM_LABEL,
                capsize=3, linewidth=1.5, linestyle="--")
    ax.axhline(oracle_p, color="gray", linestyle=":", linewidth=1.5, label=ORACLE_LABEL)
    ax.axhline(single_p, color="gray", linestyle="-.", linewidth=1.2, label=SINGLE_CENTER_LABEL)
    ax.axhline(
        nodp_p_gm,
        color="#ba7517",
        linestyle=(0, (3, 1, 1, 1)),
        linewidth=1.4,
        label=NO_DP_DISTRIBUTED_LABEL,
    )
    
    if eps_crit is not None:
        ax.axvline(eps_crit, color="red", linestyle="--", alpha=0.4, linewidth=1,
                   label=f"ε* ≈ {eps_crit}")
    ax.set_xlabel("Privacy budget ($\epsilon$)")
    ax.set_ylabel(STATISTICAL_POWER_LABEL)
    ax.set_xscale("log")
    ax.set_ylim(0.05, 1.05)

    ax = axes[1]
    ax.plot(eps_arr, f1_gm, marker="o", label=DP_DISTRIBUTED_GM_LABEL, linewidth=1.5)
    ax.plot(eps_arr, f1_am, marker="s", label=DP_DISTRIBUTED_AM_LABEL, linewidth=1.5, linestyle="--")
    ax.axhline(oracle_f1_m, color="gray", linestyle=":", linewidth=1.5, label=ORACLE_LABEL)
    ax.axhline(single_f1_m, color="gray", linestyle="-.", linewidth=1.2, label=SINGLE_CENTER_LABEL)
    ax.axhline(
        nodp_f1_gm_m,
        color="#ba7517",
        linestyle=(0, (3, 1, 1, 1)),
        linewidth=1.4,
        label=NO_DP_DISTRIBUTED_LABEL,
    )
    
    if eps_crit_f1 is not None:
        ax.axvline(eps_crit_f1, color="red", linestyle="--", alpha=0.4, linewidth=1,
                   label=f"ε* ≈ {eps_crit_f1}")
    ax.set_xlabel("Privacy budget ($\epsilon$)")
    ax.set_ylabel("F1 score")
    ax.set_xscale("log")
    ax.set_ylim(0.05, 1.05)
    ax.legend(fontsize=8, frameon=False)

    fig.suptitle(
        f"Privacy-utility tradeoff  "
        f"($N={n_individuals}$, $M_{{SNP}}={n_snps}$, $M_{{causal}}={n_causal}$, $n={n_centers}$)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "exp1_privacy_utility.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved exp1_privacy_utility.pdf  (ε* ≈ {eps_crit})")

    return dict(
        epsilons=epsilons, power_gm=power_gm, power_am=power_am,
        fdr_gm=fdr_gm, fdr_am=fdr_am,
        oracle_power=oracle_p, single_power=single_p, eps_crit=eps_crit,
        eps_crit_f1=eps_crit_f1,
        nodp_power_gm=nodp_p_gm,
        nodp_power_am=nodp_p_am,
        nodp_f1_gm=nodp_f1_gm_m,
        nodp_f1_am=nodp_f1_am_m,
    )


# ---------------------------------------------------------------------------
# Experiment 2 : Three-way comparison — single-site / DP-distributed / oracle
# ---------------------------------------------------------------------------

def experiment2_three_way(
    n_individuals_list: list[int] | None = None,
    n_snps: int = 500,
    n_causal: int = 20,
    n_centers: int = 5,
    epsilon: float = 1.0,
    n_reps: int = 4,
    seed: int = 2,
    output_dir: str = "../figures",
) -> dict:
    """
    Compare oracle, single-site, DP-distributed, and no-DP distributed
    (``K=1``, ``epsilon=np.inf``) across varying total cohort sizes.
    Shows the regimes where DP-distributed recovers near-oracle performance
    and where single-site fails.
    """
    if n_individuals_list is None:
        n_individuals_list = [2000, 4000, 8000, 16000]

    print("Experiment 2: Three-way comparison across cohort sizes")

    rows = []
    for n in n_individuals_list:
        for rep in range(n_reps):
            data = simulate_gwas_data(n, n_snps, n_causal, seed=seed + rep * 7 + n)
            centers = split_data_across_centers(data, n_centers, seed=seed + rep)
            causal_idx = data["causal_idx"]

            oracle = centralized_gwas(centers, alpha=ALPHA_GWAS)
            single = single_center_gwas(centers, alpha=ALPHA_GWAS)
            dp_res = run_dp_gwas_mle(
                centers, epsilon=epsilon, alpha=ALPHA_GWAS, K=15,
                topology="complete", seed=seed + rep * 100,
            )
            m_gm = evaluate_gwas(dp_res.selected_gm, causal_idx, n_snps)
            nodp_res = run_dp_gwas_mle(
                centers,
                epsilon=np.inf,
                alpha=ALPHA_GWAS,
                K=1,
                topology="complete",
                seed=seed + rep * 100,
            )
            m_nodp = evaluate_gwas(nodp_res.selected_gm, causal_idx, n_snps)

            rows.append(dict(
                n=n,
                oracle_power=oracle["power"], oracle_fdr=oracle["fdr"],
                single_power=single["power"], single_fdr=single["fdr"],
                dp_power=m_gm["power"], dp_fdr=m_gm["fdr"],
                nodp_power=m_nodp["power"], nodp_fdr=m_nodp["fdr"],
            ))

    import pandas as pd
    df = pd.DataFrame(rows)
    agg = df.groupby("n").agg(["mean", "sem"]).reset_index()

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    N = np.array(n_individuals_list, dtype=int)
    col_o, col_d, col_s, col_n = (
        "oracle_power",
        "dp_power",
        "single_power",
        "nodp_power",
    )
    ax.errorbar(N, agg[(col_o, "mean")], yerr=agg[(col_o, "sem")], marker="o", label=ORACLE_LABEL, linewidth=1.5)
    ax.errorbar(N, agg[(col_d, "mean")], yerr=agg[(col_d, "sem")], marker="s", label=DP_DISTRIBUTED_GM_LABEL, linewidth=1.5)
    ax.errorbar(N, agg[(col_s, "mean")], yerr=agg[(col_s, "sem")], marker="^", label=SINGLE_CENTER_LABEL, linewidth=1.5, linestyle="--")
    ax.errorbar(
        N,
        agg[(col_n, "mean")],
        yerr=agg[(col_n, "sem")],
        marker="D",
        color="#ba7517",
        label=NO_DP_DISTRIBUTED_LABEL,
        linewidth=1.5,
        linestyle=(0, (3, 1, 1, 1)),
    )
    ax.set_xlabel("Total cohort size ($N$)")
    ax.set_ylabel(STATISTICAL_POWER_LABEL)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, frameon=False)

    fig.suptitle(
        f"{STATISTICAL_POWER_LABEL} vs $N$ ($\epsilon={epsilon}$, $n={n_centers}$, $M_{{SNP}}={n_snps}$, $M_{{causal}}={n_causal}$)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "exp2_three_way.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  → saved exp2_three_way.pdf")
    return dict(df=df)

def experiment3_topology(
    n_individuals: int = 6000,
    n_snps: int = 300,
    n_causal: int = 15,
    n_centers: int = 6,
    epsilon: float = 1.0,
    T: int = 150,
    n_reps: int = 3,
    seed: int = 3,
    output_dir: str = "../figures",
) -> dict:
    """
    Compare convergence speed and final power across network topologies.
    Reports spectral gap, power, FDR, and belief-trace TV distance.

    Directly validates the communication complexity bound's dependence
    on a*_n = |lambda_2((A+I)/2)| from Table 1.
    """
    topologies = ["complete", "ring", "random", "star", "scale-free", "small-world"]
    print("Experiment 3: Network topology effects")

    results = {t: {"power": [], "fdr": [], "converged_at": [], "slem": []} for t in topologies}

    for top in topologies:
        A = make_adjacency(n_centers, topology=top, seed=seed)
        sg = spectral_gap(A)

        for rep in range(n_reps):
            data = simulate_gwas_data(n_individuals, n_snps, n_causal, seed=seed + rep * 13)
            centers = split_data_across_centers(data, n_centers, seed=rep)
            causal_idx = data["causal_idx"]

            dp_res = run_dp_gwas_mle(
                centers, epsilon=epsilon, alpha=ALPHA_GWAS, K=15, T=T,
                topology=top, seed=seed + rep * 100,
                track_convergence=True, convergence_tol=1e-3,
            )
            m = evaluate_gwas(dp_res.selected_gm, causal_idx, n_snps)
            results[top]["power"].append(m["power"])
            results[top]["fdr"].append(m["fdr"])
            results[top]["converged_at"].append(dp_res.converged_at)
            results[top]["slem"].append(1 - sg)

    # Belief trace from last rep (for plotting)
    traces = {}
    for top in topologies:
        data = simulate_gwas_data(n_individuals, n_snps, n_causal, seed=seed + 99)
        centers = split_data_across_centers(data, n_centers, seed=0)
        dp_res = run_dp_gwas_mle(
            centers, epsilon=epsilon, alpha=ALPHA_GWAS, K=5, T=T,
            topology=top, seed=seed, track_convergence=True,
        )
        traces[top] = dp_res.belief_trace

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # Power bar chart
    ax = axes[0]
    x = np.arange(len(topologies))
    powers = [np.mean(results[t]["power"]) for t in topologies]
    sems   = [np.std(results[t]["power"]) / np.sqrt(n_reps) for t in topologies]
    slem_vals = [np.mean(results[t]["slem"]) for t in topologies]
    bars = ax.bar(x, powers, yerr=sems, capsize=4,
                  color=["#1d9e75", "#534ab7", "#d85a30", "#888780", "#999999", "#ba7517"])
    ax.set_xticks(x)
    ax.set_xticklabels(topologies, rotation=90, fontsize=8)
    ax.set_ylabel(STATISTICAL_POWER_LABEL + " (GM)")
    ax.set_ylim(0, 1.1)
    ax.set_title("Power by topology")

    # Spectral gap vs convergence iteration
    ax = axes[1]
    
    records = []

    for t in topologies:
        A = make_adjacency(n_centers, topology=t, seed=seed)
        sg = spectral_gap(A)
        conv_iters = results[t]["converged_at"]

        records.append({
            'Spectral gap': sg,
            'Convergence iterations': np.mean(conv_iters),
            'Topology': t,
        })
    
    df = pd.DataFrame(records)
    sns.scatterplot(data=df, x='Spectral gap', y='Convergence iterations', style='Topology', ax=ax, s=80, palette=["#1d9e75", "#534ab7", "#d85a30", "#888780", "#999999", "#ba7517"])
    ax.set_xlabel("Spectral gap ($1 - |\\lambda_2|$)")
    ax.set_ylabel("Iterations to convergence")
    ax.set_title("Spectral gap vs convergence")

    ax.set_xscale("log")

    fig.suptitle(
        f"Network topology  ($N={n_individuals}$, $\epsilon={epsilon}$, $n={n_centers}$)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "exp3_topology.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  → saved exp3_topology.pdf")
    return results


# ---------------------------------------------------------------------------
# Experiment 4 : Heritability & MAF strata
# ---------------------------------------------------------------------------

def experiment4_stratified(
    n_individuals: int = 8000,
    n_snps: int = 600,
    n_causal: int = 20,
    n_centers: int = 5,
    epsilon: float = 1.0,
    n_reps: int = 3,
    seed: int = 4,
    output_dir: str = "../figures",
) -> dict:
    """
    Power stratified by (a) heritability h² and (b) minor allele frequency.

    Standard GWAS practice — quantifies whether the DP method maintains
    expected power gradients.  Low-MAF SNPs are harder to detect; high-h²
    traits are easier.  The DP method should preserve both gradients.
    """
    print("Experiment 4: Heritability & MAF strata")

    h2_vals = [0.2, 0.3, 0.5, 0.7]

    # --- h2 sweep ---
    h2_power_oracle, h2_power_dp, h2_power_single = [], [], []
    for h2 in h2_vals:
        pwr_o, pwr_d, pwr_s = [], [], []
        for rep in range(n_reps):
            data = simulate_gwas_data(n_individuals, n_snps, n_causal, h2=h2,
                                      seed=seed + rep * 11)
            centers = split_data_across_centers(data, n_centers, seed=rep)
            causal_idx = data["causal_idx"]

            oracle = centralized_gwas(centers, alpha=ALPHA_GWAS)
            single = single_center_gwas(centers, alpha=ALPHA_GWAS)
            dp_res = run_dp_gwas_mle(
                centers, epsilon=epsilon, alpha=ALPHA_GWAS, K=15,
                topology="complete", seed=seed + rep * 100,
            )
            m = evaluate_gwas(dp_res.selected_gm, causal_idx, n_snps)
            pwr_o.append(oracle["power"])
            pwr_d.append(m["power"])
            pwr_s.append(single["power"])

        h2_power_oracle.append(np.mean(pwr_o))
        h2_power_dp.append(np.mean(pwr_d))
        h2_power_single.append(np.mean(pwr_s))

    # --- MAF-stratified power (single run, SNPs binned by MAF) ---
    data = simulate_gwas_data(n_individuals, n_snps, n_causal,
                               maf_range=(0.01, 0.5), h2=0.3, seed=seed + 999)
    centers = split_data_across_centers(data, n_centers, seed=0)
    causal_idx = data["causal_idx"]
    mafs = data["mafs"]

    oracle = centralized_gwas(centers, alpha=ALPHA_GWAS)
    dp_res = run_dp_gwas_mle(
        centers, epsilon=epsilon, alpha=ALPHA_GWAS, K=15,
        topology="complete", seed=seed,
    )

    # Bin SNPs by MAF
    maf_bins = [0.01, 0.05, 0.1, 0.2, 0.35, 0.5]
    maf_labels = ["<5%", "5–10%", "10–20%", "20–35%", ">35%"]
    power_by_maf_oracle, power_by_maf_dp = [], []

    for lo, hi in zip(maf_bins[:-1], maf_bins[1:]):
        mask = (mafs >= lo) & (mafs < hi)
        causal_in_bin = causal_idx[np.isin(causal_idx, np.where(mask)[0])]
        if len(causal_in_bin) == 0:
            power_by_maf_oracle.append(np.nan)
            power_by_maf_dp.append(np.nan)
            continue
        pow_o = evaluate_gwas(oracle["selected"] & mask, causal_in_bin, mask.sum())["power"]
        pow_d = evaluate_gwas(dp_res.selected_gm & mask, causal_in_bin, mask.sum())["power"]
        power_by_maf_oracle.append(pow_o)
        power_by_maf_dp.append(pow_d)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.plot(h2_vals, h2_power_oracle, "o-", label=ORACLE_LABEL, linewidth=1.5)
    ax.plot(h2_vals, h2_power_dp,     "s-", label=DP_DISTRIBUTED_GM_LABEL, linewidth=1.5)
    ax.plot(h2_vals, h2_power_single,  "^--", label=SINGLE_CENTER_LABEL, linewidth=1.5)
    ax.set_xlabel("Heritability (h²)")
    ax.set_ylabel(STATISTICAL_POWER_LABEL)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1]
    x = np.arange(len(maf_labels))
    w = 0.35
    ax.bar(x - w/2, power_by_maf_oracle, w, label=ORACLE_LABEL)
    ax.bar(x + w/2, power_by_maf_dp,     w, label=DP_DISTRIBUTED_GM_LABEL)
    ax.set_xticks(x)
    ax.set_xticklabels(maf_labels, rotation=15, fontsize=8)
    ax.set_ylabel(STATISTICAL_POWER_LABEL)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8, frameon=False)

    fig.suptitle(
        f"{STATISTICAL_POWER_LABEL} by heritability & MAF strata  ($N={n_individuals}$, $n={n_centers}$)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "exp4_stratified.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  → saved exp4_stratified.pdf")
    return dict(h2_vals=h2_vals, h2_power_oracle=h2_power_oracle,
                h2_power_dp=h2_power_dp, h2_power_single=h2_power_single,
                power_by_maf_dp=power_by_maf_dp)


# ---------------------------------------------------------------------------
# Experiment 5 : Scaling with number of centers
# ---------------------------------------------------------------------------

def experiment5_scaling(
    n_individuals_total: int = 10000,
    n_snps: int = 400,
    n_causal: int = 20,
    epsilon: float = 1.0,
    n_reps: int = 3,
    seed: int = 5,
    output_dir: str = "../figures",
) -> dict:
    """
    Fix total cohort size, vary n_centers from 2 to 20.
    Measures power, FDR, and effective communication complexity K*T.
    Validates the polylog(n) overhead bound in Table 1.
    """
    n_centers_list = [2, 3, 5, 8, 10, 15, 20]
    print("Experiment 5: Scaling with number of centers")

    res = {nc: {"power": [], "fdr": [], "comm_complexity": [], "time": []}
           for nc in n_centers_list}

    for nc in n_centers_list:
        for rep in range(n_reps):
            data = simulate_gwas_data(n_individuals_total, n_snps, n_causal,
                                      seed=seed + rep * 7)
            centers = split_data_across_centers(data, nc, seed=rep)
            causal_idx = data["causal_idx"]
            K = 15

            time_start = time.time()    
            dp_res = run_dp_gwas_mle(
                centers, epsilon=epsilon, alpha=ALPHA_GWAS, K=K,
                topology="complete", seed=seed + rep * 100,
                track_convergence=False,
            )
            time_end = time.time()

            time_taken = time_end - time_start

            m = evaluate_gwas(dp_res.selected_gm, causal_idx, n_snps)
            
            T_used = dp_res.converged_at
            
            res[nc]["power"].append(m["power"])
            res[nc]["fdr"].append(m["fdr"])
            res[nc]["comm_complexity"].append(K * T_used * nc)
            res[nc]["time"].append(time_taken)
    fdrs   = [np.mean(res[nc]["fdr"])   for nc in n_centers_list]
    comms  = [np.mean(res[nc]["comm_complexity"]) for nc in n_centers_list]
    times  = [np.mean(res[nc]["time"]) for nc in n_centers_list]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    ax = axes[0]
    ax.plot(n_centers_list, comms, marker="o", linewidth=1.5, color="#534ab7")
    ax.set_xlabel("Number of centers ($n$)")
    ax.set_ylabel("Total number of pairwise belief exchanges")
    # Overlay theoretical O(n log n) curve
    n_arr = np.array(n_centers_list, dtype=float)
    scale = comms[0] / (n_arr[0] * np.log(n_arr[0]))
    ax.plot(n_arr, scale * n_arr * np.log(n_arr),
            "r--", alpha=0.5, linewidth=1, label="Theoretical upper bound: $O(polylog(n))$")
    ax.legend(fontsize=8, frameon=False)

    ax.set_xticks(n_centers_list)
    ax.set_xticklabels(n_centers_list)

    ax = axes[1]
    ax.plot(n_centers_list, times, marker="o", linewidth=1.5, color="#534ab7")
    ax.set_xlabel("Number of centers ($n$)")
    ax.set_ylabel("Time taken (seconds)")
    ax.legend(fontsize=8, frameon=False)
    ax.set_xticks(n_centers_list)
    ax.set_xticklabels(n_centers_list)

    fig.suptitle(f"Scaling with $n$ centers ($N={n_individuals_total}$, $M_{{SNP}}={n_snps}$, $M_{{causal}}={n_causal}$, $\epsilon={epsilon}$)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "exp5_scaling.pdf"), bbox_inches="tight")
    print("  → saved exp5_scaling.pdf")
    return dict(n_centers_list=n_centers_list, fdrs=fdrs, comms=comms)

# ---------------------------------------------------------------------------
# Experiment 6 : Comparison with Rizk et al. 2023 (mirrors Fig 6)
# ---------------------------------------------------------------------------

def experiment6_rizk_comparison(
    n_individuals: int = 6000,
    n_snps: int = 400,
    n_causal: int = 20,
    n_centers_list: list[int] | None = None,
    epsilons: list[float] | None = None,
    n_reps: int = 3,
    seed: int = 6,
    output_dir: str = "../figures",
) -> dict:
    """
    Average total variation distance (TVD) between each method's selected set
    and the oracle selection, as a function of epsilon and n_centers.

    Methods: DP-GWAS (GM), DP-GWAS (AM), Rizk et al. 2023.
    Mirrors Figure 6 of the paper.
    """
    if n_centers_list is None:
        n_centers_list = [5, 10, 20]
    if epsilons is None:
        epsilons = [0.2, 0.5, 1.0, 1.5]

    print("Experiment 6: Comparison with Rizk et al. 2023")

    def tvd(sel_a: np.ndarray, sel_b: np.ndarray) -> float:
        """Binary TVD between two selection vectors."""
        return float(np.mean(sel_a.astype(float) != sel_b.astype(float)))

    fig, axes = plt.subplots(1, len(n_centers_list), figsize=(4 * len(n_centers_list), 4),
                             sharey=True)
    if len(n_centers_list) == 1:
        axes = [axes]

    all_results = {}
    for nc, ax in zip(n_centers_list, axes):
        tvd_gm, tvd_am, tvd_rizk = [], [], []
        for eps in epsilons:
            tvds_gm, tvds_am, tvds_r = [], [], []
            for rep in range(n_reps):
                data = simulate_gwas_data(n_individuals, n_snps, n_causal,
                                          seed=seed + rep * 13 + nc)
                centers = split_data_across_centers(data, nc, seed=rep)
                oracle = centralized_gwas(centers, alpha=ALPHA_GWAS)

                dp_res = run_dp_gwas_mle(
                    centers, epsilon=eps, alpha=ALPHA_GWAS, K=15,
                    topology="complete", seed=seed + rep * 100,
                )
                rizk = run_rizk_baseline(
                    centers, epsilon=eps, alpha=ALPHA_GWAS,
                    T=150, seed=seed + rep * 100,
                )

                tvds_gm.append(tvd(dp_res.selected_gm, oracle["selected"]))
                tvds_am.append(tvd(dp_res.selected_am, oracle["selected"]))
                tvds_r.append(tvd(rizk["selected"],    oracle["selected"]))

            tvd_gm.append(np.mean(tvds_gm))
            tvd_am.append(np.mean(tvds_am))
            tvd_rizk.append(np.mean(tvds_r))

        ax.plot(epsilons, tvd_gm,  "o-", label="DP-GWAS (GM)",  linewidth=1.5)
        ax.plot(epsilons, tvd_am,  "s--", label="DP-GWAS (AM)", linewidth=1.5)
        ax.plot(epsilons, tvd_rizk,"^:",  label="Rizk et al.",  linewidth=1.5)
        ax.set_xlabel("ε")
        ax.set_title(f"$n={nc}$ centers")
        ax.set_ylim(0, None)
        if ax == axes[0]:
            ax.set_ylabel("TVD to oracle")
        ax.legend(fontsize=7, frameon=False)

        all_results[nc] = dict(epsilons=epsilons, tvd_gm=tvd_gm,
                               tvd_am=tvd_am, tvd_rizk=tvd_rizk)

    fig.suptitle("TVD to oracle: DP-GWAS vs Rizk et al. 2023", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "exp6_rizk_comparison.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  → saved exp6_rizk_comparison.pdf")
    return all_results


# ---------------------------------------------------------------------------
# Experiment 7 : Full GWAS evaluation metrics vs privacy budget
# ---------------------------------------------------------------------------

def experiment_gwas_metrics_vs_epsilon(
    n_individuals: int = 6000,
    n_snps: int = 500,
    n_causal: int = 20,
    n_centers: int = 5,
    n_reps: int = 3,
    epsilons: list[float] | None = None,
    seed: int = 7,
    output_dir: str = "../figures",
) -> dict:
    """
    Plot all metrics returned by ``evaluate_gwas`` (power, FDR, F1, FPR)
    as a function of ε, for both GM and AM DP-GWAS.  Oracle, single-center,
    and no-DP distributed (``K=1``, ``epsilon=np.inf``) baselines are horizontal
    reference lines (averaged over reps).
    """
    if epsilons is None:
        epsilons = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]

    print("Experiment 7: GWAS metrics vs privacy budget (ε)")

    metric_keys = ("power", "fdr", "f1", "fpr")
    results = {}
    for eps in epsilons:
        results[eps] = {f"{m}_gm": [] for m in metric_keys}
        for m in metric_keys:
            results[eps][f"{m}_am"] = []

    oracle_rows: list[dict] = []
    single_rows: list[dict] = []
    nodp_gm_rows: list[dict] = []
    nodp_am_rows: list[dict] = []

    for rep in range(n_reps):
        data = simulate_gwas_data(n_individuals, n_snps, n_causal, seed=seed + rep)
        centers = split_data_across_centers(data, n_centers, seed=seed + rep)
        causal_idx = data["causal_idx"]

        oracle = centralized_gwas(centers, alpha=ALPHA_GWAS)
        single = single_center_gwas(centers, alpha=ALPHA_GWAS)
        oracle_rows.append({k: oracle[k] for k in metric_keys})
        single_rows.append({k: single[k] for k in metric_keys})

        nodp = run_dp_gwas_mle(
            centers,
            epsilon=np.inf,
            alpha=ALPHA_GWAS,
            K=1,
            topology="complete",
            seed=seed + rep * 100,
        )
        nodp_gm_rows.append(
            {k: evaluate_gwas(nodp.selected_gm, causal_idx, n_snps)[k] for k in metric_keys}
        )
        nodp_am_rows.append(
            {k: evaluate_gwas(nodp.selected_am, causal_idx, n_snps)[k] for k in metric_keys}
        )

        for eps in epsilons:
            res = run_dp_gwas_mle(
                centers,
                epsilon=eps,
                alpha=ALPHA_GWAS,
                K=15,
                topology="complete",
                seed=seed + rep * 100,
            )
            m_gm = evaluate_gwas(res.selected_gm, causal_idx, n_snps)
            m_am = evaluate_gwas(res.selected_am, causal_idx, n_snps)
            for k in metric_keys:
                results[eps][f"{k}_gm"].append(m_gm[k])
                results[eps][f"{k}_am"].append(m_am[k])

    eps_arr = np.array(epsilons)
    mean = lambda lst: float(np.mean(lst))
    se = lambda lst: float(np.std(lst) / np.sqrt(max(len(lst), 1)))

    agg = {}
    for k in metric_keys:
        agg[f"{k}_gm_mean"] = [mean(results[e][f"{k}_gm"]) for e in epsilons]
        agg[f"{k}_am_mean"] = [mean(results[e][f"{k}_am"]) for e in epsilons]
        agg[f"{k}_gm_se"] = [se(results[e][f"{k}_gm"]) for e in epsilons]
        agg[f"{k}_am_se"] = [se(results[e][f"{k}_am"]) for e in epsilons]

    oracle_mean = {k: mean([r[k] for r in oracle_rows]) for k in metric_keys}
    single_mean = {k: mean([r[k] for r in single_rows]) for k in metric_keys}
    nodp_gm_mean = {k: mean([r[k] for r in nodp_gm_rows]) for k in metric_keys}
    nodp_am_mean = {k: mean([r[k] for r in nodp_am_rows]) for k in metric_keys}

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), squeeze=False, sharey=True)
    panels = [
        (axes[0, 0], "power", STATISTICAL_POWER_LABEL),
        (axes[0, 1], "f1", "F1 score"),
    ]

    for ax, key, ylabel in panels:
        gm_m = agg[f"{key}_gm_mean"]
        am_m = agg[f"{key}_am_mean"]
        gm_s = agg[f"{key}_gm_se"]
        am_s = agg[f"{key}_am_se"]
        ax.errorbar(
            eps_arr,
            gm_m,
            yerr=gm_s,
            marker="o",
            label=DP_DISTRIBUTED_GM_LABEL,
            capsize=3,
            linewidth=1.5,
        )
        ax.errorbar(
            eps_arr,
            am_m,
            yerr=am_s,
            marker="s",
            label=DP_DISTRIBUTED_AM_LABEL,
            capsize=3,
            linewidth=1.5,
            linestyle="--",
        )
        ax.axhline(
            oracle_mean[key],
            color="gray",
            linestyle=":",
            linewidth=1.5,
            label=ORACLE_LABEL,
        )
        ax.axhline(
            single_mean[key],
            color="gray",
            linestyle="-.",
            linewidth=1.2,
            label=SINGLE_CENTER_LABEL,
        )
        ax.axhline(
            nodp_gm_mean[key],
            color="#ba7517",
            linestyle=(0, (3, 1, 1, 1)),
            linewidth=1.35,
            label=NO_DP_DISTRIBUTED_LABEL,
        )
        
        if key == "fdr":
            ax.axhline(
                ALPHA_GWAS,
                color="red",
                linestyle="--",
                linewidth=1,
                alpha=0.6,
                label="$\alpha$ (GWAS significance threshold)",
            )
        ax.set_xlabel("Privacy budget ($\epsilon$)")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log")
       
        ax.legend(fontsize=7, frameon=False, loc="best")
        

    fig.suptitle(
        f"GWAS evaluation metrics vs privacy budget  "
        f"($N={n_individuals}$, $M_{{SNP}}={n_snps}$, $M_{{causal}}={n_causal}$, $n={n_centers}$)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "exp_gwas_metrics_vs_epsilon.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  → saved exp_gwas_metrics_vs_epsilon.pdf")

    return dict(
        epsilons=epsilons,
        oracle_mean=oracle_mean,
        single_mean=single_mean,
        nodp_gm_mean=nodp_gm_mean,
        nodp_am_mean=nodp_am_mean,
        **agg,
    )


# ---------------------------------------------------------------------------
# Experiment 8 : Full GWAS evaluation metrics vs number of centers (fixed ε)
# ---------------------------------------------------------------------------

def experiment_gwas_metrics_vs_n_centers(
    n_individuals: int = 6000,
    n_snps: int = 500,
    n_causal: int = 20,
    n_reps: int = 3,
    epsilon: float = 1.0,
    n_centers_list: list[int] | None = None,
    seed: int = 8,
    output_dir: str = "../figures",
) -> dict:
    """
    Same metrics as ``experiment_gwas_metrics_vs_epsilon``, but sweep the number
    of centers with a fixed privacy budget.  One dataset is simulated per
    replicate; the oracle (pooled GWAS) is therefore constant across ``n`` for
    that replicate and shown as a horizontal reference.  Single-center, DP,
    and no-DP distributed (``K=1``, ``epsilon=np.inf``) curves vary with ``n``.
    """
    if n_centers_list is None:
        n_centers_list = [2, 3, 5, 8, 10, 15, 20]

    print("Experiment 8: GWAS metrics vs number of centers (fixed ε)")

    metric_keys = ("power", "fdr", "f1", "fpr")
    results: dict = {nc: {f"{m}_gm": [] for m in metric_keys} for nc in n_centers_list}
    for nc in n_centers_list:
        for m in metric_keys:
            results[nc][f"{m}_am"] = []

    single_results: dict = {nc: {k: [] for k in metric_keys} for nc in n_centers_list}
    nodp_results: dict = {
        nc: {**{f"{m}_gm": [] for m in metric_keys}, **{f"{m}_am": [] for m in metric_keys}}
        for nc in n_centers_list
    }
    oracle_rows: list[dict] = []

    for rep in range(n_reps):
        data = simulate_gwas_data(n_individuals, n_snps, n_causal, seed=seed + rep)
        causal_idx = data["causal_idx"]

        centers_ref = split_data_across_centers(
            data, n_centers_list[0], seed=seed + rep
        )
        oracle = centralized_gwas(centers_ref, alpha=ALPHA_GWAS)
        oracle_rows.append({k: oracle[k] for k in metric_keys})

        for nc in n_centers_list:
            centers = split_data_across_centers(data, nc, seed=seed + rep)
            single = single_center_gwas(centers, alpha=ALPHA_GWAS)
            for k in metric_keys:
                single_results[nc][k].append(single[k])
            res = run_dp_gwas_mle(
                centers,
                epsilon=epsilon,
                alpha=ALPHA_GWAS,
                K=15,
                topology="complete",
                seed=seed + rep * 100 + nc,
            )
            m_gm = evaluate_gwas(res.selected_gm, causal_idx, n_snps)
            m_am = evaluate_gwas(res.selected_am, causal_idx, n_snps)
            for k in metric_keys:
                results[nc][f"{k}_gm"].append(m_gm[k])
                results[nc][f"{k}_am"].append(m_am[k])

            nodp = run_dp_gwas_mle(
                centers,
                epsilon=np.inf,
                alpha=ALPHA_GWAS,
                K=1,
                topology="complete",
                seed=seed + rep * 1000 + nc,
            )
            m_n_gm = evaluate_gwas(nodp.selected_gm, causal_idx, n_snps)
            m_n_am = evaluate_gwas(nodp.selected_am, causal_idx, n_snps)
            for k in metric_keys:
                nodp_results[nc][f"{k}_gm"].append(m_n_gm[k])
                nodp_results[nc][f"{k}_am"].append(m_n_am[k])

    n_arr = np.array(n_centers_list, dtype=float)
    mean = lambda lst: float(np.mean(lst))
    se = lambda lst: float(np.std(lst) / np.sqrt(max(len(lst), 1)))

    agg = {}
    for k in metric_keys:
        agg[f"{k}_gm_mean"] = [mean(results[nc][f"{k}_gm"]) for nc in n_centers_list]
        agg[f"{k}_am_mean"] = [mean(results[nc][f"{k}_am"]) for nc in n_centers_list]
        agg[f"{k}_gm_se"] = [se(results[nc][f"{k}_gm"]) for nc in n_centers_list]
        agg[f"{k}_am_se"] = [se(results[nc][f"{k}_am"]) for nc in n_centers_list]
        agg[f"nodp_{k}_gm_mean"] = [
            mean(nodp_results[nc][f"{k}_gm"]) for nc in n_centers_list
        ]
        agg[f"nodp_{k}_am_mean"] = [
            mean(nodp_results[nc][f"{k}_am"]) for nc in n_centers_list
        ]
        agg[f"nodp_{k}_gm_se"] = [
            se(nodp_results[nc][f"{k}_gm"]) for nc in n_centers_list
        ]
        agg[f"nodp_{k}_am_se"] = [
            se(nodp_results[nc][f"{k}_am"]) for nc in n_centers_list
        ]

    oracle_mean = {k: mean([r[k] for r in oracle_rows]) for k in metric_keys}
    single_mean_per_n = {
        k: [mean(single_results[nc][k]) for nc in n_centers_list]
        for k in metric_keys
    }
    single_se_per_n = {
        k: [se(single_results[nc][k]) for nc in n_centers_list]
        for k in metric_keys
    }

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), squeeze=False, sharey=True)
    panels = [
        (axes[0, 0], "power", STATISTICAL_POWER_LABEL),
        (axes[0, 1], "f1", "F1 score"),
    ]

    for i, (ax, key, ylabel) in enumerate(panels):
        gm_m = agg[f"{key}_gm_mean"]
        am_m = agg[f"{key}_am_mean"]
        gm_s = agg[f"{key}_gm_se"]
        am_s = agg[f"{key}_am_se"]
        ax.errorbar(
            n_arr,
            gm_m,
            yerr=gm_s,
            marker="o",
            label=DP_DISTRIBUTED_GM_LABEL,
            capsize=3,
            linewidth=1.5,
        )
        ax.errorbar(
            n_arr,
            am_m,
            yerr=am_s,
            marker="s",
            label=DP_DISTRIBUTED_AM_LABEL,
            capsize=3,
            linewidth=1.5,
            linestyle="--",
        )
        ax.errorbar(
            n_arr,
            agg[f"nodp_{key}_gm_mean"],
            yerr=agg[f"nodp_{key}_gm_se"],
            marker="D",
            color="#ba7517",
            label=NO_DP_DISTRIBUTED_LABEL,
            capsize=3,
            linewidth=1.35,
            linestyle=(0, (3, 1, 1, 1)),
        )
        ax.axhline(
            oracle_mean[key],
            color="gray",
            linestyle=":",
            linewidth=1.5,
            label=ORACLE_LABEL,
        )
        ax.errorbar(
            n_arr,
            single_mean_per_n[key],
            yerr=single_se_per_n[key],
            marker="^",
            color="#888780",
            label=SINGLE_CENTER_LABEL,
            capsize=3,
            linewidth=1.2,
            linestyle="-.",
        )
        ax.set_xlabel("Number of centers ($n$)")
        ax.set_ylabel(ylabel)

        if i == 0:
            ax.legend(fontsize=7, frameon=False, loc="best")

        # set xticks to be int
        ax.set_xticks(n_arr.astype(int))
        ax.set_xticklabels(n_arr.astype(int))

    fig.suptitle(
        f"GWAS evaluation metrics vs number of centers  "
        f"($N={n_individuals}$, $M_{{SNP}}={n_snps}$, $M_{{causal}}={n_causal}$, "
        f"$\\epsilon={epsilon}$)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "exp_gwas_metrics_vs_n_centers.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  → saved exp_gwas_metrics_vs_n_centers.pdf")

    return dict(
        n_centers_list=n_centers_list,
        epsilon=epsilon,
        oracle_mean=oracle_mean,
        single_mean_per_n=single_mean_per_n,
        single_se_per_n=single_se_per_n,
        **agg,
    )


# ---------------------------------------------------------------------------
# Experiment 9 : Posterior P(H1) — geometric vs arithmetic mean aggregators
# ---------------------------------------------------------------------------

def experiment_posterior_gm_am(
    n_individuals: int = 2500,
    n_snps: int = 600,
    n_causal: int = 25,
    n_centers: int = 5,
    epsilon: float = 1.0,
    seed: int = 9,
    output_dir: str = "../figures",
) -> dict:
    """
    Run DP-GWAS once and visualize Stage-2 posteriors for GM vs AM.

    ``run_dp_gwas_mle`` stores log-odds in ``log_beliefs_gm`` / ``log_beliefs_am``;
    P(H1) = sigmoid(log-odds) matches ``exp(normalized log q_1)`` in the core routine.
    """
    print("Experiment 9: Posterior GM vs AM")

    data = simulate_gwas_data(n_individuals, n_snps, n_causal, seed=seed)
    centers = split_data_across_centers(data, n_centers, seed=seed)
    causal_idx = data["causal_idx"]
    causal_mask = np.zeros(n_snps, dtype=bool)
    causal_mask[causal_idx] = True

    res = run_dp_gwas_mle(
        centers,
        epsilon=epsilon,
        alpha=ALPHA_GWAS,
        K=15,
        topology="complete",
        seed=seed,
    )
    posterior_gm = expit(res.log_beliefs_gm)
    posterior_am = expit(res.log_beliefs_am)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    nc = ~causal_mask
    ax.scatter(
        posterior_gm[nc],
        posterior_am[nc],
        s=8,
        alpha=0.35,
        c="#888780",
        label="Non-causal",
        rasterized=True,
    )
    ax.scatter(
        posterior_gm[causal_mask],
        posterior_am[causal_mask],
        s=36,
        alpha=0.9,
        c="#d85a30",
        edgecolors="k",
        linewidths=0.4,
        label="Causal",
        zorder=5,
    )
    lim = (-0.02, 1.02)
    ax.plot(lim, lim, "k--", linewidth=1, alpha=0.45, label="$y=x$")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Posterior $P(H_1)$ — GM")
    ax.set_ylabel("Posterior $P(H_1)$ — AM")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_title("Geometric vs arithmetic consensus")

    ax = axes[1]
    bins = np.linspace(0, 1, 31)
    ax.hist(
        posterior_gm[nc],
        bins=bins,
        alpha=0.55,
        color="#1d9e75",
        label="GM (non-causal)",
        density=True,
    )
    ax.hist(
        posterior_am[nc],
        bins=bins,
        alpha=0.45,
        color="#534ab7",
        label="AM (non-causal)",
        density=True,
    )
    ax.set_xlabel("Posterior $P(H_1)$")
    ax.set_ylabel("Density (non-causal SNPs)")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Marginal distributions")

    fig.suptitle(
        f"Stage-2 posteriors: GM vs AM  "
        f"($N={n_individuals}$, $M_{{SNP}}={n_snps}$, $M_{{causal}}={n_causal}$ causal SNPs, "
        f"$n={n_centers}$ centers, $\\epsilon={epsilon}$)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "exp_posterior_gm_am.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  → saved exp_posterior_gm_am.pdf")

    return dict(
        epsilon=epsilon,
        posterior_gm=posterior_gm,
        posterior_am=posterior_am,
        causal_idx=causal_idx,
        pvalues=res.pvalues_gm,
    )
