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

OUT = Path("../figures")
OUT.mkdir(exist_ok=True)

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
) -> dict:
    """
    Sweep epsilon in [0.05, 5].  For each epsilon, run DP-GWAS (GM + AM)
    and record power, FDR, and F1.  Overlay the Centralized and
    single-center baselines as horizontal dashed lines.

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

    # Oracle & single-site benchmarks (noise-free, so computed once per rep)
    oracle_power, oracle_fdr = [], []
    single_power, single_fdr = [], []

    for rep in range(n_reps):
        data = simulate_gwas_data(n_individuals, n_snps, n_causal, seed=seed + rep)
        centers = split_data_across_centers(data, n_centers, seed=seed + rep)
        causal_idx = data["causal_idx"]

        oracle = centralized_gwas(centers, alpha=ALPHA_GWAS)
        oracle_power.append(oracle["power"])
        oracle_fdr.append(oracle["fdr"])

        single = single_center_gwas(centers, alpha=ALPHA_GWAS)
        single_power.append(single["power"])
        single_fdr.append(single["fdr"])

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

    oracle_p  = mean(oracle_power)
    single_p  = mean(single_power)

    # Find critical epsilon (first eps where power_gm > single_p)
    eps_crit = None
    for i, eps in enumerate(epsilons):
        if power_gm[i] > single_p:
            eps_crit = eps
            break

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.errorbar(eps_arr, power_gm, yerr=se_pgm, marker="o", label="DP-GWAS (GM)",
                capsize=3, linewidth=1.5)
    ax.errorbar(eps_arr, power_am, yerr=se_pam, marker="s", label="DP-GWAS (AM)",
                capsize=3, linewidth=1.5, linestyle="--")
    ax.axhline(oracle_p, color="gray", linestyle=":", linewidth=1.5, label="Centralized")
    ax.axhline(single_p, color="gray", linestyle="-.", linewidth=1.2, label="Single center")
    if eps_crit is not None:
        ax.axvline(eps_crit, color="red", linestyle="--", alpha=0.4, linewidth=1,
                   label=f"ε* ≈ {eps_crit}")
    ax.set_xlabel("Privacy budget ($\epsilon$)")
    ax.set_ylabel("Statistical power")
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Power vs privacy budget")

    ax = axes[1]
    ax.plot(eps_arr, fdr_gm, marker="o", label="DP-GWAS (GM)", linewidth=1.5)
    ax.plot(eps_arr, fdr_am, marker="s", label="DP-GWAS (AM)", linewidth=1.5, linestyle="--")
    ax.axhline(oracle["fdr"], color="gray", linestyle=":", linewidth=1.5, label="Oracle")
    ax.axhline(ALPHA_GWAS, color="red", linestyle="--", linewidth=1, alpha=0.6, label="$\\alpha_{GWAS}$ (significance threshold)")
    ax.set_xlabel("Privacy budget ($\epsilon$)")
    ax.set_ylabel("False discovery rate")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("FDR vs privacy budget")

    fig.suptitle(
        f"Privacy-utility tradeoff  "
        f"($N={n_individuals}$, $M_{{SNP}}={n_snps}$, $M_{{causal}}={n_causal}$ causal SNPs, $n={n_centers}$ centers)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT / "exp1_privacy_utility.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved exp1_privacy_utility.pdf  (ε* ≈ {eps_crit})")

    return dict(
        epsilons=epsilons, power_gm=power_gm, power_am=power_am,
        fdr_gm=fdr_gm, fdr_am=fdr_am,
        oracle_power=oracle_p, single_power=single_p, eps_crit=eps_crit,
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
) -> dict:
    """
    Compare three methods across varying total cohort sizes.
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

            rows.append(dict(
                n=n,
                oracle_power=oracle["power"], oracle_fdr=oracle["fdr"],
                single_power=single["power"], single_fdr=single["fdr"],
                dp_power=m_gm["power"], dp_fdr=m_gm["fdr"],
            ))

    import pandas as pd
    df = pd.DataFrame(rows)
    agg = df.groupby("n").agg(["mean", "sem"]).reset_index()

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    N = np.array(n_individuals_list, dtype=int)
    col_o, col_d, col_s = ("oracle_power", "dp_power", "single_power")
    ax.errorbar(N, agg[(col_o, "mean")], yerr=agg[(col_o, "sem")], marker="o", label="Centralized", linewidth=1.5)
    ax.errorbar(N, agg[(col_d, "mean")], yerr=agg[(col_d, "sem")], marker="s", label=f"DP-distributed ($\epsilon={epsilon}$)", linewidth=1.5)
    ax.errorbar(N, agg[(col_s, "mean")], yerr=agg[(col_s, "sem")], marker="^", label="Single center", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Total cohort size ($N$)")
    ax.set_ylabel("Statistical power")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, frameon=False)

    fig.suptitle(
        f"Statistical Power vs $N$ ($\epsilon={epsilon}$, $n={n_centers}$, $M_{{SNP}}={n_snps}$, $M_{{causal}}={n_causal}$)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT / "exp2_three_way.pdf", bbox_inches="tight")
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

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # Convergence traces
    ax = axes[0]
    for top, tr in traces.items():
        sg_val = spectral_gap(make_adjacency(n_centers, topology=top, seed=seed))
        ax.plot(tr, label=f"{top} (spectral gap={sg_val:.2f})", linewidth=1.5)
    ax.set_xlabel("Iteration ($t$)")
    ax.set_ylabel("Belief std")
    ax.set_yscale("log")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Convergence speed by network topology")

    # Power bar chart
    ax = axes[1]
    x = np.arange(len(topologies))
    powers = [np.mean(results[t]["power"]) for t in topologies]
    sems   = [np.std(results[t]["power"]) / np.sqrt(n_reps) for t in topologies]
    slem_vals = [np.mean(results[t]["slem"]) for t in topologies]
    bars = ax.bar(x, powers, yerr=sems, capsize=4,
                  color=["#1d9e75", "#534ab7", "#d85a30", "#888780", "#999999", "#534ab7"])
    ax.set_xticks(x)
    ax.set_xticklabels(topologies)
    ax.set_ylabel("Power (GM)")
    ax.set_ylim(0, 1.1)
    ax.set_title("Power by topology")
    for bar, sl in zip(bars, slem_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"$1 - |\\lambda_2| = {sl:.2f}$", ha="center", fontsize=8, rotation=45)

    # Spectral gap vs convergence iteration
    ax = axes[2]
    sgs = [spectral_gap(make_adjacency(n_centers, t, seed=seed)) for t in topologies]
    conv_iters = [np.mean(results[t]["converged_at"]) for t in topologies]
    ax.scatter(sgs, conv_iters, s=80,
               color=["#1d9e75", "#534ab7", "#d85a30", "#888780", "#999999", "#534ab7"], zorder=3)
    for top, sg_v, ci in zip(topologies, sgs, conv_iters):
        ax.annotate(top, (sg_v, ci), textcoords="offset points",
                    xytext=(6, 3), fontsize=8)
    ax.set_xlabel("Spectral gap ($1 - |\\lambda_2|$)")
    ax.set_ylabel("Iterations to convergence")
    ax.set_title("Spectral gap vs convergence")
    ax.set_xscale("log")

    fig.suptitle(
        f"Network topology  ($N={n_individuals}$, $\epsilon={epsilon}$, $n={n_centers}$)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT / "exp3_topology.pdf", bbox_inches="tight")
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
    ax.plot(h2_vals, h2_power_oracle, "o-", label="Oracle", linewidth=1.5)
    ax.plot(h2_vals, h2_power_dp,     "s-", label=f"DP-GWAS ($\epsilon={epsilon}$)", linewidth=1.5)
    ax.plot(h2_vals, h2_power_single,  "^--", label="Single center", linewidth=1.5)
    ax.set_xlabel("Heritability (h²)")
    ax.set_ylabel("Statistical power")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Power vs heritability")

    ax = axes[1]
    x = np.arange(len(maf_labels))
    w = 0.35
    ax.bar(x - w/2, power_by_maf_oracle, w, label="Oracle")
    ax.bar(x + w/2, power_by_maf_dp,     w, label=f"DP-GWAS ($\epsilon={epsilon}$)")
    ax.set_xticks(x)
    ax.set_xticklabels(maf_labels, rotation=15, fontsize=8)
    ax.set_ylabel("Statistical power")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Power by minor allele frequency bin")

    fig.suptitle(
        f"Heritability & MAF strata  ($N={n_individuals}$, $n={n_centers}$)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT / "exp4_stratified.pdf", bbox_inches="tight")
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
            T_used = dp_res.converged_at if dp_res.converged_at < K * 500 else 200
            res[nc]["power"].append(m["power"])
            res[nc]["fdr"].append(m["fdr"])
            res[nc]["comm_complexity"].append(K * T_used * nc)
            res[nc]["time"].append(time_taken)
    powers = [np.mean(res[nc]["power"]) for nc in n_centers_list]
    fdrs   = [np.mean(res[nc]["fdr"])   for nc in n_centers_list]
    comms  = [np.mean(res[nc]["comm_complexity"]) for nc in n_centers_list]
    se_p   = [np.std(res[nc]["power"]) / np.sqrt(n_reps) for nc in n_centers_list]
    times  = [np.mean(res[nc]["time"]) for nc in n_centers_list]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.errorbar(n_centers_list, powers, yerr=se_p, marker="o", linewidth=1.5, capsize=3)
    ax.set_xlabel("Number of centers ($n$)")
    ax.set_ylabel("Statistical power (GM)")
    ax.set_ylim(0, 1.05)
    # Fit log curve to show polylog growth
    log_n = np.log(n_centers_list)
    try:
        coeffs = np.polyfit(log_n, powers, 1)
        ax.plot(n_centers_list,
                np.polyval(coeffs, log_n),
                "r--", alpha=0.5, linewidth=1, label=f"Log fit: $Power(n) = {coeffs[0]:.1g} \\log(n) + {coeffs[1]:.1g}$")
        ax.legend(fontsize=8, frameon=False)
    except Exception:
        pass

    ax.set_xticks(n_centers_list)
    ax.set_xticklabels(n_centers_list)

    ax = axes[1]
    ax.plot(n_centers_list, comms, marker="o", linewidth=1.5, color="#534ab7")
    ax.set_xlabel("Number of centers ($n$)")
    ax.set_ylabel("Total communication cost ($K \cdot T \cdot n$)")
    # Overlay theoretical O(n log n) curve
    n_arr = np.array(n_centers_list, dtype=float)
    scale = comms[0] / (n_arr[0] * np.log(n_arr[0]))
    ax.plot(n_arr, scale * n_arr * np.log(n_arr),
            "r--", alpha=0.5, linewidth=1, label="Theoretical upper bound: $O(polylog(n))$")
    ax.legend(fontsize=8, frameon=False)

    ax.set_xticks(n_centers_list)
    ax.set_xticklabels(n_centers_list)

    ax = axes[2]
    ax.plot(n_centers_list, times, marker="o", linewidth=1.5, color="#534ab7")
    ax.set_xlabel("Number of centers ($n$)")
    ax.set_ylabel("Time taken (seconds)")
    ax.legend(fontsize=8, frameon=False)
    ax.set_xticks(n_centers_list)
    ax.set_xticklabels(n_centers_list)

    fig.suptitle(f"Scaling with $n$ centers ($N={n_individuals_total}$, $M_{{SNP}}={n_snps}$, $M_{{causal}}={n_causal}$, $\epsilon={epsilon}$)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "exp5_scaling.pdf", bbox_inches="tight")
    print("  → saved exp5_scaling.pdf")
    return dict(n_centers_list=n_centers_list, powers=powers, fdrs=fdrs, comms=comms)

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
    fig.savefig(OUT / "exp6_rizk_comparison.pdf", bbox_inches="tight")
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
) -> dict:
    """
    Plot all metrics returned by ``evaluate_gwas`` (power, FDR, F1, FPR)
    as a function of ε, for both GM and AM DP-GWAS.  Oracle and single-center
    baselines are horizontal reference lines (noise-free, averaged over reps).
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

    for rep in range(n_reps):
        data = simulate_gwas_data(n_individuals, n_snps, n_causal, seed=seed + rep)
        centers = split_data_across_centers(data, n_centers, seed=seed + rep)
        causal_idx = data["causal_idx"]

        oracle = centralized_gwas(centers, alpha=ALPHA_GWAS)
        single = single_center_gwas(centers, alpha=ALPHA_GWAS)
        oracle_rows.append({k: oracle[k] for k in metric_keys})
        single_rows.append({k: single[k] for k in metric_keys})

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

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), squeeze=False, sharey=True)
    panels = [
        (axes[0, 0], "power", "Statistical power (Recall of causal SNPs)"),
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
            label="DP-GWAS (GM)",
            capsize=3,
            linewidth=1.5,
        )
        ax.errorbar(
            eps_arr,
            am_m,
            yerr=am_s,
            marker="s",
            label="DP-GWAS (AM)",
            capsize=3,
            linewidth=1.5,
            linestyle="--",
        )
        ax.axhline(
            oracle_mean[key],
            color="gray",
            linestyle=":",
            linewidth=1.5,
            label="Centralized",
        )
        ax.axhline(
            single_mean[key],
            color="gray",
            linestyle="-.",
            linewidth=1.2,
            label="Single center",
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
    fig.savefig(OUT / "exp_gwas_metrics_vs_epsilon.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  → saved exp_gwas_metrics_vs_epsilon.pdf")

    return dict(
        epsilons=epsilons,
        oracle_mean=oracle_mean,
        single_mean=single_mean,
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
) -> dict:
    """
    Same metrics as ``experiment_gwas_metrics_vs_epsilon``, but sweep the number
    of centers with a fixed privacy budget.  One dataset is simulated per
    replicate; the oracle (pooled GWAS) is therefore constant across ``n`` for
    that replicate and shown as a horizontal reference.  Single-center and DP
    curves vary with ``n``.
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

    n_arr = np.array(n_centers_list, dtype=float)
    mean = lambda lst: float(np.mean(lst))
    se = lambda lst: float(np.std(lst) / np.sqrt(max(len(lst), 1)))

    agg = {}
    for k in metric_keys:
        agg[f"{k}_gm_mean"] = [mean(results[nc][f"{k}_gm"]) for nc in n_centers_list]
        agg[f"{k}_am_mean"] = [mean(results[nc][f"{k}_am"]) for nc in n_centers_list]
        agg[f"{k}_gm_se"] = [se(results[nc][f"{k}_gm"]) for nc in n_centers_list]
        agg[f"{k}_am_se"] = [se(results[nc][f"{k}_am"]) for nc in n_centers_list]

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
        (axes[0, 0], "power", "Statistical power (Recall of causal SNPs)"),
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
            label="DP-GWAS (GM)",
            capsize=3,
            linewidth=1.5,
        )
        ax.errorbar(
            n_arr,
            am_m,
            yerr=am_s,
            marker="s",
            label="DP-GWAS (AM)",
            capsize=3,
            linewidth=1.5,
            linestyle="--",
        )
        ax.axhline(
            oracle_mean[key],
            color="gray",
            linestyle=":",
            linewidth=1.5,
            label="Centralized",
        )
        ax.errorbar(
            n_arr,
            single_mean_per_n[key],
            yerr=single_se_per_n[key],
            marker="^",
            color="#888780",
            label="Single center",
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
    fig.savefig(OUT / "exp_gwas_metrics_vs_n_centers.pdf", bbox_inches="tight")
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
    fig.savefig(OUT / "exp_posterior_gm_am.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  → saved exp_posterior_gm_am.pdf")

    return dict(
        epsilon=epsilon,
        posterior_gm=posterior_gm,
        posterior_am=posterior_am,
        causal_idx=causal_idx,
        pvalues=res.pvalues_gm,
    )
