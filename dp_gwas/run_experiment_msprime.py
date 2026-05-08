import os, sys, argparse, warnings
from itertools import product
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import seaborn as sns
from scipy.special import expit
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(__file__))
from dp_gwas_core import *

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "axes.prop_cycle": plt.cycler(
        color=["#1d9e75", "#534ab7", "#d85a30", "#888780", "#ba7517"]
    ),
})

ALPHA_LIST = [1e-7, 2e-7, 5e-7, 1e-6, 2e-6, 5e-6]
ALPHA_GWAS = 2e-7

MAF_RARE      = 0.01
R2_THRESH     = 0.1    
HERITABILITY  = 0.05   

N_SNPS        = 600
N_COMMON      = 300       
N_RARE        = 700      
N_CAUSAL      = 25
N_CENTERS     = 5
N_INDIVIDUALS = 25_000

K_ROUNDS      = 30

FIGSIZE       = 3

ROW_COLORS = {
    "all":    ("#1d9e75", "#534ab7"),
    "common": ("#1d9e75", "#534ab7"),
    "rare":   ("#1d9e75", "#534ab7"),
}

NO_DP_DISTRIBUTED_LABEL = "No DP Distributed"
DP_DISTRIBUTED_GM_LABEL = "DP Distributed (GM)"
DP_DISTRIBUTED_AM_LABEL = "DP Distributed (AM)"
SINGLE_CENTER_LABEL = "Single Center"
ORACLE_LABEL = "Centralized"
STATISTICAL_POWER_LABEL = "Statistical power"
FDR_LABEL = "False discovery rate"


def _locus_metrics(selected, causal_idx, G_std, n_snps):
    """SNP-level selection → locus-level power / FDR / F1 / FPR."""
    return evaluate_gwas_locus(selected, causal_idx, G_std, n_snps, R2_THRESH)


def _eval_stratum(selected, causal_idx, G_std, n_snps, rare_mask, common_mask):
    """
    Return locus-level metrics for all / rare / common strata.
    Always passes full G_std so that causal_idx (full-array coords) remain valid.
    """
    rare_idx   = np.where(rare_mask)[0]
    common_idx = np.where(common_mask)[0]
    causal_rare   = np.intersect1d(causal_idx, rare_idx)
    causal_common = np.intersect1d(causal_idx, common_idx)

    m_all    = _locus_metrics(selected,              causal_idx,    G_std, n_snps)
    m_rare   = _locus_metrics(selected & rare_mask,  causal_rare,   G_std, n_snps)
    m_common = _locus_metrics(selected & common_mask, causal_common, G_std, n_snps)
    return m_all, m_rare, m_common


def _simulate_rep(
    rep: int,
    n_individuals: int = N_INDIVIDUALS,
    n_snps: int = N_SNPS,
    n_causal: int = N_CAUSAL,
    n_centers: int = N_CENTERS,
    n_common: int = N_COMMON,
    n_rare: int = N_RARE,
    h2: float = HERITABILITY,
    base_seed: int = 100,
    weights: np.ndarray = None,
):
    seed = base_seed + rep
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        data = simulate_msprime_gwas_data(
            n_individuals=n_individuals,
            n_snps=n_snps,
            n_causal=n_causal,
            seed=seed,
            h2=h2,
            n_common_target=n_common,
            n_rare_target=n_rare,
        )
    centers = split_data_across_centers(data, n_centers, seed=seed, weights=weights)
    return data, centers

def exp10_power_fdr_vs_epsilon(n_reps: int = 5, output_dir: str = "figures"):
    print("\n=== Experiment: Metrics vs $\\epsilon$ ===")

    epsilons = [0.1, 0.5, 1.0, 10.0]
    strata   = ["all", "rare", "common"]
    metrics  = ["power", "fdr", "f1"]

    def _store():
        return {
            eps: {m: {s: {k: [] for k in metrics} for s in strata}
                  for m in ["gm", "am", "nodp_gm"]}
            for eps in epsilons
        }

    res = _store()
    oracle_vals = {s: {k: [] for k in metrics} for s in strata}
    single_vals = {s: {k: [] for k in metrics} for s in strata}

    for rep in tqdm(range(n_reps), desc="Replicate"):
        data, centers = _simulate_rep(rep)
        causal_idx  = data["causal_idx"]
        rare_mask   = data["rare_mask"]
        common_mask = data["common_mask"]
        G_std       = data["G_std"]
        n_snps      = G_std.shape[1]


        # Baselines
        oracle = centralized_gwas(centers, alpha=ALPHA_GWAS)
        single = single_center_gwas(centers, alpha=ALPHA_GWAS)
        for name, sel in [("oracle", oracle["selected"]), ("single", single["selected"])]:
            m_all, m_rare, m_common = _eval_stratum(sel, causal_idx, G_std, n_snps, rare_mask, common_mask)
            tgt = oracle_vals if name == "oracle" else single_vals
            for s, mv in zip(strata, [m_all, m_rare, m_common]):
                for k in metrics:
                    tgt[s][k].append(mv[k])

        # No-DP distributed
        nodp = run_dp_gwas_mle(centers, epsilon=np.inf, alpha=ALPHA_GWAS,
                               K=1, topology="complete", seed=rep * 1000)
        nd_all_gm, nd_rare_gm, nd_common_gm = _eval_stratum(
            nodp.selected_gm, causal_idx, G_std, n_snps, rare_mask, common_mask)

        # Epsilon sweep
        for eps in epsilons:
            dp = run_dp_gwas_mle(centers, epsilon=eps, alpha=ALPHA_GWAS,
                                 K=K_ROUNDS, topology="complete", seed=rep * 1000)
            m_gm_all, m_gm_rare, m_gm_common = _eval_stratum(
                dp.selected_gm, causal_idx, G_std, n_snps, rare_mask, common_mask)
            m_am_all, m_am_rare, m_am_common = _eval_stratum(
                dp.selected_am, causal_idx, G_std, n_snps, rare_mask, common_mask)
            for s, mgm, mam, mnd in zip(
                strata,
                [m_gm_all, m_gm_rare, m_gm_common],
                [m_am_all, m_am_rare, m_am_common],
                [nd_all_gm, nd_rare_gm, nd_common_gm],
            ):
                for k in metrics:
                    res[eps]["gm"][s][k].append(mgm[k])
                    res[eps]["am"][s][k].append(mam[k])
                    res[eps]["nodp_gm"][s][k].append(mnd[k])

    m_fn  = lambda lst: np.mean(lst) if lst else 0.0
    se_fn = lambda lst: np.std(lst) / np.sqrt(max(len(lst), 1)) if lst else 0.0
    eps_arr = np.array(epsilons)

    row_labels = {
        "all":    "All variants",
        "rare":   f"Rare (MAF < {MAF_RARE})",
        "common": f"Common (MAF ≥ {MAF_RARE})",
    }

    nrows, ncols = len(strata), len(metrics)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * FIGSIZE, nrows * FIGSIZE),
        squeeze=False,
    )

    for r, stratum in enumerate(strata):
        c_gm, c_am = ROW_COLORS[stratum]
        for c, metric in enumerate(metrics):
            ax = axes[r][c]

            gm_m  = [m_fn(res[e]["gm"][stratum][metric])     for e in epsilons]
            am_m  = [m_fn(res[e]["am"][stratum][metric])     for e in epsilons]
            gm_se = [se_fn(res[e]["gm"][stratum][metric])    for e in epsilons]
            am_se = [se_fn(res[e]["am"][stratum][metric])    for e in epsilons]
            nd_m  = [m_fn(res[e]["nodp_gm"][stratum][metric]) for e in epsilons]
            ora_m = m_fn(oracle_vals[stratum][metric])
            sng_m = m_fn(single_vals[stratum][metric])

            ax.errorbar(eps_arr, gm_m, yerr=gm_se, marker="o",
                        color=c_gm, capsize=3, linewidth=1.6, label=DP_DISTRIBUTED_GM_LABEL)
            # ax.errorbar(eps_arr, am_m, yerr=am_se, marker="s",
            #             color=c_am, capsize=3, linewidth=1.6, linestyle="--", label=DP_DISTRIBUTED_AM_LABEL)
            ax.plot(eps_arr, nd_m, marker="D", color="#ba7517",
                    linewidth=1.4, linestyle=(0, (3, 1, 1, 1)), label=NO_DP_DISTRIBUTED_LABEL)
            ax.axhline(ora_m, color="gray", linestyle=":", linewidth=1.5, label=ORACLE_LABEL)
            ax.axhline(sng_m, color="gray", linestyle="-.", linewidth=1.2, label=SINGLE_CENTER_LABEL)

            ax.set_ylim(-0.02, 1.05)

            ax.set_xscale("log")
            ax.set_xlabel("Privacy budget (ε)")
            ylabel = {"power": "Statistical power", "fdr": "FDR", "f1": "F1 score"}[metric]
            ax.set_ylabel(ylabel)
            ax.set_title(f"{row_labels[stratum]}")
            if r == 0 and c == len(metrics) - 1:
                ax.legend(fontsize=7, frameon=False, loc="best")

    # fig.suptitle(
    #     f"DP-GWAS (COSI, FDR, r²>{R2_THRESH}) — metrics vs ε\n"
    #     f"$N={N_INDIVIDUALS}$, {N_COMMON} common + {N_RARE} rare SNPs, "
    #     f"{N_CAUSAL} causal SNPs, {N_CENTERS} centers",
    #     fontsize=11, y=1.01,
    # )
    fig.tight_layout()
    out = os.path.join(output_dir, "exp10_power_fdr_vs_epsilon.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {out}")
    return res, oracle_vals, single_vals


# ── Plot 2: metrics vs n_centers ─────────────────────────────────────────────

def exp11_power_fdr_vs_n_centers(
    n_reps: int = 5,
    epsilon: float = 1.0,
    output_dir: str = "figures",
):
    print("\n=== Experiment 11: Metrics vs n_centers ===")

    n_centers_list = [4, 6, 8, 10]
    strata  = ["all", "rare", "common"]
    metrics = ["power", "fdr", "f1"]

    def _store():
        return {
            nc: {m: {s: {k: [] for k in metrics} for s in strata}
                 for m in ["gm", "am", "nodp_gm", "single"]}
            for nc in n_centers_list
        }

    res = _store()
    oracle_vals = {s: {k: [] for k in metrics} for s in strata}

    for rep in tqdm(range(n_reps), desc="Replicate"):
        data, _ = _simulate_rep(rep)
        causal_idx  = data["causal_idx"]
        rare_mask   = data["rare_mask"]
        common_mask = data["common_mask"]
        G_std       = data["G_std"]
        n_snps      = G_std.shape[1]


        centers_max = split_data_across_centers(data, max(n_centers_list), seed=rep)
        oracle = centralized_gwas(centers_max, alpha=ALPHA_GWAS)
        m_ora_all, m_ora_rare, m_ora_common = _eval_stratum(
            oracle["selected"], causal_idx, G_std, n_snps, rare_mask, common_mask)
        for s, mv in zip(strata, [m_ora_all, m_ora_rare, m_ora_common]):
            for k in metrics:
                oracle_vals[s][k].append(mv[k])

        for nc in n_centers_list:
            centers = split_data_across_centers(data, nc, seed=rep)

            nodp = run_dp_gwas_mle(centers, epsilon=np.inf, alpha=ALPHA_GWAS, K=1, topology="complete", seed=rep * 1000)
            single = single_center_gwas(centers, alpha=ALPHA_GWAS)
            dp = run_dp_gwas_mle(centers, epsilon=epsilon, alpha=ALPHA_GWAS, K=K_ROUNDS, topology="complete", seed=rep * 1000)

            m_nd_all, m_nd_rare, m_nd_common = _eval_stratum(
                nodp.selected_gm, causal_idx, G_std, n_snps, rare_mask, common_mask)
            m_sc_all, m_sc_rare, m_sc_common = _eval_stratum(
                single["selected"], causal_idx, G_std, n_snps, rare_mask, common_mask)
            m_gm_all, m_gm_rare, m_gm_common = _eval_stratum(
                dp.selected_gm, causal_idx, G_std, n_snps, rare_mask, common_mask)
            m_am_all, m_am_rare, m_am_common = _eval_stratum(
                dp.selected_am, causal_idx, G_std, n_snps, rare_mask, common_mask)

            for s, mgm, mam, mnd, msc in zip(
                strata,
                [m_gm_all, m_gm_rare, m_gm_common],
                [m_am_all, m_am_rare, m_am_common],
                [m_nd_all, m_nd_rare, m_nd_common],
                [m_sc_all, m_sc_rare, m_sc_common],
            ):
                for k in metrics:
                    res[nc]["gm"][s][k].append(mgm[k])
                    res[nc]["am"][s][k].append(mam[k])
                    res[nc]["nodp_gm"][s][k].append(mnd[k])
                    res[nc]["single"][s][k].append(msc[k])

    m_fn  = lambda lst: np.mean(lst) if lst else 0.0
    se_fn = lambda lst: np.std(lst) / np.sqrt(max(len(lst), 1)) if lst else 0.0
    nc_arr = np.array(n_centers_list)

    row_labels = {
        "all":    "All variants",
        "rare":   f"Rare (MAF < {MAF_RARE})",
        "common": f"Common (MAF ≥ {MAF_RARE})",
    }
    nrows, ncols = len(strata), len(metrics)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * FIGSIZE, nrows * FIGSIZE),
        squeeze=False,
    )

    for r, stratum in enumerate(strata):
        c_gm, c_am = ROW_COLORS[stratum]
        for c, metric in enumerate(metrics):
            ax = axes[r][c]

            gm_m  = [m_fn(res[nc]["gm"][stratum][metric])      for nc in n_centers_list]
            am_m  = [m_fn(res[nc]["am"][stratum][metric])      for nc in n_centers_list]
            gm_se = [se_fn(res[nc]["gm"][stratum][metric])     for nc in n_centers_list]
            am_se = [se_fn(res[nc]["am"][stratum][metric])     for nc in n_centers_list]
            nd_m  = [m_fn(res[nc]["nodp_gm"][stratum][metric]) for nc in n_centers_list]
            sc_m  = [m_fn(res[nc]["single"][stratum][metric])  for nc in n_centers_list]
            ora_m = m_fn(oracle_vals[stratum][metric])

            ax.errorbar(nc_arr, gm_m, yerr=gm_se, marker="o",
                        color=c_gm, capsize=3, linewidth=1.6, label=DP_DISTRIBUTED_GM_LABEL)
            # ax.errorbar(nc_arr, am_m, yerr=am_se, marker="s",
                        # color=c_am, capsize=3, linewidth=1.6, linestyle="--", label=DP_DISTRIBUTED_AM_LABEL)
            ax.plot(nc_arr, nd_m, marker="D", color="#ba7517",
                    linewidth=1.4, linestyle=(0, (3, 1, 1, 1)), label=NO_DP_DISTRIBUTED_LABEL)
            ax.plot(nc_arr, sc_m, marker="^", color="#888780",
                    linewidth=1.2, linestyle="-.", label=SINGLE_CENTER_LABEL)
            ax.axhline(ora_m, color="gray", linestyle=":", linewidth=1.5, label=ORACLE_LABEL)

            ax.set_ylim(-0.02, 1.05)

            ax.set_xticks(nc_arr)
            ax.set_xticklabels(nc_arr)
            ax.set_xlabel("Number of centers")
            ylabel = {"power": "Statistical power", "fdr": "FDR", "f1": "F1 score"}[metric]
            ax.set_ylabel(ylabel)
            ax.set_title(f"{row_labels[stratum]}")
            if r == 0 and c == len(metrics) - 1:
                ax.legend(fontsize=7, frameon=False, loc="best")

    # fig.suptitle(
    #     f"DP-GWAS (COSI, FDR, r²>{R2_THRESH}) — metrics vs n_centers (ε={epsilon})\n"
    #     f"$N={N_INDIVIDUALS}$, {N_COMMON} common + {N_RARE} rare SNPs, {N_CAUSAL} causal SNPs",
    #     fontsize=11, y=1.01,
    # )
    fig.tight_layout()
    out = os.path.join(output_dir, "exp11_power_fdr_vs_n_centers.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {out}")
    return res

def exp12_manhattan_plot(output_dir: str = "figures"):
    print("\n=== Experiment 12: Manhattan plot ===")

    data, centers = _simulate_rep(0)
    causal_idx  = data["causal_idx"]
    rare_mask   = data["rare_mask"]
    n_snps      = data["G_std"].shape[1]

    oracle = centralized_gwas(centers, alpha=ALPHA_GWAS)
    dp = run_dp_gwas_mle(centers, epsilon=1.0, alpha=ALPHA_GWAS,
                         K=K_ROUNDS, topology="complete", seed=0)

    pval_oracle = oracle["pvalues"]
    pval_dp     = dp.pvalues_gm

    causal_mask = np.zeros(n_snps, dtype=bool)
    causal_mask[causal_idx] = True

    x = np.arange(n_snps)
    y_oracle = -np.log10(np.clip(pval_oracle, 1e-300, 1))
    y_dp     = -np.log10(np.clip(pval_dp,     1e-300, 1))
    threshold = -np.log10(ALPHA_GWAS)

    nrows, ncols = 2, 1
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * FIGSIZE, nrows * FIGSIZE), sharex=True,
    )
    for ax, y, title in zip(
        axes,
        [y_oracle, y_dp],
        [CENTRALIZED_ORACLE_LABEL, DP_DISTRIBUTED_GM_LABEL],
    ):
        ax.scatter(x[~causal_mask & ~rare_mask], y[~causal_mask & ~rare_mask],
                   s=6, alpha=0.5, color="#7fa7c9", rasterized=True, label=COMMON_NON_CAUSAL_LABEL)
        ax.scatter(x[~causal_mask & rare_mask],  y[~causal_mask & rare_mask],
                   s=4, alpha=0.3, color="#c0c0c0", rasterized=True, label=RARE_NON_CAUSAL_LABEL)
        ax.scatter(x[causal_mask & ~rare_mask],  y[causal_mask & ~rare_mask],
                   s=40, color="#d85a30", edgecolors="k", linewidths=0.5,
                   zorder=5, label=COMMON_CAUSAL_LABEL)
        ax.scatter(x[causal_mask & rare_mask],   y[causal_mask & rare_mask],
                   s=40, color="#a050c8", edgecolors="k", linewidths=0.5,
                   zorder=5, label=RARE_CAUSAL_LABEL)
        ax.axhline(threshold, color="red", linestyle="--", linewidth=1,
                   alpha=0.8, label=ALPHA_LABEL)
        ax.set_ylabel("$-\\log_{10}(p)$")
        ax.set_title(title)
        ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper right")

    axes[-1].set_xlabel(f"SNP index (200 kb region, {N_COMMON} common + {N_RARE} rare variants)")
    # fig.suptitle(
    #     f"Manhattan plot\n"
    #     f"{N_INDIVIDUALS} individuals, {N_COMMON} common + {N_RARE} rare SNPs, {N_CAUSAL} causal SNPs, {N_CENTERS} centers",
    #     fontsize=12,
    # )
    fig.tight_layout()
    out = os.path.join(output_dir, "exp12_manhattan.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {out}")

def exp2_three_way(
    n_individuals_list: list[int] | None = None,
    epsilon: float = 1.0,
    n_reps: int = 3,
    base_seed: int = 2,
    output_dir: str = "figures",
    snp_chunk_size: int = N_SNPS,
):
    if n_individuals_list is None:
        n_individuals_list = [15000, 20000, 25000]

    print("\n=== Experiment 2: Metrics vs cohort size ===")
    rows = []
    for n in n_individuals_list:
        for rep in range(n_reps):
            data, centers = _simulate_rep(
                rep, n_individuals=n, base_seed=base_seed + rep * 7 + n,
            )
            causal_idx, G_std = data["causal_idx"], data["G_std"]
            n_snps = G_std.shape[1]

            oracle = centralized_gwas(centers, alpha=ALPHA_GWAS)
            single = single_center_gwas(centers, alpha=ALPHA_GWAS)
            dp_res = run_dp_gwas_mle(
                centers, epsilon=epsilon, alpha=ALPHA_GWAS, K=K_ROUNDS,
                topology="complete", seed=base_seed + rep * 100,
                snp_chunk_size=snp_chunk_size,
            )
            nodp_res = run_dp_gwas_mle(
                centers, epsilon=np.inf, alpha=ALPHA_GWAS, K=1,
                topology="complete", seed=base_seed + rep * 100 + 1,
                snp_chunk_size=snp_chunk_size,
            )
            m_o = _locus_metrics(oracle["selected"], causal_idx, G_std, n_snps)
            m_s = _locus_metrics(single["selected"], causal_idx, G_std, n_snps)
            m_d = _locus_metrics(dp_res.selected_gm, causal_idx, G_std, n_snps)
            m_n = _locus_metrics(nodp_res.selected_gm, causal_idx, G_std, n_snps)
            rows.append(dict(
                n=n,
                oracle_power=m_o["power"], oracle_fdr=m_o["fdr"],
                single_power=m_s["power"], single_fdr=m_s["fdr"],
                dp_power=m_d["power"], dp_fdr=m_d["fdr"],
                nodp_power=m_n["power"], nodp_fdr=m_n["fdr"],
            ))

    df = pd.DataFrame(rows)
    agg = df.groupby("n").agg(["mean", "sem"]).reset_index()
    N = np.array(n_individuals_list, dtype=int)

    nrows, ncols = 1, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * FIGSIZE, nrows * FIGSIZE))
    axes[0].errorbar(N, agg[("oracle_power", "mean")], yerr=agg[("oracle_power", "sem")], marker="o", label=ORACLE_LABEL, linewidth=1.5)
    axes[0].errorbar(N, agg[("dp_power", "mean")], yerr=agg[("dp_power", "sem")], marker="s", label=DP_DISTRIBUTED_GM_LABEL, linewidth=1.5)
    axes[0].errorbar(N, agg[("single_power", "mean")], yerr=agg[("single_power", "sem")], marker="^", label=SINGLE_CENTER_LABEL, linewidth=1.5, linestyle="--")
    axes[0].errorbar(N, agg[("nodp_power", "mean")], yerr=agg[("nodp_power", "sem")], marker="D", color="#ba7517", label=NO_DP_DISTRIBUTED_LABEL, linewidth=1.5, linestyle=(0, (3, 1, 1, 1)))
    axes[0].set_xlabel("Total cohort size ($N$)")
    axes[0].set_ylabel(STATISTICAL_POWER_LABEL)
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=8, frameon=False)

    axes[1].errorbar(N, agg[("oracle_fdr", "mean")], yerr=agg[("oracle_fdr", "sem")], marker="o", label=ORACLE_LABEL, linewidth=1.5)
    axes[1].errorbar(N, agg[("dp_fdr", "mean")], yerr=agg[("dp_fdr", "sem")], marker="s", label=DP_DISTRIBUTED_GM_LABEL, linewidth=1.5)
    axes[1].errorbar(N, agg[("single_fdr", "mean")], yerr=agg[("single_fdr", "sem")], marker="^", label=SINGLE_CENTER_LABEL, linewidth=1.5, linestyle="--")
    axes[1].errorbar(N, agg[("nodp_fdr", "mean")], yerr=agg[("nodp_fdr", "sem")], marker="D", color="#ba7517", label=NO_DP_DISTRIBUTED_LABEL, linewidth=1.5, linestyle=(0, (3, 1, 1, 1)))
    axes[1].set_xlabel("Total cohort size ($N$)")
    axes[1].set_ylabel(FDR_LABEL)
    axes[1].set_ylim(-0.02, 0.65)
    axes[1].legend(fontsize=8, frameon=False)

    # fig.suptitle(
    #     f"Power and FDR vs cohort size ($\\epsilon={epsilon}$, $n={N_CENTERS}$ centers, ${N_SNPS}$ SNPs)",
    #     fontsize=11,
    # )
    fig.tight_layout()
    out = os.path.join(output_dir, "exp2_three_way.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {out}")
    return dict(df=df)


def exp3_topology(
    n_centers: int = 6,
    epsilon: float = 1.0,
    T: int = 150,
    n_reps: int = 3,
    base_seed: int = 3,
    output_dir: str = "figures",
    snp_chunk_size: int = N_SNPS,
):
    topologies = ["complete", "ring", "random", "star", "scale-free", "small-world"]
    print("\n=== Experiment 3: Metrics vs topology ===")
    results = {t: {"power": [], "fdr": [], "converged_at": [], "slem": []} for t in topologies}

    for top in topologies:
        A = make_adjacency(n_centers, topology=top, seed=base_seed)
        sg = spectral_gap(A)
        for rep in range(n_reps):
            data, centers = _simulate_rep(rep, n_centers=n_centers, base_seed=base_seed + rep * 13)
            causal_idx, G_std = data["causal_idx"], data["G_std"]
            n_snps = G_std.shape[1]
            dp_res = run_dp_gwas_mle(
                centers, epsilon=epsilon, alpha=ALPHA_GWAS, K=K_ROUNDS, T=T,
                topology=top, seed=base_seed + rep * 100,
                track_convergence=True, convergence_tol=1e-3,
                snp_chunk_size=snp_chunk_size,
            )
            m = _locus_metrics(dp_res.selected_gm, causal_idx, G_std, n_snps)
            results[top]["power"].append(m["power"])
            results[top]["fdr"].append(m["fdr"])
            results[top]["converged_at"].append(dp_res.converged_at)
            results[top]["slem"].append(1 - sg)

    traces = {}
    for top in topologies:
        data, centers = _simulate_rep(0, n_centers=n_centers, base_seed=base_seed + 99)
        dp_res = run_dp_gwas_mle(
            centers, epsilon=epsilon, alpha=ALPHA_GWAS, K=5, T=T,
            topology=top, seed=base_seed,
            track_convergence=True, snp_chunk_size=snp_chunk_size,
        )
        traces[top] = dp_res.belief_trace

    nrows, ncols = 1, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * FIGSIZE, nrows * FIGSIZE))
    x = np.arange(len(topologies))
    powers = [np.mean(results[t]["power"]) for t in topologies]
    sems = [np.std(results[t]["power"]) / np.sqrt(n_reps) for t in topologies]
    axes[0].bar(x, powers, yerr=sems, capsize=4, color=["#1d9e75", "#534ab7", "#d85a30", "#888780", "#999999", "#ba7517"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(topologies, rotation=90, fontsize=8)
    axes[0].set_ylabel(STATISTICAL_POWER_LABEL + " (GM)")
    axes[0].set_ylim(0, 1.1)
    axes[0].set_title("Power by topology")

    fdrs = [np.mean(results[t]["fdr"]) for t in topologies]
    fdr_sems = [np.std(results[t]["fdr"]) / np.sqrt(n_reps) for t in topologies]
    axes[1].bar(x, fdrs, yerr=fdr_sems, capsize=4, color=["#1d9e75", "#534ab7", "#d85a30", "#888780", "#999999", "#ba7517"])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(topologies, rotation=90, fontsize=8)
    axes[1].set_ylabel(FDR_LABEL + " (GM)")
    axes[1].set_ylim(0, 0.65)
    axes[1].set_title("FDR by topology")

    records = []
    for t in topologies:
        A = make_adjacency(n_centers, topology=t, seed=base_seed)
        sg = spectral_gap(A)
        records.append(dict(Spectral_gap=sg, Convergence_iterations=np.mean(results[t]["converged_at"]), Topology=t))
    df_sg = pd.DataFrame(records)
    sns.scatterplot(
        data=df_sg, x="Spectral_gap", y="Convergence_iterations",
        style="Topology", ax=axes[2], s=80,
    )
    axes[2].set_xlabel("Spectral gap ($1 - |\\lambda_2|$)")
    axes[2].set_ylabel("Iterations to convergence")
    axes[2].set_title("Spectral gap vs convergence")
    axes[2].set_xscale("log")

    # fig.suptitle(f"Topology (${N_INDIVIDUALS}$ individuals, $\\epsilon={epsilon}$, $n={n_centers}$ centers)", fontsize=11)
    fig.tight_layout()
    out = os.path.join(output_dir, "exp3_topology.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {out}")
    return dict(results=results, traces=traces)


def exp4_stratified(
    n_reps: int = 3,
    epsilon: float = 1.0,
    base_seed: int = 4,
    output_dir: str = "figures",
    snp_chunk_size: int = 100,
):
    print("\n=== Experiment 4: Metrics vs heritability and MAF strata ===")
    h2_vals = [0.01, 0.05, 0.1]
    h2_power_oracle, h2_power_dp, h2_power_single = [], [], []
    h2_fdr_oracle, h2_fdr_dp, h2_fdr_single = [], [], []

    for h2 in h2_vals:
        pwr_o, pwr_d, pwr_s = [], [], []
        fdr_o, fdr_d, fdr_s = [], [], []
        for rep in range(n_reps):
            data, centers = _simulate_rep(rep, h2=h2, base_seed=base_seed + rep * 11)
            causal_idx, G_std = data["causal_idx"], data["G_std"]
            n_snps = G_std.shape[1]
            oracle = centralized_gwas(centers, alpha=ALPHA_GWAS)
            single = single_center_gwas(centers, alpha=ALPHA_GWAS)
            dp_res = run_dp_gwas_mle(
                centers, epsilon=epsilon, alpha=ALPHA_GWAS, K=K_ROUNDS,
                topology="complete", seed=base_seed + rep * 100,
                snp_chunk_size=snp_chunk_size,
            )
            mo = _locus_metrics(oracle["selected"], causal_idx, G_std, n_snps)
            md = _locus_metrics(dp_res.selected_gm, causal_idx, G_std, n_snps)
            ms = _locus_metrics(single["selected"], causal_idx, G_std, n_snps)
            pwr_o.append(mo["power"])
            pwr_d.append(md["power"])
            pwr_s.append(ms["power"])
            fdr_o.append(mo["fdr"])
            fdr_d.append(md["fdr"])
            fdr_s.append(ms["fdr"])
        h2_power_oracle.append(np.mean(pwr_o))
        h2_power_dp.append(np.mean(pwr_d))
        h2_power_single.append(np.mean(pwr_s))
        h2_fdr_oracle.append(np.mean(fdr_o))
        h2_fdr_dp.append(np.mean(fdr_d))
        h2_fdr_single.append(np.mean(fdr_s))

    data, centers = _simulate_rep(0, h2=0.3, base_seed=base_seed + 999)
    causal_idx, G_std, mafs = data["causal_idx"], data["G_std"], data["mafs"]
    n_snps = G_std.shape[1]
    oracle = centralized_gwas(centers, alpha=ALPHA_GWAS)
    dp_res = run_dp_gwas_mle(
        centers, epsilon=epsilon, alpha=ALPHA_GWAS, K=K_ROUNDS,
        topology="complete", seed=base_seed, snp_chunk_size=snp_chunk_size,
    )

    maf_bins = [0.0001, 0.01, 0.05, 0.1, 0.2, 0.35, 0.5]
    maf_labels = ["<1%", "1–5%", "5–10%", "10–20%", "20–35%", ">35%"]
    power_by_maf_oracle, power_by_maf_dp = [], []
    fdr_by_maf_oracle, fdr_by_maf_dp = [], []

    for lo, hi in zip(maf_bins[:-1], maf_bins[1:]):
        mask = (mafs >= lo) & (mafs < hi)
        causal_in_bin = causal_idx[np.isin(causal_idx, np.where(mask)[0])]
        if len(causal_in_bin) == 0:
            power_by_maf_oracle.append(np.nan)
            power_by_maf_dp.append(np.nan)
            fdr_by_maf_oracle.append(np.nan)
            fdr_by_maf_dp.append(np.nan)
            continue
        ev_o = _locus_metrics(oracle["selected"] & mask, causal_in_bin, G_std, n_snps)
        ev_d = _locus_metrics(dp_res.selected_gm & mask, causal_in_bin, G_std, n_snps)
        power_by_maf_oracle.append(ev_o["power"])
        power_by_maf_dp.append(ev_d["power"])
        fdr_by_maf_oracle.append(ev_o["fdr"])
        fdr_by_maf_dp.append(ev_d["fdr"])

    nrows, ncols = 2, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * FIGSIZE, nrows * FIGSIZE))
    axes[0, 0].plot(h2_vals, h2_power_oracle, "o-", label=ORACLE_LABEL, linewidth=1.5)
    axes[0, 0].plot(h2_vals, h2_power_dp, "s-", label=DP_DISTRIBUTED_GM_LABEL, linewidth=1.5)
    axes[0, 0].plot(h2_vals, h2_power_single, "^--", label=SINGLE_CENTER_LABEL, linewidth=1.5)
    axes[0, 0].set_xlabel("Heritability ($h^2$)")
    axes[0, 0].set_ylabel(STATISTICAL_POWER_LABEL)
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].legend(fontsize=8, frameon=False)

    xb = np.arange(len(maf_labels))
    w = 0.35
    axes[0, 1].bar(xb - w / 2, power_by_maf_oracle, w, label=ORACLE_LABEL)
    axes[0, 1].bar(xb + w / 2, power_by_maf_dp, w, label=DP_DISTRIBUTED_GM_LABEL)
    axes[0, 1].set_xticks(xb)
    axes[0, 1].set_xticklabels(maf_labels, rotation=15, fontsize=8)
    axes[0, 1].set_ylabel(STATISTICAL_POWER_LABEL)
    axes[0, 1].set_ylim(0, 1.1)
    axes[0, 1].legend(fontsize=8, frameon=False)

    axes[1, 0].plot(h2_vals, h2_fdr_oracle, "o-", label=ORACLE_LABEL, linewidth=1.5)
    axes[1, 0].plot(h2_vals, h2_fdr_dp, "s-", label=DP_DISTRIBUTED_GM_LABEL, linewidth=1.5)
    axes[1, 0].plot(h2_vals, h2_fdr_single, "^--", label=SINGLE_CENTER_LABEL, linewidth=1.5)
    axes[1, 0].set_xlabel("Heritability ($h^2$)")
    axes[1, 0].set_ylabel(FDR_LABEL)
    axes[1, 0].set_ylim(-0.02, 0.65)
    axes[1, 0].legend(fontsize=8, frameon=False)

    axes[1, 1].bar(xb - w / 2, fdr_by_maf_oracle, w, label=ORACLE_LABEL)
    axes[1, 1].bar(xb + w / 2, fdr_by_maf_dp, w, label=DP_DISTRIBUTED_GM_LABEL)
    axes[1, 1].set_xticks(xb)
    axes[1, 1].set_xticklabels(maf_labels, rotation=15, fontsize=8)
    axes[1, 1].set_ylabel(FDR_LABEL)
    axes[1, 1].set_ylim(0, 0.65)
    axes[1, 1].legend(fontsize=8, frameon=False)

    # fig.suptitle(f"Heritability and MAF strata ($n={N_CENTERS}$)", fontsize=11)
    fig.tight_layout()
    out = os.path.join(output_dir, "exp4_stratified.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {out}")
    return dict(h2_vals=h2_vals, h2_power_dp=h2_power_dp)


def exp5_scaling(
    epsilon: float = 1.0,
    n_reps: int = 2,
    base_seed: int = 5,
    output_dir: str = "figures",
    snp_chunk_size: int = 100,
):
    n_centers_list = [4, 6, 8, 10]
    topologies = ["complete", "ring", "star", "random", "small-world", "scale-free"]
    palette = ["#1d9e75", "#534ab7", "#d85a30", "#888780", "#999999", "#ba7517"]
    print("\n=== Exp5: Scaling n_centers × topology ===")

    res = {
        top: {nc: {"power": [], "fdr": [], "comm_complexity": [], "time": []} for nc in n_centers_list}
        for top in topologies
    }
    topo_seed_off = {t: i * 97 for i, t in enumerate(topologies)}

    for top in topologies:
        for nc in n_centers_list:
            for rep in range(n_reps):
                data, centers = _simulate_rep(
                    rep, n_centers=nc,
                    base_seed=base_seed + rep * 7 + topo_seed_off[top],
                )
                causal_idx, G_std = data["causal_idx"], data["G_std"]
                n_snps = G_std.shape[1]
                K = K_ROUNDS
                t0 = time.time()
                dp_res = run_dp_gwas_mle(
                    centers, epsilon=epsilon, alpha=ALPHA_GWAS, K=K,
                    topology=top, seed=base_seed + rep * 100 + topo_seed_off[top],
                    track_convergence=False, snp_chunk_size=snp_chunk_size,
                )
                dt = time.time() - t0
                m = _locus_metrics(dp_res.selected_gm, causal_idx, G_std, n_snps)
                T_used = dp_res.converged_at
                res[top][nc]["power"].append(m["power"])
                res[top][nc]["fdr"].append(m["fdr"])
                res[top][nc]["comm_complexity"].append(K * T_used * nc)
                res[top][nc]["time"].append(dt)

    by_topology = {}
    for top in topologies:
        by_topology[top] = dict(
            fdrs=[np.mean(res[top][nc]["fdr"]) for nc in n_centers_list],
            comms=[np.mean(res[top][nc]["comm_complexity"]) for nc in n_centers_list],
            times=[np.mean(res[top][nc]["time"]) for nc in n_centers_list],
            powers=[np.mean(res[top][nc]["power"]) for nc in n_centers_list],
        )

    nrows, ncols = 2, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * FIGSIZE, nrows * FIGSIZE))
    n_arr = np.array(n_centers_list, dtype=float)
    comms_complete = by_topology["complete"]["comms"]
    scale = comms_complete[0] / (n_arr[0] * np.log(max(n_arr[0], 2)))
    axes[0, 0].plot(n_arr, scale * n_arr * np.log(np.maximum(n_arr, 2)), "r--", alpha=0.5, linewidth=1, label="Ref.: $O(n\\log n)$")
    for top, col in zip(topologies, palette):
        axes[0, 0].plot(n_centers_list, by_topology[top]["comms"], marker="o", linewidth=1.5, color=col, label=top)
    axes[0, 0].set_xlabel("Number of centers ($n$)")
    axes[0, 0].set_ylabel("Belief exchanges ($K \\cdot T \\cdot n$)")
    axes[0, 0].legend(fontsize=7, frameon=False, loc="best")
    axes[0, 0].set_xticks(n_centers_list)
    axes[0, 0].set_xticklabels(n_centers_list)

    for top, col in zip(topologies, palette):
        axes[0, 1].plot(n_centers_list, by_topology[top]["times"], marker="o", linewidth=1.5, color=col, label=top)
    axes[0, 1].set_xlabel("Number of centers ($n$)")
    axes[0, 1].set_ylabel("Wall time (s)")
    axes[0, 1].legend(fontsize=7, frameon=False, loc="best")

    for top, col in zip(topologies, palette):
        axes[1, 0].plot(n_centers_list, by_topology[top]["powers"], marker="o", linewidth=1.5, color=col, label=top)
    axes[1, 0].set_xlabel("Number of centers ($n$)")
    axes[1, 0].set_ylabel(STATISTICAL_POWER_LABEL + " (GM)")
    axes[1, 0].set_ylim(0, 1.05)

    for top, col in zip(topologies, palette):
        axes[1, 1].plot(n_centers_list, by_topology[top]["fdrs"], marker="o", linewidth=1.5, color=col, label=top)
    axes[1, 1].set_xlabel("Number of centers ($n$)")
    axes[1, 1].set_ylabel(FDR_LABEL + " (GM)")
    axes[1, 1].set_ylim(-0.02, 0.65)

    # fig.suptitle(f"Scaling (${N_INDIVIDUALS}$ individuals, $\\epsilon={epsilon}$)", fontsize=11)
    fig.tight_layout()
    out = os.path.join(output_dir, "exp5_scaling.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {out}")
    return dict(by_topology=by_topology, raw=res)


def exp6_rizk_comparison(
    n_centers_list: list[int] | None = None,
    epsilons: list[float] | None = None,
    n_reps: int = 3,
    base_seed: int = 6,
    output_dir: str = "figures",
    snp_chunk_size: int = 100,
    tune: bool = False,
):
    if n_centers_list is None:
        n_centers_list = [5, 10, 20]
    if epsilons is None:
        epsilons = [0.2, 0.5, 1.0, 1.5]

    print("\n=== Exp6: TVD to oracle vs Rizk baseline ===")

    def tvd(sel_a, sel_b):
        return float(np.mean(sel_a.astype(float) != sel_b.astype(float)))

    nrows, ncols = 1, len(n_centers_list)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * FIGSIZE, nrows * FIGSIZE), sharey=True,
    )
    if len(n_centers_list) == 1:
        axes = [axes]

    all_results = {}
    for nc, ax in zip(n_centers_list, axes):
        tvd_gm, tvd_am, tvd_rizk = [], [], []
        for eps in epsilons:
            tvds_gm, tvds_am, tvds_r = [], [], []
            for rep in range(n_reps):
                data, centers = _simulate_rep(
                    rep, n_centers=nc, base_seed=base_seed + rep * 13 + nc,
                )
                oracle = centralized_gwas(centers, alpha=ALPHA_GWAS)
                dp_res = run_dp_gwas_mle(
                    centers, epsilon=eps, alpha=ALPHA_GWAS, K=K_ROUNDS,
                    topology="complete", seed=base_seed + rep * 100,
                    snp_chunk_size=snp_chunk_size,
                )
                rizk = run_rizk_baseline(
                    centers, epsilon=eps, alpha=ALPHA_GWAS, T=150, seed=base_seed + rep * 100,
                )
                tvds_gm.append(tvd(dp_res.selected_gm, oracle["selected"]))
                tvds_am.append(tvd(dp_res.selected_am, oracle["selected"]))
                tvds_r.append(tvd(rizk["selected"], oracle["selected"]))
            tvd_gm.append(np.mean(tvds_gm))
            tvd_am.append(np.mean(tvds_am))
            tvd_rizk.append(np.mean(tvds_r))

        ax.plot(epsilons, tvd_gm, "o-", label="DP-GWAS (GM)", linewidth=1.5)
        ax.plot(epsilons, tvd_am, "s--", label="DP-GWAS (AM)", linewidth=1.5)
        ax.plot(epsilons, tvd_rizk, "^:", label="Rizk et al.", linewidth=1.5)
        ax.set_xlabel("ε")
        ax.set_title(f"$n={nc}$ centers")
        ax.set_ylim(0, None)
        if ax == axes[0]:
            ax.set_ylabel("TVD to oracle selection")
        ax.legend(fontsize=7, frameon=False)
        all_results[nc] = dict(epsilons=epsilons, tvd_gm=tvd_gm, tvd_am=tvd_am, tvd_rizk=tvd_rizk)

    # fig.suptitle("Comparison with Rizk et al. 2023", fontsize=11)
    fig.tight_layout()
    out = os.path.join(output_dir, "exp6_rizk_comparison.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {out}")
    return all_results


def exp1_privacy_utility(
    n_reps: int = 3,
    epsilons: list[float] | None = None,
    base_seed: int = 7,
    output_dir: str = "figures",
    snp_chunk_size: int = N_SNPS,
    weights: np.ndarray = None,
    topology: str = "complete",
    filename: str = 'exp1_gwas_metrics_vs_epsilon.pdf',
    title: str = None,
    **kwargs
):
    if epsilons is None:
        epsilons = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0]

    print("\n=== Exp1: Metrics vs $\\epsilon$ ===")
    metric_keys = ("power", "fdr", "f1", "fpr")
    results = {eps: {f"{m}_gm": [] for m in metric_keys} for eps in epsilons}
    for eps in epsilons:
        for m in metric_keys:
            results[eps][f"{m}_am"] = []

    oracle_rows, single_rows, nodp_gm_rows, nodp_am_rows = [], [], [], []

    if weights is not None:
        n_centers = weights.shape[0]
    else:
        n_centers = N_CENTERS

    for rep in tqdm(range(n_reps), desc="exp1"):
        data, centers = _simulate_rep(rep, base_seed=base_seed + 2000, weights=weights, n_centers=n_centers)
        causal_idx, G_std = data["causal_idx"], data["G_std"]
        n_snps = G_std.shape[1]

        oracle = centralized_gwas(centers, alpha=ALPHA_GWAS)
        single = single_center_gwas(centers, alpha=ALPHA_GWAS)
        oracle_rows.append({k: _locus_metrics(oracle["selected"], causal_idx, G_std, n_snps)[k] for k in metric_keys})
        single_rows.append({k: _locus_metrics(single["selected"], causal_idx, G_std, n_snps)[k] for k in metric_keys})

        nodp = run_dp_gwas_mle(
            centers, epsilon=np.inf, alpha=ALPHA_GWAS, K=1,
            topology=topology, seed=base_seed + rep * 100, snp_chunk_size=snp_chunk_size,
            **kwargs
        )
        nodp_gm_rows.append({k: _locus_metrics(nodp.selected_gm, causal_idx, G_std, n_snps)[k] for k in metric_keys})
        nodp_am_rows.append({k: _locus_metrics(nodp.selected_am, causal_idx, G_std, n_snps)[k] for k in metric_keys})

        for eps in epsilons:
            res = run_dp_gwas_mle(
                centers, epsilon=eps, alpha=ALPHA_GWAS, K=K_ROUNDS,
                topology=topology, seed=base_seed + rep * 100, snp_chunk_size=snp_chunk_size,
                **kwargs
            )
            m_gm = _locus_metrics(res.selected_gm, causal_idx, G_std, n_snps)
            m_am = _locus_metrics(res.selected_am, causal_idx, G_std, n_snps)
            for k in metric_keys:
                results[eps][f"{k}_gm"].append(m_gm[k])
                results[eps][f"{k}_am"].append(m_am[k])

    mean = lambda lst: float(np.mean(lst))
    se = lambda lst: float(np.std(lst) / np.sqrt(max(len(lst), 1)))
    eps_arr = np.array(epsilons)
    agg = {}
    for k in metric_keys:
        agg[f"{k}_gm_mean"] = [mean(results[e][f"{k}_gm"]) for e in epsilons]
        agg[f"{k}_am_mean"] = [mean(results[e][f"{k}_am"]) for e in epsilons]
        agg[f"{k}_gm_se"] = [se(results[e][f"{k}_gm"]) for e in epsilons]
        agg[f"{k}_am_se"] = [se(results[e][f"{k}_am"]) for e in epsilons]

    oracle_mean = {k: mean([r[k] for r in oracle_rows]) for k in metric_keys}
    single_mean = {k: mean([r[k] for r in single_rows]) for k in metric_keys}
    nodp_gm_mean = {k: mean([r[k] for r in nodp_gm_rows]) for k in metric_keys}

    nrows, ncols = 2, 2
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * FIGSIZE, nrows * FIGSIZE), squeeze=False,
    )
    panels = [
        (axes[0, 0], "power", STATISTICAL_POWER_LABEL),
        (axes[0, 1], "fdr", FDR_LABEL),
        (axes[1, 0], "f1", "F1 score"),
        (axes[1, 1], "fpr", "FPR"),
    ]
    for ax, key, ylabel in panels:
        ax.errorbar(eps_arr, agg[f"{key}_gm_mean"], yerr=agg[f"{key}_gm_se"], marker="o", label=DP_DISTRIBUTED_GM_LABEL, capsize=3, linewidth=1.5)
        # ax.errorbar(eps_arr, agg[f"{key}_am_mean"], yerr=agg[f"{key}_am_se"], marker="s", label=DP_DISTRIBUTED_AM_LABEL, capsize=3, linewidth=1.5, linestyle="--")
        ax.axhline(oracle_mean[key], color="gray", linestyle=":", linewidth=1.5, label=ORACLE_LABEL)
        ax.axhline(single_mean[key], color="gray", linestyle="-.", linewidth=1.2, label=SINGLE_CENTER_LABEL)
        ax.axhline(nodp_gm_mean[key], color="#ba7517", linestyle=(0, (3, 1, 1, 1)), linewidth=1.35, label=NO_DP_DISTRIBUTED_LABEL)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel("Privacy budget ($\\epsilon$)")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log")
        ax.legend(fontsize=7, frameon=False, loc="best")

    if title is None:
        fig.suptitle(
            f"${N_INDIVIDUALS}$ individuals, ${N_SNPS}$ SNPs, ${n_centers}$ centers, {topology.capitalize()} network topology",
            fontsize=11,
        )
    else:
        fig.suptitle(title, fontsize=11)
    
    fig.tight_layout()
    out = os.path.join(output_dir, filename)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {out}")
    return dict(epsilons=epsilons, oracle_mean=oracle_mean, **agg)


def exp8_gwas_metrics_vs_n_centers(
    n_reps: int = 3,
    epsilon: float = 1.0,
    n_centers_list: list[int] | None = None,
    base_seed: int = 8,
    output_dir: str = "figures",
    snp_chunk_size: int = 100,
):
    if n_centers_list is None:
        n_centers_list = [2, 3, 5, 8, 10, 15, 20]

    print("\n=== Exp8: Metrics vs n_centers ===")
    metric_keys = ("power", "fdr", "f1", "fpr")
    results = {nc: {f"{m}_gm": [] for m in metric_keys} for nc in n_centers_list}
    for nc in n_centers_list:
        for m in metric_keys:
            results[nc][f"{m}_am"] = []
    single_results = {nc: {k: [] for k in metric_keys} for nc in n_centers_list}
    nodp_results = {
        nc: {**{f"{m}_gm": [] for m in metric_keys}, **{f"{m}_am": [] for m in metric_keys}}
        for nc in n_centers_list
    }
    oracle_rows = []

    for rep in tqdm(range(n_reps), desc="exp8"):
        data, _ = _simulate_rep(rep, base_seed=base_seed + 3000)
        causal_idx, G_std = data["causal_idx"], data["G_std"]
        n_snps = G_std.shape[1]
        centers_max = split_data_across_centers(data, max(n_centers_list), seed=base_seed + rep)
        oracle = centralized_gwas(centers_max, alpha=ALPHA_GWAS)
        oracle_rows.append({k: _locus_metrics(oracle["selected"], causal_idx, G_std, n_snps)[k] for k in metric_keys})

        for nc in n_centers_list:
            centers = split_data_across_centers(data, nc, seed=base_seed + rep)
            single = single_center_gwas(centers, alpha=ALPHA_GWAS)
            for k in metric_keys:
                single_results[nc][k].append(_locus_metrics(single["selected"], causal_idx, G_std, n_snps)[k])
            res = run_dp_gwas_mle(
                centers, epsilon=epsilon, alpha=ALPHA_GWAS, K=K_ROUNDS,
                topology="complete", seed=base_seed + rep * 100 + nc, snp_chunk_size=snp_chunk_size,
            )
            m_gm = _locus_metrics(res.selected_gm, causal_idx, G_std, n_snps)
            m_am = _locus_metrics(res.selected_am, causal_idx, G_std, n_snps)
            for k in metric_keys:
                results[nc][f"{k}_gm"].append(m_gm[k])
                results[nc][f"{k}_am"].append(m_am[k])
            nodp = run_dp_gwas_mle(
                centers, epsilon=np.inf, alpha=ALPHA_GWAS, K=1,
                topology="complete", seed=base_seed + rep * 1000 + nc, snp_chunk_size=snp_chunk_size,
            )
            m_n_gm = _locus_metrics(nodp.selected_gm, causal_idx, G_std, n_snps)
            m_n_am = _locus_metrics(nodp.selected_am, causal_idx, G_std, n_snps)
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
        agg[f"nodp_{k}_gm_mean"] = [mean(nodp_results[nc][f"{k}_gm"]) for nc in n_centers_list]
        agg[f"nodp_{k}_gm_se"] = [se(nodp_results[nc][f"{k}_gm"]) for nc in n_centers_list]

    oracle_mean = {k: mean([r[k] for r in oracle_rows]) for k in metric_keys}
    single_mean_per_n = {k: [mean(single_results[nc][k]) for nc in n_centers_list] for k in metric_keys}
    single_se_per_n = {k: [se(single_results[nc][k]) for nc in n_centers_list] for k in metric_keys}

    nrows, ncols = 1, 1
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * FIGSIZE, nrows * FIGSIZE), squeeze=False,
    )
    panels = [
        (axes[0, 0], "power", STATISTICAL_POWER_LABEL),
        (axes[0, 0], "fdr", FDR_LABEL),
    ]
    for ax, key, ylabel in panels:
        ax.errorbar(n_arr, agg[f"{key}_gm_mean"], yerr=agg[f"{key}_gm_se"], marker="o", label=DP_DISTRIBUTED_GM_LABEL, capsize=3, linewidth=1.5)
        # ax.errorbar(n_arr, agg[f"{key}_am_mean"], yerr=agg[f"{key}_am_se"], marker="s", label=DP_DISTRIBUTED_AM_LABEL, capsize=3, linewidth=1.5, linestyle="--")
        ax.errorbar(n_arr, agg[f"nodp_{key}_gm_mean"], yerr=agg[f"nodp_{key}_gm_se"], marker="D", color="#ba7517", label=NO_DP_DISTRIBUTED_LABEL, capsize=3, linewidth=1.35, linestyle=(0, (3, 1, 1, 1)))
        ax.axhline(oracle_mean[key], color="gray", linestyle=":", linewidth=1.5, label=ORACLE_LABEL)
        ax.errorbar(n_arr, single_mean_per_n[key], yerr=single_se_per_n[key], marker="^", color="#888780", label=SINGLE_CENTER_LABEL, capsize=3, linewidth=1.2, linestyle="-.")
        ax.set_xlabel("Number of centers ($n$)")
        ax.set_ylabel(ylabel)
        if key == "fdr":
            ax.axhline(ALPHA_GWAS, color="red", linestyle="--", linewidth=1, alpha=0.6, label="$\\alpha$ threshold")
            ax.set_ylim(-0.02, 0.65)
        ax.legend(fontsize=6, frameon=False, loc="best")
        ax.set_xticks(n_arr.astype(int))
        ax.set_xticklabels(n_arr.astype(int))

    fig.suptitle(
        f"Power and FDR vs $n$ (${N_INDIVIDUALS}$ individuals, ${N_SNPS}$ SNPs, $\\epsilon={epsilon}$)",
        fontsize=11,
    )
    fig.tight_layout()
    out = os.path.join(output_dir, "exp8_gwas_metrics_vs_n_centers.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {out}")
    return dict(n_centers_list=n_centers_list, oracle_mean=oracle_mean, **agg)


def exp9_posterior_gm_am(
    epsilon: float = 1.0,
    base_seed: int = 9,
    output_dir: str = "figures",
    snp_chunk_size: int = 100,
    n_reps: int = 3,
):
    print("\n=== Posterior GM vs AM ===")
    data, centers = _simulate_rep(0, base_seed=base_seed + 4000)
    causal_idx, n_snps = data["causal_idx"], data["G_std"].shape[1]
    causal_mask = np.zeros(n_snps, dtype=bool)
    causal_mask[causal_idx] = True

    res = run_dp_gwas_mle(
        centers, epsilon=epsilon, alpha=ALPHA_GWAS, K=K_ROUNDS,
        topology="complete", seed=base_seed, snp_chunk_size=snp_chunk_size,
    )
    posterior_gm = expit(res.log_beliefs_gm)
    posterior_am = expit(res.log_beliefs_am)

    nrows, ncols = 1, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * FIGSIZE, nrows * FIGSIZE))
    nc = ~causal_mask
    axes[0].scatter(posterior_gm[nc], posterior_am[nc], s=8, alpha=0.35, c="#888780", label="Non-causal", rasterized=True)
    axes[0].scatter(
        posterior_gm[causal_mask], posterior_am[causal_mask], s=36, alpha=0.9, c="#d85a30",
        edgecolors="k", linewidths=0.4, label="Causal", zorder=5,
    )
    lim = (-0.02, 1.02)
    axes[0].plot(lim, lim, "k--", linewidth=1, alpha=0.45, label="$y=x$")
    axes[0].set_xlim(lim)
    axes[0].set_ylim(lim)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel("Posterior $P(H_1)$ — GM")
    axes[0].set_ylabel("Posterior $P(H_1)$ — AM")
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    axes[0].set_title("Geometric vs arithmetic consensus")

    bins = np.linspace(0, 1, 31)
    axes[1].hist(posterior_gm[nc], bins=bins, alpha=0.55, color="#1d9e75", label="GM (non-causal)", density=True)
    axes[1].hist(posterior_am[nc], bins=bins, alpha=0.45, color="#534ab7", label="AM (non-causal)", density=True)
    axes[1].set_xlabel("Posterior $P(H_1)$")
    axes[1].set_ylabel("Density (non-causal SNPs)")
    axes[1].legend(fontsize=8, frameon=False)
    axes[1].set_title("Marginal distributions")

    fig.suptitle(
        f"Stage-2 posteriors (${N_INDIVIDUALS}$ individuals, ${N_SNPS}$ SNPs, ${N_CENTERS}$ centers, $\\epsilon={epsilon}$)",
        fontsize=11,
    )
    fig.tight_layout()
    out = os.path.join(output_dir, "exp9_posterior_gm_am.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {out}")
    return dict(posterior_gm=posterior_gm, posterior_am=posterior_am, causal_idx=causal_idx)


def _split_by_sizes(
    data: dict,
    sizes: np.ndarray,
    seed: int = 0,
) -> list[dict]:
    """
    Split data['G_std'] and data['y'] into len(sizes) centers,
    with center i receiving exactly sizes[i] individuals (sampled w/o replacement).
    """
    rng = np.random.default_rng(seed)
    n_total = data["G_std"].shape[0]
    sizes = np.array(sizes)
    total_needed = int(sizes.sum())

    # If more individuals requested than available, sample with replacement
    if total_needed > n_total:
        perm = rng.choice(n_total, size=total_needed, replace=True)
    else:
        perm = rng.choice(n_total, size=total_needed, replace=False)

    centers = []
    start = 0
    for sz in sizes:
        idx = perm[start : start + sz]
        centers.append({
            "G_std":      data["G_std"][idx],
            "y":          data["y"][idx],
            "beta":       data["beta"],
            "causal_idx": data["causal_idx"],
            "mafs":       data["mafs"],
        })
        start += sz
    return centers


def exp13_nyc_federation(
    n_snps: int = N_SNPS,
    total_individuals: int = N_INDIVIDUALS,
    epsilon: float = 1.0,
    n_reps: int = 3,
    base_seed: int = 1300,
    output_dir: str = "../figures",
) -> dict:
    
    dataset_csv = "../datasets/nyc_hospitals_w_organizations.csv"

    hospitals_df = pd.read_csv(dataset_csv)
    # sort by BEDS
    hospitals_df = hospitals_df.sort_values(by='BEDS', ascending=False)

    hospitals_df['weights'] = hospitals_df['BEDS'] / hospitals_df['BEDS'].sum()

    organizations_df = hospitals_df.groupby("ORGANIZATION")["BEDS"].sum().reset_index()
    organizations_df.columns = ["ORGANIZATION", "BEDS"]
    organizations_df = organizations_df.sort_values(by='BEDS', ascending=False)
    organizations_df['weights'] = organizations_df['BEDS'] / organizations_df['BEDS'].sum()

    organizations = organizations_df["ORGANIZATION"].unique()

    hospital_beds = organizations_df["BEDS"].values
    organization_beds = organizations_df["BEDS"].values

    delta = 0.95 - 1 / N_INDIVIDUALS


    print(f"Privacy Delta = {delta}")

    # plot bed distributions
    
    fig, ax = plt.subplots(2, 1, figsize=(FIGSIZE, 2 * FIGSIZE), squeeze=False)
    org_range = np.arange(0, len(organizations_df))
    hosp_range = np.arange(0, len(hospitals_df))

    # subdivide ax[0, 3] into 2 subplots (top and bottom)
    ax[0, 0].barh(org_range, organizations_df['BEDS'].values, color='gray')
    # remove yticks 
    ax[0, 0].set_yticks([])
    ax[0, 0].set_ylabel('Organization')
    ax[0, 0].set_xlabel('# Beds')
    ax[0, 0].set_title('Organization-level\nBed Distribution')

    ax[1, 0].barh(hosp_range, hospitals_df['BEDS'].values, color='gray')
    # remove yticks 
    ax[1, 0].set_yticks([])
    ax[1, 0].set_ylabel('Hospital')
    ax[1, 0].set_xlabel('# Beds')
    ax[1, 0].set_title('Hospital-level\nBed Distribution')
    
    fig.tight_layout()
    out = os.path.join(output_dir, "exp13_nyc_bed_distributions.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {out}")

    print('Privacy-utity tradeoff')

    # print('Hospital-level (knn, complete)')
    exp1_privacy_utility(n_reps=n_reps, weights=hospitals_df['weights'].values, filename='exp13_nyc_privacy_utility_hosp_knn.pdf', topology='knn', hospitals=hospitals_df, T=10, output_dir=output_dir, title='Hospital-level Geographical Network')
    exp1_privacy_utility(n_reps=n_reps, weights=hospitals_df['weights'].values, filename='exp13_nyc_privacy_utility_hosp_complete.pdf', topology='complete', hospitals=hospitals_df, T=10, output_dir=output_dir, title='Hospital-level Complete Network')

    print('Organization-level (complete, star)')
    exp1_privacy_utility(n_reps=n_reps, weights=organizations_df['weights'].values, filename='exp13_nyc_privacy_utility_org_complete.pdf', topology='complete', T=10, output_dir=output_dir, title='Organization-level Complete Network')
    exp1_privacy_utility(n_reps=n_reps, weights=organizations_df['weights'].values, filename='exp13_nyc_privacy_utility_org_star.pdf', topology='star', T=10, output_dir=output_dir, title='Organization-level Star Network')

    records = []

    for rep in range(n_reps):
        for network_type, weights in [('Org-level', organizations_df['weights'].values), ('Hosp-level', hospitals_df['weights'].values)]:
            data, centers = _simulate_rep(rep, base_seed=base_seed + rep * 11, weights=weights)
            causal_idx, G_std = data["causal_idx"], data["G_std"]
            n_snps = G_std.shape[1]
            res = run_dp_gwas_mle(
                centers,
                epsilon=epsilon,
                alpha=ALPHA_GWAS,
                K=K_ROUNDS,
                topology="complete",
                seed=rep * 1000 + 17,
                snp_chunk_size=N_SNPS,
            )
            m_all, m_rare, m_common = _eval_stratum(
                res.selected_gm,
                causal_idx,
                G_std,
                n_snps,
                data["rare_mask"],
                data["common_mask"],
            )

            records.append({
                'Power': m_all['power'],
                'FDR': m_all['fdr'],
                'F1': m_all['f1'],
                'Type': DP_DISTRIBUTED_GM_LABEL,
                'rep' : rep,
                'Variant Type' : 'All',
                'Network Type' : network_type
            })

            records.append({
                'Power': m_rare['power'],
                'FDR': m_rare['fdr'],
                'F1': m_rare['f1'],
                'Type': DP_DISTRIBUTED_GM_LABEL,
                'rep' : rep,
                'Variant Type' : 'Rare',
                'Network Type' : network_type
            })

            records.append({
                'Power': m_common['power'],
                'FDR': m_common['fdr'],
                'F1': m_common['f1'],
                'Type': DP_DISTRIBUTED_GM_LABEL,
                'rep' : rep,
                'Variant Type' : 'Common',
                'Network Type' : network_type
            })

            # run centralized gwas
            oracle = centralized_gwas(centers, alpha=ALPHA_GWAS)
            m_oracle_all, m_oracle_rare, m_oracle_common = _eval_stratum(
                oracle['selected'],
                causal_idx,
                G_std,
                n_snps,
                data['rare_mask'],
                data['common_mask'],
            )

            records.append({
                'Power': m_oracle_all['power'],
                'FDR': m_oracle_all['fdr'],
                'F1': m_oracle_all['f1'],
                'Type': ORACLE_LABEL,
                'rep' : rep,
                'Variant Type' : 'All',
                'Network Type' : network_type
            })

            records.append({
                'Power': m_oracle_rare['power'],
                'FDR': m_oracle_rare['fdr'],
                'F1': m_oracle_rare['f1'],
                'Type': ORACLE_LABEL,
                'rep' : rep,
                'Variant Type' : 'Rare',
                'Network Type' : network_type
            })
            
            records.append({
                'Power': m_oracle_common['power'],
                'FDR': m_oracle_common['fdr'],
                'F1': m_oracle_common['f1'],
                'Type': ORACLE_LABEL,
                'rep' : rep,
                'Variant Type' : 'Common',
                'Network Type' : network_type,
            })

            # run single center gwas
            single = single_center_gwas(centers, alpha=ALPHA_GWAS)
            m_single_all, m_single_rare, m_single_common = _eval_stratum(
                single['selected'],
                causal_idx,
                G_std,
                n_snps,
                data['rare_mask'],
                data['common_mask'],
            )

            records.append({
                'Power': m_single_all['power'],
                'FDR': m_single_all['fdr'],
                'F1': m_single_all['f1'],
                'Type': SINGLE_CENTER_LABEL,
                'rep' : rep,
                'Variant Type' : 'All',
                'Network Type' : network_type
            })

            records.append({
                'Power': m_single_rare['power'],
                'FDR': m_single_rare['fdr'],
                'F1': m_single_rare['f1'],
                'Type': SINGLE_CENTER_LABEL,
                'rep' : rep,
                'Variant Type' : 'Rare',
                'Network Type' : network_type
            })

            records.append({
                'Power': m_single_common['power'],
                'FDR': m_single_common['fdr'],
                'F1': m_single_common['f1'],
                'Type': SINGLE_CENTER_LABEL,    
                'rep' : rep,
                'Variant Type' : 'Common',
                'Network Type' : network_type
            })

            # run no dp gwas
            nodp = run_dp_gwas_mle(centers, 
                epsilon=np.inf, 
                alpha=ALPHA_GWAS, 
                K=1, 
                topology="complete", 
                seed=rep * 1000 + 17, 
                snp_chunk_size=N_SNPS
            )
            m_nodp_all, m_nodp_rare, m_nodp_common = _eval_stratum(
                nodp.selected_gm,
                causal_idx,
                G_std,
                n_snps,
                data['rare_mask'],
                data['common_mask'],
            )

            records.append({
                'Power': m_nodp_all['power'],
                'FDR': m_nodp_all['fdr'],
                'F1': m_nodp_all['f1'],
                'Type': NO_DP_DISTRIBUTED_LABEL,
                'rep' : rep,
                'Variant Type' : 'All',
                'Network Type' : network_type
            })

            records.append({
                'Power': m_nodp_rare['power'],
                'FDR': m_nodp_rare['fdr'],
                'F1': m_nodp_rare['f1'],
                'Type': NO_DP_DISTRIBUTED_LABEL,
                'rep' : rep,
                'Variant Type' : 'Rare',
                'Network Type' : network_type
            })

            records.append({
                'Power': m_nodp_common['power'],
                'FDR': m_nodp_common['fdr'],
                'F1': m_nodp_common['f1'],
                'Type': NO_DP_DISTRIBUTED_LABEL,
                'rep' : rep,
                'Variant Type' : 'Common',
                'Network Type' : network_type
            })

    df = pd.DataFrame(records)

    df_common = df[df['Variant Type'] == 'Common']
    df_rare = df[df['Variant Type'] == 'Rare']
    df_all = df[df['Variant Type'] == 'All']

    variant_list = [
        ('Common', df_common),
        ('Rare', df_rare),
        ('All', df_all),
    ]


    for variant_type, df_variant in variant_list:
        fig, ax = plt.subplots(1, 5, figsize=(4*FIGSIZE, FIGSIZE), squeeze=False, gridspec_kw={'width_ratios': [1, 1, 1, 0.5, 0.5]})
        sns.barplot(x='Network Type', y='Power', hue='Type', data=df_variant, ax=ax[0, 0], legend=False)
        ax[0, 0].set_xlabel('')
        ax[0, 0].set_ylabel('Power')
        ax[0, 0].set_ylim(0, 1)
        sns.barplot(x='Network Type', y='FDR', hue='Type', data=df_variant, ax=ax[0, 1], legend=True)
        ax[0, 1].set_xlabel('')
        ax[0, 1].set_ylabel('FDR')
        ax[0, 1].set_ylim(0, 1)
        sns.barplot(x='Network Type', y='F1', hue='Type', data=df_variant, ax=ax[0, 2], legend=False)
        ax[0, 2].set_xlabel('')
        ax[0, 2].set_ylabel('F1 Score')
        ax[0, 2].set_ylim(0, 1)
        
        # move legend outside of the axes so that it doesn't overlap with the bars
        ax[0, 1].legend(loc='upper right')
        
        # plot the number of beds in descending order for the hospitals and organizations
        hospitals_df = hospitals_df.sort_values(by='BEDS', ascending=False)
        organizations_df = organizations_df.sort_values(by='BEDS', ascending=False)

        org_range = np.arange(0, len(organizations_df))
        hosp_range = np.arange(0, len(hospitals_df))

        # subdivide ax[0, 3] into 2 subplots (top and bottom)
        ax[0, 3].barh(org_range, organizations_df['BEDS'].values, color='gray')
        # remove yticks 
        ax[0, 3].set_yticks([])
        ax[0, 3].set_xlabel('# Beds')
        ax[0, 3].set_title('Org-level\nDistribution')

        ax[0, 4].barh(hosp_range, hospitals_df['BEDS'].values, color='gray')
        # remove yticks 
        ax[0, 4].set_yticks([])
        ax[0, 4].set_xlabel('# Beds')
        ax[0, 4].set_title('Hosp-level\nDistribution')
        
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f'exp13_{variant_type}.pdf'), bbox_inches='tight')

    return df


def tune_experiment(
    n_reps: int,
    epsilon: float,
    base_seed: int = 100,
    K: int = K_ROUNDS,
    T: int = None,
    h2: float = HERITABILITY,
    n_centers: int = N_CENTERS,
    topology: str = "complete",
    convergence_tol: float = 1e-4,
    fdr_cap: float = 0.,
    fdr_penalty: float = 3.0,
):

    print(f"Tuning for epsilon={epsilon}, h2={h2}, n_centers={n_centers}, topology={topology}, convergence_tol={convergence_tol}, fdr_cap={fdr_cap}, fdr_penalty={fdr_penalty}")

    rows: list[dict] = []
    for alpha in ALPHA_LIST:
        powers, fdrs, f1s = [], [], []
        for rep in tqdm(range(n_reps), desc=f"Tune {alpha}"):
            data, centers = _simulate_rep(rep, base_seed=base_seed, h2=h2, n_centers=n_centers  )
            causal_idx, G_std = data["causal_idx"], data["G_std"]
            n_snps = G_std.shape[1]
            res = run_dp_gwas_mle(
                centers,
                epsilon=epsilon,
                alpha=alpha,
                K=K,
                T=T,
                topology=topology,
                convergence_tol=convergence_tol,
                track_convergence=False,
                seed=rep * 1000 + 17,
            )
            m_all, _, _ = _eval_stratum(
                res.selected_gm,
                causal_idx,
                G_std,
                n_snps,
                data["rare_mask"],
                data["common_mask"],
            )

            powers.append(m_all['power'])
            fdrs.append(m_all['fdr'])
            f1s.append(m_all['f1'])

        p_mean, p_se = float(np.mean(powers)), float(np.std(powers) / np.sqrt(max(len(powers), 1)))
        f_mean, f_se = float(np.mean(fdrs)), float(np.std(fdrs) / np.sqrt(max(len(fdrs), 1)))
        z_mean, z_se = float(np.mean(f1s)), float(np.std(f1s) / np.sqrt(max(len(f1s), 1)))
        score = z_mean - fdr_penalty * max(0.0, f_mean - fdr_cap)

        rows.append(
            dict(
                K=K,
                T=("auto" if T is None else int(T)),
                alpha=alpha,
                topology=topology,
                convergence_tol=convergence_tol,
                power_mean=p_mean,
                power_se=p_se,
                fdr_mean=f_mean,
                fdr_se=f_se,
                f1_mean=z_mean,
                f1_se=z_se,
                score=score,
                n_reps=n_reps,
                epsilon=epsilon,
            )
        )

    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)

    best = df.iloc[0]
    
    return best["alpha"]

def run_experiments(args: argparse.Namespace):
    output_dir = args.output_dir
    n_reps = args.n_reps
    epsilon = args.epsilon
    
    experiments = args.experiments

    os.makedirs(output_dir, exist_ok=True)
    
    if experiments == 'all' or experiments == 'regular':
        exp1_privacy_utility(n_reps=n_reps, output_dir=output_dir)
        exp2_three_way(n_reps=n_reps, epsilon=epsilon, output_dir=output_dir)
        exp3_topology(n_reps=max(2, min(n_reps, 3)), output_dir=output_dir)
        exp4_stratified(n_reps=n_reps, epsilon=epsilon, output_dir=output_dir)
        exp5_scaling(n_reps=max(2, min(n_reps, 2)), epsilon=epsilon, output_dir=output_dir)
        exp6_rizk_comparison(n_reps=n_reps, output_dir=output_dir)
        exp8_gwas_metrics_vs_n_centers(n_reps=n_reps, epsilon=epsilon, output_dir=output_dir)
        exp9_posterior_gm_am(epsilon=epsilon, output_dir=output_dir, n_reps=n_reps)
        exp10_power_fdr_vs_epsilon(n_reps=n_reps, output_dir=output_dir)
        exp11_power_fdr_vs_n_centers(n_reps=n_reps, epsilon=epsilon, output_dir=output_dir)
        exp12_manhattan_plot(output_dir=output_dir)

    if experiments == 'all' or experiments == 'nyc':
        exp13_nyc_federation(n_reps=n_reps, epsilon=epsilon, output_dir=output_dir)

def main():
    parser = argparse.ArgumentParser(
        description="DP-GWAS msprime experiments.",
    )
    parser.add_argument("--output_dir", default="../figures")
    parser.add_argument("--n_reps", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--experiments", type=str, default="all", choices=['all', 'regular', 'nyc'])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    run_experiments(args)
    print(f"\nAll figures saved to: {args.output_dir}/")

if __name__ == "__main__":
    main()
