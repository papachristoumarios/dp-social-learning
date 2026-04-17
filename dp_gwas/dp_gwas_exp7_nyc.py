"""
dp_gwas_exp7_nyc.py
===================
Experiment 7: Realistic execution of the DP-distributed GWAS framework
on an actual network of New York City-area hospitals.

Hospital locations, names, and bed counts come from the real US Hospital
Locations dataset.  Patient cohort sizes are proportional to each
hospital's reported bed count (a standard proxy for case volume in
multi-centre genomics studies).

Sub-experiments
---------------
7a. Network topology from geographic proximity (k-NN graph on GPS coords)
    vs. full connectivity — power and convergence.
7b. Privacy-utility curve for the realistic NYC network (mirrors Exp 1
    but with heterogeneous cohort sizes reflecting real bed counts).
7c. Per-hospital belief convergence traces — shows which hospitals
    drive consensus (high-degree / large-cohort hubs).
7d. County-level federation: Bronx / Brooklyn / Manhattan / Queens /
    Staten Island / Nassau / Westchester / Suffolk as natural sub-networks.

All figures are saved to the figures/ directory alongside Exp 1-6.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import networkx as nx
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.special import logsumexp

from dp_gwas_core import (
    simulate_gwas_data,
    run_dp_gwas_mle,
    centralized_gwas,
    single_center_gwas,
    evaluate_gwas,
    spectral_gap,
    make_adjacency,
    split_data_across_centers,
    score_stats_precompute,
    _score_stats_from_precomputed,
    log_belief_init,
    laplace_noise_log_belief,
    log_linear_update_all,
    sensitivity_score_stat,
    DPGWASResult,
)
from scipy.stats import chi2
import seaborn as sns

OUT = Path("../figures")
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

ALPHA_GWAS = 5e-8

# ---------------------------------------------------------------------------
# County colour palette
# ---------------------------------------------------------------------------
COUNTY_COLORS = {
    "BRONX":       "#534ab7",
    "KINGS":       "#1d9e75",
    "NEW YORK":    "#d85a30",
    "QUEENS":      "#ba7517",
    "RICHMOND":    "#888780",
    "NASSAU":      "#3b8bd4",
    "WESTCHESTER": "#d4537e",
    "SUFFOLK":     "#639922",
}

# ---------------------------------------------------------------------------
# Load and prepare the NYC hospital network
# ---------------------------------------------------------------------------

def load_nyc_hospitals(
    csv_path: str = "../datasets/us_hospital_locations.csv",
    counties: list[str] | None = None,
    min_beds: int = 50,
) -> pd.DataFrame:
    """
    Load open acute-care NY hospitals with valid bed counts.
    Returns a DataFrame sorted by county then bed count.
    """
    if counties is None:
        counties = ["BRONX", "KINGS", "NEW YORK", "QUEENS", "RICHMOND",
                    "NASSAU", "WESTCHESTER", "SUFFOLK"]

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    mask = (
        (df["STATE"] == "NY") &
        (df["STATUS"] == "OPEN") &
        (df["TYPE"].isin(["GENERAL ACUTE CARE", "CHILDREN", "WOMEN"])) &
        (df["BEDS"] >= min_beds) &
        (df["COUNTY"].isin(counties))
    )
    hospitals = df[mask].copy()
    hospitals = hospitals.sort_values(["COUNTY", "BEDS"], ascending=[True, False])
    hospitals = hospitals.reset_index(drop=True)

    # Short display name: first ≤35 chars
    hospitals["SHORT_NAME"] = hospitals["NAME"].str[:35]
    return hospitals


def build_geographic_adjacency(
    hospitals: pd.DataFrame,
    k_neighbors: int = 3,
    method: str = "knn",
) -> np.ndarray:
    """
    Build a doubly stochastic adjacency matrix from GPS coordinates.

    method='knn'   : k-nearest-neighbour graph, Metropolis-Hastings weights
    method='complete': fully connected (uniform 1/n)
    method='county': hospitals in the same county are connected
    """
    n = len(hospitals)
    coords = hospitals[["LATITUDE", "LONGITUDE"]].values

    if method == "complete":
        return np.ones((n, n)) / n

    elif method == "county":
        county = hospitals["COUNTY"].values
        adj = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(n):
                if county[i] == county[j]:
                    adj[i, j] = True
        # Metropolis-Hastings weights
        return _metropolis_hastings_bool(adj)

    else:  # knn
        # Haversine distance in km
        dist = _haversine_matrix(coords)
        adj = np.zeros((n, n), dtype=bool)
        np.fill_diagonal(adj, True)
        for i in range(n):
            nn_idx = np.argsort(dist[i])[1 : k_neighbors + 1]
            adj[i, nn_idx] = True
            adj[nn_idx, i] = True
        return _metropolis_hastings_bool(adj)


def _haversine_matrix(coords: np.ndarray) -> np.ndarray:
    """Pairwise Haversine distances in km for (lat, lon) array."""
    lat = np.radians(coords[:, 0])
    lon = np.radians(coords[:, 1])
    n = len(lat)
    D = np.zeros((n, n))
    for i in range(n):
        dlat = lat - lat[i]
        dlon = lon - lon[i]
        a = np.sin(dlat / 2) ** 2 + np.cos(lat[i]) * np.cos(lat) * np.sin(dlon / 2) ** 2
        D[i] = 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return D


def _metropolis_hastings_bool(adj: np.ndarray) -> np.ndarray:
    """Metropolis-Hastings weights for a symmetric boolean adjacency matrix."""
    n = adj.shape[0]
    degree = adj.sum(axis=1).astype(float)
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and adj[i, j]:
                A[i, j] = 1.0 / (1.0 + max(degree[i], degree[j]))
        A[i, i] = 1.0 - A[i].sum()
    return A


# ---------------------------------------------------------------------------
# Cohort sizes proportional to bed counts
# ---------------------------------------------------------------------------

def beds_to_cohort_sizes(
    beds: np.ndarray,
    total_individuals: int = 10000,
    min_per_hospital: int = 50,
) -> np.ndarray:
    """
    Allocate total_individuals to hospitals proportional to bed counts.
    Guarantees each hospital has at least min_per_hospital individuals.
    """
    beds_pos = np.maximum(beds, min_per_hospital).astype(float)
    raw = beds_pos / beds_pos.sum() * total_individuals
    sizes = np.maximum(np.round(raw).astype(int), min_per_hospital)
    # Re-normalise to exactly total_individuals
    diff = total_individuals - sizes.sum()
    idx = np.argsort(beds_pos)[::-1]
    for i in range(abs(diff)):
        sizes[idx[i % len(idx)]] += int(np.sign(diff))
    return sizes


# ---------------------------------------------------------------------------
# Custom DP-GWAS runner that accepts a prebuilt adjacency matrix
# and heterogeneous cohort sizes
# ---------------------------------------------------------------------------

def run_dp_gwas_custom_network(
    centers_data: list[dict],
    A: np.ndarray,
    epsilon: float,
    alpha: float = 5e-8,
    T: int = 200,
    K: int = 20,
    binary_trait: bool = False,
    track_per_hospital: bool = False,
    seed: int = 0,
    snp_chunk_size: int | None = None,
    convergence_tv_max_snps: int | None = None,
) -> dict:
    """
    Like run_dp_gwas_mle but accepts an arbitrary adjacency matrix A
    and additionally returns per-hospital belief traces when requested.

    ``snp_chunk_size`` bounds peak memory for the belief tensor; see
    :func:`dp_gwas_core.run_dp_gwas_mle`.
    """
    n_centers = len(centers_data)
    M = centers_data[0]["G_std"].shape[1]
    n_states = 2

    sensitivities = [
        sensitivity_score_stat(c["G_std"].shape[0], binary=binary_trait)
        for c in centers_data
    ]

    pres = [score_stats_precompute(c["y"], binary_trait) for c in centers_data]
    chunk_size = M if snp_chunk_size is None else int(snp_chunk_size)
    chunk_size = max(1, min(chunk_size, M))

    noisy_stat_sum = np.zeros(M, dtype=np.float64)
    gm_accum = np.zeros((M, n_states), dtype=np.float64)
    log_prob_am_merged = np.empty((M, n_states), dtype=np.float64)
    per_hospital_traces = {i: [] for i in range(n_centers)} if track_per_hospital else {}
    tv_trace: list[float] = []

    for j0 in range(0, M, chunk_size):
        j1 = min(j0 + chunk_size, M)
        sl = slice(j0, j1)
        B = j1 - j0

        llr_block = np.stack(
            [
                _score_stats_from_precomputed(centers_data[ic]["G_std"][:, sl], pres[ic])
                for ic in range(n_centers)
            ],
            axis=0,
        )

        am_chunk: np.ndarray | None = None
        for k in range(K):
            rng_k = np.random.default_rng(seed * 10000 + k)
            belief_rows = []
            noisy_chunk = np.zeros(B, dtype=np.float64)
            for i, cdata in enumerate(centers_data):
                delta_i = sensitivities[i]
                noise_scale = delta_i * K * n_states / epsilon
                llr_i = llr_block[i]
                noisy_chunk += llr_i + rng_k.laplace(0.0, noise_scale, B)
                lb_i = log_belief_init(llr_i)
                lb_noisy = laplace_noise_log_belief(
                    lb_i, delta_i, epsilon, n_states, K, rng_k
                )
                belief_rows.append(lb_noisy)

            noisy_stat_sum[sl] += noisy_chunk
            belief_tensor = np.stack(belief_rows, axis=0)

            for t in range(T):
                belief_tensor = log_linear_update_all(belief_tensor, A)

                if track_per_hospital and k == 0 and j0 == 0:
                    for i in range(n_centers):
                        p_alt = float(np.exp(belief_tensor[i, 0, 1]))
                        per_hospital_traces[i].append(p_alt)
                    n_tv = B if convergence_tv_max_snps is None else min(
                        B, int(convergence_tv_max_snps)
                    )
                    lbr = belief_tensor[:, :n_tv, 1] - belief_tensor[:, :n_tv, 0]
                    tv_trace.append(float(np.std(lbr)))

            gm_accum[sl] += belief_tensor.mean(axis=0)
            ls_round = logsumexp(belief_tensor, axis=0)
            if am_chunk is None:
                am_chunk = ls_round
            else:
                am_chunk = logsumexp(
                    np.stack([am_chunk, ls_round], axis=0), axis=0
                )

        log_prob_am_merged[sl] = am_chunk

    # Stage 1: calibrated p-values from noisy aggregate statistic
    # Under H0: 2 * sum_i Lambda_i ~ chi2_{n_centers}
    agg_stat = noisy_stat_sum / float(K)
    chi2_agg = np.clip(2.0 * agg_stat, 0.0, None)
    pvalues = chi2.sf(chi2_agg, df=n_centers)

    # Stage 2: belief aggregation for SNP ranking
    log_beliefs_gm = gm_accum / float(K)
    lse = logsumexp(log_beliefs_gm, axis=1, keepdims=True)
    log_beliefs_gm = log_beliefs_gm - lse
    lbr_gm = log_beliefs_gm[:, 1] - log_beliefs_gm[:, 0]

    log_prob_am = log_prob_am_merged - np.log(K * n_centers)
    lse2 = logsumexp(log_prob_am, axis=1, keepdims=True)
    log_prob_am = log_prob_am - lse2
    lbr_am = log_prob_am[:, 1] - log_prob_am[:, 0]

    # Selection: posterior ranking gated by calibrated p-value
    posterior_gm = np.exp(log_beliefs_gm[:, 1])
    posterior_am = np.exp(log_prob_am[:, 1])
    selected_gm = (posterior_gm > 0.5) & (pvalues < alpha)
    selected_am = (posterior_am > 0.5) & (pvalues < alpha)

    return dict(
        lbr_gm=lbr_gm, lbr_am=lbr_am,
        pvalues_gm=pvalues, pvalues_am=pvalues,
        selected_gm=selected_gm, selected_am=selected_am,
        per_hospital_traces=per_hospital_traces,
        tv_trace=tv_trace,
        n_total=sum(c["G_std"].shape[0] for c in centers_data),
    )


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def _draw_hospital_map(
    ax: plt.Axes,
    hospitals: pd.DataFrame,
    A: np.ndarray,
    highlight_idx: list[int] | None = None,
    title: str = "",
    edge_alpha_scale: float = 1.0,
    node_size_col: str = "BEDS",
) -> None:
    """
    Draw hospitals as scatter points on a lat/lon axes with network edges.
    Node size ∝ bed count.  Colour by county.
    """
    lons = hospitals["LONGITUDE"].values
    lats = hospitals["LATITUDE"].values
    counties = hospitals["COUNTY"].values
    beds = hospitals[node_size_col].values.astype(float)

    n = len(hospitals)
    # Draw edges first (thin, semi-transparent)
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] > 1e-6 and not (i == j):
                w = A[i, j]
                ax.plot(
                    [lons[i], lons[j]], [lats[i], lats[j]],
                    color="#aaaaaa", alpha=min(0.6 * edge_alpha_scale * w * n, 0.7),
                    linewidth=0.6, zorder=1,
                )

    # Draw nodes
    for idx, row in hospitals.iterrows():
        color = COUNTY_COLORS.get(row["COUNTY"], "#888780")
        size = 20 + (row["BEDS"] / beds.max()) * 160
        marker = "*" if (highlight_idx and idx in highlight_idx) else "o"
        ax.scatter(row["LONGITUDE"], row["LATITUDE"],
                   s=size, color=color, edgecolors="white",
                   linewidth=0.4, zorder=3, marker=marker)

    # County legend
    handles = [
        mpatches.Patch(color=c, label=county.title())
        for county, c in COUNTY_COLORS.items()
        if county in counties
    ]
    ax.legend(handles=handles, fontsize=6, loc="lower left",
              frameon=True, framealpha=0.8, ncol=2)
    ax.set_xlabel("Longitude", fontsize=8)
    ax.set_ylabel("Latitude", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    
    # despine
    sns.despine(ax=ax)


# ---------------------------------------------------------------------------
# Experiment 7a: Geographic network topologies
# ---------------------------------------------------------------------------

def exp7a_network_topologies(
    hospitals: pd.DataFrame,
    n_snps: int = 400,
    n_causal: int = 20,
    total_individuals: int = 8000,
    epsilon: float = 1.0,
    T: int = 150,
    K: int = 15,
    n_reps: int = 3,
    seed: int = 70,
) -> dict:
    """
    Compare three network topologies using the real NYC hospital graph:
    (1) k-NN geographic (k=3), (2) county-level, (3) fully connected.
    Report power, FDR, spectral gap, and convergence speed.
    """
    print("  7a: Network topology comparison (geographic / county / complete)")

    beds = hospitals["BEDS"].values
    cohort_sizes = beds_to_cohort_sizes(beds, total_individuals)
    n = len(hospitals)

    topologies = {
        "k-NN ($k = 3$)": build_geographic_adjacency(hospitals, k_neighbors=3, method="knn"),
        "County-level":     build_geographic_adjacency(hospitals, method="county"),
        "Complete":   build_geographic_adjacency(hospitals, method="complete"),
    }

    results = {t: {"power": [], "fdr": [], "sg": spectral_gap(A)}
               for t, A in topologies.items()}

    for rep in range(n_reps):
        data = simulate_gwas_data(
            total_individuals, n_snps, n_causal, seed=seed + rep * 7
        )
        # Build per-hospital data splits using real cohort sizes
        centers_data = _split_by_sizes(data, cohort_sizes, seed=rep)
        causal_idx = data["causal_idx"]
        oracle = centralized_gwas(centers_data, alpha=ALPHA_GWAS)

        for top_name, A in topologies.items():
            res = run_dp_gwas_custom_network(
                centers_data, A, epsilon=epsilon,
                alpha=ALPHA_GWAS, T=T, K=K, seed=seed + rep * 100,
            )
            m = evaluate_gwas(res["selected_gm"], causal_idx, n_snps)
            results[top_name]["power"].append(m["power"])
            results[top_name]["fdr"].append(m["fdr"])

    # --- Figure 7a ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Map for each topology
    for ax, (top_name, A) in zip(axes, topologies.items()):
        sg = spectral_gap(A)
        pwr = np.mean(results[top_name]["power"])
        _draw_hospital_map(
            ax, hospitals, A,
            title=f"{top_name}\nspectral gap = {sg:.1g}, statistical power={pwr:.2f}",
            edge_alpha_scale=(3.0 if top_name != "Complete" else 0.3),
        )
        sns.despine()

    fig.suptitle(
        f"NYC hospital network topologies  "
        f"($n = {n}$ hospitals, $\epsilon = {epsilon}$)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT / "exp7a_network_topologies.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"    → saved exp7a_network_topologies.pdf")
    return results


# ---------------------------------------------------------------------------
# Experiment 7b: Privacy-utility curve (real heterogeneous cohorts)
# ---------------------------------------------------------------------------

def exp7b_privacy_utility(
    hospitals: pd.DataFrame,
    n_snps: int = 400,
    n_causal: int = 20,
    total_individuals: int = 8000,
    epsilons: list[float] | None = None,
    T: int = 150,
    K: int = 15,
    n_reps: int = 3,
    seed: int = 71,
) -> dict:
    """
    Privacy-utility tradeoff on the real NYC network.
    Uses the k-NN geographic topology and bed-proportional cohort sizes.
    Overlays oracle (centralised), best single hospital (largest beds), and
    no-DP distributed (:func:`run_dp_gwas_custom_network` with ``K=1``,
    ``epsilon=np.inf``).
    """
    if epsilons is None:
        epsilons = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]

    print("  7b: Privacy-utility curve on real NYC network")

    beds = hospitals["BEDS"].values
    cohort_sizes = beds_to_cohort_sizes(beds, total_individuals)
    A_knn = build_geographic_adjacency(hospitals, k_neighbors=3, method="knn")
    n = len(hospitals)

    # Best single hospital = one with most beds
    best_hosp_idx = int(np.argmax(beds))
    best_hosp_name = hospitals.iloc[best_hosp_idx]["SHORT_NAME"]

    power_gm, power_am, fdr_gm, fdr_am = [], [], [], []
    oracle_powers, single_powers = [], []
    nodp_powers_gm, nodp_powers_am = [], []
    nodp_fdrs_gm, nodp_fdrs_am = [], []
    oracle_fdrs: list[float] = []

    for rep in range(n_reps):
        data = simulate_gwas_data(total_individuals, n_snps, n_causal, seed=seed + rep * 7)
        centers_data = _split_by_sizes(data, cohort_sizes, seed=rep)
        causal_idx = data["causal_idx"]

        oracle = centralized_gwas(centers_data, alpha=ALPHA_GWAS)
        single = single_center_gwas(centers_data, center_idx=best_hosp_idx, alpha=ALPHA_GWAS)
        oracle_powers.append(oracle["power"])
        single_powers.append(single["power"])
        oracle_fdrs.append(float(oracle["fdr"]))

        res_nd = run_dp_gwas_custom_network(
            centers_data,
            A_knn,
            epsilon=np.inf,
            alpha=ALPHA_GWAS,
            T=T,
            K=1,
            seed=seed + rep * 100,
        )
        m_nd_gm = evaluate_gwas(res_nd["selected_gm"], causal_idx, n_snps)
        m_nd_am = evaluate_gwas(res_nd["selected_am"], causal_idx, n_snps)
        nodp_powers_gm.append(m_nd_gm["power"])
        nodp_powers_am.append(m_nd_am["power"])
        nodp_fdrs_gm.append(m_nd_gm["fdr"])
        nodp_fdrs_am.append(m_nd_am["fdr"])

    for eps in epsilons:
        p_gm_rep, p_am_rep, f_gm_rep, f_am_rep = [], [], [], []
        for rep in range(n_reps):
            data = simulate_gwas_data(total_individuals, n_snps, n_causal, seed=seed + rep * 7)
            centers_data = _split_by_sizes(data, cohort_sizes, seed=rep)
            causal_idx = data["causal_idx"]
            res = run_dp_gwas_custom_network(
                centers_data, A_knn, epsilon=eps,
                alpha=ALPHA_GWAS, T=T, K=K, seed=seed + rep * 100,
            )
            m_gm = evaluate_gwas(res["selected_gm"], causal_idx, n_snps)
            m_am = evaluate_gwas(res["selected_am"], causal_idx, n_snps)
            p_gm_rep.append(m_gm["power"])
            p_am_rep.append(m_am["power"])
            f_gm_rep.append(m_gm["fdr"])
            f_am_rep.append(m_am["fdr"])
        power_gm.append(np.mean(p_gm_rep))
        power_am.append(np.mean(p_am_rep))
        fdr_gm.append(np.mean(f_gm_rep))
        fdr_am.append(np.mean(f_am_rep))

    oracle_p = np.mean(oracle_powers)
    single_p = np.mean(single_powers)
    nodp_p_gm = float(np.mean(nodp_powers_gm))
    nodp_p_am = float(np.mean(nodp_powers_am))
    oracle_fdr_m = float(np.mean(oracle_fdrs))
    nodp_fdr_gm = float(np.mean(nodp_fdrs_gm))
    nodp_fdr_am = float(np.mean(nodp_fdrs_am))
    eps_crit = next((e for e, p in zip(epsilons, power_gm) if p > single_p), None)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.plot(epsilons, power_gm, "o-", color="#534ab7", linewidth=1.8, label="DP-GWAS (GM)")
    ax.plot(epsilons, power_am, "s--", color="#1d9e75", linewidth=1.8, label="DP-GWAS (AM)")
    ax.axhline(oracle_p, color="#888780", linestyle=":", linewidth=1.5,
               label=f"Oracle ({n} hospitals pooled)")
    ax.axhline(single_p, color="#d85a30", linestyle="-.", linewidth=1.3,
               label=f"Best single hospital\n({best_hosp_name})")
    ax.axhline(
        nodp_p_gm,
        color="#ba7517",
        linestyle=(0, (3, 1, 1, 1)),
        linewidth=1.35,
        label="No DP (GM)",
    )
    ax.axhline(
        nodp_p_am,
        color="#ba7517",
        linestyle=(0, (1, 1)),
        linewidth=1.15,
        alpha=0.85,
        label="No DP (AM)",
    )
    if eps_crit:
        ax.axvline(eps_crit, color="red", linestyle="--", alpha=0.35, linewidth=1.2,
                   label=f"ε* ≈ {eps_crit}")
    ax.set_xlabel("Privacy budget (ε)")
    ax.set_ylabel("Statistical power")
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.10)
    ax.legend(fontsize=7.5, frameon=False)
    ax.set_title(f"Power vs ε — {n} NYC hospitals")

    ax = axes[1]
    ax.plot(epsilons, fdr_gm, "o-", color="#534ab7", linewidth=1.8, label="DP-GWAS (GM)")
    ax.plot(epsilons, fdr_am, "s--", color="#1d9e75", linewidth=1.8, label="DP-GWAS (AM)")
    ax.axhline(oracle_fdr_m, color="#888780", linestyle=":", linewidth=1.5, label="Oracle")
    ax.axhline(
        nodp_fdr_gm,
        color="#ba7517",
        linestyle=(0, (3, 1, 1, 1)),
        linewidth=1.35,
        label="No DP (GM)",
    )
    ax.axhline(
        nodp_fdr_am,
        color="#ba7517",
        linestyle=(0, (1, 1)),
        linewidth=1.15,
        alpha=0.85,
        label="No DP (AM)",
    )
    ax.set_xlabel("Privacy budget (ε)")
    ax.set_ylabel("False discovery rate")
    ax.set_xscale("log")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("FDR vs ε")

    # Annotate bed distribution as inset bar chart
    # move a bit lower
    ax_inset = axes[1].inset_axes([0.55, 0.35, 0.42, 0.48])
    county_beds = hospitals.groupby("COUNTY")["BEDS"].sum().reindex(list(COUNTY_COLORS.keys()), fill_value=0)
    ax_inset.barh(range(len(county_beds)), county_beds.values,
                  color=[COUNTY_COLORS.get(c, "#888") for c in county_beds.index], height=0.7)
    ax_inset.set_yticks(range(len(county_beds)))
    ax_inset.set_yticklabels([c.title() for c in county_beds.index], fontsize=6)
    ax_inset.set_xlabel("Total beds by county", fontsize=6)
    ax_inset.tick_params(axis="x", labelsize=6)
    ax_inset.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Privacy-utility on NYC hospital network\n"
        f"($n = {n}$, bed-proportional cohorts, k-NN topology, $\epsilon = {eps}$)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(OUT / "exp7b_privacy_utility_nyc.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"    → saved exp7b_privacy_utility_nyc.pdf  (ε* ≈ {eps_crit})")
    return dict(
        epsilons=epsilons,
        power_gm=power_gm,
        power_am=power_am,
        fdr_gm=fdr_gm,
        oracle_power=oracle_p,
        single_power=single_p,
        nodp_power_gm=nodp_p_gm,
        nodp_power_am=nodp_p_am,
        eps_crit=eps_crit,
        best_hospital=best_hosp_name,
    )


# ---------------------------------------------------------------------------
# Experiment 7c: Per-hospital convergence traces + hub analysis
# ---------------------------------------------------------------------------

def exp7c_per_hospital_convergence(
    hospitals: pd.DataFrame,
    n_snps: int = 300,
    n_causal: int = 15,
    total_individuals: int = 8000,
    epsilon: float = 1.0,
    T: int = 150,
    K: int = 5,
    seed: int = 72,
) -> dict:
    """
    Track per-hospital belief evolution over iterations.
    Highlights high-degree geographic hubs vs. small peripheral hospitals.
    Shows that hospitals with more neighbours and larger cohorts
    converge faster and anchor the consensus.
    """
    print("  7c: Per-hospital convergence traces")

    beds = hospitals["BEDS"].values
    cohort_sizes = beds_to_cohort_sizes(beds, total_individuals)
    A_knn = build_geographic_adjacency(hospitals, k_neighbors=3, method="knn")
    degree = (A_knn > 1e-6).sum(axis=1) - 1   # exclude self

    data = simulate_gwas_data(total_individuals, n_snps, n_causal, seed=seed)
    centers_data = _split_by_sizes(data, cohort_sizes, seed=0)
    causal_idx = data["causal_idx"]

    # Pick a causal SNP to track
    tracked_snp = int(causal_idx[0])

    res = run_dp_gwas_custom_network(
        centers_data, A_knn, epsilon=epsilon,
        alpha=ALPHA_GWAS, T=T, K=K, seed=seed,
        track_per_hospital=True,
    )

    # Identify hub hospitals (top 5 by degree) and peripheral (bottom 5)
    hub_idx    = np.argsort(degree)[-5:][::-1]
    periph_idx = np.argsort(degree)[:5]

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.35)

    # Panel 1: Map with hub/peripheral highlighted
    ax_map = fig.add_subplot(gs[:, 0])
    _draw_hospital_map(
        ax_map, hospitals, A_knn,
        highlight_idx=list(hub_idx),
        title="NYC k-NN network\n(★ = hub hospitals)",
        edge_alpha_scale=2.5,
    )
    # Mark peripheral in different style
    for pi in periph_idx:
        row = hospitals.iloc[pi]
        ax_map.scatter(row["LONGITUDE"], row["LATITUDE"],
                       s=60, color="white", edgecolors="#d85a30",
                       linewidth=1.5, zorder=4, marker="o")

    hub_legend    = mlines.Line2D([], [], marker="*", color="k", markersize=8,
                                  linestyle="None", label="Hub (high degree)")
    periph_legend = mlines.Line2D([], [], marker="o", color="white",
                                  markeredgecolor="#d85a30", markersize=8,
                                  linestyle="None", label="Peripheral")
    ax_map.legend(handles=[hub_legend, periph_legend], fontsize=7,
                  loc="upper right", frameon=True)

    # Panel 2: Convergence traces — hub hospitals
    ax_hub = fig.add_subplot(gs[0, 1])
    cmap_hub = plt.cm.Blues(np.linspace(0.45, 0.9, len(hub_idx)))
    for c_idx, color in zip(hub_idx, cmap_hub):
        trace = res["per_hospital_traces"][c_idx]
        lbl = hospitals.iloc[c_idx]["SHORT_NAME"]
        ax_hub.plot(trace, color=color, linewidth=1.2, label=lbl[:28],
                    alpha=0.9)
    ax_hub.set_xlabel("Iteration", fontsize=8)
    ax_hub.set_ylabel("P(alt | SNP 0)", fontsize=8)
    ax_hub.set_title("Hub hospital convergence", fontsize=9)
    ax_hub.legend(fontsize=5.5, frameon=False, loc="right")
    ax_hub.set_xscale("log")
    ax_hub.set_yscale("log")

    # Panel 3: Convergence traces — peripheral hospitals
    ax_per = fig.add_subplot(gs[1, 1])
    cmap_per = plt.cm.Oranges(np.linspace(0.45, 0.9, len(periph_idx)))
    for c_idx, color in zip(periph_idx, cmap_per):
        trace = res["per_hospital_traces"][c_idx]
        lbl = hospitals.iloc[c_idx]["SHORT_NAME"]
        ax_per.plot(trace, color=color, linewidth=1.2, label=lbl[:28],
                    alpha=0.9)
    ax_per.set_xlabel("Iteration", fontsize=8)
    ax_per.set_ylabel("P(alt | SNP 0)", fontsize=8)
    ax_per.set_title("Peripheral hospital convergence", fontsize=9)
    ax_per.legend(fontsize=5.5, frameon=False, loc="right")
    ax_per.set_xscale("log")
    ax_per.set_yscale("log")

    # Panel 4: Network TV distance over iterations
    ax_tv = fig.add_subplot(gs[0, 2])
    ax_tv.plot(res["tv_trace"], color="#534ab7", linewidth=1.5)
    ax_tv.set_xlabel("Iteration", fontsize=8)
    ax_tv.set_ylabel("Belief spread (std)", fontsize=8)
    ax_tv.set_yscale("log")
    ax_tv.set_title("Network-wide convergence", fontsize=9)

    # Panel 5: Bed count vs degree scatter
    ax_deg = fig.add_subplot(gs[1, 2])
    sc = ax_deg.scatter(beds, degree,
                        c=[list(COUNTY_COLORS.keys()).index(c)
                           if c in COUNTY_COLORS else 0
                           for c in hospitals["COUNTY"]],
                        cmap="tab10", s=30, alpha=0.8, zorder=3)
    for hi in hub_idx:
        ax_deg.scatter(beds[hi], degree[hi], s=100, marker="*",
                       color="gold", edgecolors="k", linewidth=0.5, zorder=5)
    ax_deg.set_xlabel("Bed count", fontsize=8)
    ax_deg.set_ylabel("Network degree (k-NN)", fontsize=8)
    ax_deg.set_title("Beds vs network degree", fontsize=9)

    fig.suptitle(
        "Per-hospital belief convergence on NYC network",
        fontsize=11,
    )
    fig.savefig(OUT / "exp7c_per_hospital_convergence.pdf", bbox_inches="tight")
    plt.close(fig)
    print("    → saved exp7c_per_hospital_convergence.pdf")
    return dict(hub_idx=hub_idx.tolist(), periph_idx=periph_idx.tolist(),
                tv_trace=res["tv_trace"])


# ---------------------------------------------------------------------------
# Experiment 7d: County-level federation
# ---------------------------------------------------------------------------

def exp7d_county_federation(
    hospitals: pd.DataFrame,
    n_snps: int = 400,
    n_causal: int = 20,
    total_individuals: int = 8000,
    epsilon: float = 1.0,
    T: int = 150,
    K: int = 15,
    n_reps: int = 3,
    seed: int = 73,
) -> dict:
    """
    Model the realistic regulatory scenario where hospitals primarily
    exchange data within county (data sharing agreements are easier
    intra-county under NY state law).

    Compare:
    - Full network (k-NN across county boundaries)
    - County-restricted network (no cross-county edges)
    - Each county acting independently (no federation)
    - Oracle (all data pooled)

    Annotate which counties are "information bottlenecks" (low power alone).
    """
    print("  7d: County-level federation analysis")

    beds = hospitals["BEDS"].values
    cohort_sizes = beds_to_cohort_sizes(beds, total_individuals)
    A_full   = build_geographic_adjacency(hospitals, k_neighbors=3, method="knn")
    A_county = build_geographic_adjacency(hospitals, method="county")

    county_list = sorted(hospitals["COUNTY"].unique())
    # county_idx[c] = list of hospital indices in that county
    county_idx = {c: hospitals.index[hospitals["COUNTY"] == c].tolist()
                  for c in county_list}

    res_full   = {"power": [], "fdr": []}
    res_county = {"power": [], "fdr": []}
    county_solo = {c: {"power": [], "fdr": []} for c in county_list}

    for rep in range(n_reps):
        data = simulate_gwas_data(total_individuals, n_snps, n_causal, seed=seed + rep * 11)
        centers_data = _split_by_sizes(data, cohort_sizes, seed=rep)
        causal_idx = data["causal_idx"]
        oracle = centralized_gwas(centers_data, alpha=ALPHA_GWAS)

        # Full k-NN network
        r_full = run_dp_gwas_custom_network(
            centers_data, A_full, epsilon=epsilon,
            alpha=ALPHA_GWAS, T=T, K=K, seed=seed + rep * 100,
        )
        m = evaluate_gwas(r_full["selected_gm"], causal_idx, n_snps)
        res_full["power"].append(m["power"]); res_full["fdr"].append(m["fdr"])

        # County-restricted network
        r_cnty = run_dp_gwas_custom_network(
            centers_data, A_county, epsilon=epsilon,
            alpha=ALPHA_GWAS, T=T, K=K, seed=seed + rep * 100,
        )
        m = evaluate_gwas(r_cnty["selected_gm"], causal_idx, n_snps)
        res_county["power"].append(m["power"]); res_county["fdr"].append(m["fdr"])

        # Each county independently (best single hospital in each county)
        for c in county_list:
            idxs = county_idx[c]
            if not idxs:
                continue
            # Use the hospital with most beds in this county
            best = max(idxs, key=lambda i: beds[i])
            s = single_center_gwas(centers_data, center_idx=best, alpha=ALPHA_GWAS)
            county_solo[c]["power"].append(s["power"])
            county_solo[c]["fdr"].append(s["fdr"])

    # --- Figure 7d ---
    fig = plt.figure(figsize=(16, 6))
    gs_main = fig.add_gridspec(1, 2, wspace=0.38)

    # Left: map with county colouring and connectivity
    ax_map = fig.add_subplot(gs_main[0])
    _draw_hospital_map(
        ax_map, hospitals, A_county,
        title="County-restricted network topology",
        edge_alpha_scale=4.0,
    )

    # Right: grouped bar chart comparing federation levels
    ax_bar = fig.add_subplot(gs_main[1])
    conditions = ["Oracle", "Full k-NN", "County-only"] + \
                 [f"{c.title()[:8]}\nalone" for c in county_list]
    powers_all = (
        [1.0,   # oracle normalised
         np.mean(res_full["power"]),
         np.mean(res_county["power"])]
        + [np.mean(county_solo[c]["power"]) if county_solo[c]["power"] else 0.0
           for c in county_list]
    )
    # Replace oracle with actual oracle power
    data_ref = simulate_gwas_data(total_individuals, n_snps, n_causal, seed=seed)
    centers_ref = _split_by_sizes(data_ref, cohort_sizes, seed=0)
    oracle_ref = centralized_gwas(centers_ref, alpha=ALPHA_GWAS)
    powers_all[0] = oracle_ref["power"]

    colors = (["#888780", "#534ab7", "#1d9e75"] +
              [COUNTY_COLORS.get(c, "#999") for c in county_list])
    x = np.arange(len(conditions))
    bars = ax_bar.bar(x, powers_all, color=colors, width=0.7, edgecolor="white",
                      linewidth=0.5)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(conditions, rotation=35, ha="right", fontsize=7.5)
    ax_bar.set_ylabel("Statistical power (GM)")
    ax_bar.set_ylim(0, 1.15)
    ax_bar.set_title("Power by federation level")
    ax_bar.axhline(powers_all[0], color="#888780", linestyle=":", linewidth=1, alpha=0.7)

    # Annotate bottleneck counties (power < 0.2 acting alone)
    for i, (cond, pwr) in enumerate(zip(conditions, powers_all)):
        if pwr < 0.2 and i > 2:
            ax_bar.text(x[i], pwr + 0.03, f"power < 0.2", ha="center", fontsize=9, color="#d85a30", rotation=90)

    fig.suptitle(
        "County-level federation on NYC hospital network\n"
        "($\Delta$ = counties that cannot detect signal independently)",
        fontsize=10,
    )
    fig.savefig(OUT / "exp7d_county_federation.pdf", bbox_inches="tight")
    plt.close(fig)
    print("    → saved exp7d_county_federation.pdf")
    return dict(
        res_full=res_full, res_county=res_county,
        county_solo=county_solo, county_list=county_list,
    )


# ---------------------------------------------------------------------------
# Helper: split data according to pre-specified cohort sizes
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Master runner for Exp 7
# ---------------------------------------------------------------------------

def run_experiment7(
    csv_path: str = "../datasets/us_hospital_locations.csv",
    fast: bool = False,
) -> dict:
    """Run all four sub-experiments for Exp 7."""
    print("\nExperiment 7: Realistic NYC hospital network GWAS")

    hospitals = load_nyc_hospitals(csv_path)
    print(f"  Loaded {len(hospitals)} NYC-area acute-care hospitals "
          f"across {hospitals['COUNTY'].nunique()} counties")
    print(f"  Bed range: {hospitals['BEDS'].min()}–{hospitals['BEDS'].max()} "
          f"(median {hospitals['BEDS'].median():.0f})")

    if fast:
        cfg = dict(n_snps=200, n_causal=10, total_individuals=4000,
                   T=80, K=8, n_reps=2)
    else:
        cfg = dict(n_snps=400, n_causal=20, total_individuals=8000,
                   T=150, K=15, n_reps=3)

    r7a = exp7a_network_topologies(hospitals, **cfg)
    r7b = exp7b_privacy_utility(
        hospitals,
        n_snps=cfg["n_snps"], n_causal=cfg["n_causal"],
        total_individuals=cfg["total_individuals"],
        T=cfg["T"], K=cfg["K"], n_reps=cfg["n_reps"],
        epsilons=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0] if not fast
                  else [0.1, 0.5, 1.0, 2.0],
    )
    r7c = exp7c_per_hospital_convergence(
        hospitals,
        n_snps=cfg["n_snps"], n_causal=cfg["n_causal"],
        total_individuals=cfg["total_individuals"],
        T=cfg["T"], K=min(cfg["K"], 5),
    )
    r7d = exp7d_county_federation(
        hospitals,
        n_snps=cfg["n_snps"], n_causal=cfg["n_causal"],
        total_individuals=cfg["total_individuals"],
        T=cfg["T"], K=cfg["K"], n_reps=cfg["n_reps"],
    )

    print(f"\n  Exp 7 summary:")
    print(f"    7b ε* ≈ {r7b['eps_crit']}  "
          f"(best single hospital: {r7b['best_hospital'][:40]})")
    for top, v in r7a.items():
        print(f"    7a {top:15s}: power={np.mean(v['power']):.3f}  "
              f"spectral gap={v['sg']:.3f}")

    return dict(r7a=r7a, r7b=r7b, r7c=r7c, r7d=r7d, hospitals=hospitals)
