from __future__ import annotations

import warnings
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm, chi2
from dataclasses import dataclass, field
from typing import Literal
import networkx as nx
import msprime

def _eur_demography() -> msprime.Demography:
    d = msprime.Demography()
    d.add_population(name="EUR", initial_size=512_000)
    d.add_population_parameters_change(time=150,  population="EUR", initial_size=512_000, growth_rate=0.0)
    d.add_population_parameters_change(time=700,  population="EUR", initial_size=9_600)
    d.add_population_parameters_change(time=1_500, population="EUR", initial_size=2_000)
    d.add_population_parameters_change(time=5_000, population="EUR", initial_size=14_474)
    return d

def simulate_msprime_haplotypes(
    n_individuals: int = 10_000,
    region_bp: int = 200_000,
    mu: float = 1.5e-8,
    r: float = 1e-8,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    demography = _eur_demography()
    ts = msprime.sim_ancestry(
        samples=n_individuals,
        demography=demography,
        sequence_length=region_bp,
        recombination_rate=r,
        random_seed=seed,
    )
    ts = msprime.sim_mutations(ts, rate=mu, random_seed=seed + 1, model="jc69")
    G_hap = ts.genotype_matrix().T.astype(np.int8)
    G_raw = G_hap[0::2] + G_hap[1::2]
    allele_freq = G_raw.mean(axis=0) / 2.0
    mafs = np.minimum(allele_freq, 1.0 - allele_freq)
    return G_raw, mafs

def _standardise(G_raw: np.ndarray, mafs: np.ndarray) -> np.ndarray:
    freq  = G_raw.mean(axis=0) / 2.0
    mu_g  = 2.0 * freq
    sd_g  = np.sqrt(2.0 * freq * (1.0 - freq))
    sd_g  = np.where(sd_g < 1e-8, 1.0, sd_g)
    return (G_raw.astype(np.float64) - mu_g) / sd_g

def subsample_variants_stratified(
    G_raw: np.ndarray,
    mafs: np.ndarray,
    n_common_target: int = 500,
    n_rare_target: int = 500,
    maf_common_min: float = 0.01,
    maf_min_abs: float = 1e-4,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    common_pool = np.where(mafs >= maf_common_min)[0]
    rare_pool   = np.where((mafs >= maf_min_abs) & (mafs < maf_common_min))[0]

    n_common = min(n_common_target, len(common_pool))
    n_rare   = min(n_rare_target,   len(rare_pool))

    if n_common < n_common_target:
        warnings.warn(
            f"Only {n_common} common SNPs available (wanted {n_common_target}).",
            RuntimeWarning, stacklevel=3,
        )

    sel_common = rng.choice(common_pool, size=n_common, replace=False) if n_common > 0 else np.array([], dtype=int)
    sel_rare   = rng.choice(rare_pool,   size=n_rare,   replace=False) if n_rare   > 0 else np.array([], dtype=int)

    keep = np.sort(np.concatenate([sel_common, sel_rare]))
    m    = mafs[keep]
    return G_raw[:, keep], m, m < maf_common_min, m >= maf_common_min


def evaluate_gwas_locus(
    selected: np.ndarray,
    causal_idx: np.ndarray,
    G_std: np.ndarray,
    n_snps: int,
    r2_thresh: float = 0.1,
) -> dict:
    sel_idx    = np.where(selected)[0]
    causal_set = set(int(c) for c in causal_idx)
    n_causal   = len(causal_set)
    n_sel      = len(sel_idx)

    causal_loci_hit = set() 
    fp = 0

    for s in sel_idx:
        s = int(s)
        if s in causal_set:
            causal_loci_hit.add(s)
        else:
            best_c = None
            for c in causal_idx:
                r2 = float(np.corrcoef(G_std[:, s], G_std[:, int(c)])[0, 1]) ** 2
                if r2 > r2_thresh:
                    best_c = int(c)
                    break
            if best_c is not None:
                causal_loci_hit.add(best_c)
            else:
                fp += 1

    n_hit  = len(causal_loci_hit)
    fn     = n_causal - n_hit
    tn     = n_snps - n_sel - fn         # rough lower bound
    power  = n_hit / max(n_causal, 1)
    fdr    = fp   / max(n_sel, 1)
    fpr    = fp   / max(tn + fp, 1)
    f1     = 2 * n_hit / max(2 * n_hit + fp + fn, 1)

    return dict(power=power, fdr=fdr, fpr=fpr, f1=f1,
                tp=n_hit, fp=fp, fn=fn, tn=tn, n_selected=n_sel)


def simulate_msprime_gwas_data(
    n_individuals: int = 10_000,
    n_snps: int = 1_000,
    n_causal: int = 20,
    h2: float = 0.3,
    binary_trait: bool = False,
    prevalence: float = 0.3,
    seed: int = 42,
    maf_rare_threshold: float = 0.01,
    n_common_target: int | None = None,
    n_rare_target:   int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)

    n_common = n_common_target if n_common_target is not None else n_snps // 2
    n_rare   = n_rare_target   if n_rare_target   is not None else n_snps - n_common

    G_raw, mafs_all = simulate_msprime_haplotypes(n_individuals=n_individuals, seed=seed)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        G_raw, mafs, rare_mask, common_mask = subsample_variants_stratified(
            G_raw, mafs_all,
            n_common_target=n_common,
            n_rare_target=n_rare,
            maf_common_min=maf_rare_threshold,
            seed=seed,
        )
    M = G_raw.shape[1]
    G_std = _standardise(G_raw, mafs)

    rare_idx   = np.where(rare_mask)[0]
    common_idx = np.where(common_mask)[0]
    n_causal_common = min(n_causal // 2, len(common_idx))
    n_causal_rare   = min(n_causal - n_causal_common, len(rare_idx))
    n_causal_common = n_causal - n_causal_rare

    causal_common = rng.choice(common_idx, size=n_causal_common, replace=False) if n_causal_common > 0 else np.array([], dtype=int)
    causal_rare   = rng.choice(rare_idx,   size=n_causal_rare,   replace=False) if n_causal_rare   > 0 else np.array([], dtype=int)
    causal_idx_arr = np.concatenate([causal_common, causal_rare]).astype(int)

    beta = np.zeros(M)
    per_snp_var = h2 / max(len(causal_idx_arr), 1)
    beta[causal_idx_arr] = rng.normal(0, np.sqrt(per_snp_var), size=len(causal_idx_arr))
    linear_pred = G_std @ beta

    if binary_trait:
        prob = 1.0 / (1.0 + np.exp(-linear_pred))
        prob = np.clip(prob * prevalence / max(prob.mean(), 1e-9), 0, 1)
        y = rng.binomial(1, prob).astype(float)
    else:
        y = linear_pred + rng.normal(0, np.sqrt(max(1.0 - h2, 1e-6)), size=n_individuals)

    return dict(G_std=G_std, y=y, beta=beta, causal_idx=causal_idx_arr,
                mafs=mafs, rare_mask=rare_mask, common_mask=common_mask)


def split_data_across_centers(
    data: dict,
    n_centers: int,
    mode: Literal["random", "stratified", "weigthed"] = "random",
    seed: int = 0,
    weights: np.ndarray = None,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    n = data["G_std"].shape[0]
    y = data["y"]

    if mode == "random":
        perm = rng.permutation(n)
    elif mode == "stratified":
        order = np.argsort(y)
        # Interleave: center k gets rows k, k+n_centers, k+2*n_centers, ...
        perm = order
    elif mode == "weighted":
        perm = rng.choice(n, size=n, replace=False, p=weights)
    else:
        raise ValueError(f"Unknown mode: {mode}")
        order = np.argsort(y)
        # Interleave: center k gets rows k, k+n_centers, k+2*n_centers, ...
        perm = order

    chunks = np.array_split(perm, n_centers)
    centers = []
    for idx in chunks:
        centers.append({
            "G_std": data["G_std"][idx],
            "y":     data["y"][idx],
            "beta":  data["beta"],
            "causal_idx": data["causal_idx"],
            "mafs":  data["mafs"],
        })
    return centers


def score_stats_precompute(y: np.ndarray, binary: bool = False) -> dict:
    y = np.asarray(y, dtype=np.float64)
    n = int(y.shape[0])
    if not binary:
        y_centered = y - y.mean()
        sigma2 = float(np.var(y_centered))
        if sigma2 <= 0.0:
            sigma2 = 1e-12
        return {"binary": False, "n": n, "y_centered": y_centered, "sigma2": sigma2}
    mu0 = float(y.mean())
    w = mu0 * (1.0 - mu0)
    if w <= 0.0:
        w = 1e-12
    resid = y - mu0
    return {"binary": True, "n": n, "resid": resid, "w": w}


def _score_stats_from_precomputed(G_std: np.ndarray, pre: dict) -> np.ndarray:
    n, _M = G_std.shape
    if n != pre["n"]:
        raise ValueError("G_std row count does not match precomputed n")
    if not pre["binary"]:
        dot = G_std.T @ pre["y_centered"]
        score = (dot ** 2) / (n * pre["sigma2"])
    else:
        dot = G_std.T @ pre["resid"]
        score = (dot ** 2) / (n * pre["w"])
    return score / 2.0


def compute_score_stats(
    G_std: np.ndarray,
    y: np.ndarray,
    binary: bool = False,
    *,
    precomputed: dict | None = None,
) -> np.ndarray:
    if precomputed is None:
        precomputed = score_stats_precompute(y, binary)
    return _score_stats_from_precomputed(G_std, precomputed)


def sensitivity_score_stat(n_i: int, binary: bool = False, n_snps: int = 1, n_centers: int = 1) -> float:
    lambda_max = 32.7
    return 3 * np.log(n_i) * lambda_max / n_i

def log_belief_init(log_llr: np.ndarray) -> np.ndarray:
    M = len(log_llr)
    lb = np.stack([np.zeros(M), log_llr], axis=1)   # (M, 2)
    lse = logsumexp(lb, axis=1, keepdims=True)       # (M, 1)
    return lb - lse


def laplace_noise_log_belief(
    log_belief: np.ndarray,
    sensitivity: float,
    epsilon: float,
    n_states: int,
    K: int,
    rng: np.random.Generator,
) -> np.ndarray:
    scale = sensitivity * K * n_states / epsilon
    noise = rng.laplace(loc=0.0, scale=scale, size=log_belief.shape)
    lb_noisy = log_belief + noise
    # Re-normalise
    lse = logsumexp(lb_noisy, axis=1, keepdims=True)
    return lb_noisy - lse


def _log_linear_update_tensor(log_belief_tensor: np.ndarray, A: np.ndarray) -> np.ndarray:
    n, M, S = log_belief_tensor.shape
    flat = log_belief_tensor.reshape(n, M * S)
    log_nu_raw = (A @ flat).reshape(n, M, S)
    log_nu_raw = np.clip(log_nu_raw, -700.0, 700.0)
    lse = logsumexp(log_nu_raw, axis=2, keepdims=True)
    return log_nu_raw - lse


def log_linear_update_all(log_belief_tensor: np.ndarray, A: np.ndarray) -> np.ndarray:
    return _log_linear_update_tensor(log_belief_tensor, A)


def log_linear_update(
    log_beliefs: list[np.ndarray],
    A: np.ndarray,
    center_idx: int,
) -> np.ndarray:
    L = np.stack(log_beliefs, axis=0)
    out = _log_linear_update_tensor(L, A)
    return np.ascontiguousarray(out[center_idx])


# ---------------------------------------------------------------------------
# Communication matrix constructors
# ---------------------------------------------------------------------------

def make_adjacency(n: int, topology: str = "complete", seed: int = 0, add_I=False, **kwargs) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = np.zeros((n, n))

    if topology == "complete":
        A = np.ones((n, n)) / n

    elif topology == "ring":
        for i in range(n):
            A[i, i] = 1 / 3
            A[i, (i - 1) % n] = 1 / 3
            A[i, (i + 1) % n] = 1 / 3

    elif topology == "star":
        # Hub = node 0
        A[0, :] = 1 / n
        A[:, 0] = 1 / n
        for i in range(1, n):
            A[i, i] = 1 - 1 / n
        # Make doubly stochastic (Sinkhorn)
        A = _sinkhorn(A)
    elif topology == 'star':
        G = nx.star_graph(n)
        A = nx.to_numpy_array(G)
        A = _metropolis_hastings(A)
    elif topology == 'scale-free':
        G = nx.barabasi_albert_graph(n, m=1)
        A = nx.to_numpy_array(G)
        A = _metropolis_hastings(A)
    elif topology == 'small-world':
        G = nx.watts_strogatz_graph(n, k=4, p=0.1)
        A = nx.to_numpy_array(G)
        A = _metropolis_hastings(A)

    elif topology == "random":
        # Erdos-Renyi with p = 0.5, then Metropolis-Hastings weights
        connected = False
        while not connected:
            adj = rng.random((n, n)) < 0.5
            adj = adj | adj.T
            np.fill_diagonal(adj, True)
            # Check connectivity via powers of adjacency
            reachable = np.linalg.matrix_power(adj.astype(float), n)
            connected = np.all(reachable > 0)
        A = _metropolis_hastings(adj)
    elif topology in ['knn', 'county', 'organization']:
        hospitals = kwargs.get('hospitals', None)
        if hospitals is None:
            raise ValueError("hospitals is required for knn, county, and organization topologies")
        A = build_geographic_adjacency(method=topology, **kwargs)
    else:
        raise ValueError(f"Unknown topology: {topology}")
    if add_I:
        A = A + np.eye(n)

    return A


def _metropolis_hastings(adj: np.ndarray) -> np.ndarray:
    n = adj.shape[0]
    degree = adj.sum(axis=1)
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and adj[i, j]:
                A[i, j] = 1.0 / (1 + max(degree[i], degree[j]))
        A[i, i] = 1.0 - A[i].sum()
    return A


def _sinkhorn(A: np.ndarray, n_iter: int = 1000, tol: float = 1e-9) -> np.ndarray:
    A = A.copy()
    for _ in range(n_iter):
        A /= A.sum(axis=1, keepdims=True)
        A /= A.sum(axis=0, keepdims=True)
        if np.max(np.abs(A.sum(axis=1) - 1)) < tol:
            break
    return A


def build_geographic_adjacency(
    method: str = "knn",
    **kwargs
) -> np.ndarray:

    hospitals = kwargs.get('hospitals', None)
    if hospitals is None:
        raise ValueError("hospitals is required for knn, county, and organization topologies")

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

    elif method == "organization":
        if "ORGANIZATION" not in hospitals.columns:
            raise ValueError(
                "method='organization' requires an ORGANIZATION column; "
                "use load_nyc_hospitals(..., organizations_csv=...)"
            )
        org = hospitals["ORGANIZATION"].astype(str).values
        adj = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(n):
                if org[i] == org[j]:
                    adj[i, j] = True
        return _metropolis_hastings_bool(adj)
    else:  # knn
        k_neighbors = kwargs.get('k_neighbors', 3)
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
    n = adj.shape[0]
    degree = adj.sum(axis=1).astype(float)
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and adj[i, j]:
                A[i, j] = 1.0 / (1.0 + max(degree[i], degree[j]))
        A[i, i] = 1.0 - A[i].sum()
    return A


def spectral_gap(A: np.ndarray) -> float:
    eigvals = np.sort(np.abs(np.linalg.eigvals(A)))[::-1]
    return float(1.0 - eigvals[1]) if len(eigvals) > 1 else 1.0

@dataclass
class DPGWASResult:
    log_beliefs_gm: np.ndarray
    log_beliefs_am: np.ndarray
    selected_gm: np.ndarray
    selected_am: np.ndarray
    pvalues_gm: np.ndarray
    pvalues_am: np.ndarray
    belief_trace: list
    converged_at: int


def run_dp_gwas_mle(
    centers_data: list[dict],
    epsilon: float,
    alpha: float = 5e-8,
    beta_power: float = 0.8,
    T: int | None = None,
    K: int = 20,
    topology: str = "complete",
    binary_trait: bool = False,
    track_convergence: bool = True,
    convergence_tol: float = 1e-4,
    seed: int = 0,
    snp_chunk_size: int | None = None,
    convergence_tv_max_snps: int | None = None,
    mode: str = "mle",
    **kwargs
) -> DPGWASResult:
    n_centers = len(centers_data)
    M = centers_data[0]["G_std"].shape[1]
    n_states = 2

    threshold_am = kwargs.get('threshold_am', 0.5)
    threshold_gm = kwargs.get('threshold_gm', 0.5)

    A = make_adjacency(n_centers, topology=topology, seed=seed, add_I=True, **kwargs)

    if T is None:
        slem = 1.0 - spectral_gap(A)
        slem = float(np.clip(slem, 1e-9, 1.0 - 1e-9))
        T = max(50, int(np.ceil(np.log(2.0 / max(float(alpha), 1e-15)) / np.log(1.0 / slem))))
        T = min(T, 500)

    min_ni = min(c["G_std"].shape[0] for c in centers_data)
    
    sensitivities = [
        sensitivity_score_stat(min_ni, binary=binary_trait, n_snps=M, n_centers=n_centers)
        for c in centers_data
    ]

    pres = [score_stats_precompute(c["y"], binary_trait) for c in centers_data]

    chunk_size = M if snp_chunk_size is None else int(snp_chunk_size)
    if chunk_size < 1:
        raise ValueError("snp_chunk_size must be >= 1")
    chunk_size = min(chunk_size, M)

    belief_trace: list[float] = []
    converged_at = T * K
    noisy_stat_sum = np.zeros(M, dtype=np.float64)
    gm_accum = np.zeros((M, n_states), dtype=np.float64)
    log_prob_am_merged = np.empty((M, n_states), dtype=np.float64)

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
                belief_tensor = _log_linear_update_tensor(belief_tensor, A)

                if track_convergence and k == 0 and j0 == 0:
                    n_tv = B if convergence_tv_max_snps is None else min(
                        B, int(convergence_tv_max_snps)
                    )
                    lbr = belief_tensor[:, :n_tv, 1] - belief_tensor[:, :n_tv, 0]
                    tv = float(np.std(lbr))
                    belief_trace.append(tv)
                    if tv < convergence_tol:
                        converged_at = min(converged_at, t)

            gm_accum[sl] += belief_tensor.mean(axis=0)
            ls_round = logsumexp(belief_tensor, axis=0)
            if am_chunk is None:
                am_chunk = ls_round
            else:
                am_chunk = logsumexp(
                    np.stack([am_chunk, ls_round], axis=0), axis=0
                )

        log_prob_am_merged[sl] = am_chunk

    # -----------------------------------------------------------------------
    # Stage 1: calibrated p-values
    # Under H0: 2 * sum_i Lambda_i ~ chi2_{n_centers}
    # -----------------------------------------------------------------------
    agg_stat = noisy_stat_sum / float(K)
    chi2_agg = np.clip(2.0 * agg_stat, 0.0, None)
    pvalues = chi2.sf(chi2_agg, df=n_centers)

    # -----------------------------------------------------------------------
    # Stage 2: GM and AM belief aggregation
    # -----------------------------------------------------------------------
    log_beliefs_gm = gm_accum / float(K)
    lse = logsumexp(log_beliefs_gm, axis=1, keepdims=True)
    log_beliefs_gm = log_beliefs_gm - lse
    lbr_gm = log_beliefs_gm[:, 1] - log_beliefs_gm[:, 0]

    log_prob_am = log_prob_am_merged - np.log(K * n_centers)
    lse2 = logsumexp(log_prob_am, axis=1, keepdims=True)
    log_prob_am = log_prob_am - lse2
    lbr_am = log_prob_am[:, 1] - log_prob_am[:, 0]
    
    chi2_am = np.clip(2.0 * lbr_am, 0.0, None)

    posterior_gm = np.exp(log_beliefs_gm[:, 1])
    posterior_am = np.exp(log_prob_am[:, 1])

    if mode == 'hypothesis_testing':
        # we discard am completely and use gm only
        selected_gm = pvalues < alpha
        selected_am = np.zeros_like(selected_gm)
    elif mode == 'mle':
        selected_am = posterior_am > threshold_am
        selected_gm = posterior_gm > threshold_gm

    return DPGWASResult(
        log_beliefs_gm=lbr_gm,
        log_beliefs_am=lbr_am,
        selected_gm=selected_gm,
        selected_am=selected_am,
        pvalues_gm=pvalues,
        pvalues_am=None,
        belief_trace=belief_trace,
        converged_at=converged_at,
    )

def evaluate_gwas(
    selected: np.ndarray,
    causal_idx: np.ndarray,
    n_snps: int,
    alpha: float = 5e-8,
) -> dict:
    causal_set = set(causal_idx.tolist())
    selected_set = set(np.where(selected)[0].tolist())

    tp = len(causal_set & selected_set)
    fp = len(selected_set - causal_set)
    fn = len(causal_set - selected_set)
    tn = n_snps - tp - fp - fn

    power = tp / max(len(causal_set), 1)
    fdr   = fp / max(len(selected_set), 1)
    fpr   = fp / max(tn + fp, 1)
    f1    = 2 * tp / max(2 * tp + fp + fn, 1)

    return dict(
        power=power, fdr=fdr, fpr=fpr, f1=f1,
        tp=tp, fp=fp, fn=fn, tn=tn,
        n_selected=len(selected_set),
    )


def centralized_gwas(
    centers_data: list[dict],
    alpha: float = 5e-8,
    binary_trait: bool = False,
) -> dict:
    G_all = np.vstack([c["G_std"] for c in centers_data])
    y_all = np.concatenate([c["y"] for c in centers_data])
    causal_idx = centers_data[0]["causal_idx"]
    n_snps = G_all.shape[1]

    log_llr = compute_score_stats(G_all, y_all, binary=binary_trait)
    n_total = G_all.shape[0]
    chi2_stat = np.clip(2.0 * log_llr, 0, None)
    pvalues = chi2.sf(chi2_stat, df=1)
    selected = pvalues < alpha

    metrics = evaluate_gwas(selected, causal_idx, n_snps, alpha)
    return dict(selected=selected, pvalues=pvalues, log_llr=log_llr, **metrics)


def single_center_gwas(
    centers_data: list[dict],
    center_idx: int = 0,
    alpha: float = 5e-8,
    binary_trait: bool = False,
) -> dict:
    c = centers_data[center_idx]
    causal_idx = c["causal_idx"]
    n_snps = c["G_std"].shape[1]

    log_llr = compute_score_stats(c["G_std"], c["y"], binary=binary_trait)
    chi2_stat = np.clip(2.0 * log_llr, 0, None)
    pvalues = chi2.sf(chi2_stat, df=1)
    selected = pvalues < alpha

    metrics = evaluate_gwas(selected, causal_idx, n_snps, alpha)
    return dict(selected=selected, pvalues=pvalues, log_llr=log_llr, **metrics)

def run_rizk_baseline(
    centers_data: list[dict],
    epsilon: float,
    T: int = 200,
    alpha: float = 5e-8,
    binary_trait: bool = False,
    lr: float = 0.01,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    n_centers = len(centers_data)
    M = centers_data[0]["G_std"].shape[1]
    causal_idx = centers_data[0]["causal_idx"]
    n_snps = M

    A = make_adjacency(n_centers, topology="complete", seed=seed)

    # Sensitivity of clipped gradient per SNP: 2 * B_theta * B_x / n_i
    B_theta = 1.0
    B_x = 1.0   # standardised genotypes
    # Noise scale: 2 * B_x * B_theta * T * lr / epsilon (from Rizk et al. composition)
    noise_scale = 2.0 * B_x * B_theta * T * lr / epsilon

    # Initialise: theta_i = 0 for all centers
    theta = np.zeros((n_centers, M))

    for t in range(T):
        grad = np.zeros((n_centers, M))
        for i, cdata in enumerate(centers_data):
            G, y = cdata["G_std"], cdata["y"]
            n_i = G.shape[0]
            y_c = y - y.mean()
            # Gradient of (1/n_i) sum log-likelihood w.r.t. theta_j at current theta_i
            pred = G @ theta[i]
            residual = y_c - pred
            g = (G.T @ residual) / n_i   # (M,)
            # Clip gradient L1 norm
            g_clipped = np.clip(g, -B_theta * B_x, B_theta * B_x)
            noise = rng.laplace(0, noise_scale, size=M)
            grad[i] = g_clipped + noise

        # Consensus + gradient step
        theta = A @ theta + lr * grad

    # Average over centers
    theta_avg = theta.mean(axis=0)   # (M,)
    n_total = sum(c["G_std"].shape[0] for c in centers_data)

    # Approximate p-values via Wald test
    se = noise_scale / np.sqrt(n_total)
    z = theta_avg / max(se, 1e-10)
    pvalues = 2 * norm.sf(np.abs(z))
    selected = pvalues < alpha

    metrics = evaluate_gwas(selected, causal_idx, n_snps, alpha)
    return dict(selected=selected, pvalues=pvalues, theta=theta_avg, **metrics)
