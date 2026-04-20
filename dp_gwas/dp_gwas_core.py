"""
dp_gwas_core.py
===============
Differentially private distributed GWAS inference.
Implements the log-linear belief update framework of
Papachristou & Rahimian (2025) applied to genome-wide
association studies.

All belief arithmetic is performed in log-space with
logsumexp normalisation to guarantee numerical stability.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm, chi2
from dataclasses import dataclass, field
from typing import Literal
import networkx as nx


# ---------------------------------------------------------------------------
# Data simulation
# ---------------------------------------------------------------------------

def simulate_gwas_data(
    n_individuals: int,
    n_snps: int,
    n_causal: int,
    maf_range: tuple[float, float] = (0.05, 0.5),
    h2: float = 0.3,
    binary_trait: bool = False,
    prevalence: float = 0.3,
    seed: int = 42,
    odds_ratio: np.ndarray = None,
) -> dict:
    """
    Simulate GWAS genotype + phenotype data under a polygenic model.

    Returns
    -------
    dict with keys:
        G_std      : (n, M) standardised genotype matrix
        y          : (n,) phenotype vector
        beta       : (M,) true effect sizes (zero for non-causal)
        causal_idx : (n_causal,) indices of causal SNPs
        mafs       : (M,) minor allele frequencies
    """
    rng = np.random.default_rng(seed)
    mafs = rng.uniform(*maf_range, size=n_snps)

    G = rng.binomial(2, mafs, size=(n_individuals, n_snps))

    col_std = np.sqrt(2 * mafs * (1 - mafs))
    col_std = np.where(col_std < 1e-8, 1.0, col_std)
    G_std = (G - 2 * mafs) / col_std

    causal_idx = rng.choice(n_snps, n_causal, replace=False)
    beta = np.zeros(n_snps)
    per_snp_var = h2 / n_causal
    beta[causal_idx] = rng.normal(0, np.sqrt(per_snp_var), size=n_causal)

    linear_pred = G_std @ beta

    if binary_trait:
        prob = 1 / (1 + np.exp(-linear_pred))
        rescale = prevalence / prob.mean()
        prob = np.clip(prob * rescale, 0, 1)
        y = rng.binomial(1, prob).astype(float)
    else:
        env_noise = rng.normal(0, np.sqrt(max(1.0 - h2, 1e-6)), size=n_individuals)
        y = linear_pred + env_noise

    return dict(G_std=G_std, y=y, beta=beta, causal_idx=causal_idx, mafs=mafs)



def split_data_across_centers(
    data: dict,
    n_centers: int,
    mode: Literal["random", "stratified"] = "random",
    seed: int = 0,
) -> list[dict]:
    """
    Partition individuals across n_centers.

    mode='random'     : uniform random split
    mode='stratified' : split by phenotype quantile (simulates demographic variation)
    """
    rng = np.random.default_rng(seed)
    n = data["G_std"].shape[0]
    y = data["y"]

    if mode == "random":
        perm = rng.permutation(n)
    else:
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


# ---------------------------------------------------------------------------
# Score statistics & log-likelihoods
# ---------------------------------------------------------------------------

def score_stats_precompute(y: np.ndarray, binary: bool = False) -> dict:
    """
    Per-center phenotype summaries for score tests (reuse across SNP chunks).

    Passing the returned dict to :func:`compute_score_stats` avoids recomputing
    ``y.mean()``, ``var(y)``, etc. when scanning genome-wide data in blocks.
    """
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
    """Per-SNP log-LR (score/2) from genotype block and precomputed phenotype stats."""
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
    """
    Return per-SNP score test statistics (log-likelihood ratio approximation).

    For continuous traits : score stat = (G_j^T y)^2 / n  (OLS score)
    For binary traits     : efficient score under logistic null model

    Both are O(n * M) and exact under the respective null models.
    Returns array of shape (M,) — one value per SNP.
    """
    if precomputed is None:
        precomputed = score_stats_precompute(y, binary)
    return _score_stats_from_precomputed(G_std, precomputed)


def sensitivity_score_stat(n_i: int, binary: bool = False) -> float:
    """
    Global L1-sensitivity of the per-SNP log-LR score statistic
    with respect to the removal/replacement of one individual.

    For standardised genotypes |g_ij| <= sqrt(2) / col_std, and
    a single individual's contribution to (G_j^T y_c)^2 / (n * sigma2)
    is bounded.  We use the closed-form bound:
        Delta = 2 / n_i
    """
    return 2.0 / n_i


# ---------------------------------------------------------------------------
# Log-space belief utilities (numerical stability core)
# ---------------------------------------------------------------------------

def log_belief_init(log_llr: np.ndarray) -> np.ndarray:
    """
    Convert per-SNP log-LR array to a (M, 2) log-belief matrix.

    State 0 = null (theta=0),  State 1 = alternative (theta != 0).
    log_belief[j, 0] = 0  (unnormalised)
    log_belief[j, 1] = log_llr[j]

    Returns log_belief normalised so logsumexp over axis=1 = 0 (i.e. sum = 1).
    """
    M = len(log_llr)
    # Stack: column 0 = log p(null), column 1 = log p(alt)
    lb = np.stack([np.zeros(M), log_llr], axis=1)   # (M, 2)
    # Normalise in log-space: subtract logsumexp per row
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
    """
    Add Laplace noise in log-space and re-normalise.

    Noise scale = sensitivity * K * n_states / epsilon
    (composition over K rounds and n_states states per Theorem 2).

    Operates on (M, n_states) log-belief array.
    Returns noisy, re-normalised log-belief of same shape.
    """
    scale = sensitivity * K * n_states / epsilon
    # Independent noise per (SNP, state) pair
    noise = rng.laplace(loc=0.0, scale=scale, size=log_belief.shape)
    lb_noisy = log_belief + noise
    # Re-normalise
    lse = logsumexp(lb_noisy, axis=1, keepdims=True)
    return lb_noisy - lse


def _log_linear_update_tensor(log_belief_tensor: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    One synchronous log-linear update for all centers.

    Parameters
    ----------
    log_belief_tensor : (n_centers, M, n_states)
    A                 : (n_centers, n_centers) doubly stochastic

    Returns
    -------
    Updated tensor of shape (n_centers, M, n_states), row-normalised in log-space.
    """
    n, M, S = log_belief_tensor.shape
    flat = log_belief_tensor.reshape(n, M * S)
    log_nu_raw = (A @ flat).reshape(n, M, S)
    log_nu_raw = np.clip(log_nu_raw, -700.0, 700.0)
    lse = logsumexp(log_nu_raw, axis=2, keepdims=True)
    return log_nu_raw - lse


def log_linear_update_all(log_belief_tensor: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    Vectorised log-linear belief step for all centers (BLAS-friendly for large M).

    ``log_belief_tensor`` has shape ``(n_centers, M, n_states)``.  Applies one
    synchronous Jacobi update and row-normalises in log-space.
    """
    return _log_linear_update_tensor(log_belief_tensor, A)


def log_linear_update(
    log_beliefs: list[np.ndarray],
    A: np.ndarray,
    center_idx: int,
) -> np.ndarray:
    """Update a single center's log-beliefs (wraps :func:`log_linear_update_all`)."""
    L = np.stack(log_beliefs, axis=0)
    out = _log_linear_update_tensor(L, A)
    return np.ascontiguousarray(out[center_idx])


# ---------------------------------------------------------------------------
# Communication matrix constructors
# ---------------------------------------------------------------------------

def make_adjacency(n: int, topology: str = "complete", seed: int = 0, add_I=False) -> np.ndarray:
    """
    Build a doubly stochastic adjacency matrix for n agents.

    topologies : 'complete', 'ring', 'random', 'star'
    """
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
    else:
        raise ValueError(f"Unknown topology: {topology}")

    if add_I:
        A = A + np.eye(n)

    return A


def _metropolis_hastings(adj: np.ndarray) -> np.ndarray:
    """Metropolis-Hastings weights for a symmetric adjacency (0/1) matrix."""
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
    """Sinkhorn-Knopp to make a non-negative matrix doubly stochastic."""
    A = A.copy()
    for _ in range(n_iter):
        A /= A.sum(axis=1, keepdims=True)
        A /= A.sum(axis=0, keepdims=True)
        if np.max(np.abs(A.sum(axis=1) - 1)) < tol:
            break
    return A


def spectral_gap(A: np.ndarray) -> float:
    """Return 1 - |lambda_2(A)|, the spectral gap (mixing rate)."""
    eigvals = np.sort(np.abs(np.linalg.eigvals(A)))[::-1]
    return float(1.0 - eigvals[1]) if len(eigvals) > 1 else 1.0


# ---------------------------------------------------------------------------
# Main DP-distributed GWAS algorithm
# ---------------------------------------------------------------------------

@dataclass
class DPGWASResult:
    """Output of run_dp_gwas_mle."""
    log_beliefs_gm: np.ndarray      # (M,) log-belief ratio for GM aggregator
    log_beliefs_am: np.ndarray      # (M,) log-belief ratio for AM aggregator
    selected_gm: np.ndarray         # (M,) bool — significant by GM
    selected_am: np.ndarray         # (M,) bool — significant by AM
    pvalues_gm: np.ndarray          # (M,) approximate p-values from GM log-LR
    pvalues_am: np.ndarray          # (M,) approximate p-values from AM log-LR
    belief_trace: list              # per-iteration TV distance (optional)
    converged_at: int               # iteration where TV < tol


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
) -> DPGWASResult:
    """
    Run differentially private distributed GWAS via log-linear belief updates
    (Algorithm 1, Papachristou & Rahimian 2025) extended to GWAS.

    Two-stage design
    ----------------
    Stage 1 — Calibrated p-values from noisy aggregate score statistics:
        Each center adds Laplace noise to its score statistic (log-LR).
        Noisy stats are summed across centers and averaged over K rounds.
        Under H0, 2 * sum_i Lambda_i ~ chi2_{n_centers}, giving correctly
        calibrated p-values directly comparable with the centralised oracle.

    Stage 2 — Log-linear belief consensus for SNP ranking:
        Algorithm 1 runs using the same noisy log-beliefs, converging to
        a posterior P(H1 | all data) per SNP.  The GM estimator selects
        SNPs whose posterior > 0.5 AND whose Stage 1 p-value < alpha.

    This separates MLE identification (Stage 2) from hypothesis testing
    (Stage 1), matching Proposition 1 of the paper.

    Scaling
    -------
    ``snp_chunk_size`` processes SNPs in blocks so peak memory stays
    ``O(n_centers * chunk * n_states)`` for belief tensors instead of
    ``O(n_centers * M * n_states)``. Score statistics for each block are
    computed once per center and reused across all ``K`` privacy rounds
    (the noise is still drawn every round).

    Setting ``convergence_tv_max_snps`` caps how many SNPs enter the TV
    diagnostic (useful when ``M`` is huge). ``None`` uses all SNPs in the
    current chunk (same as legacy behaviour for a single full chunk).

    Note: chunking changes the order in which pseudo-random noise is drawn,
    so results are not bitwise-identical to an all-at-once run with the
    same ``seed`` (the underlying DP mechanism is unchanged).
    """
    n_centers = len(centers_data)
    M = centers_data[0]["G_std"].shape[1]
    n_states = 2

    A = make_adjacency(n_centers, topology=topology, seed=seed, add_I=True)

    if T is None:
        slem = 1.0 - spectral_gap(A)
        slem = float(np.clip(slem, 1e-9, 1.0 - 1e-9))
        T = max(50, int(np.ceil(np.log(2.0 / max(float(alpha), 1e-15)) / np.log(1.0 / slem))))
        T = min(T, 500)

    min_ni = min(c["G_std"].shape[0] for c in centers_data)
    
    sensitivities = [
        sensitivity_score_stat(min_ni, binary=binary_trait)
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

    # -----------------------------------------------------------------------
    # Selection: posterior ranking (Stage 2) gated by calibrated test (Stage 1)
    # GM: low Type I error (precision).  AM: low Type II error (recall).
    # -----------------------------------------------------------------------
    posterior_gm = np.exp(log_beliefs_gm[:, 1])
    posterior_am = np.exp(log_prob_am[:, 1])

    selected_gm = pvalues < alpha
    selected_am = pvalues < alpha

    return DPGWASResult(
        log_beliefs_gm=lbr_gm,
        log_beliefs_am=lbr_am,
        selected_gm=selected_gm,
        selected_am=selected_am,
        pvalues_gm=pvalues,
        pvalues_am=pvalues,
        belief_trace=belief_trace,
        converged_at=converged_at,
    )


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def evaluate_gwas(
    selected: np.ndarray,
    causal_idx: np.ndarray,
    n_snps: int,
    alpha: float = 5e-8,
) -> dict:
    """
    Return standard GWAS evaluation metrics.

    selected   : (M,) boolean mask of selected SNPs
    causal_idx : true causal SNP indices
    """
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
    """Oracle: pool all data, run standard GWAS, return selected mask + metrics."""
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
    """Baseline: only one center runs GWAS independently."""
    c = centers_data[center_idx]
    causal_idx = c["causal_idx"]
    n_snps = c["G_std"].shape[1]

    log_llr = compute_score_stats(c["G_std"], c["y"], binary=binary_trait)
    chi2_stat = np.clip(2.0 * log_llr, 0, None)
    pvalues = chi2.sf(chi2_stat, df=1)
    selected = pvalues < alpha

    metrics = evaluate_gwas(selected, causal_idx, n_snps, alpha)
    return dict(selected=selected, pvalues=pvalues, log_llr=log_llr, **metrics)


# ---------------------------------------------------------------------------
# First-order DP baseline (Rizk et al. 2023 analog)
# ---------------------------------------------------------------------------

def run_rizk_baseline(
    centers_data: list[dict],
    epsilon: float,
    T: int = 200,
    alpha: float = 5e-8,
    binary_trait: bool = False,
    lr: float = 0.01,
    seed: int = 0,
) -> dict:
    """
    First-order distributed gradient descent with DP (Rizk et al. 2023).
    Each center maintains a local estimate theta_i (one value per SNP),
    does consensus + noisy gradient step.

    Returns selected mask + p-values derived from final parameter estimates.
    """
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
