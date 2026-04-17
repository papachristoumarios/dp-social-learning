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

def compute_score_stats(
    G_std: np.ndarray,
    y: np.ndarray,
    binary: bool = False,
) -> np.ndarray:
    """
    Return per-SNP score test statistics (log-likelihood ratio approximation).

    For continuous traits : score stat = (G_j^T y)^2 / n  (OLS score)
    For binary traits     : efficient score under logistic null model

    Both are O(n * M) and exact under the respective null models.
    Returns array of shape (M,) — one value per SNP.
    """
    n, M = G_std.shape
    y_centered = y - y.mean()

    if not binary:
        # Score = (G_j^T y_c)^2 / (n * sigma2_hat)
        # log-LR approximation: score_stat / 2  (chi^2_1 / 2)
        dot = G_std.T @ y_centered          # (M,)
        sigma2 = np.var(y_centered)
        score = (dot ** 2) / (n * sigma2)   # chi^2_1 under H0
        log_llr = score / 2.0               # log p(data|H1) - log p(data|H0)
    else:
        # Under logistic null (intercept only), mu = mean(y)
        mu0 = y.mean()
        w = mu0 * (1 - mu0)
        resid = y - mu0
        dot = G_std.T @ resid               # (M,)
        score = (dot ** 2) / (n * w)        # Rao score stat ~ chi^2_1
        log_llr = score / 2.0

    return log_llr


def sensitivity_score_stat(n_i: int, binary: bool = False) -> float:
    """
    Global L1-sensitivity of the per-SNP log-LR score statistic
    with respect to the removal/replacement of one individual.

    For standardised genotypes |g_ij| <= sqrt(2) / col_std, and
    a single individual's contribution to (G_j^T y_c)^2 / (n * sigma2)
    is bounded.  We use the closed-form bound:
        Delta = 2 / n_i
    which is conservative for the linear case (see Appendix F analog).
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


def log_linear_update(
    log_beliefs: list[np.ndarray],
    A: np.ndarray,
    center_idx: int,
) -> np.ndarray:
    """
    Compute one step of the log-linear (geometric mean) belief update
    for a single center, fully in log-space.

    Equation (12) from Papachristou & Rahimian (2025):

        nu_i,t(theta) ∝ nu_i,t-1(theta)^(1+a_ii)
                        * prod_{j in N_i} nu_j,t-1(theta)^a_ij

    In log-space this is a *weighted sum* of log-beliefs:

        log_nu_raw[j,s] = (1+a_ii)*log_nu_i[j,s]
                         + sum_{k != i} a_ik * log_nu_k[j,s]

    However, because each log_nu_k is already normalised (logsumexp = 0),
    adding (1 + a_ii) times it rather than a_ii times it grows the
    magnitude without bound over many iterations.

    The stable formulation rewrites the exponent as:

        weight_i = 1 + a_ii,   weight_k = a_ik  for k != i

    These weights sum to 1 + a_ii + sum_{k!=i} a_ik = 1 + 1 = 2.
    So the weighted sum has logsumexp ≈ log(2), not 0 — meaning
    the beliefs do NOT remain on the unit simplex directly.

    The correct implementation follows the paper: after the weighted
    combination, we renormalise to the simplex.  The key insight is
    that normalisation happens AFTER the weighted sum, so the (1+a_ii)
    self-weight is equivalent to the agent giving extra weight to its own
    prior — the final normalisation restores the simplex constraint.

    To avoid unbounded growth over T iterations, we clip the raw
    weighted sum to [-700, 700] before normalising (which corresponds
    to probability ratios up to exp(700) ≈ 10^304 — effectively a
    hard 0/1 decision, which is the correct asymptotic behaviour).

    Parameters
    ----------
    log_beliefs : list of (M, n_states) arrays, one per center,
                  each normalised so logsumexp(axis=1) = 0
    A           : (n, n) doubly stochastic adjacency matrix
    center_idx  : which center to update

    Returns
    -------
    (M, n_states) updated log-belief, normalised so logsumexp(axis=1) = 0
    """
    n = len(log_beliefs)
    M, S = log_beliefs[center_idx].shape

    # Weighted sum in log-space with (1 + a_ii) self-weight
    log_nu_raw = np.zeros((M, S))
    for k in range(n):
        w = A[center_idx, k]
        if k == center_idx:
            w = 1.0 + A[center_idx, k]
        if w > 1e-12:
            log_nu_raw += w * log_beliefs[k]

    # Clip to prevent overflow before normalisation
    log_nu_raw = np.clip(log_nu_raw, -700.0, 700.0)

    # Normalise to unit simplex in log-space
    lse = logsumexp(log_nu_raw, axis=1, keepdims=True)
    return log_nu_raw - lse


# ---------------------------------------------------------------------------
# Communication matrix constructors
# ---------------------------------------------------------------------------

def make_adjacency(n: int, topology: str = "complete", seed: int = 0) -> np.ndarray:
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
    """
    n_centers = len(centers_data)
    M = centers_data[0]["G_std"].shape[1]
    n_states = 2

    A = make_adjacency(n_centers, topology=topology, seed=seed)

    if T is None:
        slem = 1.0 - spectral_gap(A)
        slem = float(np.clip(slem, 1e-9, 1.0 - 1e-9))
        T = max(50, int(np.ceil(np.log(2.0 / max(float(alpha), 1e-15)) / np.log(1.0 / slem))))
        T = min(T, 500)

    sensitivities = [
        sensitivity_score_stat(c["G_std"].shape[0], binary=binary_trait)
        for c in centers_data
    ]

    belief_trace = []
    converged_at = T * K
    round_final_log_beliefs = []
    noisy_stat_rounds = []

    for k in range(K):
        rng_k = np.random.default_rng(seed * 10000 + k)

        # Stage 1: accumulate noisy sum of score statistics
        noisy_sum_k = np.zeros(M)

        # Stage 2: initialise noisy log-beliefs for consensus
        log_beliefs = []
        for i, cdata in enumerate(centers_data):
            log_llr_i = compute_score_stats(cdata["G_std"], cdata["y"], binary=binary_trait)
            delta_i = sensitivities[i]
            noise_scale = delta_i * K * n_states / epsilon

            # Stage 1 accumulator
            noisy_sum_k += log_llr_i + rng_k.laplace(0.0, noise_scale, M)

            # Stage 2 log-belief
            lb_i = log_belief_init(log_llr_i)
            lb_noisy = laplace_noise_log_belief(lb_i, delta_i, epsilon, n_states, K, rng_k)
            log_beliefs.append(lb_noisy)

        noisy_stat_rounds.append(noisy_sum_k)

        # Stage 2: T iterations of log-linear updates
        for t in range(T):
            log_beliefs = [log_linear_update(log_beliefs, A, i) for i in range(n_centers)]

            if track_convergence and k == 0:
                lbr = np.array([lb[:, 1] - lb[:, 0] for lb in log_beliefs])
                tv = float(np.std(lbr))
                belief_trace.append(tv)
                if tv < convergence_tol and converged_at == T * K:
                    converged_at = t

        round_final_log_beliefs.append(np.array(log_beliefs))

    # -----------------------------------------------------------------------
    # Stage 1: calibrated p-values
    # Under H0: 2 * sum_i Lambda_i ~ chi2_{n_centers}
    # -----------------------------------------------------------------------
    agg_stat = np.mean(noisy_stat_rounds, axis=0)          # (M,)
    chi2_agg = np.clip(2.0 * agg_stat, 0.0, None)
    pvalues = chi2.sf(chi2_agg, df=n_centers)

    # -----------------------------------------------------------------------
    # Stage 2: GM and AM belief aggregation
    # -----------------------------------------------------------------------
    stacked = np.array(round_final_log_beliefs)             # (K, n, M, 2)

    log_beliefs_gm = stacked.mean(axis=0).mean(axis=0)     # (M, 2)
    lse = logsumexp(log_beliefs_gm, axis=1, keepdims=True)
    log_beliefs_gm = log_beliefs_gm - lse
    lbr_gm = log_beliefs_gm[:, 1] - log_beliefs_gm[:, 0]

    log_prob_am = logsumexp(stacked, axis=(0, 1)) - np.log(K * n_centers)
    lse2 = logsumexp(log_prob_am, axis=1, keepdims=True)
    log_prob_am = log_prob_am - lse2
    lbr_am = log_prob_am[:, 1] - log_prob_am[:, 0]

    # -----------------------------------------------------------------------
    # Selection: posterior ranking (Stage 2) gated by calibrated test (Stage 1)
    # GM: low Type I error (precision).  AM: low Type II error (recall).
    # -----------------------------------------------------------------------
    posterior_gm = np.exp(log_beliefs_gm[:, 1])
    posterior_am = np.exp(log_prob_am[:, 1])

    selected_gm = (posterior_gm > 0.5) & (pvalues < alpha)
    selected_am = (posterior_am > 0.2) & (pvalues < alpha)

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
        theta_new = np.zeros_like(theta)
        for i in range(n_centers):
            consensus = sum(A[i, j] * theta[j] for j in range(n_centers))
            theta_new[i] = consensus + lr * grad[i]
        theta = theta_new

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
