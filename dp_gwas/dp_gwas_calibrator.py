import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_directory', type=str, default='../datasets/IHAC_meta')
    parser.add_argument('--output_directory', type=str, default='../datasets/IHAC_meta')
    parser.add_argument('--chr_min', type=int, default=1)
    parser.add_argument('--chr_max', type=int, default=22)
    return parser.parse_args()

args = parse_args()

# ---------------------------------------------------------------------------
# Genome-wide SNP count per chromosome (approx., HapMap3 common variants)
# Used for h² scaling when only a subset of chromosomes is provided.
# ---------------------------------------------------------------------------
CHR_SNP_COUNTS = {
    1: 220000,  2: 232000,  3: 194000,  4: 185000,  5: 171000,
    6: 166000,  7: 147000,  8: 144000,  9: 115000, 10: 131000,
   11: 130000, 12: 124000, 13:  94000, 14:  87000, 15:  81000,
   16:  87000, 17:  77000, 18:  77000, 19:  57000, 20:  63000,
   21:  38000, 22:  37000,
}
GENOME_WIDE_SNPS = sum(CHR_SNP_COUNTS.values())   # ~2.9 M (HapMap3 scale)



def or_to_liability_beta(
    log_or: np.ndarray,
    prevalence: float,
    sample_prevalence: float = 0.5,
) -> np.ndarray:
    """
    Convert observed log-OR from a case-control GWAS to liability-scale
    effect sizes (betas) suitable for the simulator.
 
    Uses the Lee et al. (2011) liability-scale correction:
 
        beta_liab = beta_obs * [ K*(1-K) / z ] / [ K_s*(1-K_s) / z_s ]
 
    Where:
        K    = population prevalence
        K_s  = sample prevalence (0.5 for 1:1 design)
        z    = N(0,1) PDF at liability threshold for K   → phi(Phi^-1(1-K))
        z_s  = N(0,1) PDF at liability threshold for K_s → phi(Phi^-1(1-K_s))
 
    With 1:1 sampling (K_s = 0.5):
        - threshold_s = 0  (by symmetry of the normal)
        - z_s = phi(0) = 1/sqrt(2*pi) ~ 0.3989
        - the correction factor simplifies considerably
 
    Parameters
    ----------
    log_or            : array of log-OR values (the OR column in your data)
    prevalence        : population prevalence of the trait (K)
    sample_prevalence : case fraction in the sample (default 0.5 for 1:1)
 
    Returns
    -------
    beta_liab : liability-scale effect sizes, ready for simulate_gwas_data()
    """
    from scipy.stats import norm
 
    K   = prevalence
    K_s = sample_prevalence
 
    # Liability thresholds
    t   = norm.ppf(1 - K)        # population threshold
    t_s = norm.ppf(1 - K_s)      # sample threshold (= 0.0 for 1:1)
 
    # PDF heights at thresholds
    z   = norm.pdf(t)             # phi(t)
    z_s = norm.pdf(t_s)           # phi(0) = 0.3989 for 1:1
 
    # Lee et al. correction factor
    correction = (K * (1 - K) / z) / (K_s * (1 - K_s) / z_s)
 
    return np.asarray(log_or, dtype=float) * correction


# ---------------------------------------------------------------------------
# Core simulator (unchanged from original)
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
    if odds_ratio is not None:
        beta[causal_idx] = or_to_liability_beta(odds_ratio[causal_idx], prevalence)
    else:
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


# ---------------------------------------------------------------------------
# Data loading: single file or list of per-chromosome files
# ---------------------------------------------------------------------------

def load_meta_files(
    meta_input: Union[str, list],
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load one or more .meta files into a single DataFrame.

    Parameters
    ----------
    meta_input : str  -> single file path (one or all chromosomes)
                 list -> list of per-chromosome file paths (chr1-chr22)
    """
    if isinstance(meta_input, (str, Path)):
        paths = [meta_input]
    else:
        paths = list(meta_input)

    chunks = []
    for p in paths:
        df_chr = pd.read_csv(p, sep=r"\s+")
        chunks.append(df_chr)
        if verbose:
            chrs = df_chr["CHR"].unique().tolist()
            print(f"  Loaded {Path(p).name}: {len(df_chr):,} SNPs  (chr {chrs})")

    return pd.concat(chunks, ignore_index=True)


# ---------------------------------------------------------------------------
# Heritability estimation with chromosome-aware scaling
# ---------------------------------------------------------------------------

def estimate_h2(
    df: pd.DataFrame,
    n_individuals: int,
    observed_chrs: list,
) -> tuple:
    """
    Estimate genome-wide h² from summary statistics.

    Uses the infinitesimal approximation:
        Var(beta_obs) ~ h2/M_genome + 1/N
        h2 ~ M_genome * max(Var(beta_obs) - 1/N, 0)

    Because beta variance is estimated from only the observed chromosomes,
    we first estimate h² for those chromosomes, then scale up to genome-wide
    by dividing by the fraction of the genome they represent.

    Returns (h2_estimate, genome_fraction_covered)
    """
    n_cohorts_max = int(df["N"].max())
    # Use only SNPs with full cohort coverage for cleaner variance estimate
    df_full = df[df["N"] == n_cohorts_max].copy()
    beta_var = df_full["OR"].var()   # OR column = log-OR (beta)

    # Fraction of genome covered by observed chromosomes
    observed_snps_expected = sum(CHR_SNP_COUNTS.get(c, 0) for c in observed_chrs)
    genome_fraction = observed_snps_expected / GENOME_WIDE_SNPS

    M_observed = len(df)

    # h² for observed chromosomes only
    h2_partial = M_observed * max(beta_var - 1.0 / n_individuals, 0)

    # Scale up to genome-wide h²
    h2_genome = h2_partial / genome_fraction if genome_fraction > 0 else h2_partial
    h2_est = float(np.clip(h2_genome, 0.01, 0.99))

    return h2_est, genome_fraction


# ---------------------------------------------------------------------------
# Main calibration function
# ---------------------------------------------------------------------------

def calibrate_from_meta(
    meta_input: Union[str, list],
    n_per_cohort: int = 3_000,
    prevalence: float = 0.3,
    p_threshold: float = 5e-8,
    verbose: bool = True,
) -> tuple:
    """
    Derive simulator parameters from one or more GWAS meta-analysis files.

    Parameters
    ----------
    meta_input    : single .meta file path, OR list of per-chromosome paths
                    (chr1-chr22). Files are concatenated before calibration.
    n_per_cohort  : assumed individuals per cohort (to estimate total N)
    prevalence    : disease prevalence for binary trait simulation
    p_threshold   : p-value threshold to define causal SNPs
                    (default: 5e-8 = genome-wide significance;
                     automatically falls back to 5e-5 if no hits found)
    verbose       : print calibration summary

    Returns
    -------
    params      : dict of calibrated parameters for simulate_gwas_data()
    causal_snps : DataFrame of SNPs passing p_threshold
    """
    if verbose:
        print("\nLoading meta-analysis file(s)...")

    df = load_meta_files(meta_input, verbose=verbose)

    # ------------------------------------------------------------------
    # 1. Chromosome coverage
    # ------------------------------------------------------------------
    observed_chrs = sorted(df["CHR"].dropna().astype(int).unique().tolist())
    n_chrs = len(observed_chrs)
    is_genome_wide = (n_chrs == 22)

    # ------------------------------------------------------------------
    # 2. Sample size
    # ------------------------------------------------------------------
    n_cohorts_max = int(df["N"].max())
    n_individuals = n_cohorts_max * n_per_cohort

    # ------------------------------------------------------------------
    # 3. SNP count
    # ------------------------------------------------------------------
    n_snps_observed = len(df)

    # ------------------------------------------------------------------
    # 4. Heritability (chromosome-aware scaling)
    # ------------------------------------------------------------------
    h2_est, genome_fraction = estimate_h2(df, n_individuals, observed_chrs)

    # ------------------------------------------------------------------
    # 5. Causal SNPs — with automatic fallback threshold
    # ------------------------------------------------------------------
    causal_snps = df[df["P"] < p_threshold].copy()
    used_threshold = p_threshold

    if len(causal_snps) == 0:
        used_threshold = 5e-5
        causal_snps = df[df["P"] < used_threshold].copy()
        if verbose:
            print(f"\n  [Note] No GWS hits (P<{p_threshold:.0e}); "
                  f"falling back to suggestive threshold "
                  f"(P<{used_threshold:.0e}): {len(causal_snps)} SNPs")

    n_causal = max(len(causal_snps), 1)

    # Per-chromosome causal counts
    causal_per_chr = (
        causal_snps.groupby("CHR")
        .size()
        .reindex(observed_chrs, fill_value=0)
        .astype(int)
    )

    # ------------------------------------------------------------------
    # 6. Heterogeneity summary
    # ------------------------------------------------------------------
    i2_mean   = df["I"].mean()
    i2_median = df["I"].median()

    # ------------------------------------------------------------------
    # 7. Beta distribution
    # ------------------------------------------------------------------
    df_valid  = df[df["OR"].notna() & (df["OR"] != 0)]
    beta_var  = df_valid["OR"].var()

    # ------------------------------------------------------------------
    # Package output
    # ------------------------------------------------------------------
    params = dict(
        n_individuals = n_individuals,
        n_snps        = n_snps_observed,
        n_causal      = n_causal,
        maf_range     = (0.05, 0.50),
        h2            = h2_est,
        binary_trait  = True,
        prevalence    = prevalence,
    )

    if verbose:
        chrs_str = (
            "1-22 (genome-wide)" if is_genome_wide
            else ", ".join(map(str, observed_chrs))
        )
        print()
        print("=" * 60)
        print("  GWAS Simulator — Calibrated Parameters")
        print("=" * 60)
        print(f"  Chromosomes          : {chrs_str}")
        print(f"  Genome fraction      : {genome_fraction*100:.1f}%")
        print(f"  SNPs observed        : {n_snps_observed:,}")
        print(f"  Max cohorts (N)      : {n_cohorts_max}")
        print(f"  Assumed N/cohort     : {n_per_cohort:,}")
        print(f"  Total individuals    : {n_individuals:,}")
        print(f"  Beta variance        : {beta_var:.6f}")
        print(f"  h² estimate          : {h2_est:.4f}")
        print(f"  Causal SNPs (P<{used_threshold:.0e}): {n_causal}")
        print(f"  MAF range            : (0.05, 0.50)")
        print(f"  Binary trait         : True")
        print(f"  Prevalence           : {prevalence}")
        print(f"  Mean I²              : {i2_mean:.2f}%")
        print(f"  Median I²            : {i2_median:.2f}%")
        print()
        print("  Causal SNPs per chromosome:")
        for chrom, count in causal_per_chr.items():
            bar = "█" * min(count, 40)
            print(f"    chr{chrom:>2}  {count:>4}  {bar}")
        print("=" * 60)

    return params, causal_snps


# ---------------------------------------------------------------------------
# Usage examples
# ---------------------------------------------------------------------------



if __name__ == "__main__":

    import os

    args = parse_args()
    input_directory = args.input_directory
    output_directory = args.output_directory
    chr_min = args.chr_min
    chr_max = min(args.chr_max, 22)

    os.makedirs(output_directory, exist_ok=True)


    chr_files = [
        os.path.join(input_directory, f"ihac2.chr{c}.eur.meta") for c in range(chr_min, chr_max + 1)
    ]


    params, causal_snps = calibrate_from_meta(
        meta_input   = chr_files,
        n_per_cohort = 3_000,
        prevalence   = 0.3,
        p_threshold  = 5e-8,
        verbose      = True,
    )

    sim_params = params.copy()
    sim_params["odds_ratio"] = causal_snps["OR"].values.astype(float)

    print(f"\nRunning demo simulation: {sim_params['n_snps']:,} SNPs, "
          f"{sim_params['n_individuals']:,} individuals, "
          f"{sim_params['n_causal']} causal SNPs ...")

    results = simulate_gwas_data(**sim_params, seed=42)

    print("\n--- Simulation Output ---")
    print(f"  Genotype matrix     : {results['G_std'].shape}")
    print(f"  Phenotype prevalence: {results['y'].mean():.3f} "
          f"(target {params['prevalence']})")
    print(f"  Non-zero betas      : {(results['beta'] != 0).sum()}")
    print(f"  Beta range          : [{results['beta'].min():.4f}, "
          f"{results['beta'].max():.4f}]")

    print(f"\n--- Causal SNPs from real data ---")
    cols = ["CHR", "SNP", "A1", "A2", "N", "P", "OR", "I"]
    print(causal_snps[cols].sort_values(["CHR", "P"]).to_string(index=False))

