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

def main(args: argparse.Namespace) -> None:
    size = args.size
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n=== Running experiments for: {size} ===\n")

    from dp_gwas_experiments import (
        experiment1_privacy_utility, experiment2_three_way,
        experiment3_topology, experiment4_stratified,
        experiment5_scaling, experiment6_rizk_comparison,
        experiment_gwas_metrics_vs_epsilon,
        experiment_gwas_metrics_vs_n_centers,
        experiment_posterior_gm_am,
    )
    from dp_gwas_exp7_nyc import run_experiment7

    if size == 'small':
        n_individuals = 1000
        n_snps = 100
        n_causal = 25
        n_centers_list = [5, 10, 20]
        n_centers = 5
        n_reps = 5
        epsilons = [0.2, 0.5, 1.0, 1.5]
        r1 = experiment1_privacy_utility(n_individuals=n_individuals, n_snps=n_snps, n_causal=n_causal, n_centers=n_centers, n_reps=n_reps, output_dir=args.output_dir, snp_chunk_size=args.snp_chunk_size)
        r2 = experiment2_three_way(n_individuals_list=[5000, 10000, 15000, 20000, 25000], n_snps=n_snps, n_causal=n_causal, n_centers=n_centers, n_reps=n_reps, output_dir=args.output_dir, snp_chunk_size=args.snp_chunk_size)
        r3 = experiment3_topology(n_individuals=n_individuals, n_snps=n_snps, n_causal=n_causal, n_centers=30, n_reps=n_reps, T=200, output_dir=args.output_dir, snp_chunk_size=args.snp_chunk_size)
        r4 = experiment4_stratified(n_individuals=n_individuals, n_snps=n_snps, n_causal=n_causal, n_reps=n_reps, output_dir=args.output_dir, snp_chunk_size=args.snp_chunk_size)
        r5 = experiment5_scaling(n_individuals_total=n_individuals, n_snps=n_snps, n_causal=n_causal, n_reps=n_reps, output_dir=args.output_dir, snp_chunk_size=args.snp_chunk_size)
        r6 = experiment6_rizk_comparison(n_individuals=n_individuals, n_snps=n_snps, n_causal=n_causal,
                                          n_centers_list=n_centers_list, epsilons=epsilons, n_reps=n_reps, output_dir=args.output_dir, snp_chunk_size=args.snp_chunk_size)
        r6b = experiment_gwas_metrics_vs_epsilon(
            n_individuals=n_individuals, n_snps=n_snps, n_causal=n_causal, n_centers=n_centers, n_reps=n_reps,
            output_dir=args.output_dir, snp_chunk_size=args.snp_chunk_size
        )
        r6c = experiment_gwas_metrics_vs_n_centers(
            n_individuals=n_individuals, n_snps=n_snps, n_causal=n_causal, n_reps=n_reps,
            epsilon=1.0,
            output_dir=args.output_dir, snp_chunk_size=args.snp_chunk_size
        )
        r9 = experiment_posterior_gm_am(
            n_individuals=n_individuals, n_snps=n_snps, n_causal=n_causal, n_centers=n_centers, epsilon=1.0,
            output_dir=args.output_dir, snp_chunk_size=args.snp_chunk_size
        )
        r7 = run_experiment7(size='small', output_dir=args.output_dir, snp_chunk_size=args.snp_chunk_size)
    elif size == 'small_nyc':
        r7 = run_experiment7(size='small', output_dir=args.output_dir, snp_chunk_size=args.snp_chunk_size)
    elif size == 'medium':
        n_individuals = 25000
        n_snps = 600
        n_causal = 25
        n_centers_list = [5, 10, 20]
        n_centers = 5
        n_reps = 5
        epsilons = [0.2, 0.5, 1.0, 1.5]
        r1 = experiment1_privacy_utility(n_individuals=n_individuals, n_snps=n_snps, n_causal=n_causal, n_centers=n_centers, n_reps=n_reps, output_dir=args.output_dir, snp_chunk_size=args.snp_chunk_size)
        r2 = experiment2_three_way(n_individuals_list=[5000, 10000, 15000, 20000, 25000], n_snps=n_snps, n_causal=n_causal, n_centers=n_centers, n_reps=n_reps, output_dir=args.output_dir, snp_chunk_size=args.snp_chunk_size)
        r3 = experiment3_topology(n_individuals=n_individuals, n_snps=n_snps, n_causal=n_causal, n_centers=30, n_reps=n_reps, T=200, output_dir=args.output_dir)
        r4 = experiment4_stratified(n_individuals=n_individuals, n_snps=n_snps, n_causal=n_causal, n_reps=n_reps, output_dir=args.output_dir, snp_chunk_size=args.snp_chunk_size)
        r5 = experiment5_scaling(n_individuals_total=n_individuals, n_snps=n_snps, n_causal=n_causal, n_reps=n_reps, output_dir=args.output_dir, snp_chunk_size=args.snp_chunk_size)
        r6 = experiment6_rizk_comparison(n_individuals=n_individuals, n_snps=n_snps, n_causal=n_causal,
                                          n_centers_list=n_centers_list, epsilons=epsilons, n_reps=n_reps, output_dir=args.output_dir, snp_chunk_size=args.snp_chunk_size)
        r6b = experiment_gwas_metrics_vs_epsilon(
            n_individuals=n_individuals, n_snps=n_snps, n_causal=n_causal, n_centers=n_centers, n_reps=n_reps,
            output_dir=args.output_dir,
            snp_chunk_size=args.snp_chunk_size
        )
        r6c = experiment_gwas_metrics_vs_n_centers(
            n_individuals=n_individuals, n_snps=n_snps, n_causal=n_causal, n_reps=n_reps,
            epsilon=1.0,
            output_dir=args.output_dir,
            snp_chunk_size=args.snp_chunk_size
        )
        r9 = experiment_posterior_gm_am(
            n_individuals=n_individuals, n_snps=n_snps, n_causal=n_causal, n_centers=n_centers, epsilon=1.0,
            output_dir=args.output_dir,
            snp_chunk_size=args.snp_chunk_size
        )
        r7 = run_experiment7(size='medium', output_dir=args.output_dir, snp_chunk_size=args.snp_chunk_size)
    elif size == 'medium_nyc':
        r7 = run_experiment7(size='medium', output_dir=args.output_dir, snp_chunk_size=args.snp_chunk_size)
    elif size == 'large':
        n_individuals = 25000
        n_snps = 100000
        n_causal = 25
        n_reps = 5
        epsilons = [0.2, 0.5, 1.0, 1.5]
        n_centers_list = [5, 10, 20]
        n_centers = 5

        r1 = experiment1_privacy_utility(n_individuals=n_individuals, n_snps=n_snps, n_causal=n_causal, n_centers=n_centers, n_reps=n_reps, output_dir=args.output_dir, snp_chunk_size=args.snp_chunk_size)
        r2 = experiment2_three_way(n_individuals_list=[10000, 20000, 30000, 40000, 50000], n_snps=n_snps, n_causal=n_causal, n_centers=n_centers, n_reps=n_reps, output_dir=args.output_dir, snp_chunk_size=args.snp_chunk_size)
        r3 = experiment3_topology(n_individuals=n_individuals, n_snps=n_snps, n_causal=n_causal, n_centers=30, n_reps=n_reps, T=200, output_dir=args.output_dir, snp_chunk_size=args.snp_chunk_size)
        r4 = experiment4_stratified(n_individuals=n_individuals, n_snps=n_snps, n_causal=n_causal, n_reps=n_reps, output_dir=args.output_dir, snp_chunk_size=args.snp_chunk_size)
        r5 = experiment5_scaling(n_individuals_total=n_individuals, n_snps=n_snps, n_causal=n_causal, n_reps=n_reps, output_dir=args.output_dir, snp_chunk_size=args.snp_chunk_size)
        r6 = experiment6_rizk_comparison(n_individuals=n_individuals, n_snps=n_snps, n_causal=n_causal, n_centers_list=n_centers_list, epsilons=epsilons, n_reps=n_reps, output_dir=args.output_dir, snp_chunk_size=args.snp_chunk_size)
        r6b = experiment_gwas_metrics_vs_epsilon(n_individuals=n_individuals, n_snps=n_snps, n_causal=n_causal, n_centers=n_centers, n_reps=n_reps, epsilons=epsilons, output_dir=args.output_dir, snp_chunk_size=args.snp_chunk_size)
        r6c = experiment_gwas_metrics_vs_n_centers(
            n_individuals=n_individuals, n_snps=n_snps, n_causal=n_causal, n_reps=n_reps,
            epsilon=1.0,
            output_dir=args.output_dir,
            snp_chunk_size=args.snp_chunk_size
        )
        r9 = experiment_posterior_gm_am(
            n_individuals=n_individuals, n_snps=n_snps, n_causal=n_causal, n_centers=n_centers, epsilon=1.0,
            output_dir=args.output_dir,
            snp_chunk_size=args.snp_chunk_size
        )
        r7 = run_experiment7(size='large', output_dir=args.output_dir, snp_chunk_size=args.snp_chunk_size)
    elif size == 'large_nyc':
        r7 = run_experiment7(size='large', output_dir=args.output_dir, snp_chunk_size=args.snp_chunk_size)
    elif size == 'real_world':
        n_individuals = 25000
        n_snps = 1000000
        n_causal = 25
        n_reps = 5
        epsilons = [0.2, 0.5, 1.0, 1.5]
        n_centers_list = [5, 10, 20]
        n_centers = 5
        args.snp_chunk_size = 100000

        r1 = experiment1_privacy_utility(n_individuals=n_individuals, n_snps=n_snps, n_causal=n_causal, n_centers=n_centers, n_reps=n_reps, output_dir=args.output_dir, snp_chunk_size=args.snp_chunk_size)
        r2 = experiment2_three_way(n_individuals_list=[10000, 20000, 30000, 40000, 50000], n_snps=n_snps, n_causal=n_causal, n_centers=n_centers, n_reps=n_reps, output_dir=args.output_dir, snp_chunk_size=args.snp_chunk_size)
        r3 = experiment3_topology(n_individuals=n_individuals, n_snps=n_snps, n_causal=n_causal, n_centers=30, n_reps=n_reps, T=200, output_dir=args.output_dir, snp_chunk_size=args.snp_chunk_size)
        r4 = experiment4_stratified(n_individuals=n_individuals, n_snps=n_snps, n_causal=n_causal, n_reps=n_reps, output_dir=args.output_dir, snp_chunk_size=args.snp_chunk_size)
        r5 = experiment5_scaling(n_individuals_total=n_individuals, n_snps=n_snps, n_causal=n_causal, n_reps=n_reps, output_dir=args.output_dir, snp_chunk_size=args.snp_chunk_size)
        r6 = experiment6_rizk_comparison(n_individuals=n_individuals, n_snps=n_snps, n_causal=n_causal, n_centers_list=n_centers_list, epsilons=epsilons, n_reps=n_reps, output_dir=args.output_dir, snp_chunk_size=args.snp_chunk_size)
        r6b = experiment_gwas_metrics_vs_epsilon(n_individuals=n_individuals, n_snps=n_snps, n_causal=n_causal, n_centers=n_centers, n_reps=n_reps, epsilons=epsilons, output_dir=args.output_dir, snp_chunk_size=args.snp_chunk_size)
        r6c = experiment_gwas_metrics_vs_n_centers(
            n_individuals=n_individuals, n_snps=n_snps, n_causal=n_causal, n_reps=n_reps,
            epsilon=1.0,
            output_dir=args.output_dir,
            snp_chunk_size=args.snp_chunk_size
        )
        r9 = experiment_posterior_gm_am(
            n_individuals=n_individuals, n_snps=n_snps, n_causal=n_causal, n_centers=n_centers, epsilon=1.0,
            output_dir=args.output_dir,
            snp_chunk_size=args.snp_chunk_size
        )
        r7 = run_experiment7(size='real_world', output_dir=args.output_dir, snp_chunk_size=args.snp_chunk_size)
    elif size == 'real_world_nyc':
        r7 = run_experiment7(size='real_world', output_dir=args.output_dir, snp_chunk_size=args.snp_chunk_size)
    
    
    print(f"\nAll figures saved to: {args.output_dir}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', choices=['small', 'small_nyc', 'medium', 'medium_nyc', 'large', 'large_nyc', 'real_world_nyc'], default='small')
    parser.add_argument('--output_dir', type=str, default='../figures')
    parser.add_argument('--snp_chunk_size', type=int, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
