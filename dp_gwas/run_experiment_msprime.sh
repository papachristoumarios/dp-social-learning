#!/bin/bash

#SBATCH --job-name=dp-gwas
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=../logs/%x-%j.out
#SBATCH --error=../logs/%x-%j.err

module load mamba/latest

source activate dp-social-learning

OUTDIR="../figures"

mkdir -p $OUTDIR

python -u run_experiment_msprime.py --output_dir $OUTDIR

echo "Completed. Results are in: ${OUTDIR}"
