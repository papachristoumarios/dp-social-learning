#!/bin/bash

#SBATCH --job-name=dp-gwas-$SIZE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=../logs/%x-%j.out
#SBATCH --error=../logs/%x-%j.err

SIZE=$1

module load mamba/latest


source activate dp-social-learning

OUTDIR="../figures/${SIZE}"

python -u run_experiments.py --size $SIZE --output_dir $OUTDIR

echo "Completed. Results are in: ${OUTDIR}"
