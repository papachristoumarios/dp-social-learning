# dp-social-learning

Code and data for distributed survival analysis and differentially private federated GWAS simulations. The notebook `clinical_trials.ipynb` reproduces figures for the survival / clinical-trials experiments. The `dp_gwas` package runs msprime-based federation simulations and writes figures under `figures/`.

## Setup

Use Python 3.10+ (recommended). From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

To run the clinical trials notebook, install Jupyter as well:

```bash
pip install jupyter
```

`msprime` may require a C toolchain on some platforms; if `pip install` fails, install build essentials (e.g. `gcc`) or use a conda environment with `msprime` from conda-forge.

## Clinical trials experiment (`clinical_trials.ipynb`)

From the repository root, start Jupyter and open the notebook:

```bash
jupyter notebook clinical_trials.ipynb
# or: jupyter lab clinical_trials.ipynb
```

Execute cells in order. The notebook expects data paths and outputs consistent with the project layout under `datasets/` where applicable.

## DP-GWAS experiment (`dp_gwas`)

Run the driver script **from the `dp_gwas` directory** so default paths (`../figures`, `../datasets/...`) resolve correctly:

```bash
cd dp_gwas
python run_experiment_msprime.py
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--output_dir` | `../figures` | Directory for PDF outputs |
| `--n_reps` | `5` | Monte Carlo replicates (some experiments cap this internally) |
| `--epsilon` | `1.0` | Privacy budget where used |
| `--experiments` | `all` | `all` (full suite), `regular` (simulated msprime experiments only), or `nyc` (NYC hospital / organization federation, needs dataset) |

Examples:

```bash
# Full pipeline (simulations + NYC block if dataset present)
python run_experiment_msprime.py --experiments all --n_reps 5 --epsilon 1.0

# Simulated experiments only (no NYC-specific CSV)
python run_experiment_msprime.py --experiments regular

# NYC federation plots only (expects ../datasets/nyc_hospitals_w_organizations.csv)
python run_experiment_msprime.py --experiments nyc
```

Figures are written under the chosen `output_dir` (by default, `figures/` at the repo root).
