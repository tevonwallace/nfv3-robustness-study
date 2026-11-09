# What Moves Adversarial Robustness in NetFlow Intrusion Detection? An Empirical Study of Constraint Enforcement, Feature Selection, and Adversarial Training

This repository contains the code to reproduce the experiments in the paper
*"What Moves Adversarial Robustness in NetFlow Intrusion Detection? An Empirical
Study of Constraint Enforcement, Feature Selection, and Adversarial Training."*

The study evaluates three techniques on standardized NetFlow NIDS data:
1. **Constraint-aware evaluation** — enforcing NetFlow feature constraints on
   adversarial examples via a Constraint Enforcement Layer (CEL).
2. **Class-aware feature selection** — a faithful reproduction of ACAFS
   (Mchina et al., *Frontiers in Big Data*, 2026).
3. **Adversarial training** — a PGD adversarial-training defense baseline.

These are tested against FGSM, PGD, and the Carlini–Wagner (C&W) L2 evasion
attacks across a range of perturbation budgets, on two datasets:
NF-UNSW-NB15-v3 and NF-CSE-CIC-IDS2018-v3.

---

## Repository structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
└── src/
    └── acafs/
      ├── acafs_nfv3.py                  # ACAFS reproduction (library module)
      ├── run_acafs_robustness.py        # ACAFS adversarial robustness experiment
      ├── run_cw_attack.py               # Carlini–Wagner (C&W) L2 attack experiment
      └── run_adversarial_training.py    # PGD adversarial-training defense baseline
    └── constraint-enforcement/
      ├── cel_nfv3.py                    # Constraint Enforcement Layer (library module)
      ├── check_constraints.py           # Validates derived-value constraints
      ├── check_structural_constraints.py# Validates structural constraints
      └── run_nfv3_experiment.py         # Constraint-enforcement robustness experiment
```

**Note:** `acafs_nfv3.py` and `cel_nfv3.py` are library modules; they are imported
by the experiment scripts and are not executed directly. `run_cw_attack.py` and
`run_adversarial_training.py` additionally import helpers from
`run_acafs_robustness.py`, so keep all four `acafs/` scripts in the same
directory and run them from inside `src/acafs/` so the imports resolve.

---

## Datasets

This repository does **not** include the datasets. Download them from the
University of Queensland NIDS datasets collection:

- NF-UNSW-NB15-v3 and NF-CSE-CIC-IDS2018-v3:
  https://staff.itee.uq.edu.au/marius/NIDS_datasets/

Place the CSV files where you will point the scripts (e.g., a local `data/`
directory, which is git-ignored). When using these datasets, cite the original
dataset papers as required by the dataset authors:

- M. Sarhan, S. Layeghy, N. Moustafa, and M. Portmann, "NetFlow Datasets for
  Machine Learning-Based Network Intrusion Detection Systems," 2021.
- M. Sarhan, S. Layeghy, and M. Portmann, "Towards a Standard Feature Set for
  Network Intrusion Detection System Datasets," *Mobile Networks and
  Applications*, 2022.

---

## Setup

Requires Python 3.9+.

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Reproducing the paper's results

Run all commands from inside the directory that holds each script (so the module
imports resolve).

### Table 1 — which NetFlow constraints hold

```bash
python check_constraints.py /path/to/NF-UNSW-NB15-v3.csv
python check_structural_constraints.py /path/to/NF-UNSW-NB15-v3.csv
python check_structural_constraints.py /path/to/NF-CSE-CIC-IDS2018-v3.csv
```

Reports, per candidate constraint, the fraction of flows that satisfy it.
Structural ordering/subset constraints hold; derived-value equalities do not.

### Table 2 — constraint enforcement robustness gap

```bash
python run_nfv3_experiment.py /path/to/NF-UNSW-NB15-v3.csv --sample 100000
python run_nfv3_experiment.py /path/to/NF-CSE-CIC-IDS2018-v3.csv --sample 100000
```

Reports the violation rate and the robustness gap (evasion under unconstrained
minus constraint-valid attacks). Uses `cel_nfv3.py`.

### Tables 3–5 — ACAFS vs. all-features baseline (clean accuracy and FGSM/PGD evasion)

```bash
python run_acafs_robustness.py /path/to/NF-UNSW-NB15-v3.csv --sample 150000
python run_acafs_robustness.py /path/to/NF-CSE-CIC-IDS2018-v3.csv --sample 150000
```

Runs ACAFS feature selection, trains an ACAFS-selected model and an all-features
baseline (both SMOTE-Tomek balanced), and reports clean per-class accuracy and
per-class evasion under FGSM and PGD across an eps sweep. Uses `acafs_nfv3.py`.

### Table 6 — robustness under the C&W L2 attack

```bash
python run_cw_attack.py /path/to/NF-UNSW-NB15-v3.csv --sample 150000 --cw_sample 1000
python run_cw_attack.py /path/to/NF-CSE-CIC-IDS2018-v3.csv --sample 150000 --cw_sample 1000
```

Trains the ACAFS and baseline models and reports per-class evasion under the
Carlini–Wagner L2 attack on a random test subsample. Confirms the feature-
selection null result holds under a strong optimization attack. Imports helpers
from `run_acafs_robustness.py` and `acafs_nfv3.py`.

### Figure 1 and Tables 7–8 — PGD adversarial-training defense baseline

```bash
python run_adversarial_training.py /path/to/NF-UNSW-NB15-v3.csv --sample 150000 --with_cw
python run_adversarial_training.py /path/to/NF-CSE-CIC-IDS2018-v3.csv --sample 150000 --with_cw
```

Trains a standard model and a PGD-adversarially-trained model on the same
SMOTE-Tomek-balanced all-features data, and reports clean per-class accuracy and
per-class evasion under FGSM, PGD (and, with `--with_cw`, C&W) across the eps
sweep. Imports helpers from `run_acafs_robustness.py` and `acafs_nfv3.py`.

---

## Command-line options

| Option | Scripts | Meaning | Default |
|--------|---------|---------|---------|
| `--sample N` | all `run_*` | Randomly subsample N flows (0 = all) | 150000 |
| `--epochs N` | all `run_*` | Training epochs | 15 |
| `--seed N` | all `run_*` | Random seed | 42 |
| `--eps E` | `run_nfv3_experiment` | Perturbation budget | 0.1 |
| `--eps_sweep "a,b,c"` | `run_acafs_robustness`, `run_adversarial_training` | Comma-separated eps values | 0.05,0.1,0.2,0.3 |
| `--min_class N` | `run_acafs_robustness`, `run_cw_attack`, `run_adversarial_training` | Drop classes with < N samples | 100 |
| `--cw_sample N` | `run_cw_attack`, `run_adversarial_training` | Test flows attacked with C&W | 1000 |
| `--cw_max_iter N` | `run_cw_attack`, `run_adversarial_training` | C&W optimization iterations | 10 |
| `--cw_bsteps N` | `run_cw_attack`, `run_adversarial_training` | C&W binary-search steps | 5 |
| `--at_eps E` | `run_adversarial_training` | L∞ budget used during adversarial training | 0.1 |
| `--at_steps N` | `run_adversarial_training` | PGD steps during adversarial training | 10 |
| `--with_cw` | `run_adversarial_training` | Also compare standard vs. AT under C&W | off |

---

## Notes on reproducibility

- Random seeds are fixed, but exact numbers may vary slightly across hardware,
  library versions, and OS due to nondeterminism in floating-point and threaded
  operations (XGBoost, SMOTE-Tomek, PyTorch).
- Classes with very few samples are excluded (`--min_class`); the scripts print
  which classes were dropped. Interpret small retained classes with caution.
- C&W is evaluated on a random subsample (`--cw_sample`, default 1000) because it
  solves a per-sample optimization; per-class support within the subsample is
  reported, and minority classes with few sampled instances are interpreted with
  caution.
- Adversarial training is performed at a single training budget
  (`--at_eps`, default 0.1); the defense reduces evasion within that budget and
  degrades beyond it.
- `cel_nfv3.py` enforces only the constraints that were empirically validated to
  hold in the data (structural ordering and subset constraints, with duration
  tolerances and the characterized IAT special case). Derived-value equalities
  that the flow exporter does not preserve are deliberately not enforced.

---

## Citation

If you use this code, please cite the paper (details to be added upon
publication) and the dataset papers listed above.