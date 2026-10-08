# An Empirical Audit of Feature Selection and Constraint Enforcement for Adversarial Robustness in NetFlow Intrusion Detection

This repository contains the code to reproduce the experiments in the paper
*"An Empirical Audit of Feature Selection and Constraint Enforcement for Adversarial Robustness in NetFlow Intrusion Detection."*

The study evaluates two techniques on standardized NetFlow NIDS data:
1. **Constraint-aware evaluation** — enforcing NetFlow feature constraints on
   adversarial examples via a Constraint Enforcement Layer (CEL).
2. **Class-aware feature selection** — a faithful reproduction of ACAFS
   (Mchina et al., *Frontiers in Big Data*, 2026).

Both are tested against FGSM and PGD evasion attacks across a range of
perturbation budgets, on two datasets: NF-UNSW-NB15-v3 and NF-CSE-CIC-IDS2018-v3.

---

## Repository structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
└── src/
    └── acafs/
      ├── acafs_nfv3.py                  # ACAFS reproduction
      └── run_acafs_robustness.py        # ACAFS adversarial robustness experiment
    └── constraint-enforcement/
      ├── cel_nfv3.py                    # Constraint Enforcement Layer
      ├── check_constraints.py           # Validates derived-value constraints
      ├── check_structural_constraints.py# Validates structural constraints
      ├── run_nfv3_experiment.py         # Constraint-enforcement robustness experiment
```

**Note:** `cel_nfv3.py` and `acafs_nfv3.py` are library modules. They are imported
by the experiment scripts and are not executed directly. Run the scripts from
inside `src/` (or keep all six files in the same directory) so the imports resolve.

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

Run all commands from inside `src/` (so the module imports resolve).

### Table I — which NetFlow constraints hold

```bash
python check_constraints.py /path/to/NF-UNSW-NB15-v3.csv
python check_structural_constraints.py /path/to/NF-UNSW-NB15-v3.csv
python check_structural_constraints.py /path/to/NF-CSE-CIC-IDS2018-v3.csv
```

These report, per candidate constraint, the fraction of flows that satisfy it.
Structural ordering/subset constraints hold; derived-value equalities do not.

### Table II — constraint enforcement robustness gap

```bash
python run_nfv3_experiment.py /path/to/NF-UNSW-NB15-v3.csv --sample 150000
python run_nfv3_experiment.py /path/to/NF-CSE-CIC-IDS2018-v3.csv --sample 150000
```

Reports the violation rate and the robustness gap (evasion under unconstrained
minus constraint-valid attacks). Uses `cel_nfv3.py`.

### Tables III–V — ACAFS adversarial robustness

```bash
python run_acafs_robustness.py /path/to/NF-UNSW-NB15-v3.csv --sample 150000
python run_acafs_robustness.py /path/to/NF-CSE-CIC-IDS2018-v3.csv --sample 150000
```

Runs ACAFS feature selection, trains an ACAFS-selected model and an
all-features baseline (both SMOTE-Tomek balanced), and reports clean per-class
accuracy and per-class evasion under FGSM and PGD across an eps sweep. Uses
`acafs_nfv3.py`.

---

## Command-line options

| Option | Scripts | Meaning | Default |
|--------|---------|---------|---------|
| `--sample N` | run_* | Randomly subsample N flows (0 = all) | 150000 |
| `--epochs N` | run_* | Training epochs | 15 |
| `--eps E` | run_nfv3_experiment | Perturbation budget | 0.1 |
| `--eps_sweep "a,b,c"` | run_acafs_robustness | Comma-separated eps values | 0.05,0.1,0.2,0.3 |
| `--min_class N` | run_acafs_robustness | Drop classes with < N samples | 100 |
| `--seed N` | all run_* | Random seed | 42 |

---

## Notes on reproducibility

- Random seeds are fixed, but exact numbers may vary slightly across hardware,
  library versions, and OS due to nondeterminism in floating-point and threaded
  operations (XGBoost, SMOTE-Tomek, PyTorch).
- Classes with very few samples are excluded (`--min_class`); the scripts print
  which classes were dropped. Interpret small retained classes with caution.
- `cel_nfv3.py` enforces only the constraints that were empirically validated to
  hold in the data (structural ordering and subset constraints, with duration
  tolerances and the characterized IAT special case). Derived-value equalities
  that the flow exporter does not preserve are deliberately not enforced.

---

## Citation

If you use this code, please cite the paper (details to be added upon
publication) and the dataset papers listed above.
