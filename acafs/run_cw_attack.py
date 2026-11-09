"""
run_cw_attack.py
----------------
Adds the Carlini-Wagner L2 (C&W) attack to the NF-v3 robustness study.

Operations:
  1. Load NF-v3 dataset, drop tiny classes, split, min-max scale.
  2. ACAFS feature selection on the train split; SMOTE-Tomek balance BOTH the
     ACAFS feature space and the all-features baseline.
  3. Train the ACAFS model and the all-features baseline.
  4. Attack BOTH with C&W L2 and report PER-CLASS evasion.

C&W is SLOW (an optimization loop per sample). We therefore attack a random
subsample of the test set (--cw_sample, default 1000).

Usage:
    python run_cw_attack.py NF-UNSW-NB15-v3.csv  --sample 150000 --cw_sample 1000
    python run_cw_attack.py NF-CICIDS2018-v3.csv --sample 150000 --cw_sample 1000
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Reuse the EXACT helpers from the main experiment so preprocessing, model

from acafs_nfv3 import ACAFS, balance_smote_tomek
from run_acafs_robustness import (
    load_multiclass, filter_small_classes, DNN, train_dnn, per_class_evasion,
)


def clean_accuracy(model, Xte_feat, yte, classes):
    with torch.no_grad():
        pred = model(torch.tensor(Xte_feat, dtype=torch.float32)).argmax(1).numpy()
    out = {}
    for ci, cname in enumerate(classes):
        mask = (yte == ci)
        if mask.sum() == 0:
            continue
        out[cname] = (int(mask.sum()), float((pred[mask] == ci).mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--sample", type=int, default=150_000)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_class", type=int, default=100)
    ap.add_argument("--cw_sample", type=int, default=1000,
                    help="Number of test flows to attack with C&W (it is slow). "
                         "These are drawn at random from the test split with the "
                         "fixed seed. Per-class support attacked is reported.")
    ap.add_argument("--cw_max_iter", type=int, default=10,
                    help="C&W optimization iterations per binary-search step.")
    ap.add_argument("--cw_bsteps", type=int, default=5,
                    help="C&W binary-search steps on the trade-off constant.")
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)

    print(f"Loading {args.csv} ...")
    X, y, feats, classes = load_multiclass(args.csv, args.sample, args.seed)
    print(f"  {X.shape[0]:,} samples, {X.shape[1]} features, {len(classes)} classes (raw)")

    X, y, classes, dropped = filter_small_classes(X, y, classes, args.min_class)
    if dropped:
        print(f"  dropped {len(dropped)} classes with < {args.min_class} samples: {dropped}")
    n_classes = len(classes)
    print(f"  {X.shape[0]:,} samples, {n_classes} classes after filtering")

    counts = pd.Series(y).value_counts().sort_values()
    minority = [ci for ci in counts.index if counts[ci] < counts.median()]
    minority_names = [classes[ci] for ci in minority]

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=args.seed, stratify=y)
    scaler = MinMaxScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr).astype(np.float32)
    Xte_s = scaler.transform(Xte).astype(np.float32)

    print("\n=== ACAFS feature selection ===")
    acafs = ACAFS(minority_classes=minority, verbose=True)
    sel = acafs.fit(Xtr_s, ytr, feature_names=feats)
    Xtr_sel, Xte_sel = Xtr_s[:, sel], Xte_s[:, sel]

    print("\n=== SMOTE-Tomek balancing (both feature spaces) ===")
    try:
        Xtr_bal, ytr_bal = balance_smote_tomek(Xtr_sel, ytr, args.seed)
        print(f"  ACAFS-selected balanced train: {Xtr_bal.shape[0]:,} samples")
    except Exception as e:
        print(f"  ACAFS balancing skipped ({e}); using unbalanced")
        Xtr_bal, ytr_bal = Xtr_sel, ytr
    try:
        Xtr_base_bal, ytr_base_bal = balance_smote_tomek(Xtr_s, ytr, args.seed)
        print(f"  all-features baseline balanced train: {Xtr_base_bal.shape[0]:,} samples")
    except Exception as e:
        print(f"  baseline balancing skipped ({e}); using unbalanced")
        Xtr_base_bal, ytr_base_bal = Xtr_s, ytr

    print("\n=== Training models ===")
    model_acafs = train_dnn(Xtr_bal, ytr_bal, n_classes, args.epochs, args.seed)
    model_base = train_dnn(Xtr_base_bal, ytr_base_bal, n_classes, args.epochs, args.seed)

    from art.estimators.classification import PyTorchClassifier
    from art.attacks.evasion import CarliniL2Method

    def wrap(model, dim):
        return PyTorchClassifier(
            model=model, loss=nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
            input_shape=(dim,), nb_classes=n_classes, clip_values=(0.0, 1.0))

    art_acafs = wrap(model_acafs, len(sel))
    art_base = wrap(model_base, X.shape[1])

    print("\n=== Clean per-class accuracy (no attack) ===")
    ca_ac = clean_accuracy(model_acafs, Xte_sel, yte, classes)
    ca_bs = clean_accuracy(model_base, Xte_s, yte, classes)
    print(f"  {'class':<24} {'support':>8} {'ACAFS_acc':>10} {'Base_acc':>10}")
    for cname in classes:
        if cname in ca_ac:
            sup, aa = ca_ac[cname]
            _, ab = ca_bs.get(cname, (0, float('nan')))
            tag = "  <-- minority" if cname in minority_names else ""
            print(f"  {cname:<24} {sup:>8,} {aa:>10.3f} {ab:>10.3f}{tag}")

    # ---- C&W attack on a random test subsample (it is slow) -----------
    rng = np.random.default_rng(args.seed)
    n_sub = min(args.cw_sample, Xte_s.shape[0])
    sub = rng.choice(Xte_s.shape[0], n_sub, replace=False)
    Xsub_s, Xsub_sel, ysub = Xte_s[sub], Xte_sel[sub], yte[sub]

    cw_ac = CarliniL2Method(art_acafs, targeted=False,
                            max_iter=args.cw_max_iter,
                            binary_search_steps=args.cw_bsteps,
                            batch_size=128, verbose=False)
    cw_bs = CarliniL2Method(art_base, targeted=False,
                            max_iter=args.cw_max_iter,
                            binary_search_steps=args.cw_bsteps,
                            batch_size=128, verbose=False)

    Xadv_ac = cw_ac.generate(Xsub_sel)
    Xadv_bs = cw_bs.generate(Xsub_s)
    ev_ac = per_class_evasion(model_acafs, Xsub_sel, Xadv_ac, ysub, classes)
    ev_bs = per_class_evasion(model_base, Xsub_s, Xadv_bs, ysub, classes)

    print(f"\n--- C&W L2 ---")
    print(f"  {'class':<24} {'support':>8} {'ACAFS_ev':>9} {'Base_ev':>9}")
    for cname in classes:
        if cname in ev_ac:
            sup, ea = ev_ac[cname]
            _, eb = ev_bs.get(cname, (0, float('nan')))
            tag = "  <-- minority" if cname in minority_names else ""
            print(f"  {cname:<24} {sup:>8,} {ea:>9.3f} {eb:>9.3f}{tag}")

if __name__ == "__main__":
    main()