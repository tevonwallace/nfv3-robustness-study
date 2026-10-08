"""
The ACAFS adversarial-robustness experiment

Pipeline:
  1. Load NF-v3 dataset (multi-class labels via the 'Attack' column)
  2. Run ACAFS feature selection (acafs_nfv3.py) on the training split
  3. SMOTE-Tomek balance the selected training features
  4. Train a classifier on the ACAFS-selected features
  5. Attack it with FGSM / PGD
  6. Measure PER-CLASS evasion — do minority classes collapse first?
  7. Compare against a baseline model using ALL features

"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, accuracy_score

from acafs_nfv3 import ACAFS, balance_smote_tomek

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["XGBOOST_NUM_THREADS"] = "1"


DROP_COLS = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT",
             "FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS", "Dataset"]

class DNN(nn.Module):
    def __init__(self, d, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_classes))

    def forward(self, x):
        return self.net(x)


def load_multiclass(path, sample, seed):
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()

    # Multi-class label: use 'Attack' text column (has attack category names)
    if "Attack" in df.columns:
        label_raw = df["Attack"].astype(str)
    elif "Label" in df.columns:
        label_raw = df["Label"].astype(str)
    else:
        raise ValueError(f"No Attack/Label column. Columns: {list(df.columns)[:8]}")

    # map class names to integers
    classes = sorted(label_raw.unique())
    cmap = {c: i for i, c in enumerate(classes)}
    y = label_raw.map(cmap).values

    drop = [c for c in DROP_COLS if c in df.columns] + \
           [c for c in ["Attack", "Label"] if c in df.columns]
    Xdf = df.drop(columns=drop)
    Xdf = Xdf.replace([np.inf, -np.inf], np.nan).fillna(0)
    feats = list(Xdf.columns)
    X = Xdf.values.astype(float)
    # y currently indexes into `classes`; keep the names around
    class_names_all = classes

    if sample and sample < len(X):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), sample, replace=False)
        X, y = X[idx], y[idx]

    return X, y, feats, class_names_all


def filter_small_classes(X, y, class_names, min_count):
    """
    Drop classes with fewer than `min_count` samples (in the sampled data),
    then remap the remaining labels to a contiguous 0..C-1 range so XGBoost
    and the DNN don't choke on gaps. Returns filtered X, y, and the surviving
    class-name list (in new-index order).

    Classes with only a handful of samples cannot be split into train/test or
    cross-validation folds and are excluded, which we report in the paper.
    """
    counts = pd.Series(y).value_counts()
    keep_old = sorted([c for c in counts.index if counts[c] >= min_count])
    dropped = [(class_names[c], int(counts[c]))
               for c in counts.index if counts[c] < min_count]
    keep_mask = np.isin(y, keep_old)
    X_f = X[keep_mask]
    y_old = y[keep_mask]
    # remap old label -> new contiguous index
    remap = {old: new for new, old in enumerate(keep_old)}
    y_f = np.array([remap[v] for v in y_old])
    surviving_names = [class_names[c] for c in keep_old]
    return X_f, y_f, surviving_names, dropped


def train_dnn(X, y, n_classes, epochs, seed, device="cpu"):
    torch.manual_seed(seed)
    model = DNN(X.shape[1], n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xt, yt),
        batch_size=512, shuffle=True)
    for _ in range(epochs):
        model.train()
        for xb, yb in dl:
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
    model.eval()
    return model


def per_class_evasion(model, X_clean, X_adv, y, classes, device="cpu"):
    """For each true attack class, fraction now misclassified after attack."""
    with torch.no_grad():
        pred_clean = model(torch.tensor(X_clean, dtype=torch.float32)).argmax(1).numpy()
        pred_adv = model(torch.tensor(X_adv, dtype=torch.float32)).argmax(1).numpy()
    out = {}
    for ci, cname in enumerate(classes):
        mask = (y == ci)
        if mask.sum() == 0:
            continue
        # evasion = correctly classified clean but wrong after attack
        correct_clean = (pred_clean[mask] == ci)
        wrong_adv = (pred_adv[mask] != ci)
        if correct_clean.sum() > 0:
            evasion = (wrong_adv & correct_clean).sum() / correct_clean.sum()
        else:
            evasion = float("nan")
        out[cname] = (int(mask.sum()), float(evasion))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--sample", type=int, default=150_000)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--eps_sweep", type=str, default="0.05,0.1,0.2,0.3",
                    help="Comma-separated eps values for the sweep.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_class", type=int, default=100,
                    help="Drop classes with fewer than this many samples "
                         "(too small to split/train/attack meaningfully).")
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)

    print(f"Loading {args.csv} ...")
    X, y, feats, classes = load_multiclass(args.csv, args.sample, args.seed)
    print(f"  {X.shape[0]:,} samples, {X.shape[1]} features, "
          f"{len(classes)} classes (raw)")

    # Drop classes too small to split/train on, remap to contiguous labels.
    X, y, classes, dropped = filter_small_classes(
        X, y, classes, args.min_class)
    if dropped:
        print(f"  dropped {len(dropped)} classes with < {args.min_class} "
              f"samples: {dropped}")
    n_classes = len(classes)
    print(f"  {X.shape[0]:,} samples, {n_classes} classes after filtering")

    # class distribution -> identify minority classes (bottom half by count)
    counts = pd.Series(y).value_counts().sort_values()
    print("  class distribution:")
    for ci, cnt in counts.items():
        print(f"    {classes[ci]:<24} {cnt:>8,} ({100*cnt/len(y):.2f}%)")
    minority = [ci for ci in counts.index
                if counts[ci] < counts.median()]
    minority_names = [classes[ci] for ci in minority]
    print(f"  minority classes: {minority_names}")

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=args.seed, stratify=y)

    scaler = MinMaxScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr).astype(np.float32)
    Xte_s = scaler.transform(Xte).astype(np.float32)

    # ---- ACAFS feature selection (on training split) -----------------
    print("\n=== ACAFS feature selection ===")
    acafs = ACAFS(minority_classes=minority, verbose=True)
    sel = acafs.fit(Xtr_s, ytr, feature_names=feats)
    print(f"  selected features: {acafs.selected_names()}")

    Xtr_sel = Xtr_s[:, sel]
    Xte_sel = Xte_s[:, sel]

    # ---- SMOTE-Tomek balance BOTH feature spaces ---------------------
    # Both the ACAFS-selected model and the all-features baseline receive
    # the SAME balancing treatment, so the only difference between them is
    # the feature set (ACAFS-selected vs all). This makes the comparison
    # fair: any difference in robustness is attributable to feature
    # selection, not to one model having seen balanced data and the other not.
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

    # ---- Train ACAFS model and baseline (both balanced) --------------
    print("\n=== Training models ===")
    model_acafs = train_dnn(Xtr_bal, ytr_bal, n_classes, args.epochs, args.seed)
    model_base = train_dnn(Xtr_base_bal, ytr_base_bal, n_classes, args.epochs, args.seed)

    from art.estimators.classification import PyTorchClassifier
    from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent

    def wrap(model, dim):
        return PyTorchClassifier(
            model=model, loss=nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
            input_shape=(dim,), nb_classes=n_classes, clip_values=(0.0, 1.0))

    art_acafs = wrap(model_acafs, len(sel))
    art_base = wrap(model_base, X.shape[1])

    # ---- Clean per-class accuracy (before any attack) ----------------
    # This tells us whether each class is detectable at all, which is
    # required to interpret the evasion numbers: 100% evasion on a class
    # the model detects well is a real robustness failure; 100% evasion on
    # a class it never detected is not informative.
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

    # ---- Attacks: FGSM + PGD across an EPS SWEEP ---------------------
    eps_list = [float(e) for e in args.eps_sweep.split(",")]
    print(f"\n=== Adversarial evasion across eps sweep {eps_list} ===")

    for atk_name, mk in [("FGSM", FastGradientMethod), ("PGD", ProjectedGradientDescent)]:
        for eps in eps_list:
            if atk_name == "FGSM":
                a_ac = mk(art_acafs, eps=eps, eps_step=eps, batch_size=512)
                a_bs = mk(art_base, eps=eps, eps_step=eps, batch_size=512)
            else:
                a_ac = mk(art_acafs, eps=eps, eps_step=eps/4, max_iter=20, batch_size=512)
                a_bs = mk(art_base, eps=eps, eps_step=eps/4, max_iter=20, batch_size=512)

            Xadv_ac = a_ac.generate(Xte_sel)
            Xadv_bs = a_bs.generate(Xte_s)
            ev_ac = per_class_evasion(model_acafs, Xte_sel, Xadv_ac, yte, classes)
            ev_bs = per_class_evasion(model_base, Xte_s, Xadv_bs, yte, classes)

            print(f"\n--- {atk_name}, eps={eps} ---")
            print(f"  {'class':<24} {'support':>8} {'ACAFS_ev':>9} {'Base_ev':>9}")
            for cname in classes:
                if cname in ev_ac:
                    sup, ea = ev_ac[cname]
                    _, eb = ev_bs.get(cname, (0, float('nan')))
                    tag = "  <-- minority" if cname in minority_names else ""
                    print(f"  {cname:<24} {sup:>8,} {ea:>9.3f} {eb:>9.3f}{tag}")

    print("\nInterpretation:")
    print("  Clean acc  = per-class accuracy with no attack (is the class detectable?)")
    print("  ACAFS_ev / Base_ev = per-class evasion at each eps for each model")


if __name__ == "__main__":
    main()