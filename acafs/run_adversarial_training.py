"""
run_adversarial_training.py
---------------------------
Adds a defense baseline: PGD adversarial training (Madry et al., 2018).

Design (identical preprocessing to run_acafs_robustness.py for comparability):
  - Uses the ALL-FEATURES space (the standard NetFlow NIDS), SMOTE-Tomek
    balanced, to isolate the effect of the DEFENSE (not feature selection).
  - "Standard" model: ordinary training (train_dnn), same as the paper baseline.
  - "AT" model: Madry-style PGD adversarial training at a fixed L-inf budget
    (--at_eps), implemented transparently in PyTorch.
  - Both evaluated on: clean per-class accuracy, and per-class evasion under
    FGSM, PGD, and (optionally) C&W.

Usage:
    python run_adversarial_training.py NF-UNSW-NB15-v3.csv  --sample 150000
    python run_adversarial_training.py NF-CICIDS2018-v3.csv --sample 150000
    # add C&W to the comparison (slower):
    python run_adversarial_training.py NF-UNSW-NB15-v3.csv --with_cw --cw_sample 1000
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from acafs_nfv3 import balance_smote_tomek
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


def pgd_perturb(model, xb, yb, eps, alpha, steps):
    """
    Madry-style L-inf PGD used INSIDE adversarial training.
    Transparent implementation (no ART) so the training loop is auditable.
    Operates in the normalized [0,1] feature space.
    """
    crit = nn.CrossEntropyLoss()
    x0 = xb.detach()
    # random start inside the eps-ball, then clip to valid [0,1]
    delta = torch.empty_like(x0).uniform_(-eps, eps)
    x_adv = torch.clamp(x0 + delta, 0.0, 1.0).detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = crit(model(x_adv), yb)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        # project back into the eps-ball around x0, then into [0,1]
        x_adv = torch.max(torch.min(x_adv, x0 + eps), x0 - eps)
        x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()
    return x_adv


def train_dnn_adversarial(X, y, n_classes, epochs, seed, eps, steps,
                          alpha=None, device="cpu"):
    """Train a DNN on PGD adversarial examples (Madry adversarial training)."""
    if alpha is None:
        alpha = eps / 4.0
    torch.manual_seed(seed)
    model = DNN(X.shape[1], n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xt, yt), batch_size=512, shuffle=True)
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            x_adv = pgd_perturb(model, xb, yb, eps, alpha, steps)
            opt.zero_grad()
            crit(model(x_adv), yb).backward()
            opt.step()
        print(f"    AT epoch {ep+1}/{epochs} done")
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--sample", type=int, default=150_000)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_class", type=int, default=100)
    ap.add_argument("--eps_sweep", type=str, default="0.05,0.1,0.2,0.3")
    ap.add_argument("--at_eps", type=float, default=0.1,
                    help="L-inf budget used DURING adversarial training.")
    ap.add_argument("--at_steps", type=int, default=10,
                    help="PGD steps used during adversarial training.")
    ap.add_argument("--with_cw", action="store_true",
                    help="Also compare under the C&W L2 attack (slower).")
    ap.add_argument("--cw_sample", type=int, default=1000)
    ap.add_argument("--cw_max_iter", type=int, default=10)
    ap.add_argument("--cw_bsteps", type=int, default=5)
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

    # All-features space, SMOTE-Tomek balanced (isolates the DEFENSE effect).
    print("\n=== SMOTE-Tomek balancing (all-features) ===")
    try:
        Xtr_bal, ytr_bal = balance_smote_tomek(Xtr_s, ytr, args.seed)
        print(f"  balanced train: {Xtr_bal.shape[0]:,} samples")
    except Exception as e:
        print(f"  balancing skipped ({e}); using unbalanced")
        Xtr_bal, ytr_bal = Xtr_s, ytr

    print("\n=== Training STANDARD model (ordinary training) ===")
    model_std = train_dnn(Xtr_bal, ytr_bal, n_classes, args.epochs, args.seed)

    print(f"\n=== Training ADVERSARIALLY-TRAINED model "
          f"(PGD-AT, eps={args.at_eps}, steps={args.at_steps}) ===")
    model_at = train_dnn_adversarial(
        Xtr_bal, ytr_bal, n_classes, args.epochs, args.seed,
        eps=args.at_eps, steps=args.at_steps)

    from art.estimators.classification import PyTorchClassifier
    from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent

    def wrap(model):
        return PyTorchClassifier(
            model=model, loss=nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
            input_shape=(X.shape[1],), nb_classes=n_classes, clip_values=(0.0, 1.0))

    art_std, art_at = wrap(model_std), wrap(model_at)

    # ---- Clean per-class accuracy: did AT cost clean detection? -------
    print("\n=== Clean per-class accuracy (no attack) ===")
    ca_std = clean_accuracy(model_std, Xte_s, yte, classes)
    ca_at = clean_accuracy(model_at, Xte_s, yte, classes)
    print(f"  {'class':<24} {'support':>8} {'STD_acc':>9} {'AT_acc':>9}")
    for cname in classes:
        if cname in ca_std:
            sup, s = ca_std[cname]
            _, a = ca_at.get(cname, (0, float('nan')))
            tag = "  <-- minority" if cname in minority_names else ""
            print(f"  {cname:<24} {sup:>8,} {s:>9.3f} {a:>9.3f}{tag}")

    # ---- Evasion under FGSM / PGD across the eps sweep ----------------
    eps_list = [float(e) for e in args.eps_sweep.split(",")]
    print(f"\n=== Evasion: STANDARD vs ADVERSARIALLY-TRAINED, eps sweep {eps_list} ===")
    for atk_name, mk in [("FGSM", FastGradientMethod), ("PGD", ProjectedGradientDescent)]:
        for eps in eps_list:
            if atk_name == "FGSM":
                a_std = mk(art_std, eps=eps, eps_step=eps, batch_size=512)
                a_at = mk(art_at, eps=eps, eps_step=eps, batch_size=512)
            else:
                a_std = mk(art_std, eps=eps, eps_step=eps/4, max_iter=20, batch_size=512)
                a_at = mk(art_at, eps=eps, eps_step=eps/4, max_iter=20, batch_size=512)
            Xadv_std = a_std.generate(Xte_s)
            Xadv_at = a_at.generate(Xte_s)
            ev_std = per_class_evasion(model_std, Xte_s, Xadv_std, yte, classes)
            ev_at = per_class_evasion(model_at, Xte_s, Xadv_at, yte, classes)
            print(f"\n--- {atk_name}, eps={eps} ---")
            print(f"  {'class':<24} {'support':>8} {'STD_ev':>8} {'AT_ev':>8}")
            for cname in classes:
                if cname in ev_std:
                    sup, es = ev_std[cname]
                    _, ea = ev_at.get(cname, (0, float('nan')))
                    tag = "  <-- minority" if cname in minority_names else ""
                    print(f"  {cname:<24} {sup:>8,} {es:>8.3f} {ea:>8.3f}{tag}")

    # ---- Optional: C&W comparison ------------------------------------
    if args.with_cw:
        from art.attacks.evasion import CarliniL2Method
        rng = np.random.default_rng(args.seed)
        n_sub = min(args.cw_sample, Xte_s.shape[0])
        sub = rng.choice(Xte_s.shape[0], n_sub, replace=False)
        Xsub, ysub = Xte_s[sub], yte[sub]
        print(f"\n=== C&W L2: STANDARD vs AT (subsample of {n_sub:,} flows) ===")
        print("  NOTE: support below is per-class count in the C&W subsample.")
        cw_std = CarliniL2Method(art_std, targeted=False, max_iter=args.cw_max_iter,
                                 binary_search_steps=args.cw_bsteps,
                                 batch_size=128, verbose=False)
        cw_at = CarliniL2Method(art_at, targeted=False, max_iter=args.cw_max_iter,
                                binary_search_steps=args.cw_bsteps,
                                batch_size=128, verbose=False)
        Xadv_std = cw_std.generate(Xsub)
        Xadv_at = cw_at.generate(Xsub)
        ev_std = per_class_evasion(model_std, Xsub, Xadv_std, ysub, classes)
        ev_at = per_class_evasion(model_at, Xsub, Xadv_at, ysub, classes)
        print(f"\n--- C&W L2 ---")
        print(f"  {'class':<24} {'support':>8} {'STD_ev':>8} {'AT_ev':>8}")
        for cname in classes:
            if cname in ev_std:
                sup, es = ev_std[cname]
                _, ea = ev_at.get(cname, (0, float('nan')))
                tag = "  <-- minority" if cname in minority_names else ""
                print(f"  {cname:<24} {sup:>8,} {es:>8.3f} {ea:>8.3f}{tag}")

if __name__ == "__main__":
    main()