"""
Runs the core Paper-2 experiment on one NF-v3 dataset:

  1. Load + preprocess (drop identifiers, split, scale)
  2. Train a classifier on clean data
  3. Generate adversarial examples (FGSM/PGD) via ART
  4. Measure violation rate (how many adv examples break NF-v3 constraints)
  5. Apply the CEL, measure evasion rate before vs after (the robustness gap)

Reports, per attack:
  - violation_rate      : % of adv examples that were unrealizable
  - evasion_unconstrained: attack success before CEL
  - evasion_constrained : attack success after CEL
  - gap                 : the overstatement of the threat

"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from cel_nfv3 import NFv3ConstraintLayer

# Identifier / leakage columns to drop (not learnable features)
DROP_COLS = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT",
             "Attack", "Dataset", "FLOW_START_MILLISECONDS",
             "FLOW_END_MILLISECONDS"]
LABEL_CANDIDATES = ["Label", "label", "LABEL"]


class DNN(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 2))

    def forward(self, x):
        return self.net(x)


def load_nfv3(path, sample, seed):
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()

    label_col = next((c for c in LABEL_CANDIDATES if c in df.columns), None)
    if label_col is None:
        raise ValueError(f"No label column found. Columns: {list(df.columns)[:10]}...")
    y = (df[label_col].astype(int) != 0).astype(int).values if df[label_col].dtype != object \
        else (df[label_col].astype(str).str.upper() != "BENIGN").astype(int).values

    drop = [c for c in DROP_COLS if c in df.columns] + [label_col]
    X_df = df.drop(columns=drop)
    X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    feature_names = list(X_df.columns)
    X = X_df.values.astype(float)

    if sample and sample < len(X):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), sample, replace=False)
        X, y = X[idx], y[idx]

    return X, y, feature_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--sample", type=int, default=100_000)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading {args.csv} ...")
    X, y, feature_names = load_nfv3(args.csv, args.sample, args.seed)
    print(f"  {X.shape[0]:,} samples, {X.shape[1]} features, "
          f"attack ratio {y.mean():.3f}")

    # Split on ORIGINAL scale (CEL needs original-scale values)
    Xtr_o, Xte_o, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=args.seed, stratify=y)

    # Scale for the classifier; keep original-scale test set for the CEL
    scaler = MinMaxScaler().fit(Xtr_o)
    Xtr = scaler.transform(Xtr_o).astype(np.float32)
    Xte = scaler.transform(Xte_o).astype(np.float32)

    # Train DNN
    print("Training DNN ...")
    model = DNN(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    Xt = torch.tensor(Xtr); yt = torch.tensor(ytr)
    ds = torch.utils.data.TensorDataset(Xt, yt)
    dl = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=True)
    for ep in range(args.epochs):
        model.train()
        for xb, yb in dl:
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
    model.eval()

    with torch.no_grad():
        clean_pred = model(torch.tensor(Xte)).argmax(1).numpy()
    print(f"  Clean test: acc={accuracy_score(yte, clean_pred):.4f} "
          f"f1={f1_score(yte, clean_pred, zero_division=0):.4f}")

    # --- Attacks via ART -------------------------------------------------
    from art.estimators.classification import PyTorchClassifier
    from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent

    art_clf = PyTorchClassifier(
        model=model, loss=crit,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        input_shape=(X.shape[1],), nb_classes=2, clip_values=(0.0, 1.0))

    cel = NFv3ConstraintLayer(feature_names)

    # only attack the ATTACK-class test samples (evasion target)
    atk_mask = yte == 1
    Xte_atk = Xte[atk_mask]
    Xte_atk_o = Xte_o[atk_mask]      # original-scale counterpart
    yte_atk = yte[atk_mask]

    def evasion_rate(pred):
        # fraction of true-attack samples classified benign
        return float((pred == 0).mean())

    print("\n%-6s %10s %14s %13s %8s" %
          ("attack", "violation", "evasion_unc", "evasion_con", "gap"))

    for name, atk in [
        ("FGSM", FastGradientMethod(art_clf, eps=args.eps, eps_step=args.eps, batch_size=512)),
        ("PGD",  ProjectedGradientDescent(art_clf, eps=args.eps, eps_step=args.eps/4,
                                          max_iter=20, batch_size=512)),
    ]:
        # Generate adversarial examples in NORMALISED space (ART clip 0-1)
        Xadv_norm = atk.generate(Xte_atk)

        # Bring both clean and adv back to ORIGINAL scale for the CEL
        Xadv_o = scaler.inverse_transform(Xadv_norm)

        # violation rate (before CEL), on original scale
        vrate = cel.violation_rate(Xte_atk_o, Xadv_o)

        # evasion before CEL
        with torch.no_grad():
            pred_unc = model(torch.tensor(Xadv_norm.astype(np.float32))).argmax(1).numpy()
        ev_unc = evasion_rate(pred_unc)

        # apply CEL on original scale, renormalise, re-evaluate
        Xadv_con_o = cel.enforce(Xte_atk_o, Xadv_o)
        Xadv_con_norm = scaler.transform(Xadv_con_o).astype(np.float32)
        Xadv_con_norm = np.clip(Xadv_con_norm, 0.0, 1.0)
        with torch.no_grad():
            pred_con = model(torch.tensor(Xadv_con_norm)).argmax(1).numpy()
        ev_con = evasion_rate(pred_con)

        print("%-6s %9.3f %13.3f %12.3f %8.3f" %
              (name, vrate, ev_unc, ev_con, ev_unc - ev_con))

    print("\nInterpretation:")
    print("  violation  = fraction of adversarial examples that were unrealizable")
    print("  evasion_unc= attack success as normally reported")
    print("  evasion_con= attack success after enforcing NF-v3 constraints")
    print("  gap        = how much the unconstrained number overstates the threat")


if __name__ == "__main__":
    main()