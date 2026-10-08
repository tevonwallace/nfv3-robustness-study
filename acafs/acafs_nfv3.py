"""
Faithful reproduction of the ACAFS framework (Mchina et al., Frontiers in Big
Data, 2026) adapted to the NF-v3 NetFlow feature sxpace.

ACAFS = Adaptive Class-Aware Feature Selection. Reproduced exactly from the
paper's Section 2.3:

  Stage 1a  Global Mutual Information per feature
  Stage 1b  Permutation-based adaptive threshold:
            permute labels 100x, recompute MI, tau = 99th percentile of null
  Stage 1c  Class-Aware Composite MI:
            Composite(f) = alpha*MI_global(f) + (1-alpha)*min_c(MI_OvR(f,c))
            alpha = 0.6
  Stage 1d  Per-class coverage: guarantee ceil(log2(C)) features per class
  Stage 2   XGBoost refinement: choose final k by macro-F1 + minority recall,
            within tolerance eps = 0.01
  Post      SMOTE-Tomek balancing on the SELECTED feature space (train only)

Dependencies: numpy, pandas, scikit-learn, xgboost, imbalanced-learn
"""

import numpy as np
import pandas as pd
from math import log2, ceil
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

try:
    from imblearn.combine import SMOTETomek
    HAVE_IMBLEARN = True
except ImportError:
    HAVE_IMBLEARN = False


# Paper-specified constants (Section 2.3 / 2.7)
ALPHA = 0.6              # Composite MI weight (global vs minority)
N_PERMUTATIONS = 100     # permutation null distribution size
NULL_PERCENTILE = 99     # tau = 99th percentile of null MI
EPS_TOLERANCE = 0.01     # Stage 2 admissible macro-F1 margin
RANDOM_SEED = 42


class ACAFS:
    """
    Adaptive Class-Aware Feature Selection, reproduced from Mchina et al. 2026.
    """

    def __init__(self, alpha=ALPHA, n_permutations=N_PERMUTATIONS,
                 null_percentile=NULL_PERCENTILE, eps=EPS_TOLERANCE,
                 minority_classes=None, random_state=RANDOM_SEED,
                 max_stage1_features=40, verbose=True):
        self.alpha = alpha
        self.n_permutations = n_permutations
        self.null_percentile = null_percentile
        self.eps = eps
        self.minority_classes = minority_classes  # list of class labels to weight in Stage 2
        self.random_state = random_state
        self.max_stage1_features = max_stage1_features
        self.verbose = verbose

        self.selected_features_ = None
        self.feature_names_ = None
        self.tau_ = None

    def _log(self, *a):
        if self.verbose:
            print(*a)

    # ---- Stage 1a: global MI ------------------------------------------
    def _global_mi(self, X, y):
        return mutual_info_classif(X, y, random_state=self.random_state)

    # ---- Stage 1b: permutation-based adaptive threshold ---------------
    def _permutation_threshold(self, X, y):
        rng = np.random.default_rng(self.random_state)
        null_scores = []
        for i in range(self.n_permutations):
            y_perm = rng.permutation(y)
            mi_perm = mutual_info_classif(X, y_perm, random_state=self.random_state + i)
            null_scores.append(mi_perm)
        null_scores = np.concatenate(null_scores)
        tau = np.percentile(null_scores, self.null_percentile)
        return tau

    # ---- Stage 1c: class-aware composite MI ---------------------------
    def _composite_mi(self, X, y, mi_global):
        classes = np.unique(y)
        # One-vs-Rest MI per class
        ovr_mi = np.zeros((len(classes), X.shape[1]))
        for ci, c in enumerate(classes):
            y_bin = (y == c).astype(int)
            ovr_mi[ci] = mutual_info_classif(
                X, y_bin, random_state=self.random_state)
        min_ovr = ovr_mi.min(axis=0)  # worst-case class-specific relevance
        composite = self.alpha * mi_global + (1 - self.alpha) * min_ovr
        return composite, ovr_mi, classes

    # ---- Stage 1d: per-class coverage --------------------------------
    def _per_class_coverage(self, ovr_mi, classes, already_selected):
        """Guarantee ceil(log2(C)) top features per class."""
        C = len(classes)
        min_per_class = ceil(log2(C)) if C > 1 else 1
        covered = set(already_selected)
        for ci in range(len(classes)):
            top_for_class = np.argsort(ovr_mi[ci])[::-1][:min_per_class]
            covered.update(top_for_class.tolist())
        return sorted(covered)

    # ---- Stage 2: XGBoost refinement ---------------------------------
    def _xgb_refine(self, X, y, candidate_idx, composite):
        """Choose final feature count k by macro-F1 + minority recall."""
        # rank candidates by composite score
        ranked = sorted(candidate_idx,
                        key=lambda j: composite[j], reverse=True)
        best_score = -1
        best_k = len(ranked)
        best_macro = -1
        results = []

        skf = StratifiedKFold(n_splits=3, shuffle=True,
                              random_state=self.random_state)
        # evaluate increasing k
        k_values = range(max(5, ceil(log2(len(np.unique(y))))), len(ranked) + 1, 2)
        for k in k_values:
            feats = ranked[:k]
            macro_f1s, min_recalls = [], []
            for tr, va in skf.split(X, y):
                clf = xgb.XGBClassifier(
                    n_estimators=100, max_depth=6,
                    random_state=self.random_state,
                    tree_method="hist", verbosity=0, n_jobs=1)
                # remap labels to 0..C-1 for xgboost
                classes = np.unique(y)
                cmap = {c: i for i, c in enumerate(classes)}
                y_tr = np.array([cmap[v] for v in y[tr]])
                y_va = np.array([cmap[v] for v in y[va]])
                clf.fit(X[tr][:, feats], y_tr)
                pred = clf.predict(X[va][:, feats])
                macro_f1s.append(f1_score(y_va, pred, average="macro",
                                          zero_division=0))
                # minority-class recall
                if self.minority_classes:
                    min_idx = [cmap[c] for c in self.minority_classes
                               if c in cmap]
                    if min_idx:
                        rec = recall_score(y_va, pred, labels=min_idx,
                                           average="macro", zero_division=0)
                        min_recalls.append(rec)
            macro = np.mean(macro_f1s)
            minrec = np.mean(min_recalls) if min_recalls else 0.0
            # paper's objective: macro-F1 + minority recall, penalize instability
            std_pen = np.std(macro_f1s)
            score = macro + minrec - std_pen
            results.append((k, macro, minrec, score))
            if macro > best_macro:
                best_macro = macro

        # apply tolerance: smallest k whose macro-F1 within eps of best, best score
        admissible = [r for r in results if r[1] >= best_macro - self.eps]
        if admissible:
            chosen = max(admissible, key=lambda r: r[3])
        else:
            chosen = max(results, key=lambda r: r[3])
        best_k = chosen[0]
        self._log(f"  Stage 2: chose k={best_k} "
                  f"(macro-F1={chosen[1]:.4f}, min-recall={chosen[2]:.4f})")
        return ranked[:best_k]

    # ---- Full pipeline -----------------------------------------------
    def fit(self, X, y, feature_names=None):
        """
        Run the full ACAFS selection. X: (n,d) numpy, y: (n,) labels.
        Returns list of selected feature indices; also stored in
        self.selected_features_.
        """
        self.feature_names_ = (list(feature_names)
                               if feature_names is not None
                               else [f"f{i}" for i in range(X.shape[1])])

        self._log("Stage 1a: global MI ...")
        mi_global = self._global_mi(X, y)

        self._log("Stage 1b: permutation threshold ...")
        self.tau_ = self._permutation_threshold(X, y)
        significant = np.where(mi_global > self.tau_)[0]
        self._log(f"  tau={self.tau_:.5f}, "
                  f"{len(significant)}/{X.shape[1]} features significant")

        self._log("Stage 1c: class-aware composite MI ...")
        composite, ovr_mi, classes = self._composite_mi(X, y, mi_global)

        # rank significant features by composite, cap candidate pool
        sig_ranked = sorted(significant, key=lambda j: composite[j],
                            reverse=True)[:self.max_stage1_features]

        self._log("Stage 1d: per-class coverage ...")
        candidates = self._per_class_coverage(ovr_mi, classes, sig_ranked)
        self._log(f"  candidate pool: {len(candidates)} features")

        self._log("Stage 2: XGBoost refinement ...")
        selected = self._xgb_refine(X, y, candidates, composite)

        self.selected_features_ = sorted(selected)
        self._log(f"ACAFS selected {len(self.selected_features_)} features.")
        return self.selected_features_

    def transform(self, X):
        if self.selected_features_ is None:
            raise RuntimeError("Call fit() first.")
        return X[:, self.selected_features_]

    def selected_names(self):
        return [self.feature_names_[i] for i in self.selected_features_]


def balance_smote_tomek(X, y, random_state=RANDOM_SEED):
    if not HAVE_IMBLEARN:
        raise ImportError("imbalanced-learn required: pip install imbalanced-learn")
    st = SMOTETomek(random_state=random_state)
    X_bal, y_bal = st.fit_resample(X, y)
    return X_bal, y_bal