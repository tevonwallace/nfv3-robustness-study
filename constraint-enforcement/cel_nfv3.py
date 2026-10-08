"""
Constraint Enforcement Layer (CEL) for the standardized NF-v3 NetFlow feature
space (NF-UNSW-NB15-v3 / NF-CSE-CIC-IDS2018-v3).

  ENFORCED (held at ~100% on both datasets):
    - Ordering:  SHORTEST_FLOW_PKT <= LONGEST_FLOW_PKT
                 MIN_IP_PKT_LEN   <= MAX_IP_PKT_LEN
                 MIN_TTL          <= MAX_TTL
    - Subset:    RETRANSMITTED_IN_PKTS  <= IN_PKTS
                 RETRANSMITTED_IN_BYTES <= IN_BYTES
                 RETRANSMITTED_OUT_PKTS <= OUT_PKTS
                 RETRANSMITTED_OUT_BYTES<= OUT_BYTES
    - Directional duration (with 1 ms tolerance):
                 DURATION_IN  <= FLOW_DURATION_MILLISECONDS
                 DURATION_OUT <= FLOW_DURATION_MILLISECONDS
    - Non-negativity on all counts/sizes.
    - IAT ordering (min <= avg <= max) enforced EXCEPT the characterized
      min==max special case, where nProbe reports avg below min/max.

  NOT ENFORCED (empirically failed — see analysis):
    - sum(NUM_PKTS_*_BYTES) == IN_PKTS + OUT_PKTS   (independent counters)
    - SRC_TO_DST_AVG_THROUGHPUT == IN_BYTES*8/dur   (nProbe internal averaging)

The CEL operates on the ORIGINAL feature scale (not normalised), because the
constraints are defined over real feature semantics. Normalise afterward for
the classifier.
"""

import numpy as np


# Millisecond rounding tolerance for duration constraints (from data: off-by-1).
DURATION_TOL_MS = 1.0


class NFv3ConstraintLayer:
    """
    Enforces the empirically-validated NF-v3 constraints on adversarial
    examples by projecting them back onto the space of realizable flows.
    """

    # Ordered pairs (low, high): enforce low <= high
    ORDER_PAIRS = [
        ("SHORTEST_FLOW_PKT", "LONGEST_FLOW_PKT"),
        ("MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN"),
        ("MIN_TTL", "MAX_TTL"),
    ]

    # Subset pairs (part, whole): enforce part <= whole
    SUBSET_PAIRS = [
        ("RETRANSMITTED_IN_PKTS", "IN_PKTS"),
        ("RETRANSMITTED_IN_BYTES", "IN_BYTES"),
        ("RETRANSMITTED_OUT_PKTS", "OUT_PKTS"),
        ("RETRANSMITTED_OUT_BYTES", "OUT_BYTES"),
    ]

    # Directional duration <= total flow duration (with tolerance)
    DURATION_PAIRS = [
        ("DURATION_IN", "FLOW_DURATION_MILLISECONDS"),
        ("DURATION_OUT", "FLOW_DURATION_MILLISECONDS"),
    ]

    # IAT triples (min, avg, max): enforce min <= avg <= max, with the
    # nProbe min==max special case left alone.
    IAT_TRIPLES = [
        ("SRC_TO_DST_IAT_MIN", "SRC_TO_DST_IAT_AVG", "SRC_TO_DST_IAT_MAX"),
        ("DST_TO_SRC_IAT_MIN", "DST_TO_SRC_IAT_AVG", "DST_TO_SRC_IAT_MAX"),
    ]

    # Features that must be non-negative
    NONNEG = [
        "IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS",
        "FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT",
        "MIN_TTL", "MAX_TTL", "SHORTEST_FLOW_PKT", "LONGEST_FLOW_PKT",
        "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN",
        "RETRANSMITTED_IN_PKTS", "RETRANSMITTED_IN_BYTES",
        "RETRANSMITTED_OUT_PKTS", "RETRANSMITTED_OUT_BYTES",
    ]

    def __init__(self, feature_names):
        """
        feature_names : list[str] — column order of the feature matrix the
                        CEL will receive (original scale).
        """
        self.feature_names = list(feature_names)
        self.idx = {name: i for i, name in enumerate(self.feature_names)}

    # ------------------------------------------------------------------
    def _has(self, *cols):
        return all(c in self.idx for c in cols)

    # ------------------------------------------------------------------
    def enforce(self, X_original, X_adv):
        """
        Project adversarial examples onto the valid space.

        X_original : (n, d) clean samples (original scale)
        X_adv      : (n, d) adversarial samples (original scale)
        returns    : (n, d) constraint-valid adversarial samples
        """
        X = X_adv.copy().astype(float)
        Xo = X_original.astype(float)

        # 1. Non-negativity — clip up to 0
        for c in self.NONNEG:
            if c in self.idx:
                j = self.idx[c]
                X[:, j] = np.maximum(X[:, j], 0.0)

        # 2. Ordering: if low > high, the attack broke the order.
        #    Restore both endpoints to their original values (minimal, safe fix).
        for lo, hi in self.ORDER_PAIRS:
            if self._has(lo, hi):
                jl, jh = self.idx[lo], self.idx[hi]
                broken = X[:, jl] > X[:, jh]
                X[broken, jl] = Xo[broken, jl]
                X[broken, jh] = Xo[broken, jh]

        # 3. Subset: part must be <= whole. Clip part down to whole.
        for part, whole in self.SUBSET_PAIRS:
            if self._has(part, whole):
                jp, jw = self.idx[part], self.idx[whole]
                X[:, jp] = np.minimum(X[:, jp], X[:, jw])

        # 4. Directional duration <= flow duration (+ tolerance).
        #    Clip the directional value down to (flow_dur + tol).
        for d_dir, d_tot in self.DURATION_PAIRS:
            if self._has(d_dir, d_tot):
                jd, jt = self.idx[d_dir], self.idx[d_tot]
                X[:, jd] = np.minimum(X[:, jd], X[:, jt] + DURATION_TOL_MS)

        # 5. IAT ordering: min <= avg <= max, except the nProbe min==max case.
        for mn, av, mx in self.IAT_TRIPLES:
            if self._has(mn, av, mx):
                jm, ja, jx = self.idx[mn], self.idx[av], self.idx[mx]
                # rows where the special case does NOT apply
                normal = ~np.isclose(X[:, jm], X[:, jx])
                # clip avg into [min, max] on normal rows only
                lo = np.minimum(X[:, jm], X[:, jx])
                hi = np.maximum(X[:, jm], X[:, jx])
                clipped = np.clip(X[:, ja], lo, hi)
                X[normal, ja] = clipped[normal]

        return X

    # ------------------------------------------------------------------
    def violation_rate(self, X_original, X_adv):
        """
        Fraction of adversarial rows that violate at least one enforced
        constraint BEFORE projection. This is the paper's key measurement.
        """
        X = X_adv.astype(float)
        Xo = X_original.astype(float)
        n = len(X)
        violated = np.zeros(n, dtype=bool)

        for c in self.NONNEG:
            if c in self.idx:
                violated |= X[:, self.idx[c]] < 0

        for lo, hi in self.ORDER_PAIRS:
            if self._has(lo, hi):
                violated |= X[:, self.idx[lo]] > X[:, self.idx[hi]]

        for part, whole in self.SUBSET_PAIRS:
            if self._has(part, whole):
                violated |= X[:, self.idx[part]] > X[:, self.idx[whole]]

        for d_dir, d_tot in self.DURATION_PAIRS:
            if self._has(d_dir, d_tot):
                violated |= X[:, self.idx[d_dir]] > X[:, self.idx[d_tot]] + DURATION_TOL_MS

        for mn, av, mx in self.IAT_TRIPLES:
            if self._has(mn, av, mx):
                jm, ja, jx = self.idx[mn], self.idx[av], self.idx[mx]
                special = np.isclose(X[:, jm], X[:, jx])
                lo = np.minimum(X[:, jm], X[:, jx])
                hi = np.maximum(X[:, jm], X[:, jx])
                out_of_range = (X[:, ja] < lo) | (X[:, ja] > hi)
                violated |= (out_of_range & ~special)

        return float(violated.mean())

    # ------------------------------------------------------------------
    def per_constraint_violation(self, X_adv, X_original=None):
        """
        Diagnostic: fraction of rows violating each individual constraint,
        for the paper's per-constraint breakdown table.
        """
        X = X_adv.astype(float)
        out = {}

        for c in self.NONNEG:
            if c in self.idx:
                out[f"nonneg:{c}"] = float((X[:, self.idx[c]] < 0).mean())

        for lo, hi in self.ORDER_PAIRS:
            if self._has(lo, hi):
                out[f"order:{lo}<= {hi}"] = float(
                    (X[:, self.idx[lo]] > X[:, self.idx[hi]]).mean())

        for part, whole in self.SUBSET_PAIRS:
            if self._has(part, whole):
                out[f"subset:{part}<= {whole}"] = float(
                    (X[:, self.idx[part]] > X[:, self.idx[whole]]).mean())

        for d_dir, d_tot in self.DURATION_PAIRS:
            if self._has(d_dir, d_tot):
                out[f"dur:{d_dir}<= {d_tot}"] = float(
                    (X[:, self.idx[d_dir]] > X[:, self.idx[d_tot]] + DURATION_TOL_MS).mean())

        for mn, av, mx in self.IAT_TRIPLES:
            if self._has(mn, av, mx):
                jm, ja, jx = self.idx[mn], self.idx[av], self.idx[mx]
                special = np.isclose(X[:, jm], X[:, jx])
                lo = np.minimum(X[:, jm], X[:, jx])
                hi = np.maximum(X[:, jm], X[:, jx])
                oor = ((X[:, ja] < lo) | (X[:, ja] > hi)) & ~special
                out[f"iat:{mn}<= {av}<= {mx}"] = float(oor.mean())

        return out