"""
Tests STRUCTURAL (ordering + subset/bound) constraints on an NF-v3 NIDS
dataset. Unlike derived-value equalities (buckets sum, throughput=bytes/dur),
these are LOGICALLY NECESSARY relationships that a correct exporter cannot
violate — so they should hold at ~100%. Any that don't reveal either a
data-quality issue or a wrong assumption about the field semantics.

For each constraint it reports the fraction of rows that satisfy it, and shows
a few violating rows so you can see the pattern.
"""

import argparse
import numpy as np
import pandas as pd

def pct(n, total):
    return f"{n:,} / {total:,} ({100.0 * n / total:.3f}%)" if total else "0"

def check_le(df, left, right, total, label, examples=3):
    """Check left <= right. Returns nothing; prints a report."""
    if left not in df.columns or right not in df.columns:
        missing = [c for c in (left, right) if c not in df.columns]
        print(f"  [SKIP] {label}: missing {missing}")
        return
    a = df[left].astype(float)
    b = df[right].astype(float)
    ok = (a <= b)
    n_ok = ok.sum()
    print(f"  {label}")
    print(f"      {left} <= {right}: {pct(n_ok, total)}")
    if n_ok < total:
        viol = df[~ok].head(examples)
        for i in viol.index:
            print(f"        VIOLATION: {left}={df[left][i]}  >  {right}={df[right][i]}")


def check_between(df, lo, mid, hi, total, label, examples=3):
    """Check lo <= mid <= hi (an average bounded by its min and max)."""
    for c in (lo, mid, hi):
        if c not in df.columns:
            print(f"  [SKIP] {label}: missing {c}")
            return
    a = df[lo].astype(float)
    m = df[mid].astype(float)
    b = df[hi].astype(float)
    ok = (a <= m) & (m <= b)
    n_ok = ok.sum()
    print(f"  {label}")
    print(f"      {lo} <= {mid} <= {hi}: {pct(n_ok, total)}")
    if n_ok < total:
        viol = df[~ok].head(examples)
        for i in viol.index:
            print(f"        VIOLATION: min={df[lo][i]}  avg={df[mid][i]}  max={df[hi][i]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--sample", type=int, default=300_000,
                    help="Sample this many rows (0 = all). Default 300k.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"Loading {args.csv} ...")
    df = pd.read_csv(args.csv, low_memory=False)
    df.columns = df.columns.str.strip()
    print(f"  Loaded {len(df):,} rows, {df.shape[1]} columns")

    if args.sample and args.sample < len(df):
        df = df.sample(args.sample, random_state=args.seed).reset_index(drop=True)
        print(f"  Sampled {len(df):,} rows")
    total = len(df)
    print()

    # ------------------------------------------------------------------
    # A. ORDERING CONSTRAINTS (min <= max, min <= avg <= max)
    #    Logically necessary — a minimum cannot exceed a maximum.
    # ------------------------------------------------------------------
    print("=" * 70)
    print("A. ORDERING CONSTRAINTS (min <= max)")
    print("=" * 70)
    check_le(df, "SHORTEST_FLOW_PKT", "LONGEST_FLOW_PKT", total,
             "Packet size: shortest <= longest")
    check_le(df, "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN", total,
             "IP packet length: min <= max")
    check_le(df, "MIN_TTL", "MAX_TTL", total,
             "TTL: min <= max")
    print()

    print("=" * 70)
    print("A2. AVG BOUNDED BY MIN/MAX (min <= avg <= max)")
    print("=" * 70)
    check_between(df, "SRC_TO_DST_IAT_MIN", "SRC_TO_DST_IAT_AVG",
                  "SRC_TO_DST_IAT_MAX", total,
                  "Src->Dst inter-arrival time: min <= avg <= max")
    check_between(df, "DST_TO_SRC_IAT_MIN", "DST_TO_SRC_IAT_AVG",
                  "DST_TO_SRC_IAT_MAX", total,
                  "Dst->Src inter-arrival time: min <= avg <= max")
    print()

    # ------------------------------------------------------------------
    # B. SUBSET / BOUND CONSTRAINTS (part <= whole)
    #    Retransmitted packets/bytes are a subset of total packets/bytes.
    # ------------------------------------------------------------------
    print("=" * 70)
    print("B. SUBSET CONSTRAINTS (retransmitted <= total)")
    print("=" * 70)
    check_le(df, "RETRANSMITTED_IN_PKTS", "IN_PKTS", total,
             "Retransmitted in-pkts <= in-pkts")
    check_le(df, "RETRANSMITTED_IN_BYTES", "IN_BYTES", total,
             "Retransmitted in-bytes <= in-bytes")
    check_le(df, "RETRANSMITTED_OUT_PKTS", "OUT_PKTS", total,
             "Retransmitted out-pkts <= out-pkts")
    check_le(df, "RETRANSMITTED_OUT_BYTES", "OUT_BYTES", total,
             "Retransmitted out-bytes <= out-bytes")
    print()

    # ------------------------------------------------------------------
    # C. PER-DIRECTION DURATION vs TOTAL (directional <= total flow dur)
    #    A one-directional stream cannot last longer than the whole flow.
    # ------------------------------------------------------------------
    print("=" * 70)
    print("C. DIRECTIONAL DURATION <= TOTAL FLOW DURATION")
    print("=" * 70)
    check_le(df, "DURATION_IN", "FLOW_DURATION_MILLISECONDS", total,
             "Client->server stream duration <= flow duration")
    check_le(df, "DURATION_OUT", "FLOW_DURATION_MILLISECONDS", total,
             "Server->client stream duration <= flow duration")
    print()

    # ------------------------------------------------------------------
    # D. NON-NEGATIVITY (counts and sizes cannot be negative)
    #    Basic validity floor — should be 100%.
    # ------------------------------------------------------------------
    print("=" * 70)
    print("D. NON-NEGATIVITY (counts/sizes >= 0)")
    print("=" * 70)
    for col in ["IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS",
                "FLOW_DURATION_MILLISECONDS", "MIN_TTL", "MAX_TTL"]:
        if col in df.columns:
            n_ok = (df[col].astype(float) >= 0).sum()
            print(f"  {col} >= 0: {pct(n_ok, total)}")
    print()

    print("=" * 70)
    print("Interpretation:")
    print("  ~100%  -> logically-necessary constraint confirmed; enforce in CEL.")
    print("  <100%  -> either a data-quality issue or the field means something")
    print("            other than assumed; inspect the violating rows before use.")
    print("=" * 70)


if __name__ == "__main__":
    main()