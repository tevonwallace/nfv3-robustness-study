"""
Verifies three candidate consistency constraints on an NF-v3 NIDS dataset
(NF-UNSW-NB15-v3 or NF-CSE-CIC-IDS2018-v3) before using them in the CEL.

For each constraint it reports:
  - how many rows satisfy it exactly,
  - how many satisfy it within a relative tolerance,
  - a few example rows that fail

"""

import argparse
import numpy as np
import pandas as pd


def pct(n, total):
    return f"{n:,} / {total:,} ({100.0 * n / total:.2f}%)" if total else "0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Path to an NF-v3 CSV file")
    ap.add_argument("--sample", type=int, default=200_000,
                    help="Randomly sample this many rows (0 = use all). "
                         "Default 200k keeps it fast on multi-million-row files.")
    ap.add_argument("--tol", type=float, default=0.01,
                    help="Relative tolerance for the approximate checks "
                         "(0.01 = within 1%%). Default 0.01.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # --- Load ---------------------------------------------------------------
    print(f"Loading {args.csv} ...")
    df = pd.read_csv(args.csv, low_memory=False)
    # Strip whitespace from column names (the IAT columns have trailing spaces)
    df.columns = df.columns.str.strip()
    print(f"  Loaded {len(df):,} rows, {df.shape[1]} columns")

    if args.sample and args.sample < len(df):
        df = df.sample(args.sample, random_state=args.seed).reset_index(drop=True)
        print(f"  Sampled {len(df):,} rows for checking")

    total = len(df)
    print(f"  Relative tolerance for approximate checks: {args.tol}")
    print()

    # Helper: relative-closeness mask, safe against divide-by-zero
    def approx_equal(a, b, tol):
        a = a.astype(float)
        b = b.astype(float)
        denom = np.maximum(np.abs(b), 1e-9)
        return np.abs(a - b) / denom <= tol

    # =======================================================================
    # Constraint 1: packet-size buckets sum to total packets
    #   NUM_PKTS_UP_TO_128 + ... + NUM_PKTS_1024_TO_1514  ==  IN_PKTS + OUT_PKTS
    # =======================================================================
    print("=" * 70)
    print("Constraint 1: packet-size buckets sum to total packets")
    print("  sum(NUM_PKTS_*_BYTES)  ==  IN_PKTS + OUT_PKTS")
    print("=" * 70)

    bucket_cols = [
        "NUM_PKTS_UP_TO_128_BYTES",
        "NUM_PKTS_128_TO_256_BYTES",
        "NUM_PKTS_256_TO_512_BYTES",
        "NUM_PKTS_512_TO_1024_BYTES",
        "NUM_PKTS_1024_TO_1514_BYTES",
    ]
    needed = bucket_cols + ["IN_PKTS", "OUT_PKTS"]
    if all(c in df.columns for c in needed):
        bucket_sum = df[bucket_cols].sum(axis=1)
        total_pkts = df["IN_PKTS"] + df["OUT_PKTS"]

        exact = (bucket_sum == total_pkts).sum()
        approx = approx_equal(bucket_sum, total_pkts, args.tol).sum()

        print(f"  Exact match : {pct(exact, total)}")
        print(f"  Within tol  : {pct(approx, total)}")

        # Show a few failing rows
        fail = df[bucket_sum != total_pkts].head(5)
        if len(fail):
            print("  Example mismatches (bucket_sum vs IN+OUT pkts):")
            for i in fail.index:
                print(f"    bucket_sum={int(bucket_sum[i])}  "
                      f"IN+OUT={int(total_pkts[i])}  "
                      f"(IN={int(df['IN_PKTS'][i])}, OUT={int(df['OUT_PKTS'][i])})")
    else:
        missing = [c for c in needed if c not in df.columns]
        print(f"  SKIPPED — missing columns: {missing}")
    print()

    # =======================================================================
    # Constraint 2: flow duration matches end - start
    #   FLOW_DURATION_MILLISECONDS  ==  FLOW_END_MS - FLOW_START_MS
    # =======================================================================
    print("=" * 70)
    print("Constraint 2: duration matches end - start")
    print("  FLOW_DURATION_MILLISECONDS  ==  FLOW_END_MS - FLOW_START_MS")
    print("=" * 70)

    needed = ["FLOW_DURATION_MILLISECONDS",
              "FLOW_END_MILLISECONDS", "FLOW_START_MILLISECONDS"]
    if all(c in df.columns for c in needed):
        computed = df["FLOW_END_MILLISECONDS"] - df["FLOW_START_MILLISECONDS"]
        stated = df["FLOW_DURATION_MILLISECONDS"]

        exact = (stated == computed).sum()
        approx = approx_equal(stated, computed, args.tol).sum()
        # Also report exact-within-1ms, since millisecond rounding is common
        within_1ms = (np.abs(stated.astype(float) - computed.astype(float)) <= 1).sum()

        print(f"  Exact match     : {pct(exact, total)}")
        print(f"  Within 1 ms     : {pct(within_1ms, total)}")
        print(f"  Within tol      : {pct(approx, total)}")

        fail = df[stated != computed].head(5)
        if len(fail):
            print("  Example mismatches (stated vs end-start, ms):")
            for i in fail.index:
                print(f"    stated={stated[i]}  end-start={computed[i]}")
    else:
        missing = [c for c in needed if c not in df.columns]
        print(f"  SKIPPED — missing columns: {missing}")
    print()

    # =======================================================================
    # Constraint 3: throughput matches bytes / duration
    #   SRC_TO_DST_AVG_THROUGHPUT (bps)  ==  IN_BYTES * 8 / (duration_sec)
    #   Note: throughput is in bits/sec, so bytes * 8. Duration in seconds.
    # =======================================================================
    print("=" * 70)
    print("Constraint 3: throughput matches bytes / duration")
    print("  SRC_TO_DST_AVG_THROUGHPUT (bps)  ~=  IN_BYTES * 8 / duration_sec")
    print("=" * 70)

    needed = ["SRC_TO_DST_AVG_THROUGHPUT", "IN_BYTES",
              "FLOW_DURATION_MILLISECONDS"]
    if all(c in df.columns for c in needed):
        # duration in seconds; guard against zero-duration flows
        dur_sec = df["FLOW_DURATION_MILLISECONDS"].astype(float) / 1000.0
        valid = dur_sec > 0
        sub = df[valid].copy()
        dur_sec_v = dur_sec[valid]

        # Try bits/sec (bytes * 8) — the label says "thpt (bps)" = bits per sec
        computed_bps = sub["IN_BYTES"].astype(float) * 8.0 / dur_sec_v
        stated = sub["SRC_TO_DST_AVG_THROUGHPUT"].astype(float)

        approx_bps = (approx_equal(stated, computed_bps, args.tol)).sum()

        # Also try bytes/sec interpretation in case the label is loose
        computed_Bps = sub["IN_BYTES"].astype(float) / dur_sec_v
        approx_Bps = (approx_equal(stated, computed_Bps, args.tol)).sum()

        vtotal = len(sub)
        print(f"  Rows with duration>0 : {pct(vtotal, total)}")
        print(f"  Match as BITS/sec  (bytes*8/sec): {pct(approx_bps, vtotal)}")
        print(f"  Match as BYTES/sec (bytes/sec)  : {pct(approx_Bps, vtotal)}")

        # Show a few examples so you can eyeball the ratio
        print("  Example rows (stated | bytes*8/sec | bytes/sec):")
        for i in sub.index[:5]:
            b = float(df['IN_BYTES'][i])
            d = float(df['FLOW_DURATION_MILLISECONDS'][i]) / 1000.0
            if d > 0:
                print(f"    stated={df['SRC_TO_DST_AVG_THROUGHPUT'][i]:.1f}  "
                      f"bits/s={b*8/d:.1f}  bytes/s={b/d:.1f}")
    else:
        missing = [c for c in needed if c not in df.columns]
        print(f"  SKIPPED — missing columns: {missing}")
    print()

    print("=" * 70)
    print("Interpretation guide:")
    print("  - >99% exact  -> enforce as a hard equality in the CEL.")
    print("  - high within-tol but low exact -> enforce with a tolerance.")
    print("  - low match   -> the exporter computes it differently; do NOT")
    print("                   treat as a constraint (or investigate the formula).")
    print("=" * 70)


if __name__ == "__main__":
    main()