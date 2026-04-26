"""
analyze_spread_log.py
─────────────────────
Summarize the spread log from Postgres (Railway) or local CSV.

Answers:
  1. Typical opening spread (minute 0-1)?
  2. Does spread narrow as the window ages?
  3. Does mid-price at open already reflect the prior direction?
     (i.e. is the mean-reversion signal already priced in?)
  4. Mid price by streak length?
  5. Is the edge viable after the real spread?

Usage:
    # Uses DATABASE_URL if set, otherwise reads data/spread_log.csv
    python analyze_spread_log.py
"""

import os
import sys
import numpy as np
import pandas as pd
from dotenv import load_dotenv

_here = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_here, ".env"))


def load_data() -> pd.DataFrame:
    db_url = os.getenv("DATABASE_URL", "")
    if db_url:
        try:
            import psycopg2
            conn = psycopg2.connect(db_url, sslmode="require")
            df = pd.read_sql("SELECT * FROM spread_log ORDER BY timestamp_utc",
                             conn)
            conn.close()
            print(f"Loaded {len(df):,} rows from Postgres")
            return df
        except Exception as e:
            print(f"[db] Could not load from Postgres: {e}")
            print("Falling back to CSV…")

    csv_path = os.path.join(_here, "data", "spread_log.csv")
    if not os.path.exists(csv_path):
        print(f"No spread log at {csv_path}")
        print("Run observe_live_spreads.py first.")
        sys.exit(1)
    df = pd.read_csv(csv_path, parse_dates=["timestamp_utc"])
    print(f"Loaded {len(df):,} rows from CSV")
    return df


def section(title: str):
    print(f"\n── {title} {'─' * max(0, 54 - len(title))}")


def main():
    df = load_data()
    print(f"Windows covered: {df['ticker'].nunique()}")
    print(f"Date range: {df['timestamp_utc'].min()} → {df['timestamp_utc'].max()}")

    if len(df) < 5:
        print("\nNot enough data yet — keep the observer running.")
        return

    df["min_bucket"] = df["minutes_elapsed"].clip(upper=4).astype(int)

    # ── 1. Spread by minute ────────────────────────────────────────────────────
    section("Spread by minute into window")
    sprd = (df[df["spread_cents"].notna()]
            .groupby("min_bucket")["spread_cents"]
            .agg(n="count", avg="mean", med="median", std="std"))
    print(sprd.round(2).to_string())

    first_min = df[df["minutes_elapsed"] <= 1.0]
    if len(first_min):
        avg_sprd = first_min["spread_cents"].mean()
        med_sprd = first_min["spread_cents"].median()
        print(f"\nOpening (≤1 min): avg={avg_sprd:.2f}¢  median={med_sprd:.2f}¢")

    # ── 2. Does mid already reflect prior direction? ───────────────────────────
    section("Mid price by prior result (opening minute)")
    sub = first_min[first_min["mid_price"].notna()].copy()
    if len(sub):
        grp = (sub.groupby("prior_result")["mid_price"]
               .agg(n="count", avg="mean", std="std"))
        grp["avg_cents"] = (grp["avg"] * 100).round(1)
        print(grp[["n", "avg_cents", "std"]].to_string())

        up_avg   = sub.loc[sub["prior_result"] == "yes", "mid_price"].mean()
        down_avg = sub.loc[sub["prior_result"] == "no",  "mid_price"].mean()

        if not np.isnan(up_avg) and not np.isnan(down_avg):
            diff = (up_avg - down_avg) * 100
            print(f"\n  After UP prior   → avg mid: {up_avg*100:.1f}¢")
            print(f"  After DOWN prior → avg mid: {down_avg*100:.1f}¢")
            print(f"  Difference: {diff:+.1f}¢")
            if abs(diff) < 1.5:
                print("\n  ✓ Market does NOT price in prior direction (~50¢ regardless)")
                print("    → Mean-reversion edge is real; market not arbitraging it")
            else:
                print(f"\n  ✗ Market already shifts {abs(diff):.1f}¢ for prior direction")
                print("    → Edge may be reduced or gone")

    # ── 3. Mid by streak ──────────────────────────────────────────────────────
    section("Mid price by streak length (opening minute)")
    if len(sub):
        sub = sub.copy()
        sub["streak_bucket"] = pd.cut(sub["streak_len"],
                                      bins=[-1, 0, 1, 2, 3, 100],
                                      labels=["0", "1", "2", "3", "4+"])
        sg = (sub.groupby("streak_bucket", observed=True)["mid_price"]
              .agg(n="count", avg="mean"))
        sg["avg_cents"] = (sg["avg"] * 100).round(1)
        print(sg[["n", "avg_cents"]].to_string())
        print()
        print("  Interpretation: if mid drifts TOWARD prior direction with streak,")
        print("  the market is already pricing it in and fade edge is reduced.")
        print("  If mid stays near 50¢ even at streak≥4 → large untapped edge.")

    # ── 4. Volume by minute ───────────────────────────────────────────────────
    section("Volume (fp) by minute into window")
    if "volume_fp" in df.columns and df["volume_fp"].notna().any():
        vg = (df[df["volume_fp"].notna()]
              .groupby("min_bucket")["volume_fp"]
              .agg(n="count", avg="mean", med="median"))
        print(vg.round(0).to_string())

    # ── 5. Edge viability ─────────────────────────────────────────────────────
    section("Edge viability after real spread")
    if len(first_min) and first_min["spread_cents"].notna().any():
        med_sprd  = first_min["spread_cents"].median()
        half_sprd = med_sprd / 2  # taker cost = cross half the spread

        print(f"  Median opening spread : {med_sprd:.2f}¢")
        print(f"  Taker half-spread cost: {half_sprd:.2f}¢")
        print()
        for label, gross_edge in [
            ("Baseline (52.5% WR)", 2.5),
            ("Streak ≥ 3  (57% WR)", 7.0),
            ("Streak ≥ 4  (60.5%)",  10.5),
            ("Combined RF (60.2%)",   10.2),
        ]:
            net = gross_edge - half_sprd
            status = "✓ VIABLE" if net > 0 else "✗ wiped out"
            print(f"  {label:<26}: gross={gross_edge:.1f}¢  "
                  f"net={net:+.1f}¢  {status}")

    print(f"\n{'─'*58}")
    print(f"Total windows sampled: {df['ticker'].nunique()}")
    print(f"Total rows logged    : {len(df):,}")


if __name__ == "__main__":
    main()
