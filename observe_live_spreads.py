"""
observe_live_spreads.py
───────────────────────
Real-time Kalshi spread observer for KXBTC15M contracts.

Every 15-minute window:
  1. Detects the current open KXBTC15M contract
  2. Samples YES bid/ask, spread, volume, BTC spot — every 30s for 3 min
  3. Computes the streak from the last N settled windows
  4. Writes each row to Postgres (Railway) if DATABASE_URL is set,
     otherwise falls back to data/spread_log.csv

Deploy to Railway:
  - Add a Postgres plugin → DATABASE_URL is auto-set
  - Set env vars: KALSHI_KEY_ID, KALSHI_PRIVATE_KEY  (PEM content, not file path)
  - Procfile: worker: python observe_live_spreads.py

Run locally:
  - Set KALSHI_KEY_ID and KALSHI_PRIVATE_KEY_PATH in .env
  - python observe_live_spreads.py
"""

from __future__ import annotations

import os
import sys
import time
import base64
import signal
import datetime
import textwrap
from typing import Optional, List, Dict, Any

import requests
from dotenv import load_dotenv
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend

# ── env ───────────────────────────────────────────────────────────────────────

_here = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_here, ".env"))

KALSHI_BASE     = "https://api.elections.kalshi.com/trade-api/v2"
BINANCE_BASE    = "https://api.binance.us/api/v3"
SERIES          = "KXBTC15M"
STREAK_LOOKBACK = 20
SAMPLE_MINUTES  = 3       # sample each window for this many minutes
POLL_INTERVAL   = 30      # seconds between samples within a window
MAIN_LOOP_SLEEP = 15      # seconds between main-loop ticks

OUTPUT_DIR = os.path.join(_here, "data")
CSV_LOG    = os.path.join(OUTPUT_DIR, "spread_log.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Kalshi auth ───────────────────────────────────────────────────────────────

_PRIVATE_KEY = None

def _load_key():
    global _PRIVATE_KEY
    if _PRIVATE_KEY is not None:
        return _PRIVATE_KEY

    # Railway: key content stored in env var KALSHI_PRIVATE_KEY
    pem_content = os.getenv("KALSHI_PRIVATE_KEY", "")
    if pem_content:
        # Env vars can't have real newlines easily; allow \n as literal escape
        pem_content = pem_content.replace("\\n", "\n")
        _PRIVATE_KEY = serialization.load_pem_private_key(
            pem_content.encode(), password=None, backend=default_backend()
        )
        return _PRIVATE_KEY

    # Local: path from .env
    path = os.getenv("KALSHI_PRIVATE_KEY_PATH", "")
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            _PRIVATE_KEY = serialization.load_pem_private_key(
                f.read(), password=None, backend=default_backend()
            )
        return _PRIVATE_KEY

    raise RuntimeError(
        "No private key found.\n"
        "  Railway: set KALSHI_PRIVATE_KEY env var (PEM content)\n"
        "  Local:   set KALSHI_PRIVATE_KEY_PATH in .env"
    )


def _headers(method: str, path: str) -> Dict[str, str]:
    key_id = os.getenv("KALSHI_KEY_ID", "")
    ts     = str(int(time.time() * 1000))
    msg    = (ts + method.upper() + path).encode()
    sig    = _load_key().sign(
        msg,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    return {
        "KALSHI-ACCESS-KEY":       key_id,
        "KALSHI-ACCESS-TIMESTAMP": ts,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
        "Content-Type":            "application/json",
    }


def _get(path: str, params: Dict = None, retries: int = 3) -> Dict:
    url = KALSHI_BASE + path
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=_headers("GET", path),
                             params=params, timeout=10)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            print(f"  [HTTP {r.status_code}] {path}: {r.text[:200]}")
            return {}
        except requests.RequestException as e:
            print(f"  [req error] {e}")
            time.sleep(1)
    return {}

# ── Binance ───────────────────────────────────────────────────────────────────

def btc_spot_price() -> Optional[float]:
    try:
        r = requests.get(f"{BINANCE_BASE}/ticker/price",
                         params={"symbol": "BTCUSDT"}, timeout=5)
        if r.status_code == 200:
            return float(r.json()["price"])
    except Exception:
        pass
    return None

# ── Kalshi helpers ────────────────────────────────────────────────────────────

def now_utc() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def parse_dt(s: str) -> Optional[datetime.datetime]:
    if not s:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ"):
        try:
            return datetime.datetime.strptime(s, fmt).replace(
                tzinfo=datetime.timezone.utc)
        except ValueError:
            pass
    return None


def fetch_active_markets(series: str = SERIES) -> List[Dict]:
    data = _get("/markets", params={"series_ticker": series,
                                    "status": "open", "limit": 50})
    return sorted(data.get("markets", []),
                  key=lambda m: m.get("close_time", ""))


def fetch_recent_settled(series: str = SERIES,
                         n: int = STREAK_LOOKBACK) -> List[Dict]:
    data = _get("/markets", params={"series_ticker": series,
                                    "status": "settled", "limit": n})
    mkts = data.get("markets", [])
    return sorted(mkts, key=lambda m: m.get("close_time", ""), reverse=True)


def get_market_snapshot(ticker: str) -> Dict:
    """Return bid/ask/volume/floor from market detail (always populated)."""
    data = _get(f"/markets/{ticker}")
    m = data.get("market", {})
    bid   = m.get("yes_bid_dollars")
    ask   = m.get("yes_ask_dollars")
    vol   = m.get("volume_fp")
    last  = m.get("last_price_dollars")
    floor = m.get("floor_strike")
    return {
        "yes_bid":     float(bid)   if bid   is not None else None,
        "yes_ask":     float(ask)   if ask   is not None else None,
        "volume_fp":   float(vol)   if vol   is not None else None,
        "last_price":  float(last)  if last  is not None else None,
        "floor_strike": float(floor) if floor is not None else None,
    }


def compute_streak(settled: List[Dict]):
    directions = []
    for m in settled:
        r = m.get("result", "")
        if r == "yes":
            directions.append("up")
        elif r == "no":
            directions.append("down")

    if not directions:
        return 0, "unknown"

    cur    = directions[0]
    streak = 0
    for d in directions:
        if d == cur:
            streak += 1
        else:
            break
    return streak, cur

# ── Storage: Postgres + CSV fallback ─────────────────────────────────────────

LOG_COLS = [
    "timestamp_utc", "ticker", "window_open_utc", "window_close_utc",
    "minutes_elapsed", "yes_bid", "yes_ask", "spread_cents", "mid_price",
    "btc_spot", "floor_strike", "volume_fp", "last_price",
    "streak_len", "streak_dir", "prior_result",
]

_pg_conn = None

def _get_pg_conn():
    global _pg_conn
    if _pg_conn is not None:
        try:
            # Test connection is still alive
            _pg_conn.cursor().execute("SELECT 1")
            return _pg_conn
        except Exception:
            _pg_conn = None

    db_url = os.getenv("DATABASE_URL", "")
    if not db_url:
        return None

    try:
        import psycopg2
        _pg_conn = psycopg2.connect(db_url, sslmode="require")
        _pg_conn.autocommit = True
        return _pg_conn
    except Exception as e:
        print(f"  [db] Could not connect to Postgres: {e}")
        return None


def ensure_table():
    conn = _get_pg_conn()
    if conn is None:
        return
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS spread_log (
                    id               SERIAL PRIMARY KEY,
                    timestamp_utc    TIMESTAMPTZ NOT NULL,
                    ticker           TEXT,
                    window_open_utc  TIMESTAMPTZ,
                    window_close_utc TIMESTAMPTZ,
                    minutes_elapsed  REAL,
                    yes_bid          REAL,
                    yes_ask          REAL,
                    spread_cents     REAL,
                    mid_price        REAL,
                    btc_spot         REAL,
                    floor_strike     REAL,
                    volume_fp        REAL,
                    last_price       REAL,
                    streak_len       INTEGER,
                    streak_dir       TEXT,
                    prior_result     TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_spread_log_ticker
                    ON spread_log (ticker);
                CREATE INDEX IF NOT EXISTS idx_spread_log_ts
                    ON spread_log (timestamp_utc DESC);
            """)
        print("  [db] Table ready: spread_log")
    except Exception as e:
        print(f"  [db] Table creation failed: {e}")


def write_row(row: Dict[str, Any]):
    """Write to Postgres if available, always write to CSV."""
    # ── Postgres ──────────────────────────────────────────────────────────────
    conn = _get_pg_conn()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO spread_log (
                        timestamp_utc, ticker, window_open_utc, window_close_utc,
                        minutes_elapsed, yes_bid, yes_ask, spread_cents, mid_price,
                        btc_spot, floor_strike, volume_fp, last_price,
                        streak_len, streak_dir, prior_result
                    ) VALUES (
                        %(timestamp_utc)s, %(ticker)s,
                        %(window_open_utc)s, %(window_close_utc)s,
                        %(minutes_elapsed)s, %(yes_bid)s, %(yes_ask)s,
                        %(spread_cents)s, %(mid_price)s, %(btc_spot)s,
                        %(floor_strike)s, %(volume_fp)s, %(last_price)s,
                        %(streak_len)s, %(streak_dir)s, %(prior_result)s
                    )
                """, row)
            return  # Success — skip CSV
        except Exception as e:
            print(f"  [db] Insert failed: {e} — falling back to CSV")

    # ── CSV fallback ──────────────────────────────────────────────────────────
    import csv
    file_exists = os.path.exists(CSV_LOG)
    with open(CSV_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in LOG_COLS})

# ── Dashboard ─────────────────────────────────────────────────────────────────

def print_dashboard(ticker: str, mins_in: float, snap: Dict,
                    btc: Optional[float], streak: int,
                    streak_dir: str, prior_result: str):
    bid   = snap.get("yes_bid")
    ask   = snap.get("yes_ask")
    mid   = (bid + ask) / 2 if bid and ask else None
    sprd  = (ask - bid) * 100 if bid and ask else None
    floor = snap.get("floor_strike")
    vol   = snap.get("volume_fp")
    last  = snap.get("last_price")

    bid_s   = f"{bid*100:.1f}¢"  if bid   else "?"
    ask_s   = f"{ask*100:.1f}¢"  if ask   else "?"
    mid_s   = f"{mid*100:.1f}¢"  if mid   else "?"
    sprd_s  = f"{sprd:.1f}¢"     if sprd  is not None else "?"
    btc_s   = f"${btc:,.0f}"     if btc   else "?"
    floor_s = f"${floor:,.2f}"   if floor else "?"
    vol_s   = f"{vol:,.0f}"      if vol   else "?"
    last_s  = f"{last*100:.1f}¢" if last  else "?"

    arrows = ("↑" if streak_dir == "up" else "↓") * min(streak, 8)
    prior_s = ("✅ YES(UP)" if prior_result == "yes"
               else "❌ NO(DN)" if prior_result == "no" else "?")

    print(f"\n{'─'*58}")
    print(f"  {ticker}")
    print(f"  +{mins_in:.1f} min | {now_utc().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"  BTC spot : {btc_s:<10}  Target: {floor_s}")
    print(f"  {'─'*44}")
    print(f"  Bid: {bid_s:<8}  Ask: {ask_s:<8}  Spread: {sprd_s}")
    print(f"  Mid: {mid_s:<8}  Last: {last_s:<7}  Vol: {vol_s}")
    print(f"  {'─'*44}")
    print(f"  Prior: {prior_s:<12}  Streak: {arrows} ({streak}×{streak_dir})")

    if streak >= 4:
        fade = "DOWN" if streak_dir == "up" else "UP"
        print(f"  ⚡ SIGNAL: streak≥4 → fade → bet {fade}  (hist WR ~60.5%)")
    elif streak >= 3:
        fade = "DOWN" if streak_dir == "up" else "UP"
        print(f"  ⚡ SIGNAL: streak≥3 → fade → bet {fade}  (hist WR ~57%)")
    else:
        print(f"  · No signal")
    print(f"{'─'*58}")

# ── Observation loop ──────────────────────────────────────────────────────────

def observe_window(market: Dict, settled: List[Dict]):
    ticker   = market["ticker"]
    close_dt = parse_dt(market.get("close_time"))
    open_dt  = parse_dt(market.get("open_time")) or now_utc()

    streak_len, streak_dir = compute_streak(settled)
    prior_result = settled[0].get("result", "") if settled else ""

    start_time = now_utc()
    deadline   = start_time + datetime.timedelta(minutes=SAMPLE_MINUTES)

    print(f"\n🔔 Window open: {ticker}  streak={streak_len}×{streak_dir}")

    while now_utc() < deadline:
        snap    = get_market_snapshot(ticker)
        btc     = btc_spot_price()
        ts_now  = now_utc()
        mins_in = (ts_now - start_time).total_seconds() / 60

        bid  = snap.get("yes_bid")
        ask  = snap.get("yes_ask")
        mid  = (bid + ask) / 2 if bid and ask else None
        sprd = (ask - bid) * 100 if bid and ask else None

        print_dashboard(ticker, mins_in, snap, btc,
                        streak_len, streak_dir, prior_result)

        row: Dict[str, Any] = {
            "timestamp_utc":    ts_now.isoformat(),
            "ticker":           ticker,
            "window_open_utc":  open_dt.isoformat(),
            "window_close_utc": close_dt.isoformat() if close_dt else None,
            "minutes_elapsed":  round(mins_in, 3),
            "yes_bid":          bid,
            "yes_ask":          ask,
            "spread_cents":     sprd,
            "mid_price":        mid,
            "btc_spot":         btc,
            "floor_strike":     snap.get("floor_strike"),
            "volume_fp":        snap.get("volume_fp"),
            "last_price":       snap.get("last_price"),
            "streak_len":       streak_len,
            "streak_dir":       streak_dir,
            "prior_result":     prior_result,
        }
        write_row(row)

        backend = "postgres" if _get_pg_conn() else "csv"
        print(f"  ↳ saved to {backend}")

        time.sleep(POLL_INTERVAL)

    print(f"  ✓ Sampling done for {ticker}")


# ── Main ──────────────────────────────────────────────────────────────────────

_running = True

def _sigint(sig, frame):
    global _running
    print("\nCtrl-C — shutting down.")
    _running = False

signal.signal(signal.SIGINT, _sigint)


def main():
    print("=" * 58)
    print("  Kalshi KXBTC15M Live Spread Observer")
    backend = "Postgres" if os.getenv("DATABASE_URL") else f"CSV ({CSV_LOG})"
    print(f"  Storage : {backend}")
    print(f"  Sampling: {SAMPLE_MINUTES} min per window, every {POLL_INTERVAL}s")
    print("  Press Ctrl-C to stop.")
    print("=" * 58)

    # Validate credentials
    key_id = os.getenv("KALSHI_KEY_ID", "")
    if not key_id:
        print("ERROR: KALSHI_KEY_ID not set")
        sys.exit(1)
    try:
        _load_key()
        print(f"  Auth OK (key_id={key_id[:8]}…)")
    except Exception as e:
        print(f"ERROR loading private key: {e}")
        sys.exit(1)

    # Init DB table
    ensure_table()

    observed:          set  = set()
    last_settled_fetch: float = 0.0
    settled:           List  = []

    while _running:
        # Refresh settled list every 5 min
        if time.time() - last_settled_fetch > 300:
            fresh = fetch_recent_settled()
            if fresh:
                settled = fresh
                last_settled_fetch = time.time()
                sl, sd = compute_streak(settled)
                print(f"\n[{now_utc().strftime('%H:%M:%S')}] "
                      f"Settled refreshed: {len(settled)} markets | "
                      f"streak={sl}×{sd}")

        open_markets = fetch_active_markets()

        for mkt in open_markets:
            ticker   = mkt.get("ticker", "")
            close_dt = parse_dt(mkt.get("close_time"))

            if ticker in observed or close_dt is None:
                continue

            secs_left   = (close_dt - now_utc()).total_seconds()
            mins_elapsed = (15 * 60 - secs_left) / 60

            if secs_left < 0 or secs_left > 15 * 60:
                continue  # future window or already closed

            observed.add(ticker)

            if mins_elapsed <= SAMPLE_MINUTES + 0.5:
                observe_window(mkt, settled)
                last_settled_fetch = 0  # force refresh after window closes
            else:
                print(f"  [skip] {ticker} — missed open ({mins_elapsed:.1f} min elapsed)")
            break

        # Trim old tickers
        if len(observed) > 200:
            observed = set(list(observed)[-100:])

        time.sleep(MAIN_LOOP_SLEEP)

    print("Observer stopped.")


if __name__ == "__main__":
    main()
