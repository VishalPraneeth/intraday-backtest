"""
load_data_new.py
================
Standalone, drop-in alternative to ``load_data()`` in ``all_setups.py`` for
pulling ~200 trading sessions of 5-minute NSE OHLCV data *without paid APIs*.

Nothing else in the project needs to change. This file is self-contained and
deliberately does NOT import ``all_setups`` (that module does
``from openalgo import api`` at import time, so importing it fails when the
OpenAlgo stack is unavailable).

Why a new loader?
-----------------
* OpenAlgo / DuckDB cache is not working in this workspace.
* yfinance only serves the last ~60 calendar days of 5-minute bars, so it
  cannot cover ~200 sessions. It is kept here only as a zero-credential
  smoke-test provider.

Provider 1: ``upstox``  (default, recommended)
----------------------------------------------
Free Upstox account -> free Historical Candle V3 API.
* 5-minute candles available from January 2022.
* Max 1 month per request for 1-15 minute intervals (this module auto-chunks).
* Rate limit for historical candles: 50/sec, 500/min (plenty).
* Instrument keys are ISIN based (``NSE_EQ|INE848E01016``). The public NSE
  instrument master is downloaded and cached automatically, so you keep
  passing plain symbols like ``SBIN``.
  The raw master is a ~45 MB JSON with 84k+ records (all segments, futures,
  options). Only the ~2.7k ``NSE_EQ``/``EQ`` rows are needed, so the raw file
  is parsed once and discarded; the cache is a ~170 KB CSV
  (``.instrument_cache/upstox_nse_eq.csv``) you can open in Excel.
  Use ``--check-instruments``, ``--search`` or ``--dump-instruments`` to
  inspect / validate it.

    https://upstox.com/developer/api-documentation/v3/get-historical-candle-data

Auth: generate a daily access token at https://account.upstox.com/developer/apps
and put it in env ``UPSTOX_ACCESS_TOKEN`` (it expires ~03:30 IST next day),
or in a file named ``.upstox_token`` next to this script.

Provider 2: ``yahoo``   (fallback, no credentials)
--------------------------------------------------
Last 60 calendar days only. Useful to validate schema / signal plumbing.

Other free-with-an-account options if you later want a second opinion:
  * Angel One SmartAPI  - free, FIVE_MINUTE candles, programmatic TOTP login.
  * Fyers API v3        - free, historical intraday included with account.
Avoid for "free": Dhan Data APIs and Zerodha Kite Connect both require a paid
data / subscription plan. NSE bhavcopy is end-of-day only.

Usage
-----
    # one-off validation: prints a coverage report
    python load_data_new.py --symbol SBIN --start 2025-06-01 --end 2026-01-31

    # compare against Yahoo over the overlapping 60-day window
    python load_data_new.py --symbol SBIN --cross-check

    # or use it exactly where load_data() was used
    from load_data_new import load_data_new
    raw = load_data_new("SBIN", cfg)

Environment
-----------
    UPSTOX_ACCESS_TOKEN   bearer token (required for provider="upstox")
    DATA_SOURCE           default provider: "upstox" | "yahoo"
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

__all__ = ["load_data_new", "LoadCfg", "export_json", "load_json"]

IST = "Asia/Kolkata"
OHLCV = ["open", "high", "low", "close", "volume"]

UPSTOX_MASTER_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
UPSTOX_CANDLE_URL = (
    "https://api.upstox.com/v3/historical-candle/"
    "{instrument_key}/minutes/{interval}/{to_date}/{from_date}"
)
CACHE_DIR = Path(__file__).resolve().parent / ".instrument_cache"
INSTRUMENT_CSV = CACHE_DIR / "upstox_nse_eq.csv"
LEGACY_MASTER = CACHE_DIR / "upstox_nse.json"
INSTRUMENT_COLUMNS = ["trading_symbol", "instrument_key", "isin", "name"]


# ---------------------------------------------------------------------------
# Config fallback (duck-typed; all_setups.Config works just as well)
# ---------------------------------------------------------------------------
@dataclass
class LoadCfg:
    """Minimal stand-in when you don't want to import all_setups.Config."""

    session_start: str = "09:15"
    session_end: str = "15:30"
    interval_minutes: int = 5
    lookback_days: int = 300          # ~200 trading sessions


def _cfg_attr(cfg, name: str, default):
    return getattr(cfg, name, default) if cfg is not None else default


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _norm_symbol(symbol: str) -> str:
    sym = symbol.strip().upper()
    for suffix in (".NS", ".NSE", ".BO", ".BSE"):
        if sym.endswith(suffix):
            return sym[: -len(suffix)]
    return sym


def _as_date(value, default: Optional[date] = None) -> date:
    if value is None:
        return default
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return pd.Timestamp(value).date()


def _month_chunks(start: date, end: date) -> List[Tuple[date, date]]:
    """Split a range into calendar-month slices (Upstox 1-month limit)."""
    chunks: List[Tuple[date, date]] = []
    cur = date(start.year, start.month, 1)
    while cur <= end:
        nxt = date(cur.year + (cur.month == 12), (cur.month % 12) + 1, 1)
        a, b = max(start, cur), min(end, nxt - timedelta(days=1))
        if a <= b:
            chunks.append((a, b))
        cur = nxt
    return chunks


def _finalize(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Normalise any provider output to the exact frame all_setups expects."""
    if df is None or df.empty:
        raise ValueError("provider returned no rows")

    df = df.copy()
    for col in OHLCV:
        if col not in df.columns:
            raise ValueError(f"provider frame is missing '{col}' column")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=OHLCV)

    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        idx = idx.tz_localize(IST)
    else:
        idx = idx.tz_convert(IST)
    df.index = idx
    df.index.name = "timestamp"

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    start = _cfg_attr(cfg, "session_start", "09:15")
    end = _cfg_attr(cfg, "session_end", "15:30")
    df = df.between_time(start, end, inclusive="left")
    return df[OHLCV]


def _http_json(url: str, headers: Dict[str, str], timeout: int = 40,
               retries: int = 4) -> dict:
    last = None
    for attempt in range(retries):
        resp = requests.get(url, headers=headers, timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 429 or 500 <= resp.status_code < 600:
            last = f"HTTP {resp.status_code}: {resp.text[:200]}"
            time.sleep(1.5 * (attempt + 1))
            continue
        raise RuntimeError(f"HTTP {resp.status_code} for {url}: {resp.text[:300]}")
    raise RuntimeError(f"request failed after {retries} tries: {last}")


# ---------------------------------------------------------------------------
# Provider: Upstox  (free account, 5m from Jan 2022)
# ---------------------------------------------------------------------------
def _upstox_token() -> str:
    token = os.environ.get("UPSTOX_ACCESS_TOKEN", "").strip()
    if token:
        return token
    token_file = Path(__file__).resolve().parent / ".upstox_token"
    if token_file.exists():
        return token_file.read_text(encoding="utf-8").strip()
    raise RuntimeError(
        "UPSTOX_ACCESS_TOKEN is not set.\n"
        "  1. Open https://account.upstox.com/developer/apps\n"
        "  2. Create an app (redirect URL can be http://127.0.0.1)\n"
        "  3. Generate an access token and either\n"
        "       set UPSTOX_ACCESS_TOKEN=..., or\n"
        "       save it in a file named .upstox_token next to this script.\n"
        "  Note: the token expires around 03:30 IST the next day."
    )


def _upstox_instruments(force: bool = False, verbose: bool = False) -> pd.DataFrame:
    """
    Compact NSE cash-equity map: ``trading_symbol -> instrument_key``.

    Upstox's public master is a ~45 MB JSON containing 84k+ records for every
    segment (EQ, futures, options, indices...). We only ever need the ~2.7k
    ``NSE_EQ`` / ``EQ`` rows, so the raw payload is parsed once and never
    persisted. The on-disk cache is a small CSV that opens in Excel.
    """
    fresh = (
        INSTRUMENT_CSV.exists()
        and not force
        and (time.time() - INSTRUMENT_CSV.stat().st_mtime) < 12 * 3600
    )
    if fresh:
        return pd.read_csv(INSTRUMENT_CSV, dtype=str).fillna("")

    resp = requests.get(UPSTOX_MASTER_URL, timeout=120)
    resp.raise_for_status()
    payload = json.loads(gzip.decompress(resp.content).decode("utf-8"))

    rows = [
        {
            "trading_symbol": str(r.get("trading_symbol") or "").strip().upper(),
            "instrument_key": r.get("instrument_key") or "",
            "isin": r.get("isin") or "",
            "name": r.get("name") or "",
        }
        for r in payload
        if r.get("segment") == "NSE_EQ" and r.get("instrument_type") == "EQ"
    ]
    table = pd.DataFrame(rows, columns=INSTRUMENT_COLUMNS)
    table = table[table["trading_symbol"] != ""]
    table = table.drop_duplicates("instrument_key").sort_values("trading_symbol")
    table = table.reset_index(drop=True)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(INSTRUMENT_CSV, index=False, encoding="utf-8")

    if LEGACY_MASTER.exists():                 # remove old ~45 MB raw cache
        try:
            LEGACY_MASTER.unlink()
        except OSError:
            pass

    if verbose:
        print(f"instrument map rebuilt: {len(table):,} NSE equities -> {INSTRUMENT_CSV}")
    return table


def _upstox_instrument_key(symbol: str) -> str:
    """Resolve 'SBIN' / 'SBIN.NS' -> 'NSE_EQ|INE062A01020'."""
    raw = symbol.strip()
    if "|" in raw:                       # already an instrument key
        return raw

    sym = _norm_symbol(symbol)
    table = _upstox_instruments()

    exact = table.loc[table["trading_symbol"] == sym, "instrument_key"]
    if len(exact):
        return str(exact.iloc[0])

    prefix_mask = table["trading_symbol"].str.startswith(sym, na=False)
    prefix = table.loc[prefix_mask]
    if len(prefix) == 1:
        return str(prefix["instrument_key"].iloc[0])
    if len(prefix):
        names = ", ".join(prefix["trading_symbol"].head(8))
        raise KeyError(f"ambiguous symbol {symbol!r}; candidates: {names}")

    raise KeyError(
        f"symbol {symbol!r} not found in the Upstox NSE_EQ map "
        f"({len(table):,} equities). Try --refresh-instruments."
    )


def _fetch_upstox(symbol: str, start: date, end: date,
                  interval_minutes: int = 5, pause: float = 0.25) -> pd.DataFrame:
    token = _upstox_token()
    key = _upstox_instrument_key(symbol)
    headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}

    frames = []
    for a, b in _month_chunks(start, end):
        url = UPSTOX_CANDLE_URL.format(
            instrument_key=quote(key, safe=""),
            interval=interval_minutes,
            to_date=b.isoformat(),
            from_date=a.isoformat(),
        )
        payload = _http_json(url, headers)

        if payload.get("status") != "success":
            raise RuntimeError(f"Upstox error for {symbol} {a}..{b}: {payload}")

        candles = (payload.get("data") or {}).get("candles") or []
        if candles:
            frames.append(pd.DataFrame(
                candles,
                columns=["timestamp", "open", "high", "low", "close", "volume", "oi"],
            ))
        time.sleep(pause)                # be polite; limit is 50/sec

    if not frames:
        raise ValueError(f"Upstox returned no candles for {symbol} ({start} .. {end})")

    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.set_index("timestamp")[OHLCV]


# ---------------------------------------------------------------------------
# Provider: Yahoo  (no credentials, last ~60 days of 5m only)
# ---------------------------------------------------------------------------
def _fetch_yahoo(symbol: str, period: str = "60d") -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pip install yfinance to use provider='yahoo'") from exc

    ysym = symbol if symbol.upper().endswith(".NS") else f"{_norm_symbol(symbol)}.NS"
    raw = None
    for _ in range(2):
        raw = yf.download(ysym, interval="5m", period=period,
                          auto_adjust=False, progress=False, threads=False)
        if raw is not None and not raw.empty:
            break
    if raw is None or raw.empty:
        raise ValueError(f"Yahoo returned no 5m data for {ysym}")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = OHLCV
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.tz_convert(IST)


# ---------------------------------------------------------------------------
# Public entry point (same signature/shape as all_setups.load_data)
# ---------------------------------------------------------------------------
def load_data_new(symbol: str, cfg=None, provider: Optional[str] = None,
                  start=None, end=None) -> pd.DataFrame:
    """
    Return a tz-aware IST ``DatetimeIndex`` frame with columns
    ``open, high, low, close, volume`` clipped to the session window.

    Parameters
    ----------
    symbol : str
        Plain NSE symbol (``"SBIN"``), Yahoo style (``"SBIN.NS"``) or an
        Upstox instrument key (``"NSE_EQ|INE062A01020"``).
    cfg : any object with ``session_start`` / ``session_end`` attributes.
        ``all_setups.Config`` works; ``None`` uses ``LoadCfg()``.
    provider : {"upstox", "yahoo"}; defaults to $DATA_SOURCE or "upstox".
    start, end : date | str | None; defaults to the last ``lookback_days``.
    """
    cfg = cfg if cfg is not None else LoadCfg()
    provider = (provider or os.environ.get("DATA_SOURCE") or "upstox").strip().lower()

    end_d = _as_date(end, default=pd.Timestamp.now(tz=IST).date())
    start_d = _as_date(start, default=end_d - timedelta(
        days=int(_cfg_attr(cfg, "lookback_days", 300))
    ))
    if start_d > end_d:
        raise ValueError(f"start {start_d} is after end {end_d}")

    if provider == "upstox":
        raw = _fetch_upstox(
            symbol, start_d, end_d,
            interval_minutes=int(_cfg_attr(cfg, "interval_minutes", 5)),
        )
    elif provider == "yahoo":
        raw = _fetch_yahoo(symbol)
    else:
        raise ValueError(f"unknown provider {provider!r}; use 'upstox' or 'yahoo'")

    df = _finalize(raw, cfg)
    if df.empty:
        raise ValueError(f"no bars left after session filter for {symbol}")
    return df


# ---------------------------------------------------------------------------
# JSON dataset I/O
# ---------------------------------------------------------------------------
JSON_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
JSON_SCHEMA_VERSION = 1


def to_json_payload(df: pd.DataFrame, symbol: str, interval: str = "5m",
                    source: str = "", cfg=None) -> dict:
    """Convert a candle frame into the self-describing dataset schema."""
    rows = [
        [ts.isoformat(), float(o), float(h), float(l), float(c), float(v)]
        for ts, o, h, l, c, v in zip(
            df.index, df["open"], df["high"], df["low"], df["close"], df["volume"]
        )
    ]
    return {
        "schema_version": JSON_SCHEMA_VERSION,
        "symbol": _norm_symbol(symbol),
        "exchange": "NSE",
        "interval": interval,
        "timezone": IST,
        "session": {
            "start": _cfg_attr(cfg, "session_start", "09:15"),
            "end": _cfg_attr(cfg, "session_end", "15:30"),
        },
        "source": source,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sessions": int(df.index.normalize().nunique()),
        "columns": JSON_COLUMNS,
        "rows": rows,
    }


def export_json(df: pd.DataFrame, path, symbol: str, interval: str = "5m",
                source: str = "", cfg=None, indent: Optional[int] = None) -> Path:
    """Write one symbol's candles to JSON. Returns the path written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = to_json_payload(df, symbol, interval=interval, source=source, cfg=cfg)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":") if indent is None else None,
                  indent=indent)
    return path


def load_json(path, cfg=None) -> pd.DataFrame:
    """
    Read a dataset written by ``export_json`` back into the exact frame
    ``all_setups`` expects (tz-aware IST index, OHLCV columns). This is the
    function the backtester should call once it is switched to JSON input.
    """
    with Path(path).open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    columns = payload.get("columns", JSON_COLUMNS)
    df = pd.DataFrame(payload["rows"], columns=columns)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(IST)
    df = df.set_index("timestamp")
    for col in OHLCV:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return _finalize(df[OHLCV], cfg if cfg is not None else LoadCfg())


# ---------------------------------------------------------------------------
# Validation helpers / CLI
# ---------------------------------------------------------------------------
def coverage_report(df: pd.DataFrame, label: str = "") -> pd.DataFrame:
    """Print + return per-session bar counts, useful for spotting gaps."""
    per_day = (df.groupby(df.index.date)
                 .agg(bars=("close", "size"),
                      first=("close", lambda s: s.index[0].strftime("%H:%M")),
                      last=("close", lambda s: s.index[-1].strftime("%H:%M")))
                 .rename_axis("date"))

    head = f"coverage report {label}".strip()
    print(f"\n{head}")
    print("-" * len(head))
    print(f"sessions      : {len(per_day)}")
    print(f"bars          : {len(df):,}")
    print(f"range         : {df.index[0]}  ->  {df.index[-1]}")
    print(f"bars/session  : min={per_day['bars'].min()}  "
          f"median={int(per_day['bars'].median())}  max={per_day['bars'].max()}")
    print(f"first bar seen: {sorted(per_day['first'].unique())[:3]}")
    print(f"last  bar seen: {sorted(per_day['last'].unique())[-3:]}")
    print(f"duplicate ts  : {int(df.index.duplicated().sum())}")
    print(f"NaN rows      : {int(df.isna().any(axis=1).sum())}")

    short = per_day[per_day["bars"] < per_day["bars"].median()]
    if len(short):
        print(f"\n{len(short)} session(s) below median bar count (check for gaps):")
        print(short.head(15).to_string())
    return per_day


def instrument_report(table: Optional[pd.DataFrame] = None,
                      symbols: Optional[List[str]] = None) -> pd.DataFrame:
    """Sanity-check the compact symbol -> instrument key map."""
    table = _upstox_instruments() if table is None else table
    dupes = table["trading_symbol"][table["trading_symbol"].duplicated()]

    print("\ninstrument map report")
    print("---------------------")
    print(f"cache file    : {INSTRUMENT_CSV}")
    if INSTRUMENT_CSV.exists():
        print(f"cache size    : {INSTRUMENT_CSV.stat().st_size / 1024:.1f} KB")
    print(f"rows          : {len(table):,}")
    print(f"duplicate sym : {dupes.nunique()}")
    print(f"blank keys    : {int((table['instrument_key'] == '').sum())}")
    print(f"blank isin    : {int((table['isin'] == '').sum())}")

    if symbols:
        wanted = [_norm_symbol(s) for s in symbols]
        hit = table[table["trading_symbol"].isin(wanted)]
        missing = sorted(set(wanted) - set(hit["trading_symbol"]))
        print(f"\nresolved {len(hit)}/{len(wanted)} requested symbols")
        if len(hit):
            print(hit.to_string(index=False))
        if missing:
            print(f"missing: {', '.join(missing)}")
    return table


def _cross_check(symbol: str, cfg) -> None:
    """Compare Upstox vs Yahoo on the overlapping ~60-day window."""
    try:
        ydf = _finalize(_fetch_yahoo(symbol), cfg)
    except Exception as exc:
        print(f"\ncross-check skipped (Yahoo unavailable: {exc})")
        return

    udf = _finalize(
        _fetch_upstox(symbol,
                      ydf.index[0].date(),
                      ydf.index[-1].date()),
        cfg,
    )

    joined = pd.DataFrame({
        "upstox": udf["close"],
        "yahoo": ydf["close"],
    }).dropna()
    if joined.empty:
        print("\ncross-check: no overlapping timestamps")
        return

    diff_bp = (joined["upstox"] / joined["yahoo"] - 1).abs() * 1e4
    print("\ncross-check Upstox vs Yahoo (close, overlapping window)")
    print("-" * 52)
    print(f"overlapping bars : {len(joined):,}")
    print(f"correlation      : {joined.corr().iloc[0, 1]:.6f}")
    print(f"mean |diff|      : {diff_bp.mean():.2f} bp")
    print(f"p95  |diff|      : {diff_bp.quantile(0.95):.2f} bp")
    print(f"max  |diff|      : {diff_bp.max():.2f} bp")


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Validate a free 5m data source.")
    ap.add_argument("--symbol", default="SBIN")
    ap.add_argument("--provider", choices=["upstox", "yahoo"], default=None)
    ap.add_argument("--start", default=None, help="YYYY-MM-DD")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD")
    ap.add_argument("--days", type=int, default=300)
    ap.add_argument("--out", default=None, help="optional CSV path")
    ap.add_argument("--cross-check", action="store_true")
    ap.add_argument("--check-instruments", action="store_true",
                    help="print a validation report for the compact symbol map")
    ap.add_argument("--refresh-instruments", action="store_true",
                    help="re-download and rebuild the compact symbol map")
    ap.add_argument("--search", default=None, metavar="TEXT",
                    help="search the instrument map and exit")
    ap.add_argument("--dump-instruments", nargs="?", const=str(INSTRUMENT_CSV),
                    default=None, metavar="CSV",
                    help="write the compact symbol map to CSV and exit")
    ap.add_argument("--symbols", nargs="*", default=None,
                    help="multiple symbols (used with --export-json)")
    ap.add_argument("--all-stocks", action="store_true",
                    help="use the STOCKS list from all_setups.py")
    ap.add_argument("--export-json", default=None, metavar="DIR",
                    help="write one JSON dataset per symbol into DIR")
    args = ap.parse_args()

    if args.refresh_instruments or args.check_instruments or args.search or args.dump_instruments:
        table = _upstox_instruments(force=args.refresh_instruments, verbose=True)
        if args.check_instruments:
            instrument_report(table, symbols=[args.symbol])
        if args.search:
            q = args.search.strip().upper()
            hits = table[table["trading_symbol"].str.contains(q, na=False)]
            print(hits.head(25).to_string(index=False))
            print(f"\n{len(hits)} match(es) for {q!r}")
        if args.dump_instruments:
            Path(args.dump_instruments).parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(args.dump_instruments, index=False)
            print(f"\nwrote {args.dump_instruments} ({len(table):,} rows)")
        if not (args.check_instruments or args.search):
            print(table.head(10).to_string(index=False))
        return

    cfg = LoadCfg(lookback_days=args.days)
    provider = (args.provider or os.environ.get("DATA_SOURCE") or "upstox").lower()

    if args.export_json:
        symbols = args.symbols or ([args.symbol] if not args.all_stocks
                                   else _stock_list())
        _export_many(symbols, Path(args.export_json), cfg, provider,
                     start=args.start, end=args.end)
        return

    df = load_data_new(args.symbol, cfg, provider=args.provider,
                       start=args.start, end=args.end)
    coverage_report(df, label=f"[{args.symbol}]")

    if args.out:
        out = df.copy()
        out.index = out.index.tz_localize(None)
        out.to_csv(args.out)
        print(f"\nwrote {args.out}")

    if args.cross_check:
        _cross_check(args.symbol, cfg)


def _stock_list() -> List[str]:
    """Reuse all_setups.STOCKS when available so the list never drifts."""
    try:
        from all_setups import STOCKS
        return list(STOCKS)
    except Exception:
        return ["KALYANKJIL", "BAJFINANCE", "ADANIENT", "TATASTEEL",
                "HINDALCO", "SBIN", "ICICIBANK", "AXISBANK", "MOTHERSON"]


def _export_many(symbols: List[str], outdir: Path, cfg, provider: str,
                 start=None, end=None) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": JSON_SCHEMA_VERSION,
        "provider": provider,
        "interval": "5m",
        "timezone": IST,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": {},
    }

    for sym in symbols:
        try:
            df = load_data_new(sym, cfg, provider=provider, start=start, end=end)
            path = outdir / f"{_norm_symbol(sym)}_5m.json"
            export_json(df, path, sym, source=provider, cfg=cfg)

            sessions = int(df.index.normalize().nunique())
            manifest["files"][_norm_symbol(sym)] = {
                "file": path.name,
                "rows": int(len(df)),
                "sessions": sessions,
                "start": df.index[0].isoformat(),
                "end": df.index[-1].isoformat(),
                "bytes": path.stat().st_size,
            }
            print(f"  {sym:<12} {sessions:>4} sessions  {len(df):>6,} bars  "
                  f"-> {path.name} ({path.stat().st_size / 1024:,.0f} KB)")
        except Exception as exc:
            print(f"  {sym:<12} FAILED: {exc}")
            manifest["files"][_norm_symbol(sym)] = {"error": str(exc)}

    with (outdir / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nmanifest -> {outdir / 'manifest.json'}")


if __name__ == "__main__":
    _cli()
