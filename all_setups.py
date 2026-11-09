"""
intraday_backtester.py
======================
Modular backtester for three 5-minute intraday setups (NSE stocks):

    S1 = Momentum Pull Back
         - pivot version : rally -> stall just under a pivot -> absorption trigger
         - NPR version   : "non-pivot reversal", same structure, no pivot needed,
                           early session only (trigger by 10:30)
    S2 = Opening Flag Breakout
    S3 = Sandwich Pattern

Pipeline:  load_data -> add_indicators -> add_global_filters
           -> setupN_*() boolean entry signals -> simulate_trades
           -> enrich (follow-through label, target ladder, prior-day context) -> Excel

Usage
-----
    python intraday_backtester.py                      # live data (Upstox via load_data_new.py), all stocks
    python intraday_backtester.py --demo               # offline synthetic data (smoke test)
    python intraday_backtester.py --symbols SBIN --session-sheets S2
    python intraday_backtester.py --no-npr             # pivot-only S1

NOTES / LIMITS
--------------
* Data comes from load_data_new.py (Upstox historical candles). Everything downstream expects a
  tz-aware IST OHLCV frame with columns open/high/low/close/volume.
* Signals are generated on the UNDERLYING. Real option P&L needs option-chain history
  (strike, expiry, IV, bid/ask). Treat results as "does the underlying signal have an edge".
* All rules are causal (no look-ahead). Same-bar SL/target ambiguity is resolved
  pessimistically (stop first).
* Daily-context fields joined to trades (volatility regime, trend vs SMA20, ...) always come from
  the PREVIOUS completed session, never from the trade's own day.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import zlib
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger("backtest")
IST = "Asia/Kolkata"

STOCKS = [
    "KALYANKJIL",
    "BAJFINANCE",
    "ADANIENT",
    "TATASTEEL",
    "HINDALCO",
    "SBIN",
    "ICICIBANK",
    "AXISBANK",
    "MOTHERSON",
]


# =============================================================================
# 1. CONFIG  (every threshold lives here -> easy to sweep / walk-forward)
# =============================================================================
@dataclass
class Config:
    # ---- data
    interval: str = "5m"
    period: str = "60d"
    session_start: str = "09:15"
    session_end: str = "15:30"

    # ---- indicators
    vol_sma: int = 20
    atr_len: int = 14
    ema_fast: int = 20
    ema_slow: int = 50
    rsi_len: int = 14
    # daily volatility bucket (ATR% of daily close)
    daily_vol_low_pct: float = 1.0      # ATR% < this -> "Low"
    daily_vol_high_pct: float = 2.0     # ATR% > this -> "High"
    atr_pctile_lookback: int = 60       # sessions used for the daily ATR% percentile rank
    regime_low_pctile: float = 33.3     # percentile <= this -> "Low (compression)"
    regime_high_pctile: float = 66.7    # percentile >= this -> "High (expansion)"

    # ---- global pre-filters (used by S2 / S3, and by S1 only if s1_use_global_filters)
    base_bars: int = 6                  # morning base window: 6 bars = 30 min, 12 = 60 min
    base_buffer_pct: float = 0.0        # tolerance when judging "unbroken"
    latest_entry: str = "14:30"
    use_volume_filter: bool = True      # Volume > SMA(Volume, 20)
    use_momentum_filter: bool = True    # close vs VWAP + EMA slope + RSI
    rsi_long_min: float = 50.0
    rsi_short_max: float = 50.0
    use_volatility_filter: bool = True  # ATR% band: skip dead / chaotic tape
    atr_pct_min: float = 0.0015
    atr_pct_max: float = 0.0200
    max_gap_pct: Optional[float] = 3.0  # skip sessions that gap more than this (None = off)

    # ---- S1 : pivot pull-back (stall at pivot + absorption)
    pivot_levels: Tuple[str, ...] = ("P", "R1", "R2", "R3", "R4", "R5",
                                     "S1", "S2", "S3", "S4", "S5")
    s1_touch_pct: float = 0.15          # closes up to X% beyond the pivot still count as "touching"
    s1_use_global_filters: bool = False  # False = S1 uses its own filters (+ ATR% band, gap, time)
    s1_min_bar: int = 3                 # earliest bar_no a trigger may fire
    s1_first_low_buf_atr: float = 0.10  # bar-0 extreme "untested": later prints may undercut by this many ATR
    s1_rvol_days: int = 15              # baseline = avg cumulative volume of prior N sessions, same bar
    s1_rvol_min: float = 1.0            # today's cumulative volume / baseline must be >= this
    s1_rally_bars: int = 10             # look-back window for the leg before the pullback
    s1_rally_min_atr: float = 1.5       # leg (low -> peak) must be >= this many ATR
    s1_peak_recent_bars: int = 3        # pullback must start within this many bars of the leg peak
    s1_zone_atr: float = 1.0            # pullback closes must sit within this many ATR below the pivot
    s1_pb_max_bars: int = 4             # max pullback candles before the trigger candle
    s1_max_retrace: float = 0.6         # pullback may not retrace more than 60% of the leg

    # ---- S1 : NPR (non-pivot reversal) - same structure, no pivot, early session only
    s1_npr_enabled: bool = True
    s1_npr_until: str = "10:30"         # last bar (HH:MM) on which an NPR trigger may fire

    # ---- S2 opening flag
    s2_pole_window_bars: int = 9        # pole must END within first 45 min
    s2_pole_max_bars: int = 3           # pole = 1..3 same-colour candles
    s2_pole_k_atr: float = 1.5          # net move > K * ATR
    s2_pctile: float = 0.80             # ...or single body >= rolling 80th pct (causal)
    s2_pctile_lookback: int = 375       # ~5 sessions of 5-min bars
    s2_flag_min: int = 3
    s2_flag_max: int = 8
    s2_flag_max_range_atr: float = 1.2  # flag height (HH-LL) < this * ATR
    s2_flag_max_retrace: float = 0.5    # flag may not retrace > 50% of pole
    s2_flag_vol_decay: float = 0.8      # mean flag volume <= 0.8 * mean pole volume

    # ---- S3 sandwich
    s3_trend_mode: str = "ema"          # "ema" or "hierarchy"
    s3_max_mid_range_atr: float = 1.2   # counter-candle must be modest (a trap, not a reversal)
    s3_min_body_atr: float = 0.25       # trigger candle needs conviction

    # ---- data window
    lookback_days: int = 300            # ~200 trading sessions of 5m bars

    # ---- shared stop / exit model
    sl_buffer_atr: float = 0.10
    min_risk_atr: float = 0.20          # reject stops that are too tight (noise)
    max_risk_atr: float = 3.00          # reject stops that are too wide
    entry_mode: str = "close"           # "close" (as specified) or "next_open" (more realistic)
    slippage_pct: float = 0.02          # per side, in %
    rr_target: float = 2.0
    be_trigger_r: Optional[float] = 1.0  # move stop to entry after +1R (None = off)
    eod_exit: str = "15:15"
    max_trades_per_day: int = 2         # per setup

    # ---- ATR trailing exit model (run side-by-side with the static model)
    run_trailing: bool = True
    trail_atr_mult: float = 1.5         # stop trails at extreme -/+ k * ATR(14)
    trail_activate_r: float = 0.0       # start trailing after this much favourable R
    trail_keep_target: bool = False     # False = ride the trend until trail / EOD

    # ---- follow-through label + target ladder
    ft_min_r: float = 1.0               # follow-through also needs this much favourable move (in R)
    ladder_max_r: int = 6               # target ladder T1..T6

    # ---- output
    out_dir: str = "backtest_output"
    session_sheet_setup: Optional[str] = None  # None = all setups, or "S1"/"S2"/"S3"


# =============================================================================
# 2. DATA
# =============================================================================
# Data source: Upstox free Historical Candle V3 API (~200 sessions of 5m).
# See load_data_new.py for token setup, instrument-key resolution and caching.
from load_data_new import load_data_new as _upstox_history


def load_data(symbol: str, cfg: Config) -> pd.DataFrame:
    """
    Fetch ~200 sessions of 5-minute NSE data via Upstox.

    Requires UPSTOX_ACCESS_TOKEN in the environment or a .upstox_token file
    next to this script (token expires ~03:30 IST the next day).
    """
    return _upstox_history(symbol, cfg, provider="upstox")


def make_synthetic(symbol: str, n_days: int = 30) -> pd.DataFrame:
    """Offline demo data with impulse+flag days, so every setup has something to find."""
    rng = np.random.default_rng(zlib.crc32(symbol.encode()))
    days = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    rows, idx, px = [], [], 1000.0
    for d in days:
        px *= 1 + rng.normal(0, 0.006)
        impulse = rng.random() < 0.55
        direction = rng.choice([-1, 1])
        drift = direction * 0.0002
        times = pd.date_range(f"{d.date()} 09:15", periods=75, freq="5min", tz=IST)
        for i, t in enumerate(times):
            vol = 120_000 * rng.lognormal(0, 0.35)
            if impulse and i < 3:
                ret, vol = direction * 0.0045 + rng.normal(0, 0.0005), vol * 2.5
            elif impulse and i < 10:
                ret, vol = rng.normal(0, 0.0004), vol * 0.45
            else:
                ret = drift + rng.normal(0, 0.0012)
            o, c = px, px * (1 + ret)
            hi = max(o, c) * (1 + abs(rng.normal(0, 0.0005)))
            lo = min(o, c) * (1 - abs(rng.normal(0, 0.0005)))
            rows.append((o, hi, lo, c, vol))
            idx.append(t)
            px = c
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx),
                        columns=["open", "high", "low", "close", "volume"])


# =============================================================================
# 3. INDICATORS & GLOBAL FILTERS
# =============================================================================
def add_indicators(raw: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = raw.copy()
    df["date"] = df.index.date
    g = df.groupby("date", sort=False)
    df["bar_no"] = g.cumcount()
    o, h, l, c, v = (df[k] for k in ("open", "high", "low", "close", "volume"))

    df["vol_sma"] = v.rolling(cfg.vol_sma).mean()
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / cfg.atr_len, adjust=False, min_periods=cfg.atr_len).mean()
    df["ema_fast"] = c.ewm(span=cfg.ema_fast, adjust=False).mean()
    df["ema_slow"] = c.ewm(span=cfg.ema_slow, adjust=False).mean()
    df["ema_slope"] = df["ema_fast"].diff(3)

    delta = c.diff()
    up, dn = delta.clip(lower=0), -delta.clip(upper=0)
    rs = up.ewm(alpha=1 / cfg.rsi_len, adjust=False).mean() / dn.ewm(alpha=1 / cfg.rsi_len, adjust=False).mean()
    df["rsi"] = 100 - 100 / (1 + rs)

    tp = (h + l + c) / 3
    df["vwap"] = (tp * v).groupby(df["date"]).cumsum() / v.groupby(df["date"]).cumsum()

    df["day_high"] = g["high"].cummax()
    df["day_low"] = g["low"].cummin()
    morning = df[df["bar_no"] < cfg.base_bars].groupby("date")
    df["morning_high"] = df["date"].map(morning["high"].max())
    df["morning_low"] = df["date"].map(morning["low"].min())

    # body size, rolling quantile of PRIOR bars only (causal stand-in for "80th pct of the day")
    body = (c - o).abs()
    df["body_q"] = body.rolling(cfg.s2_pctile_lookback, min_periods=75).quantile(cfg.s2_pctile).shift(1)

    # standard floor-trader pivots from the previous session (extended to R5/S5)
    daily = df.groupby("date").agg(h=("high", "max"), l=("low", "min"),
                                   c=("close", "last"), o=("open", "first"))
    p = daily.shift(1)
    P = (p["h"] + p["l"] + p["c"]) / 3
    rng_ = p["h"] - p["l"]
    R1, S1 = 2 * P - p["l"], 2 * P - p["h"]
    R2, S2 = P + rng_, P - rng_
    R3 = p["h"] + 2 * (P - p["l"])
    S3 = p["l"] - 2 * (p["h"] - P)
    R4, S4 = R3 + rng_, S3 - rng_
    R5, S5 = R4 + rng_, S4 - rng_
    levels = {"P": P, "R1": R1, "R2": R2, "R3": R3, "R4": R4, "R5": R5,
              "S1": S1, "S2": S2, "S3": S3, "S4": S4, "S5": S5}
    for name, ser in levels.items():
        df[name] = df["date"].map(ser)
    df["gap_pct"] = df["date"].map((daily["o"] - p["c"]) / p["c"] * 100)
    return df


def add_global_filters(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()
    hhmm = df.index.strftime("%H:%M")
    time_ok = (df["bar_no"] >= cfg.base_bars) & (hhmm <= cfg.latest_entry)
    vol_ok = (df["volume"] > df["vol_sma"]) if cfg.use_volume_filter else True
    atr_pct = df["atr"] / df["close"]
    vola_ok = atr_pct.between(cfg.atr_pct_min, cfg.atr_pct_max) if cfg.use_volatility_filter else True
    gap_ok = ~(df["gap_pct"].abs() > cfg.max_gap_pct) if cfg.max_gap_pct is not None else True

    # morning base unbroken: nothing has traded beyond the morning extreme since the base window
    buf = cfg.base_buffer_pct / 100
    base_long = df["day_low"] >= df["morning_low"] * (1 - buf)
    base_short = df["day_high"] <= df["morning_high"] * (1 + buf)

    mom_long = (df["close"] > df["vwap"]) & (df["ema_slope"] > 0) & (df["rsi"] >= cfg.rsi_long_min)
    mom_short = (df["close"] < df["vwap"]) & (df["ema_slope"] < 0) & (df["rsi"] <= cfg.rsi_short_max)

    f_long = time_ok & vol_ok & vola_ok & gap_ok & base_long
    f_short = time_ok & vol_ok & vola_ok & gap_ok & base_short
    if cfg.use_momentum_filter:
        f_long, f_short = f_long & mom_long, f_short & mom_short
    df["f_long"], df["f_short"] = f_long.astype(bool), f_short.astype(bool)
    return df


def add_daily_indicators(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Daily-timeframe technical panel built from the stock's own OHLC series.

    Core   : TR, ATR14, SMA20, RSI14, EMA12/26, MACD (+signal/hist), Bollinger (20, 2s),
             ATR% and an ATR%-based volatility bucket, prev_close, gap_%.
    Context: day_chg_%, HL_range, HL_range_%, dist_SMA20_%, price_vs_SMA20, Stoch_K/D,
             BB_pctB, ATR_pctile (rolling rank of ATR%), vol_regime, and text reads
             (rsi_read, macd_read, bb_read).

    NOTE: every value in a row is known only at that session's CLOSE. When joining to intraday
    trades use prior_daily() (shifted one session) - never the same-date row.
    """
    daily = (df.groupby("date")
               .agg(open=("open", "first"), high=("high", "max"),
                    low=("low", "min"), close=("close", "last"),
                    volume=("volume", "sum"))
               .reset_index())

    d = daily.copy()
    pc = d["close"].shift(1)

    tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(),
                    (d["low"] - pc).abs()], axis=1).max(axis=1)
    d["TR"] = tr
    d["ATR14"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    d["SMA20"] = d["close"].rolling(20).mean()

    delta = d["close"].diff()
    up, dn = delta.clip(lower=0), -delta.clip(upper=0)
    rs = (up.ewm(alpha=1 / 14, adjust=False).mean()
          / dn.ewm(alpha=1 / 14, adjust=False).mean())
    d["RSI14"] = 100 - 100 / (1 + rs)

    d["EMA12"] = d["close"].ewm(span=12, adjust=False).mean()
    d["EMA26"] = d["close"].ewm(span=26, adjust=False).mean()
    d["MACD"] = d["EMA12"] - d["EMA26"]
    d["MACD_signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_hist"] = d["MACD"] - d["MACD_signal"]

    mid = d["SMA20"]
    sd = d["close"].rolling(20).std(ddof=0)
    d["BB_mid"] = mid
    d["BB_upper"] = mid + 2 * sd
    d["BB_lower"] = mid - 2 * sd
    d["BB_width_%"] = (d["BB_upper"] - d["BB_lower"]) / mid * 100

    d["ATR%"] = d["ATR14"] / d["close"] * 100
    d["vol_bucket"] = pd.cut(
        d["ATR%"],
        bins=[-np.inf, cfg.daily_vol_low_pct, cfg.daily_vol_high_pct, np.inf],
        labels=["Low", "Normal", "High"],
    ).astype(object).fillna("n/a")

    d["prev_close"] = pc
    d["gap_%"] = (d["open"] - pc) / pc * 100

    # ---- extra context fields
    d["day_chg_%"] = (d["close"] / pc - 1) * 100
    d["HL_range"] = d["high"] - d["low"]
    d["HL_range_%"] = d["HL_range"] / d["close"] * 100
    d["dist_SMA20_%"] = (d["close"] / d["SMA20"] - 1) * 100
    d["price_vs_SMA20"] = np.where(d["SMA20"].isna(), "n/a",
                                   np.where(d["close"] > d["SMA20"], "Above", "Below"))

    ll, hh = d["low"].rolling(14).min(), d["high"].rolling(14).max()
    d["Stoch_K"] = (d["close"] - ll) / (hh - ll).replace(0, np.nan) * 100
    d["Stoch_D"] = d["Stoch_K"].rolling(3).mean()

    d["BB_pctB"] = (d["close"] - d["BB_lower"]) / (d["BB_upper"] - d["BB_lower"]).replace(0, np.nan) * 100

    def _rank_last(x: np.ndarray) -> float:
        if np.isnan(x[-1]):
            return np.nan
        v = x[~np.isnan(x)]
        return float((v <= x[-1]).mean() * 100)

    d["ATR_pctile"] = d["ATR%"].rolling(cfg.atr_pctile_lookback, min_periods=20).apply(_rank_last, raw=True)
    d["vol_regime"] = pd.cut(
        d["ATR_pctile"],
        bins=[-np.inf, cfg.regime_low_pctile, cfg.regime_high_pctile, np.inf],
        labels=["Low (compression)", "Normal", "High (expansion)"],
    ).astype(object).fillna("n/a")

    d["rsi_read"] = np.select([d["RSI14"].isna(), d["RSI14"] >= 70, d["RSI14"] <= 30],
                              ["n/a", "Overbought", "Oversold"], default="Neutral")
    d["macd_read"] = np.where(d["MACD"] > d["MACD_signal"],
                              "Bullish (MACD > Signal)", "Bearish (MACD < Signal)")
    d["bb_read"] = np.select([d["BB_pctB"].isna(), d["BB_pctB"] > 100, d["BB_pctB"] < 0],
                             ["n/a", "Above upper band (breakout)", "Below lower band (breakdown)"],
                             default="Inside bands")
    return d


PRIOR_CTX_COLS = ["vol_bucket", "vol_regime", "price_vs_SMA20", "rsi_read",
                  "bb_read", "ATR_pctile", "dist_SMA20_%"]


def prior_daily(daily: pd.DataFrame) -> pd.DataFrame:
    """Daily context shifted one session, indexed by date: row D holds what was known at D-1's close."""
    return daily.set_index("date")[PRIOR_CTX_COLS].shift(1)


# =============================================================================
# 4. SETUP SIGNAL GENERATORS  (each returns the same signal schema)
# =============================================================================
def _empty_signals(n: int):
    return (np.zeros(n, bool), np.zeros(n, bool),
            np.full(n, np.nan), np.full(n, np.nan), np.array([""] * n, dtype=object))


def _pack(index, le, se, lsl, ssl, ref) -> pd.DataFrame:
    return pd.DataFrame({"long_entry": le, "short_entry": se,
                         "long_sl": lsl, "short_sl": ssl, "ref": ref}, index=index)


def _sessions(df: pd.DataFrame):
    starts = np.flatnonzero(df["bar_no"].values == 0)
    return list(zip(starts, np.append(starts[1:], len(df))))


def _mirror(df: pd.DataFrame) -> Dict[int, dict]:
    """Shorts are longs on a sign-flipped price series -> one code path for both directions."""
    o, h, l, c, v = (df[k].values for k in ("open", "high", "low", "close", "volume"))
    pc = np.r_[np.nan, c[:-1]]
    return {
        1: dict(o=o, h=h, l=l, c=c, v=v, pc=pc, f=df["f_long"].values),
        -1: dict(o=-o, h=-l, l=-h, c=-c, v=v, pc=-pc, f=df["f_short"].values),
    }


# ---------------- S1: momentum pull-back (pivot) + NPR -----------------------
def _rvol(df: pd.DataFrame, lookback: int, min_days: int = 5) -> np.ndarray:
    """Today's cumulative volume up to this bar / mean cumulative volume at the same bar of prior sessions."""
    cum = df.groupby("date", sort=False)["volume"].cumsum()
    tbl = (pd.DataFrame({"date": df["date"].values, "bar": df["bar_no"].values, "cum": cum.values})
             .pivot_table(index="date", columns="bar", values="cum", aggfunc="last")
             .sort_index())
    base = tbl.shift(1).rolling(lookback, min_periods=min_days).mean()   # prior sessions only (causal)
    ratio = tbl / base
    di = ratio.index.get_indexer(df["date"].values)
    bi = ratio.columns.get_indexer(df["bar_no"].values)
    return ratio.values[di, bi]


def setup1_pullback(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    LONG (shorts are the mirror):
      1. bar-0 (09:15) low has not been broken (within s1_first_low_buf_atr)
      2. RVOL >= s1_rvol_min  (today heavier than recent sessions)
      3. a leg (>= s1_rally_min_atr ATR) up into the area
      4. 1..s1_pb_max_bars pullback candles
           - pivot version: closes sit just below a pivot (down to s1_zone_atr ATR under it,
             may touch / poke up to s1_touch_pct above it)             ref = pivot name
           - NPR version  : no pivot needed, trigger must fire by s1_npr_until   ref = "NPR"
      5. trigger: bullish candle closing above the high of the pullback candle(s)
      stop = pullback low - sl_buffer_atr * ATR
    """
    n, atr = len(df), df["atr"].values
    sides = _mirror(df)
    lv = {k: df[k].values for k in cfg.pivot_levels}
    sess = _sessions(df)

    # ---- own filters (time / volatility / gap) + relative volume
    hhmm = df.index.strftime("%H:%M")
    ok = (df["bar_no"].values >= cfg.s1_min_bar) & np.asarray(hhmm <= cfg.latest_entry)
    if cfg.use_volatility_filter:
        ok &= (df["atr"] / df["close"]).between(cfg.atr_pct_min, cfg.atr_pct_max).values
    if cfg.max_gap_pct is not None:
        ok &= ~(df["gap_pct"].abs() > cfg.max_gap_pct).values
    rvol = _rvol(df, cfg.s1_rvol_days)
    ok_rvol = np.nan_to_num(rvol, nan=0.0) >= cfg.s1_rvol_min

    le, se, lsl, ssl, ref = _empty_signals(n)
    pb_ext = np.full(n, np.nan)          # real-price extreme of the pullback (for the follow-through label)
    zone = cfg.s1_zone_atr * atr
    LB = cfg.s1_rally_bars

    for side, A in sides.items():
        o, h, l, c = A["o"], A["h"], A["l"], A["c"]
        f = (A["f"] if cfg.s1_use_global_filters else ok) & ok_rvol

        # bar-0 extreme untested: running session extreme (from bar 0) never undercuts bar-0's by more than the buffer
        l_first, run_min = np.empty(n), np.empty(n)
        for s, e in sess:
            l_first[s:e] = l[s]
            run_min[s:e] = np.minimum.accumulate(l[s:e])
        first_ok = run_min >= l_first - cfg.s1_first_low_buf_atr * atr

        # ------------------------- pivot version -------------------------
        for name in cfg.pivot_levels:
            L = side * lv[name]
            tol = np.abs(L) * cfg.s1_touch_pct / 100
            in_zone = (c <= L + tol) & (c >= L - zone)      # NaN -> False

            for s, e in sess:
                done = False
                for p in np.flatnonzero(in_zone[s:e]) + s:
                    if done:
                        break
                    if p - s < 2:                              # need bars for a leg
                        continue
                    if not (c[p] < o[p] or c[p] < c[p - 1]):   # first pullback candle: red or lower close
                        continue
                    a = max(s, p - LB)
                    i_hi = a + int(np.argmax(h[a:p]))          # leg peak
                    if p - i_hi > cfg.s1_peak_recent_bars:
                        continue
                    leg_lo, peak = l[a:i_hi + 1].min(), h[i_hi]
                    if not (peak - leg_lo >= cfg.s1_rally_min_atr * atr[p - 1]):
                        continue
                    if not (peak >= L[p] - zone[p]):           # leg actually reached the pivot zone
                        continue

                    for j in range(p + 1, min(e, p + cfg.s1_pb_max_bars + 1)):
                        if not in_zone[j - 1]:                 # pullback left the zone
                            break
                        pbh, pbl = h[p:j].max(), l[p:j].min()
                        top = max(peak, pbh)
                        if top - pbl > cfg.s1_max_retrace * (top - leg_lo):   # too deep
                            break
                        if c[j] > pbh and c[j] > o[j] and f[j] and first_ok[j]:
                            if not le[j] and not se[j]:
                                sl_m = pbl - cfg.sl_buffer_atr * atr[j]
                                if side == 1:
                                    le[j], lsl[j] = True, sl_m
                                else:
                                    se[j], ssl[j] = True, -sl_m
                                ref[j], pb_ext[j] = name, side * pbl
                            done = True                        # one trade per level & direction per day
                            break

        # --------------------------- NPR version ---------------------------
        # Same structure, minus the pivot-zone test, early session only. Pivot signals already
        # written on the same bar win. One NPR per direction per day.
        if cfg.s1_npr_enabled:
            for s, e in sess:
                done = False
                for p in range(s + 2, e):
                    if done or hhmm[p] > cfg.s1_npr_until:
                        break
                    if not (c[p] < o[p] or c[p] < c[p - 1]):
                        continue
                    a = max(s, p - LB)
                    i_hi = a + int(np.argmax(h[a:p]))
                    if p - i_hi > cfg.s1_peak_recent_bars:
                        continue
                    leg_lo, peak = l[a:i_hi + 1].min(), h[i_hi]
                    if not (peak - leg_lo >= cfg.s1_rally_min_atr * atr[p - 1]):
                        continue

                    for j in range(p + 1, min(e, p + cfg.s1_pb_max_bars + 1)):
                        pbh, pbl = h[p:j].max(), l[p:j].min()
                        top = max(peak, pbh)
                        if top - pbl > cfg.s1_max_retrace * (top - leg_lo):
                            break
                        if (c[j] > pbh and c[j] > o[j] and f[j] and first_ok[j]
                                and hhmm[j] <= cfg.s1_npr_until):
                            if not le[j] and not se[j]:
                                sl_m = pbl - cfg.sl_buffer_atr * atr[j]
                                if side == 1:
                                    le[j], lsl[j] = True, sl_m
                                else:
                                    se[j], ssl[j] = True, -sl_m
                                ref[j], pb_ext[j] = "NPR", side * pbl
                            done = True
                            break

    sig = _pack(df.index, le, se, lsl, ssl, ref)
    sig["pb_ext"], sig["rvol"] = pb_ext, rvol
    return sig


# ---------------- S2: opening flag breakout ---------------------------------
def _s2_flag(p, e, ph, pl, pv, A, atr, cfg):
    o, h, l, c, v, f = (A[k] for k in ("o", "h", "l", "c", "v", "f"))
    pole_h = ph - pl
    if not pole_h > 0:
        return None
    for j in range(p + 1 + cfg.s2_flag_min, min(e, p + 2 + cfg.s2_flag_max)):
        fh, fl = h[p + 1:j].max(), l[p + 1:j].min()
        if fh - fl > cfg.s2_flag_max_range_atr * atr[p]:     # not tight any more
            return None
        if ph - fl > cfg.s2_flag_max_retrace * pole_h:       # gave back too much of the pole
            return None
        if c[j] < fl:                                        # broke the wrong way: pattern dead
            return None
        if v[p + 1:j].mean() > cfg.s2_flag_vol_decay * pv:   # volume has not dried up (yet)
            continue
        if c[j] > fh and c[j] > o[j] and f[j]:
            return j, fl - cfg.sl_buffer_atr * atr[j]
    return None


def setup2_flag(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    n, atr, body_q = len(df), df["atr"].values, df["body_q"].values
    sides = _mirror(df)
    le, se, lsl, ssl, ref = _empty_signals(n)
    for s, e in _sessions(df):
        for side, A in sides.items():
            hit = False
            for p in range(s, min(e, s + cfg.s2_pole_window_bars)):
                if hit:
                    break
                if not A["c"][p] > A["o"][p]:
                    continue
                for k in range(1, cfg.s2_pole_max_bars + 1):
                    st = p - k + 1
                    if st < s or not A["c"][st] > A["o"][st]:
                        break
                    net = A["c"][p] - A["o"][st]
                    ok = net > cfg.s2_pole_k_atr * atr[p]
                    if k == 1 and not ok and not np.isnan(body_q[p]):
                        ok = net >= body_q[p]
                    if not ok:
                        continue
                    ph, pl = A["h"][st:p + 1].max(), A["l"][st:p + 1].min()
                    res = _s2_flag(p, e, ph, pl, A["v"][st:p + 1].mean(), A, atr, cfg)
                    if res is None:
                        continue
                    t, sl = res
                    if not le[t] and not se[t]:
                        if side == 1:
                            le[t], lsl[t], ref[t] = True, sl, f"pole{k}"
                        else:
                            se[t], ssl[t], ref[t] = True, -sl, f"pole{k}"
                    hit = True
                    break
    return _pack(df.index, le, se, lsl, ssl, ref)


# ---------------- S3: sandwich ----------------------------------------------
def setup3_sandwich(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    o, h, l, c, atr = (df[k] for k in ("open", "high", "low", "close", "atr"))
    sh = lambda s, k: s.shift(k, fill_value=False)
    red, green = c < o, c > o

    if cfg.s3_trend_mode == "ema":
        up = (df["ema_fast"] > df["ema_slow"]) & (df["ema_slope"] > 0)
        dn = (df["ema_fast"] < df["ema_slow"]) & (df["ema_slope"] < 0)
    else:  # high/low hierarchy: last 3 bars vs the 3 before
        hh, ll = h.rolling(3).max(), l.rolling(3).min()
        up = (hh > hh.shift(3)) & (ll > ll.shift(3))
        dn = (hh < hh.shift(3)) & (ll < ll.shift(3))
    up, dn = sh(up, 2), sh(dn, 2)                      # trend must exist BEFORE the sandwich

    mid_ok = (h - l).shift(1) <= cfg.s3_max_mid_range_atr * atr
    body_ok = (c - o).abs() >= cfg.s3_min_body_atr * atr
    same_session = df["bar_no"] >= 2

    short = (sh(red, 2) & sh(green, 1) & red & (c < l.shift(1)) & dn
             & mid_ok & body_ok & same_session & df["f_short"])
    long_ = (sh(green, 2) & sh(red, 1) & green & (c > h.shift(1)) & up
             & mid_ok & body_ok & same_session & df["f_long"])

    ssl = (h.shift(1) + cfg.sl_buffer_atr * atr).where(short)
    lsl = (l.shift(1) - cfg.sl_buffer_atr * atr).where(long_)
    ref = np.where(long_ | short, cfg.s3_trend_mode, "").astype(object)
    return _pack(df.index, long_.values, short.values, lsl.values, ssl.values, ref)


SETUPS: Dict[str, Tuple[str, Callable]] = {
    "S1": ("Momentum Pullback", setup1_pullback),
    "S2": ("Opening Flag", setup2_flag),
    "S3": ("Sandwich", setup3_sandwich),
}
VARIANT_NAMES = {
    "S1-Pivot": "Momentum Pullback (pivot)",
    "S1-NPR": "Non-Pivot Reversal",
    "S2": SETUPS["S2"][0],
    "S3": SETUPS["S3"][0],
}


def _variant(t: pd.DataFrame) -> np.ndarray:
    """S1 trades split into pivot vs NPR (by `ref`); S2 / S3 unchanged."""
    return np.where(t["setup"] == "S1",
                    np.where(t["ref"] == "NPR", "S1-NPR", "S1-Pivot"),
                    t["setup"])


# =============================================================================
# 5. TRADE SIMULATION  (event-driven, one position at a time per setup)
# =============================================================================
TRADE_COLS = ["setup", "model", "date", "side", "ref", "signal_time", "entry_time", "entry", "sl",
              "target", "exit_time", "exit", "reason", "R", "ret_pct", "mfe_R", "mae_R", "bars_held"]
FT_COLS = ["ft3", "ft5"]
CTX_COLS = ["ctx_vol_bucket", "ctx_regime", "ctx_trend", "ctx_align", "ctx_rsi_read",
            "ctx_bb_read", "ctx_atr_pctile", "ctx_dist_sma20_%"]


def _ladder_cols(k_max: int) -> list:
    return ["eod_R", "max_R"] + [f"T{k}_R" for k in range(1, k_max + 1)]


def _empty_trades() -> pd.DataFrame:
    """Typed empty frame, so empty setups never poison concat / .round()."""
    df = pd.DataFrame({c: pd.Series(dtype="float64") for c in
                       ("entry", "sl", "target", "exit", "R", "ret_pct", "mfe_R", "mae_R", "bars_held")})
    for c in ("setup", "model", "side", "ref", "reason"):
        df[c] = pd.Series(dtype="object")
    for c in ("signal_time", "entry_time", "exit_time"):
        df[c] = pd.Series(dtype="datetime64[ns, Asia/Kolkata]")
    df["date"] = pd.Series(dtype="object")
    return df[TRADE_COLS]


def _empty_trades_ext(k_max: int = 6) -> pd.DataFrame:
    t = _empty_trades()
    for c in FT_COLS + _ladder_cols(k_max):
        t[c] = pd.Series(dtype="float64")
    for c in CTX_COLS:
        t[c] = pd.Series(dtype="object")
    return t


def _concat(frames, k_max: int = 6) -> pd.DataFrame:
    """Concatenate trade frames, skipping empty ones (avoids dtype / FutureWarning noise)."""
    frames = [f for f in frames if f is not None and not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else _empty_trades_ext(k_max)


def simulate_trades(df: pd.DataFrame, sig: pd.DataFrame, cfg: Config, setup: str,
                    model: str = "static") -> pd.DataFrame:
    """
    Event-driven simulation, one position at a time per setup.

    model="static"     fixed stop + rr_target (+ optional move-to-BE).
    model="atr_trail"  same entry, but the stop trails the favourable extreme
                       by ``trail_atr_mult`` * ATR(14). By default the fixed
                       target is disabled so a good move is ridden until the
                       trail is hit or the EOD exit fires.
    """
    O, H, L, C = (df[k].values for k in ("open", "high", "low", "close"))
    atr, times = df["atr"].values, df.index
    hhmm = times.strftime("%H:%M")
    slip = cfg.slippage_pct / 100
    le, se = sig["long_entry"].values, sig["short_entry"].values
    lsl, ssl, ref = sig["long_sl"].values, sig["short_sl"].values, sig["ref"].values
    trailing = model == "atr_trail"
    trades = []

    def exit_reason(stop: float, initial_sl: float, moved_be: bool) -> str:
        if trailing and (stop - initial_sl) * side > 1e-9:
            return "TRAIL"
        return "BE" if moved_be else "SL"

    for s, e in _sessions(df):
        eod = s + int((hhmm[s:e] <= cfg.eod_exit).sum()) - 1
        n_tr, i = 0, s
        while i < e and n_tr < cfg.max_trades_per_day:
            side = 1 if le[i] else (-1 if se[i] else 0)
            if side == 0:
                i += 1
                continue
            ei = i if cfg.entry_mode == "close" else i + 1
            if ei > eod:
                break
            entry = (C[i] if cfg.entry_mode == "close" else O[ei]) * (1 + side * slip)
            sl = lsl[i] if side == 1 else ssl[i]
            risk = (entry - sl) * side
            a = atr[i]
            if not risk > 0 or (not np.isnan(a) and not (cfg.min_risk_atr * a <= risk <= cfg.max_risk_atr * a)):
                i += 1
                continue

            target = entry + side * cfg.rr_target * risk
            stop, moved_be, mfe, mae = sl, False, 0.0, 0.0
            anchor = entry                    # best favourable price reached so far
            px, reason, xi = None, None, None

            for k in range(ei + 1 if cfg.entry_mode == "close" else ei, eod + 1):
                adverse = ((entry - L[k]) if side == 1 else (H[k] - entry)) / risk
                mae = max(mae, adverse)

                if (L[k] <= stop) if side == 1 else (H[k] >= stop):          # stop first (pessimistic)
                    px = min(O[k], stop) if side == 1 else max(O[k], stop)
                    reason, xi = exit_reason(stop, sl, moved_be), k
                    break

                if (not trailing or cfg.trail_keep_target) and \
                        ((H[k] >= target) if side == 1 else (L[k] <= target)):
                    px = max(O[k], target) if side == 1 else min(O[k], target)
                    reason, xi = "TARGET", k
                    break

                fav = ((H[k] - entry) if side == 1 else (entry - L[k])) / risk
                mfe = max(mfe, fav)

                # stop is managed at the CLOSE of bar k -> only active from bar k+1
                if cfg.be_trigger_r is not None and fav >= cfg.be_trigger_r:
                    stop = max(stop, entry) if side == 1 else min(stop, entry)
                    moved_be = True

                if trailing and fav >= cfg.trail_activate_r and not np.isnan(atr[k]):
                    if side == 1:
                        anchor = max(anchor, H[k])
                        stop = max(stop, anchor - cfg.trail_atr_mult * atr[k])
                    else:
                        anchor = min(anchor, L[k])
                        stop = min(stop, anchor + cfg.trail_atr_mult * atr[k])

            if px is None:
                px, reason, xi = C[eod], "EOD", eod
            px *= (1 - side * slip)
            R = (px - entry) * side / risk
            trades.append(dict(
                setup=setup, model=model, date=df["date"].values[i],
                side="LONG" if side == 1 else "SHORT",
                ref=ref[i], signal_time=times[i], entry_time=times[ei], entry=entry, sl=sl,
                target=target, exit_time=times[xi], exit=px, reason=reason, R=R,
                ret_pct=(px - entry) * side / entry * 100, mfe_R=mfe, mae_R=mae, bars_held=xi - ei))
            n_tr += 1
            i = xi + 1

    return pd.DataFrame(trades, columns=TRADE_COLS) if trades else _empty_trades()


def attach_exit_flags(sig: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    sig = sig.copy()
    sig["long_exit"], sig["short_exit"] = False, False
    for _, t in trades.iterrows():
        col = "long_exit" if t["side"] == "LONG" else "short_exit"
        sig.loc[t["exit_time"], col] = True
    return sig


# =============================================================================
# 5b. TRADE ENRICHMENT  (follow-through label, target ladder, prior-day context)
# =============================================================================
def _eod_index_map(df: pd.DataFrame, cfg: Config) -> dict:
    hhmm = df.index.strftime("%H:%M")
    return {df["date"].values[s]: s + int((hhmm[s:e] <= cfg.eod_exit).sum()) - 1
            for s, e in _sessions(df)}


def add_followthrough(trades: pd.DataFrame, df: pd.DataFrame, sig: pd.DataFrame,
                      cfg: Config, ks: Tuple[int, ...] = (3, 5)) -> pd.DataFrame:
    """
    ft3 / ft5 = 1.0 when, over the next k bars, price (a) never revisits the pullback extreme
    AND (b) moves at least cfg.ft_min_r * risk in the trade direction. 0.0 otherwise, NaN when
    the session ends before k bars remain or the setup has no pullback extreme (S2 / S3).
    """
    t = trades.copy()
    for k in ks:
        t[f"ft{k}"] = np.nan
    if t.empty or "pb_ext" not in sig.columns:
        return t
    n = len(df)
    pos = df.index.get_indexer(t["signal_time"])
    H, L, dates = df["high"].values, df["low"].values, df["date"].values
    ext = sig["pb_ext"].values[pos]
    entry = t["entry"].values
    risk = (t["entry"] - t["sl"]).abs().values
    sides = t["side"].values
    for k in ks:
        vals = np.full(len(t), np.nan)
        for r, (i, x, en, rk, sd) in enumerate(zip(pos, ext, entry, risk, sides)):
            if i < 0 or np.isnan(x) or not rk > 0 or i + k >= n or dates[i + k] != dates[i]:
                continue
            seg = slice(i + 1, i + 1 + k)
            if sd == "LONG":
                held, move = L[seg].min() > x, (H[seg].max() - en) / rk
            else:
                held, move = H[seg].max() < x, (en - L[seg].min()) / rk
            vals[r] = float(held and move >= cfg.ft_min_r)
        t[f"ft{k}"] = vals
    return t


def add_target_ladder(trades: pd.DataFrame, df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Fixed-target ladder per trade (entry, stop and slippage as in the simulator, no BE / trail):
        eod_R  = outcome with NO target: stopped out (about -1R) or closed at the EOD exit
        max_R  = max favourable excursion (in R) before the stop was hit
        Tk_R   = +k if k*R was reached before the stop, otherwise eod_R
    Totalling a Tk_R column shows what a fixed k*R target would have earned.
    """
    K = cfg.ladder_max_r
    cols = _ladder_cols(K)
    t = trades.copy()
    if t.empty:
        for c in cols:
            t[c] = pd.Series(dtype="float64")
        return t

    O, H, L, C = (df[k].values for k in ("open", "high", "low", "close"))
    slip = cfg.slippage_pct / 100
    eod_map = _eod_index_map(df, cfg)
    pos_entry = df.index.get_indexer(t["entry_time"])
    out = np.full((len(t), len(cols)), np.nan)

    for r, (ei, sd, entry, sl, d) in enumerate(zip(pos_entry, t["side"].values, t["entry"].values,
                                                  t["sl"].values, t["date"].values)):
        side = 1 if sd == "LONG" else -1
        risk = (entry - sl) * side
        eod = eod_map.get(d)
        if ei < 0 or eod is None or not risk > 0:
            continue
        start = ei + 1 if cfg.entry_mode == "close" else ei
        if start > eod:
            continue

        first, stop_bar, stop_px, mfe = {}, None, None, 0.0
        for b in range(start, eod + 1):
            if (L[b] <= sl) if side == 1 else (H[b] >= sl):          # stop first (pessimistic)
                stop_bar = b
                stop_px = min(O[b], sl) if side == 1 else max(O[b], sl)
                break
            fav = ((H[b] - entry) if side == 1 else (entry - L[b])) / risk
            mfe = max(mfe, fav)
            for k in range(1, K + 1):
                if k not in first and fav >= k:
                    first[k] = b

        end_px = (stop_px if stop_bar is not None else C[eod]) * (1 - side * slip)
        eod_r = (end_px - entry) * side / risk
        out[r, 0], out[r, 1] = eod_r, mfe
        for k in range(1, K + 1):
            if k in first:
                tgt = entry + side * k * risk
                fill = max(O[first[k]], tgt) if side == 1 else min(O[first[k]], tgt)
                out[r, 1 + k] = (fill * (1 - side * slip) - entry) * side / risk
            else:
                out[r, 1 + k] = eod_r

    for j, c in enumerate(cols):
        t[c] = out[:, j]
    return t


def attach_context(trades: pd.DataFrame, prior: pd.DataFrame) -> pd.DataFrame:
    """Join PRIOR-session daily context to each trade (no same-day information)."""
    t = trades.copy()
    if t.empty:
        for c in CTX_COLS:
            t[c] = pd.Series(dtype="object")
        return t

    def m(col):
        return t["date"].map(prior[col])

    t["ctx_vol_bucket"] = m("vol_bucket").fillna("n/a")
    t["ctx_regime"] = m("vol_regime").fillna("n/a")
    t["ctx_trend"] = m("price_vs_SMA20").fillna("n/a")
    with_trend = (((t["side"] == "LONG") & (t["ctx_trend"] == "Above")) |
                  ((t["side"] == "SHORT") & (t["ctx_trend"] == "Below")))
    t["ctx_align"] = np.where(t["ctx_trend"].isin(["Above", "Below"]),
                              np.where(with_trend, "With trend", "Against trend"), "n/a")
    t["ctx_rsi_read"] = m("rsi_read").fillna("n/a")
    t["ctx_bb_read"] = m("bb_read").fillna("n/a")
    t["ctx_atr_pctile"] = m("ATR_pctile")
    t["ctx_dist_sma20_%"] = m("dist_SMA20_%")
    return t


def enrich_trades(t: pd.DataFrame, df: pd.DataFrame, sig: pd.DataFrame,
                  prior: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    t = add_followthrough(t, df, sig, cfg)
    t = add_target_ladder(t, df, cfg)
    return attach_context(t, prior)


# =============================================================================
# 6. STATS
# =============================================================================
def _stats(g: pd.DataFrame) -> dict:
    g = g.sort_values("entry_time")
    r = g["R"]
    wins, losses = r[r > 0].sum(), -r[r < 0].sum()
    eq = r.cumsum()
    out = {"trades": len(g), "win_rate_%": round((r > 0).mean() * 100, 1),
           "avg_R": round(r.mean(), 3), "total_R": round(r.sum(), 2),
           "profit_factor": round(wins / losses, 2) if losses > 0 else np.nan,
           "max_drawdown_R": round((eq.cummax() - eq).max(), 2),
           "avg_MFE_R": round(g["mfe_R"].mean(), 2), "avg_MAE_R": round(g["mae_R"].mean(), 2),
           "pct_trail": round((g["reason"] == "TRAIL").mean() * 100, 1),
           "pct_target": round((g["reason"] == "TARGET").mean() * 100, 1),
           "avg_bars_held": round(g["bars_held"].mean(), 1)}
    for k in (3, 5):
        col = f"ft{k}"
        out[f"ft{k}_%"] = round(g[col].mean() * 100, 1) if col in g and g[col].notna().any() else np.nan
    return out


MODEL_LABELS = {"static": "Static SL+Target", "atr_trail": "ATR(14) Trail"}


def summarize(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame({"note": ["no trades generated"]})
    t = trades.assign(variant=_variant(trades))
    rows = []
    for (variant, model), g in t.groupby(["variant", "model"], sort=False):
        for side, gg in [("ALL", g)] + list(g.groupby("side")):
            rows.append({"setup": variant, "name": VARIANT_NAMES.get(variant, variant),
                         "model": model, "model_name": MODEL_LABELS.get(model, model),
                         "side": side, **_stats(gg)})
    return pd.DataFrame(rows)


def ladder_summary(trades: pd.DataFrame) -> pd.DataFrame:
    """Long-format target-ladder table (static-model trades): per setup variant x target."""
    cols = [c for c in trades.columns if re.fullmatch(r"T\d+_R", str(c))]
    if trades.empty or not cols or "eod_R" not in trades.columns:
        return pd.DataFrame({"note": ["no ladder data"]})
    t = trades[trades["model"] == "static"] if "model" in trades.columns else trades
    t = t.dropna(subset=["eod_R"])
    if t.empty:
        return pd.DataFrame({"note": ["no ladder data"]})
    t = t.assign(variant=_variant(t))
    ks = sorted(int(c[1:-2]) for c in cols)
    targets = [("No target (EOD)", "eod_R")] + [(f"T{k}", f"T{k}_R") for k in ks]
    rows = []
    for v, g in list(t.groupby("variant", sort=False)) + [("ALL", t)]:
        for label, col in targets:
            r = g[col].dropna()
            wins, losses = r[r > 0].sum(), -r[r < 0].sum()
            rows.append({"setup": v, "target": label, "trades": len(r),
                         "win_%": round((r > 0).mean() * 100, 1),
                         "avg_R": round(r.mean(), 3), "total_R": round(r.sum(), 2),
                         "profit_factor": round(wins / losses, 2) if losses > 0 else np.nan})
    return pd.DataFrame(rows)


# =============================================================================
# 7. EXCEL OUTPUT  (one workbook per stock)
# =============================================================================
def _naive(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if isinstance(df[col].dtype, pd.DatetimeTZDtype):
            df[col] = df[col].dt.tz_localize(None)
    return df


def _profit_factor(t: pd.DataFrame) -> float:
    if t.empty:
        return float("nan")
    wins = t.loc[t["R"] > 0, "R"].sum()
    losses = -t.loc[t["R"] < 0, "R"].sum()
    return wins / losses if losses > 0 else float("nan")


def _eligible(results: dict, model: str, min_trades: int = 10) -> list:
    """Setups with enough trades to be called 'best'."""
    keys = [k for k in results if len(results[k][model]) >= min_trades]
    return keys or list(results)


def build_inference(symbol: str, df: pd.DataFrame, results: dict, cfg: Config) -> pd.DataFrame:
    """Overall, human-readable takeaways written under the Summary table."""
    static = _concat([v["static"] for v in results.values()], cfg.ladder_max_r)
    trail = _concat([v["atr_trail"] for v in results.values()], cfg.ladder_max_r)
    daily = add_daily_indicators(df, cfg)

    def totals(t):
        if t.empty:
            return 0, 0.0, 0.0
        return len(t), float(t["R"].sum()), float((t["R"] > 0).mean() * 100)

    n_s, r_s, w_s = totals(static)
    n_t, r_t, w_t = totals(trail)

    rows = []

    def add(metric, value, comment=""):
        rows.append({"Metric": metric, "Value": value, "Comment": comment})

    add("Symbol", symbol, "NSE cash equity")
    add("Sessions", int(df["date"].nunique()),
        f"{df.index[0].date()} -> {df.index[-1].date()}")
    add("Bars (5m)", f"{len(df):,}", "session-clipped 09:15-15:30")
    add("Daily volatility mix",
        ", ".join(f"{k}={int(v)}" for k, v in
                  daily["vol_bucket"].value_counts().items() if k != "n/a"),
        f"ATR%: <{cfg.daily_vol_low_pct}% Low | "
        f"{cfg.daily_vol_low_pct}-{cfg.daily_vol_high_pct}% Normal | "
        f">{cfg.daily_vol_high_pct}% High")
    add("Daily regime mix",
        ", ".join(f"{k}={int(v)}" for k, v in
                  daily["vol_regime"].value_counts().items() if k != "n/a"),
        f"ATR% percentile over {cfg.atr_pctile_lookback} sessions")

    add("Static model", f"{n_s} trades",
        f"win {w_s:.1f}% | total {r_s:+.2f}R | PF {_profit_factor(static):.2f}")
    add("ATR trail model", f"{n_t} trades",
        f"win {w_t:.1f}% | total {r_t:+.2f}R | PF {_profit_factor(trail):.2f}")
    add("Trailing impact", f"{r_t - r_s:+.2f}R",
        "positive = trail beat static")

    for k in results:
        s_r = float(results[k]["static"]["R"].sum())
        t_r = float(results[k]["atr_trail"]["R"].sum())
        best = "static" if s_r >= t_r else "atr_trail"
        add(f"Best exit model {k}", MODEL_LABELS[best],
            f"static {s_r:+.2f}R vs trail {t_r:+.2f}R")

    best_static = max(_eligible(results, "static"),
                      key=lambda k: float(results[k]["static"]["R"].sum()))
    best_trail = max(_eligible(results, "atr_trail"),
                     key=lambda k: float(results[k]["atr_trail"]["R"].sum()))
    add("Best setup (static)", f"{best_static} {SETUPS[best_static][0]}",
        f"{float(results[best_static]['static']['R'].sum()):+.2f}R (setups with >=10 trades)")
    add("Best setup (trail)", f"{best_trail} {SETUPS[best_trail][0]}",
        f"{float(results[best_trail]['atr_trail']['R'].sum()):+.2f}R (setups with >=10 trades)")

    if not trail.empty:
        add("Trail exit share", f"{(trail['reason'] == 'TRAIL').mean() * 100:.1f}%",
            "of trailing trades closed by the trail")

    # how much favourable excursion was handed back before the exit
    for label, t in (("static", static), ("trail", trail)):
        if not t.empty:
            add(f"Give-back ({label})", f"{(t['mfe_R'] - t['R']).mean():+.2f}R/trade",
                "avg max-favourable R not captured")

    # best fixed target per setup variant (target ladder, static-model entries)
    lad = ladder_summary(static)
    if "target" in lad.columns:
        for v, g in lad[lad["target"] != "No target (EOD)"].groupby("setup", sort=False):
            b = g.loc[g["avg_R"].idxmax()]
            add(f"Best fixed target {v}", b["target"],
                f"avg {b['avg_R']:+.3f}R | win {b['win_%']:.1f}% | n={int(b['trades'])}")

    # S1 follow-through (pullback held AND moved >= ft_min_r within k bars)
    s1 = static[static["setup"] == "S1"]
    if not s1.empty:
        for k in (3, 5):
            col = f"ft{k}"
            if s1[col].notna().any():
                add(f"S1 follow-through {k} bars", f"{s1[col].mean() * 100:.1f}%",
                    f"pullback not revisited and >= {cfg.ft_min_r}R move within {k} bars")

    # performance split by PRIOR-session daily context (no same-day information)
    for col, label in (("ctx_vol_bucket", "Vol bucket (prior day)"),
                       ("ctx_regime", "Regime (prior day)"),
                       ("ctx_align", "Trend vs 20SMA (prior day)")):
        vals = sorted(v for v in pd.concat([static[col], trail[col]]).dropna().unique() if v != "n/a")
        for v in vals:
            s_b, t_b = static[static[col] == v], trail[trail[col] == v]
            add(f"{label}: {v}", f"{len(s_b)} static / {len(t_b)} trail",
                f"static {s_b['R'].sum():+.2f}R (avg {s_b['R'].mean() if len(s_b) else 0:+.2f}) | "
                f"trail {t_b['R'].sum():+.2f}R (avg {t_b['R'].mean() if len(t_b) else 0:+.2f})")

    verdict = ("trailing reduced the bleed" if r_t > r_s else
               "static was better on this sample" if r_s > r_t else
               "no difference between models")
    add("Reading", verdict, "check per-setup rows before scaling to more symbols")
    return pd.DataFrame(rows)


def write_excel(symbol: str, df: pd.DataFrame, results: dict, cfg: Config, path: str) -> None:
    from openpyxl.styles import Font, PatternFill

    sel = [cfg.session_sheet_setup] if cfg.session_sheet_setup else list(results)
    selected = {k: results[k] for k in sel}
    all_trades = _concat(
        [selected[k]["static"] for k in sel] + [selected[k]["atr_trail"] for k in sel],
        cfg.ladder_max_r,
    )
    summary_df = summarize(all_trades)
    inference_df = build_inference(symbol, df, selected, cfg)
    ladder_df = ladder_summary(all_trades)
    daily_df = add_daily_indicators(df, cfg)

    hdr = PatternFill("solid", fgColor="D9E1F2")
    green, red = PatternFill("solid", fgColor="C6EFCE"), PatternFill("solid", fgColor="FFC7CE")
    font, bold = Font(name="Arial", size=9), Font(name="Arial", size=9, bold=True)

    def colour(ws, names, rule):
        heads = [c.value for c in ws[1]]
        idx = [i for i, h in enumerate(heads) if h in names]
        for row in ws.iter_rows(min_row=2):
            for i in idx:
                fill = rule(row[i].value)
                if fill is not None:
                    row[i].fill = fill

    def sign_rule(v):
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return green if v > 0 else (red if v < 0 else None)
        return None

    def above_rule(v):
        return green if v == "Above" else (red if v == "Below" else None)

    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        summary_df.to_excel(xw, sheet_name="Summary", index=False)
        inference_df.to_excel(xw, sheet_name="Summary", index=False,
                              startrow=len(summary_df) + 3)
        _naive(all_trades).to_excel(xw, sheet_name="Trades_All", index=False)
        ladder_df.to_excel(xw, sheet_name="Target_Ladder", index=False)
        daily_df.round(4).to_excel(xw, sheet_name="Daily_Indicators", index=False)

        for ws in xw.book.worksheets:                      # light formatting
            ws.freeze_panes = "B2"
            for cell in ws[1]:
                cell.font, cell.fill = bold, hdr
            for col in ws.columns:
                w = max(len(str(c.value)) if c.value is not None else 0 for c in col[:60])
                ws.column_dimensions[col[0].column_letter].width = min(max(9, w + 2), 26)
            for row in ws.iter_rows(min_row=2):
                for cell in row:
                    cell.font = font
            # style the inference header on the Summary sheet
            if ws.title == "Summary":
                for row in ws.iter_rows(min_row=2):
                    if row[0].value == "Metric":
                        for cell in row:
                            cell.font, cell.fill = bold, hdr
                        break
            if ws.title == "Trades_All":
                colour(ws, {"R"} | set(_ladder_cols(cfg.ladder_max_r)) - {"max_R"}, sign_rule)
            if ws.title == "Target_Ladder":
                colour(ws, {"avg_R", "total_R"}, sign_rule)
            if ws.title == "Daily_Indicators":
                colour(ws, {"price_vs_SMA20"}, above_rule)


# =============================================================================
# 8. RUNNER
# =============================================================================
def run_symbol(raw: pd.DataFrame, cfg: Config):
    df = add_global_filters(add_indicators(raw, cfg), cfg)
    prior = prior_daily(add_daily_indicators(df, cfg))
    results = {}
    for key, (_, fn) in SETUPS.items():
        sig = fn(df, cfg)
        static = simulate_trades(df, sig, cfg, key, model="static")
        trail = (simulate_trades(df, sig, cfg, key, model="atr_trail")
                 if cfg.run_trailing else _empty_trades())
        static = enrich_trades(static, df, sig, prior, cfg)
        trail = enrich_trades(trail, df, sig, prior, cfg)
        results[key] = {
            "sig": attach_exit_flags(sig, static),
            "static": static,
            "atr_trail": trail,
        }
    return df, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="use synthetic data (offline)")
    ap.add_argument("--symbols", nargs="*", default=STOCKS)
    ap.add_argument("--out", default=Config.out_dir)
    ap.add_argument("--session-sheets", choices=["ALL", "S1", "S2", "S3"], default="ALL")
    ap.add_argument("--trail-atr", type=float, default=Config.trail_atr_mult,
                    help="ATR trailing distance in ATR(14) units (default 1.5)")
    ap.add_argument("--trail-activate-r", type=float, default=Config.trail_activate_r,
                    help="start trailing after this much favourable R (default 0)")
    ap.add_argument("--no-trail", action="store_true", help="skip the ATR trailing model")
    ap.add_argument("--no-npr", action="store_true", help="disable the S1 non-pivot reversal reference")
    args = ap.parse_args()

    cfg = Config(out_dir=args.out,
                 session_sheet_setup=None if args.session_sheets == "ALL" else args.session_sheets,
                 trail_atr_mult=args.trail_atr,
                 trail_activate_r=args.trail_activate_r,
                 run_trailing=not args.no_trail,
                 s1_npr_enabled=not args.no_npr)
    os.makedirs(cfg.out_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    portfolio = []
    for sym in args.symbols:
        try:
            raw = make_synthetic(sym) if args.demo else load_data(sym, cfg)
            df, results = run_symbol(raw, cfg)
            path = os.path.join(cfg.out_dir, f"{sym.replace('.', '_')}_backtest.xlsx")
            write_excel(sym, df, results, cfg, path)
            tr = _concat(
                [v["static"] for v in results.values()] + [v["atr_trail"] for v in results.values()],
                cfg.ladder_max_r,
            )
            tr.insert(0, "symbol", sym)
            portfolio.append(tr)
            log.info("%-15s %4d sessions | trades S1/S2/S3 = %s (static) / %s (trail) -> %s",
                     sym, df["date"].nunique(),
                     "/".join(str(len(results[k]["static"])) for k in SETUPS),
                     "/".join(str(len(results[k]["atr_trail"])) for k in SETUPS), path)
        except Exception as ex:  # keep going if one ticker fails
            log.warning("%-15s FAILED: %s", sym, ex)

    if portfolio:
        allt = pd.concat([p for p in portfolio if not p.empty], ignore_index=True) \
            if any(not p.empty for p in portfolio) else portfolio[0]
        with pd.ExcelWriter(os.path.join(cfg.out_dir, "_portfolio_summary.xlsx"), engine="openpyxl") as xw:
            summarize(allt).to_excel(xw, sheet_name="All_Stocks", index=False)
            rows = [{"symbol": s, **r} for s, g in allt.groupby("symbol")
                    for r in summarize(g).to_dict("records")]
            pd.DataFrame(rows).to_excel(xw, sheet_name="By_Stock", index=False)
            ladder_summary(allt).to_excel(xw, sheet_name="Target_Ladder", index=False)


if __name__ == "__main__":
    main()