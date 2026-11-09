# Intraday Backtesting Engine (`all_setups.py`)

A robust, zero-paid-API algorithmic backtesting pipeline for Indian Equities (NSE) using 5-minute OHLCV candle data. The framework is split into a modular data ingestion engine (`load_data_new.py`) and a comprehensive strategy backtester (`all_setups.py`) featuring multiple intraday setups, dynamic ATR trailing stops, global filters, and Excel report generation.

---

## 🏗️ Project Architecture

1. **`load_data_new.py` (Data Pipeline):**
   - Standalone, zero-credential data fetcher using the free Upstox Historical Candle V3 API (with Yahoo Finance as a 60-day fallback).
   - Automatically downloads and parses the NSE cash equity instrument master into a lightweight local cache (`.instrument_cache/upstox_nse_eq.csv`).
   - Automatically chunks monthly requests, normalizes data to Indian Standard Time (`Asia/Kolkata`), and clips bars to regular market hours (`09:15` - `15:30`).

2. **`all_setups.py` (Strategy Engine & Backtester):**
   - Implements core intraday setups: **S1 Momentum Pullback**, **S2 Opening Flag**, and **S3 Sandwich**.
   - Applies strict global filters (Volume SMA20, VWAP/EMA/RSI momentum checks, ATR% bands, and gap caps).
   - Executes trade simulation with slippage, fixed stop-loss/2R targets, break-even management, and ratcheted ATR trailing stops.
   - Generates multi-sheet Excel workbooks summarizing per-setup performance, trade logs, and daily technical indicators.

---

## 🔑 0) One-Time Authentication
Generate a token from the [Upstox Developer Dashboard](https://account.upstox.com/developer/apps) and configure it using **one** of these methods:

* **Token File (Recommended):** 
  ```bash
  echo YOUR_TOKEN > .upstox_token

## 1) Data Validation & Utility Commands (`load_data_new.py`)

| Command | Description |
| :--- | :--- |
| `python load_data_new.py --symbol SBIN --provider upstox` | Validates connection and prints a per-session coverage report for a symbol. |
| `python load_data_new.py --symbol SBIN --provider upstox --cross-check` | Compares Upstox vs. Yahoo Finance over the overlapping ~60-day window. |
| `python load_data_new.py --export-json data_json --all-stocks --provider upstox --days 300` | Seeds offline JSON datasets for all default stocks for archive/reruns. |
| `python load_data_new.py --check-instruments --symbol SBIN` | Validates the compact NSE symbol-to-instrument key cache. |
| `python load_data_new.py --search KALYAN` | Searches the local instrument map for matching tickers. |
| `python load_data_new.py --refresh-instruments` | Re-downloads and rebuilds the local symbol map cache. |
| `python load_data_new.py --dump-instruments symbols.csv` | Exports the full compact instrument map to a CSV file. |

---

## 2) Running Backtests (`all_setups.py`)

| # | Objective | Command |
| :--- | :--- | :--- |
| **1** | **Single Stock (Default)** | `python all_setups.py --symbols SBIN --out backtest_upstox` |
| **2** | **Single Stock (Tuned ATR Trail)** | `python all_setups.py --symbols SBIN --trail-atr 2.0 --trail-activate-r 0.5 --out backtest_upstox` |
| **3** | **Single Setup Only** | `python all_setups.py --symbols SBIN --session-sheets S1 --out backtest_upstox` *(Options: `S1`, `S2`, `S3`)* |
| **4** | **All 9 Default Stocks** | `python all_setups.py --out backtest_upstox` |
| **5** | **Multi-Symbol List** | `python all_setups.py --symbols SBIN ICICIBANK TATASTEEL HINDALCO --out backtest_upstox` |
| **6** | **Static Model Only** | `python all_setups.py --symbols SBIN --no-trail --out backtest_upstox` *(Disables ATR trailing simulation)* |
| **7** | **Offline Smoke Test** | `python all_setups.py --demo --symbols SBIN --out _demo_out` *(Uses synthetic offline data; no network/token required)* |