#!/usr/bin/env python3
"""
OKX SOL-USDT-SWAP Futures Bot (demo) — v2
- Uses 3x isolated leverage
- Each trade NOTIONAL ≈ 500 PHP (converted to USD at runtime)
- SAR-based signals: 15m reversal + 1h trend alignment
- TP = +3.7% ; SL = -2%
- Stop after 3 consecutive losses
- Manual console control: start/stop/status/reset_losses/exit
- Optional Telegram alerts
- Optional Tor proxy (set USE_TOR = True)
"""

import os
import time
import json
import math
import base64
import hmac
import hashlib
import logging
import threading
from queue import Queue, Empty
from datetime import datetime, timezone
from pathlib import Path

import requests
import pandas as pd
import pandas_ta as ta
from dotenv import load_dotenv

# -------------------------
# Load environment
# -------------------------
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# OKX API keys (use demo keys here)
API_KEY = os.getenv("API_KEY_DEMO") or os.getenv("API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY_DEMO") or os.getenv("SECRET_KEY")
PASSPHRASE = os.getenv("PASSPHRASE_DEMO") or os.getenv("PASSPHRASE")

TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN")
TG_CHAT_ID = os.getenv("TG_CHAT_ID")

if not (API_KEY and SECRET_KEY and PASSPHRASE):
    raise SystemExit("Missing OKX API keys in ghosttrader0123.env (API_KEY_DEMO/SECRET_KEY_DEMO/PASSPHRASE_DEMO)")

# -------------------------
# Config (tweakable)
# -------------------------
BASE_URL = "https://www.okx.com"
USE_TOR = False
PROXIES = {}

# Instrument & strategy
INST_ID = "ETH-USDT-SWAP"       # the perpetual futures instrument
BAR_SHORT = "1m"
BAR_LONG = "3m"
CANDLE_LIMIT = 500

# Money / leverage constraints
PHP_PER_TRADE = 100000             # fixed PHP per trade
LEVERAGE = 3                    # 3x
ISOLATED = True                 # isolated margin
TD_MODE = "isolated" if ISOLATED else "cross"

# TP/SL
TP_PCT = 0.037                  # +3.7
SL_PCT = -0.02                  # -2.0

# Loss control
MAX_CONSECUTIVE_LOSSES = 3

# Polling
POLL_INTERVAL = 10              # seconds between strategy checks
POLL_INTERVAL_SHORT = 10        # seconds during active trade monitoring

# Order rounding
QTY_DECIMALS = 3                # round quantity to 3 decimals (reasonable for SOL)

# Logging
LOGFILE = Path(__file__).parent / "okx_sol_futures_bot_v2.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOGFILE, encoding="utf-8")]
)
logger = logging.getLogger("okx_sol_v2")

# -------------------------
# State
# -------------------------
state = {
    "running": False,
    "halted": False,
    "consecutive_losses": 0,
    "positions": {},        # track live positions by inst_id (local mirror)
    "last_trade": None
}
control_q = Queue()

# -------------------------
# Utilities: alerts
# -------------------------
def send_telegram(msg: str):
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        logger.info("Telegram not configured; message: %s", msg)
        return
    try:
        url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TG_CHAT_ID, "text": msg}
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code != 200:
            logger.warning("TG send failed: %s %s", resp.status_code, resp.text)
    except Exception as e:
        logger.exception("Failed to send telegram: %s", e)

def notify(msg: str):
    logger.info("NOTIFY: %s", msg)
    send_telegram(msg)

# -------------------------
# OKX auth & requests
# -------------------------
def _ts() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + "Z"

def okx_sign(timestamp: str, method: str, request_path: str, body: str = "") -> str:
    message = timestamp + method.upper() + request_path + (body or "")
    mac = hmac.new(SECRET_KEY.encode(), message.encode(), hashlib.sha256)
    return base64.b64encode(mac.digest()).decode()

def okx_request(method: str, path: str, params=None, data=None, auth: bool = True, timeout: int = 30):
    url = BASE_URL + path
    body_str = json.dumps(data) if data else ""
    headers = {"Content-Type": "application/json"}
    if auth:
        ts = _ts()
        headers.update({
            "OK-ACCESS-KEY": API_KEY,
            "OK-ACCESS-SIGN": okx_sign(ts, method, path, body_str),
            "OK-ACCESS-TIMESTAMP": ts,
            "OK-ACCESS-PASSPHRASE": PASSPHRASE
        })
    try:
        if USE_TOR:
            resp = requests.request(method, url, headers=headers, params=params if method == "GET" else None,
                                    json=data if method != "GET" else None, proxies=PROXIES, timeout=timeout)
        else:
            resp = requests.request(method, url, headers=headers, params=params if method == "GET" else None,
                                    json=data if method != "GET" else None, timeout=timeout)
    except Exception as e:
        logger.exception("Network error calling OKX %s %s: %s", method, path, e)
        return None

    try:
        j = resp.json()
    except Exception:
        logger.error("Non-JSON response from OKX (%s): %s", resp.status_code, resp.text[:400])
        return None

    # helpful debug for non-zero code
    if isinstance(j, dict) and j.get("code") and j.get("code") != "0":
        logger.warning("OKX API returned code != 0: %s", j)
    return j

# -------------------------
# Exchange rate helper (PHP -> USD)
# - uses exchangerate.host free API at runtime
# -------------------------
def fetch_php_to_usd() -> float:
    """
    Fetch live conversion rate PHP -> USD using exchangerate.host.
    Returns USD per PHP (float), or default fallback if fail.
    """
    try:
        r = requests.get("https://api.exchangerate.host/convert?from=PHP&to=USD&amount=1", timeout=10)
        if r.status_code == 200:
            jr = r.json()
            if jr.get("result") is not None:
                return float(jr["result"])
    except Exception as e:
        logger.warning("Failed to fetch exchange rate: %s", e)

    # fallback conservative default (approx) — keep as last resort
    fallback = 0.0171
    logger.warning("Using fallback PHP->USD rate: %s", fallback)
    return fallback

# -------------------------
# Market data helpers
# -------------------------
def fetch_candles(inst_id: str, bar: str, limit: int = CANDLE_LIMIT) -> pd.DataFrame | None:
    """
    Fetch latest candle data from OKX for a given instrument and timeframe.
    """
    params = {"instId": inst_id, "bar": bar, "limit": limit}
    res = okx_request("GET", "/api/v5/market/candles", params=params, auth=False)
    if not res:
        return None

    raw = res.get("data") if isinstance(res, dict) and "data" in res else res
    if not raw:
        logger.error("Empty candles response: %s", res)
        return None

    try:
        # ✅ Create list of dictionaries approach (bypasses column count validation)
        candles = []
        for candle in raw:
            if len(candle) >= 6:  # Ensure we have at least the basic OHLCV data
                candle_dict = {
                    'ts': candle[0],
                    'open': candle[1],
                    'high': candle[2],
                    'low': candle[3],
                    'close': candle[4],
                    'vol': candle[5]
                }
                candles.append(candle_dict)
        
        df = pd.DataFrame(candles)
        
    except Exception as e:
        logger.error("❌ Failed to parse candles: %s", e)
        return None

    # ✅ Convert timestamp
    if 'ts' in df.columns:
        df['ts'] = pd.to_datetime(df['ts'].astype(float), unit='ms', errors='coerce')
        df.set_index('ts', inplace=True)
        df.sort_index(inplace=True)

    # ✅ Convert numeric columns
    numeric_cols = ['open', 'high', 'low', 'close', 'vol']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    logger.info(f"Fetched {len(df)} candles for {inst_id} ({bar})")
    return df


def get_ticker_last(inst_id: str) -> float | None:
    res = okx_request("GET", "/api/v5/market/ticker", params={"instId": inst_id}, auth=False)
    if not res:
        return None
    data = res.get("data") if isinstance(res, dict) and "data" in res else res
    if isinstance(data, list) and len(data) > 0:
        try:
            return float(data[0].get("last") or data[0].get("lastPrice") or data[0].get("c"))
        except Exception:
            return None
    return None

# -------------------------
# Indicators & strategy
# -------------------------
def compute_sar_series(df: pd.DataFrame) -> pd.Series:
    """Compute Parabolic SAR (PSAR) using pandas_ta"""
    psar = ta.psar(high=df['high'], low=df['low'], close=df['close'], af=0.03, max_af=0.3)
    # psar returns a DataFrame with columns like 'PSARl_0.02_0.2', 'PSARs_0.02_0.2', 'PSARaf_0.02_0.2', 'PSARr_0.02_0.2'
    # We want the main PSAR values - typically the first column contains the SAR values
    if isinstance(psar, pd.DataFrame) and not psar.empty:
        # Get the first column which should contain the SAR values
        sar_series = psar.iloc[:, 0]
        # Rename it for clarity
        sar_series.name = 'psar'
        return sar_series
    else:
        logger.error("Failed to compute PSAR")
        return pd.Series(index=df.index, name='psar')

def detect_signals(short_df: pd.DataFrame, long_df: pd.DataFrame):
    """
    Returns ('buy'/'sell'/None, reason)
    - short_df: 15m candles
    - long_df: 1h candles
    """
    if short_df is None or long_df is None or len(short_df) < 3 or len(long_df) < 3:
        return None, "insufficient data"

    short_sar = compute_sar_series(short_df)
    long_sar = compute_sar_series(long_df)

    # Check if we have valid SAR data
    if short_sar.empty or long_sar.empty:
        return None, "failed to compute SAR indicators"

    s_prev_close = short_df['close'].iloc[-2]
    s_last_close = short_df['close'].iloc[-1]
    s_prev_sar = short_sar.iloc[-2]
    s_last_sar = short_sar.iloc[-1]

    # 15m reversal detection
    short_sig = None
    if (s_prev_close < s_prev_sar) and (s_last_close > s_last_sar):
        short_sig = "buy"
    elif (s_prev_close > s_prev_sar) and (s_last_close < s_last_sar):
        short_sig = "sell"

    # 1h trend check: simple check using last close vs last sar and slope of last closes
    l_last_close = long_df['close'].iloc[-1]
    l_prev_close = long_df['close'].iloc[-5] if len(long_df) >= 6 else long_df['close'].iloc[-2]
    l_last_sar = long_sar.iloc[-1]

    trend_up = (l_last_close > l_last_sar) and (l_last_close > l_prev_close)
    trend_down = (l_last_close < l_last_sar) and (l_last_close < l_prev_close)

    if short_sig == "buy" and trend_up:
        return "buy", "15m PSAR bullish reversal + 1h trend up"
    if short_sig == "sell" and trend_down:
        return "sell", "15m PSAR bearish reversal + 1h trend down"

    return None, f"short_sig={short_sig}, trend_up={trend_up}, trend_down={trend_down}"

def detect_signals(short_df: pd.DataFrame, long_df: pd.DataFrame):
    """
    Returns ('buy'/'sell'/None, reason)
    - short_df: 15m candles
    - long_df: 1h candles
    """
    if short_df is None or long_df is None or len(short_df) < 3 or len(long_df) < 3:
        return None, "insufficient data"

    short_sar = compute_sar_series(short_df)
    long_sar = compute_sar_series(long_df)

    s_prev_close = short_df['close'].iloc[-2]
    s_last_close = short_df['close'].iloc[-1]
    s_prev_sar = short_sar.iloc[-2]
    s_last_sar = short_sar.iloc[-1]

    # 15m reversal detection
    short_sig = None
    if (s_prev_close < s_prev_sar) and (s_last_close > s_last_sar):
        short_sig = "buy"
    elif (s_prev_close > s_prev_sar) and (s_last_close < s_last_sar):
        short_sig = "sell"

    # 1h trend check: simple check using last close vs last sar and slope of last closes
    l_last_close = long_df['close'].iloc[-1]
    l_prev_close = long_df['close'].iloc[-5] if len(long_df) >= 6 else long_df['close'].iloc[-2]
    l_last_sar = long_sar.iloc[-1]

    trend_up = (l_last_close > l_last_sar) and (l_last_close > l_prev_close)
    trend_down = (l_last_close < l_last_sar) and (l_last_close < l_prev_close)

    if short_sig == "buy" and trend_up:
        return "buy", "15m SAR bullish reversal + 1h trend up"
    if short_sig == "sell" and trend_down:
        return "sell", "15m SAR bearish reversal + 1h trend down"

    return None, f"short_sig={short_sig}, trend_up={trend_up}, trend_down={trend_down}"

# -------------------------
# Futures helpers: set leverage and place orders
# -------------------------
def set_isolated_leverage(inst_id: str, leverage: int):
    """
    Set leverage in isolated margin mode for the instrument.
    """
    path = "/api/v5/account/set-leverage"
    data = {"instId": inst_id, "lever": str(leverage), "mgnMode": "isolated" if ISOLATED else "cross"}
    res = okx_request("POST", path, data=data, auth=True)
    logger.info("Set leverage response: %s", res)
    return res

def place_market_order_futures(inst_id: str, side: str, sz: str, pos_side: str):
    """
    Place a market order for futures.
    side: 'buy' or 'sell'
    pos_side: 'long' or 'short' (OKX expects posSide for hedge mode; even in single-side, it's fine)
    """
    path = "/api/v5/trade/order"
    payload = {
        "instId": inst_id,
        "tdMode": TD_MODE,      # isolated/cross
        "side": side,           # buy / sell
        "ordType": "market",
        "sz": str(sz),
        "posSide": pos_side
    }
    res = okx_request("POST", path, data=payload, auth=True)
    logger.info("Order placed: %s", res)
    return res

def fetch_position_for_inst(inst_id: str):
    res = okx_request("GET", "/api/v5/account/positions", auth=True)
    if not res or 'data' not in res:
        return None
    for p in res['data']:
        if p.get("instId") == inst_id:
            return p
    return None

# -------------------------
# PnL helpers
# -------------------------
def calc_pnl_pct(entry_price: float, mark_price: float, pos_side: str) -> float:
    # pos_side: 'long' or 'short'
    if entry_price == 0:
        return 0.0
    if pos_side == "long":
        return (mark_price - entry_price) / entry_price
    else:
        return (entry_price - mark_price) / entry_price

# -------------------------
# Trade lifecycle: open -> monitor -> close (TP/SL)
# -------------------------
def open_and_monitor_trade(direction: str, notional_usd: float):
    """
    direction: 'buy' -> long ; 'sell' -> short
    notional_usd: USD amount to risk before leverage
    """
    if state["halted"]:
        logger.info("Bot halted due to consecutive losses. Not opening new trades.")
        return

    # 1) Get live price
    price = get_ticker_last(INST_ID)
    if price is None:
        logger.error("Cannot fetch market price; aborting trade open.")
        return

    # 2) Compute quantity in base (SOL) taking leverage into account:
    # effective buying power = notional_usd * leverage
    effective_usd = notional_usd * LEVERAGE
    qty = effective_usd / price   # number of SOL to long/short
    qty = round(qty, QTY_DECIMALS)
    if qty <= 0:
        logger.error("Computed qty <= 0 (price=%s notional=%s qty=%s). Aborting.", price, notional_usd, qty)
        return

    # set leverage (OKX requires set-leverage call)
    set_isolated_leverage(INST_ID, LEVERAGE)

    # Map direction -> side & pos_side
    if direction == "buy":
        side = "buy"
        pos_side = "long"
    else:
        side = "sell"
        pos_side = "short"

    logger.info("Placing market order: %s %s contracts (approx) at price=%s", side, qty, price)
    order_res = place_market_order_futures(INST_ID, side, str(qty), pos_side)
    # Note: exact fill/entry price ideally parsed from fills; we use mark price as approximation then verify via positions endpoint
    time.sleep(1)  # small wait for position to register

    # Attempt to fetch position details for entry price
    pos = fetch_position_for_inst(INST_ID)
    entry_price = None
    if pos:
        # OKX position fields may include 'avgPx' or 'avgPxBL' depending response; try 'avgPx'
        try:
            # 'avgPx' often is str of average entry price; fallback to mark price
            entry_price = float(pos.get("avgPx") or pos.get("avgPxReal") or price)
        except Exception:
            entry_price = price
    else:
        entry_price = price

    trade = {
        "direction": direction,
        "side": side,
        "pos_side": pos_side,
        "qty": qty,
        "entry_price": float(entry_price),
        "opened_at": datetime.now(timezone.utc).isoformat(),
        "closed": False
    }
    state["positions"][INST_ID] = trade
    state["last_trade"] = trade
    notify(f"Opened {pos_side} {qty} {INST_ID} at {entry_price:.6f} (notional ${notional_usd:.2f}, {LEVERAGE}x isolated)")

    # monitor PnL
    logger.info("Monitoring PnL for TP=%+.2f%% SL=%+.2f%%", TP_PCT * 100, SL_PCT * 100)
    while True:
        # If bot manually paused, still keep monitoring but wait
        if not state["running"]:
            logger.info("Bot paused; monitoring still active but waiting until resumed.")
            while not state["running"]:
                time.sleep(1)
        time.sleep(POLL_INTERVAL_SHORT)

        mark = get_ticker_last(INST_ID)
        if mark is None:
            logger.warning("Unable to fetch mark price; skipping this PnL check.")
            continue

        pnl = calc_pnl_pct(trade["entry_price"], mark, trade["pos_side"])
        logger.info("Live PnL: %.4f%% (entry=%s mark=%s)", pnl * 100, trade["entry_price"], mark)

        # Take Profit
        if pnl >= TP_PCT:
            close_side = "sell" if trade["pos_side"] == "long" else "buy"
            logger.info("TP reached (%.4f%%). Closing position.", pnl * 100)
            place_market_order_futures(INST_ID, close_side, str(qty), trade["pos_side"])
            trade["closed"] = True
            trade["exit_price"] = mark
            trade["pnl_pct"] = pnl
            trade["exit_time"] = datetime.now(timezone.utc).isoformat()
            notify(f"TP hit: {trade['pos_side']} entry {trade['entry_price']} -> exit {mark} PnL={pnl*100:.2f}%")
            break

        # Stop Loss
        if pnl <= SL_PCT:
            close_side = "sell" if trade["pos_side"] == "long" else "buy"
            logger.info("SL reached (%.4f%%). Closing position.", pnl * 100)
            place_market_order_futures(INST_ID, close_side, str(qty), trade["pos_side"])
            trade["closed"] = True
            trade["exit_price"] = mark
            trade["pnl_pct"] = pnl
            trade["exit_time"] = datetime.now(timezone.utc).isoformat()
            notify(f"SL hit: {trade['pos_side']} entry {trade['entry_price']} -> exit {mark} PnL={pnl*100:.2f}%")
            break

    # finalize result
    if trade.get("pnl_pct") is not None and trade["pnl_pct"] < 0:
        state["consecutive_losses"] += 1
    else:
        state["consecutive_losses"] = 0

    logger.info("Trade closed. PnL=%.4f%% . Consecutive losses=%d", trade.get("pnl_pct", 0.0) * 100, state["consecutive_losses"])

    # Halt if too many losses
    if state["consecutive_losses"] >= MAX_CONSECUTIVE_LOSSES:
        state["halted"] = True
        state["running"] = False
        notify(f"HALT: {state['consecutive_losses']} consecutive losing trades. Bot halted.")

    # cleanup local position
    state["positions"].pop(INST_ID, None)

# -------------------------
# Console control thread
# -------------------------
def console_thread():
    logger.info("Console ready. Type: start / stop / status / reset_losses / exit")
    while True:
        try:
            cmd = input("> ").strip().lower()
        except EOFError:
            break
        if cmd:
            control_q.put(cmd)
            if cmd == "exit":
                break
    logger.info("Console thread exiting.")

# -------------------------
# Main orchestrator
# -------------------------
def main_loop():
    logger.info("Bot started (demo-mode). INST=%s LEVERAGE=%dx PHP_PER_TRADE=%s", INST_ID, LEVERAGE, PHP_PER_TRADE)

    # Get live USD rate once at start and also compute dynamic each cycle
    while True:
        # process console commands
        try:
            cmd = control_q.get_nowait()
            if cmd == "start":
                if state["halted"]:
                    logger.info("Bot halted due to losses. Reset using 'reset_losses' first.")
                else:
                    state["running"] = True
                    logger.info("Bot set to RUNNING.")
            elif cmd == "stop":
                state["running"] = False
                logger.info("Bot PAUSED.")
            elif cmd == "status":
                logger.info("Status running=%s halted=%s consecutive_losses=%d positions=%s last_trade=%s",
                            state["running"], state["halted"], state["consecutive_losses"],
                            json.dumps(state.get("positions")), json.dumps(state.get("last_trade")))
            elif cmd == "reset_losses":
                state["consecutive_losses"] = 0
                state["halted"] = False
                logger.info("Consecutive losses reset. Bot un-halted.")
            elif cmd == "exit":
                logger.info("Exit requested. Shutting down.")
                break
            else:
                logger.info("Unknown command: %s", cmd)
        except Empty:
            pass

        if not state["running"]:
            time.sleep(1)
            continue

        # Fetch candles and evaluate signals
        try:
            short_df = fetch_candles(INST_ID, BAR_SHORT, limit=300)
            long_df = fetch_candles(INST_ID, BAR_LONG, limit=400)
            if short_df is None or long_df is None:
                logger.warning("Failed to fetch candles; retrying after sleep.")
                time.sleep(POLL_INTERVAL)
                continue

            # --- NEW FEATURE: Display latest price and SAR levels ---
            latest_price = get_ticker_last(INST_ID)
            if latest_price:
                # Compute SARs
                short_sar = compute_sar_series(short_df)
                long_sar = compute_sar_series(long_df)
                if not short_sar.empty and not long_sar.empty:
                    short_sar_val = short_sar.iloc[-1]
                    long_sar_val = long_sar.iloc[-1]
                    logger.info(f"📊 Price={latest_price:.4f} | Short SAR={short_sar_val:.4f} | Long SAR={long_sar_val:.4f}")
                else:
                    logger.warning("SAR computation failed or empty.")
            else:
                logger.warning("Could not fetch latest price.")

            # --- Existing strategy check ---
            signal, reason = detect_signals(short_df, long_df)
            logger.info("Strategy check signal=%s reason=%s", signal, reason)


            if signal in ("buy", "sell"):
                # compute notional in USD from PHP_PER_TRADE using live rate
                php2usd = fetch_php_to_usd()   # USD per PHP
                notional_usd = PHP_PER_TRADE * php2usd
                logger.info("PHP->USD rate %.6f : notional USD %.2f for PHP %d", php2usd, notional_usd, PHP_PER_TRADE)

                # Check if no existing open position for instrument
                if INST_ID in state["positions"]:
                    logger.info("Already have an open position on %s; skipping new entry.", INST_ID)
                else:
                    # Open and monitor trade (this blocks until trade closed)
                    open_and_monitor_trade(signal, notional_usd)

                # small cooldown after a trade cycle
                logger.info("Cooldown sleeping %s seconds.", POLL_INTERVAL)
                time.sleep(POLL_INTERVAL)
            else:
                # no trade
                logger.info("No aligned signal. Sleeping %s seconds.", POLL_INTERVAL)
                time.sleep(POLL_INTERVAL)

        except Exception as e:
            logger.exception("Unexpected error in main loop: %s", e)
            time.sleep(POLL_INTERVAL)

# -------------------------
# Startup
# -------------------------
if __name__ == "__main__":
    # quick connectivity check (non-fatal)
    try:
        if USE_TOR:
            r = requests.get("https://www.okx.com", proxies=PROXIES, timeout=10)
        else:
            r = requests.get("https://www.okx.com", timeout=10)
        logger.info("Connectivity test HTTP %s", r.status_code)
    except Exception as e:
        logger.warning("Connectivity test failed: %s", e)

    # start console thread
    t = threading.Thread(target=console_thread, daemon=True)
    t.start()

    try:
        main_loop()
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Exiting.")
    notify("Bot stopped (script exit).")


