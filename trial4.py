import requests
import certifi
import json
import base64
import hmac
import hashlib
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timezone
from dotenv import load_dotenv
from pathlib import Path
import os
import time

# === CONFIG ===
INSTRUMENT_ID = "SOL-USDT-SWAP"  # Solana perpetual futures
LIVE_MODE = False                 # ⚠️ Set to False for demo (paper), True for live trades
USE_TOR = False                    # ✅ Toggle Tor proxy ON/OFF
USD_AMOUNT = 1000
LEVERAGE = 3
TPSL = True                       # ✅ Toggle Take Profit / Stop Loss ON/OFF
TP = 0.037                        # 4% Take Profit
SL = 0.018                        # 2% Stop Loss

SMALL_INTERVAL = "15m"         # can be 1m, 3m, 5m, 15m, 1H, 4H, 1D, etc.
LARGE_INTERVAL = "1H" 
REFRESH_SEC = 300        # how often to refresh
LIMIT = 300             # number of candles to fetch

# === Tor Proxy (default port 9150) ===
TOR_PROXIES = {
    'http': 'socks5h://127.0.0.1:9150',
    'https': 'socks5h://127.0.0.1:9150'
}
proxies = TOR_PROXIES if USE_TOR else None

BASE_URL = "https://www.okx.com"

# === Load API credentials ===
env_path = Path(__file__).parent / "demo_keys.env"
load_dotenv(dotenv_path=env_path)

API_KEY = os.getenv("API_KEY_DEMO")
SECRET_KEY = os.getenv("SECRET_KEY_DEMO")
PASSPHRASE = os.getenv("PASSPHRASE_DEMO")

# ========================================================#
# === Unified OKX Request Function (Public + Private) === #
# ========================================================#
def okx_request(method, path, params=None, auth=False):
    """Handles both public and authenticated OKX API requests."""
    url = BASE_URL + path
    timestamp = datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace("+00:00", "Z")

    # --- Build request body ---
    if method.upper() == "GET" and params:
        query_string = "?" + "&".join([f"{k}={v}" for k, v in params.items()])
        request_path = path + query_string
        body_str = ""
    else:
        request_path = path
        body_str = json.dumps(params) if params else ""

    # --- Build headers ---
    headers = {"Content-Type": "application/json"}

    # Add authentication headers if required
    if auth:
        message = f"{timestamp}{method.upper()}{request_path}{body_str}"
        signature = base64.b64encode(
            hmac.new(SECRET_KEY.encode('utf-8'), message.encode('utf-8'), hashlib.sha256).digest()
        ).decode()

        headers.update({
            "OK-ACCESS-KEY": API_KEY,
            "OK-ACCESS-SIGN": signature,
            "OK-ACCESS-TIMESTAMP": timestamp,
            "OK-ACCESS-PASSPHRASE": PASSPHRASE
        })

        if not LIVE_MODE:
            headers["x-simulated-trading"] = "1"

    # --- Choose proxy based on config ---
    
    proxy_label = "🧅 via Tor" if USE_TOR else "🌐 direct"

    # --- Make request ---
    try:
        if method.upper() == "GET":
            r = requests.get(url, headers=headers, params=params, proxies=proxies, verify=certifi.where(), timeout=45)
        else:
            r = requests.post(url, headers=headers, data=body_str, proxies=proxies, verify=certifi.where(), timeout=45)

        print(f"📡 {method.upper()} {path} → {r.status_code} {proxy_label}")
        return r.json()
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return None
#======================================#



#======================================#
# === Place Market Order Function ===  #
#======================================#
def place_order(inst_id, side, pos_side, sol_qty, leverage, live_mode):
    """
    Places a market order (LONG or SHORT) on OKX.

    Args:
        inst_id (str): Instrument ID (e.g., "SOL-USDT-SWAP")
        side (str): "buy" for long, "sell" for short
        pos_side (str): "long" or "short"
        sol_qty (float): Position size in units of the coin
        leverage (int or str): Leverage multiplier
        live_mode (bool): True = live, False = demo
        trade_direction (str): "long" or "short"
    """
    mode_label = "LIVE" if live_mode else "DEMO"
    print(f"\n🚀 Placing {mode_label} {pos_side.upper()} order on {inst_id}...")

    order_body = {
        "instId": inst_id,
        "tdMode": "isolated",
        "side": side,
        "ordType": "market",
        "sz": sol_qty,
        "lever": str(leverage),
        "posSide": pos_side
    }

    response = okx_request("POST", "/api/v5/trade/order", order_body, auth=True)

    print("\n✅ Order Response:")
    print(json.dumps(response, indent=2))

    
    # === Wait for order fill ===
    time.sleep(3)
    positions = okx_request("GET", "/api/v5/account/positions", {"instType": "SWAP", "instId": INSTRUMENT_ID}, auth=True)
    print("\n📊 Open Positions:")
    print(json.dumps(positions, indent=2))


    # === Get filled order to find entry price ===
    filled_orders = okx_request("GET", "/api/v5/trade/orders-history", {"instType": "SWAP", "instId": INSTRUMENT_ID, "limit": "1"}, auth=True)
    orders_data = filled_orders.get("data", [])

    if not orders_data:
        print("⚠️ No filled orders found yet. Retrying...")
        time.sleep(5)
        filled_orders = okx_request("GET", "/api/v5/trade/orders-history", {"instType": "SWAP", "instId": INSTRUMENT_ID, "limit": "1"}, auth=True)
        orders_data = filled_orders.get("data", [])

    if not orders_data:
        raise Exception("❌ Still no filled orders found — check if your market order executed.")

    last_order = orders_data[0]
    entry_price = float(last_order["avgPx"])
    print(f"🎯 Entry Price: {entry_price}")

    # === Only execute TP/SL if enabled ===
    if TPSL:
        print("\n⚙️ TP/SL enabled — calculating and placing OCO order...")

        # === Calculate TP and SL dynamically ===
        if pos_side == "long":
            take_profit_price = round(entry_price * (1 + TP), 4)  # +4% profit
            stop_loss_price   = round(entry_price * (1 - SL), 4)  # -2% loss
        else:
            take_profit_price = round(entry_price * (1 - TP), 4)  # +4% profit (short)
            stop_loss_price   = round(entry_price * (1 + SL), 4)  # -2% loss (short)

        print(f"📈 Take Profit: {take_profit_price}")
        print(f"📉 Stop Loss: {stop_loss_price}")

        # === Fetch position to get size ===
        positions = okx_request(
            "GET",
            "/api/v5/account/positions",
            {"instId": INSTRUMENT_ID},
            auth=True
        )
        pos_data = positions.get("data", [])

        if not pos_data:
            raise Exception("❌ No open position found for TP/SL placement.")

        pos_sz = pos_data[0]["pos"]
        print(f"📊 Position size for TP/SL: {pos_sz}")

        # === Place OCO (TP + SL) Algo Order ===
        if pos_side == "long":  
            opposite_side = "sell"
        else:
            opposite_side = "buy"   

        algo_body = {
            "instId": INSTRUMENT_ID,
            "tdMode": "isolated",
            "side": opposite_side,
            "posSide": pos_side,
            "ordType": "oco",         # One-Cancels-the-Other
            "sz": pos_sz,
            "tpTriggerPx": str(take_profit_price),
            "tpOrdPx": "-1",          # market order when triggered
            "slTriggerPx": str(stop_loss_price),
            "slOrdPx": "-1"           # market order when triggered
        }

        tp_sl_response = okx_request("POST", "/api/v5/trade/order-algo", algo_body, auth=True)
        print("\n✅ TP/SL Algo Order Response:")
        print(json.dumps(tp_sl_response, indent=2))
    else:
        print("\n⚠️ TP/SL disabled — skipping take-profit and stop-loss setup.")
#======================================#



#======================================#
# ===       Fetching Candles      ===  #
#======================================#
def fetch_candles(inst_id, bar="1m", limit=100):
    """Fetch historical kline data from OKX via Tor."""
    url = f"{BASE_URL}/api/v5/market/candles?instId={inst_id}&bar={bar}&limit={limit}"
    r = requests.get(url, proxies=proxies, verify=certifi.where(), timeout=30)
    data = r.json().get("data", [])
    if not data:
        raise Exception("Failed to fetch candle data.")
    
    # Only take first 5 columns: ts, open, high, low, close
    df = pd.DataFrame(data, columns=["ts","open","high","low","close"] + [f"extra{i}" for i in range(len(data[0])-5)])
    
    df = df.astype({
        "open": float,
        "high": float,
        "low": float,
        "close": float
    })
    df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="ms")
    df = df.sort_values("ts").reset_index(drop=True)
    return df
#======================================#



#======================================#
# ===         Calculate SAR       ===  #
#======================================#
def calc_sar(df, af=0.02, af_max=0.2):
    """Compute Parabolic SAR using pandas_ta functional API (psar)."""
    psar = ta.psar(high=df["high"], low=df["low"], close=df["close"], af=af, max_af=af_max)
    df["SAR"] = psar.apply(lambda row: row.dropna().iloc[0], axis=1)
    df["SAR"] = df["SAR"].round(2)

    return df
#======================================#



#======================================#
# ===        Detect Reversal      ===  #
#======================================#
def detect_reversal(df):
    """Detect SAR flip: when SAR crosses from below to above price or vice versa."""
    if len(df) < 2:
        return None
    prev = df.iloc[-2]
    curr = df.iloc[-1]

    prev_bullish = prev["SAR"] < prev["close"]
    curr_bullish = curr["SAR"] < curr["close"]

    if prev_bullish != curr_bullish:
        return "BULLISH" if curr_bullish else "BEARISH"
    return None
#======================================#



#======================================#
# ===   SAR Monitoring + Trading    ===#
#======================================#
def monitor_sar_and_trade():

    """Monitor SAR signals on multiple timeframes and trade when aligned."""
    print("\n📡 Starting SAR monitor loop...")
    open_position = False
    last_signal = None

    while True:
        try:
            # === Fetch small and large timeframe candles ===
            small_df = fetch_candles(INSTRUMENT_ID, SMALL_INTERVAL, LIMIT)
            large_df = fetch_candles(INSTRUMENT_ID, LARGE_INTERVAL, LIMIT)

            # === Calculate SAR ===
            small_df = calc_sar(small_df)
            large_df = calc_sar(large_df)

            # === Detect reversals on small timeframe ===
            reversal = detect_reversal(small_df)
            current_price = float(small_df.iloc[-1]["close"])
            small_sar = float(small_df.iloc[-1]["SAR"])
            large_sar = float(large_df.iloc[-1]["SAR"])

            large_bullish = large_sar < current_price
            large_bearish = large_sar > current_price

            print(f"\n🕒 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                  f"Price: {current_price:.4f} | Small SAR: {small_sar:.2f} | Large SAR: {large_sar:.2f}")
            print(f"Reversal: {reversal} | Large Trend: {'BULLISH' if large_bullish else 'BEARISH'}")

            # === Check if there's already an open position ===
            positions = okx_request("GET", "/api/v5/account/positions", {"instId": INSTRUMENT_ID}, auth=True)
            pos_data = positions.get("data", [])
            has_open_pos = any(float(p.get("pos", 0)) != 0 for p in pos_data)

            if has_open_pos:
                if not open_position:
                    print("📈 Position opened — monitoring until TP/SL hit.")
                    open_position = True
                else:
                    print("⏳ Still in position, waiting for TP/SL fill...")
                time.sleep(REFRESH_SEC)
                continue
            else:
                if open_position:
                    print(f"✅ Position closed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    open_position = False

            # === Decision Logic ===
            if not open_position and reversal:
                # Small timeframe just reversed — check alignment with large timeframe
                if reversal == "BULLISH" and large_bullish:
                    trade_direction = "long"
                elif reversal == "BEARISH" and large_bearish:
                    trade_direction = "short"
                else:
                    trade_direction = None

                if trade_direction:
                    # === Define sides ===
                    if trade_direction == "long":
                        side = "buy"
                        pos_side = "long"
                        print("🟢 Going LONG on SOL-USDT-SWAP")
                    else:
                        side = "sell"
                        pos_side = "short"
                        print("🔴 Going SHORT on SOL-USDT-SWAP")

                    # === Fetch balance and compute SOL quantity ===
                    account = okx_request("GET", "/api/v5/account/balance", None, auth=True)
                    if "data" in account and len(account["data"]) > 0:
                        usdt_bal = float(account["data"][0]["details"][0]["cashBal"])
                    else:
                        usdt_bal = USD_AMOUNT  # fallback to config

                    sol_qty = round((USD_AMOUNT * LEVERAGE) / current_price, 2)
                    print(f"💰 Trade size: {sol_qty} SOL @ {current_price}")

                    # === Place the trade ===
                    global opposite_side  
                    place_order(INSTRUMENT_ID, side, pos_side, sol_qty, LEVERAGE, LIVE_MODE)
                    print(f"🕒 Trade opened at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    open_position = True
                else:
                    print("⚖️ Reversal detected but not aligned with large trend — skipping.")
            else:
                print("📉 No actionable signal.")

        except Exception as e:
            print(f"❌ Error in monitor loop: {e}")

        time.sleep(REFRESH_SEC)


def simulate_monitor_sar_and_trade():
    """Simulate SAR signals and trades without sending real orders."""
    print("\n📡 Starting SAR monitor simulation loop...")
    open_position = False
    current_position = None  # Store current simulated position info

    while True:
        try:
            # === Fetch small and large timeframe candles ===
            small_df = fetch_candles(INSTRUMENT_ID, SMALL_INTERVAL, LIMIT)
            large_df = fetch_candles(INSTRUMENT_ID, LARGE_INTERVAL, LIMIT)

            # === Calculate SAR ===
            small_df = calc_sar(small_df)
            large_df = calc_sar(large_df)

            # === Detect reversals on small timeframe ===
            reversal = detect_reversal(small_df)
            current_price = float(small_df.iloc[-1]["close"])
            small_sar = float(small_df.iloc[-1]["SAR"])
            large_sar = float(large_df.iloc[-1]["SAR"])

            large_bullish = large_sar < current_price
            large_bearish = large_sar > current_price

            print(f"\n🕒 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                  f"Price: {current_price:.4f} | Small SAR: {small_sar:.2f} | Large SAR: {large_sar:.2f}")
            print(f"Reversal: {reversal} | Large Trend: {'BULLISH' if large_bullish else 'BEARISH'}")

            # === Check simulated open position ===
            if open_position:
                # Simulate TP/SL hit
                tp_hit = current_position['trade_direction'] == "long" and current_price >= current_position['tp']
                sl_hit = current_position['trade_direction'] == "long" and current_price <= current_position['sl']
                tp_hit = tp_hit or (current_position['trade_direction'] == "short" and current_price <= current_position['tp'])
                sl_hit = sl_hit or (current_position['trade_direction'] == "short" and current_price >= current_position['sl'])

                if tp_hit:
                    print(f"✅ TP hit for {current_position['trade_direction'].upper()} @ {current_price} | "
                          f"Size: {current_position['sol_qty']} SOL | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    open_position = False
                    current_position = None
                elif sl_hit:
                    print(f"❌ SL hit for {current_position['trade_direction'].upper()} @ {current_price} | "
                          f"Size: {current_position['sol_qty']} SOL | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    open_position = False
                    current_position = None
                else:
                    print("⏳ Position still open, monitoring TP/SL...")
                time.sleep(REFRESH_SEC)
                continue

            # === Decision Logic ===
            if not open_position and reversal:
                # Small timeframe just reversed — check alignment with large timeframe
                if reversal == "BULLISH" and large_bullish:
                    trade_direction = "long"
                elif reversal == "BEARISH" and large_bearish:
                    trade_direction = "short"
                else:
                    trade_direction = None

                if trade_direction:
                    # === Compute SOL quantity ===
                    sol_qty = round((USD_AMOUNT * LEVERAGE) / current_price, 3)

                    # === Calculate TP/SL ===
                    if trade_direction == "long":
                        tp_price = round(current_price * (1 + TP), 4)
                        sl_price = round(current_price * (1 - SL), 4)
                    else:
                        tp_price = round(current_price * (1 - TP), 4)
                        sl_price = round(current_price * (1 + SL), 4)

                    # === Simulate placing trade ===
                    current_position = {
                        "trade_direction": trade_direction,
                        "sol_qty": sol_qty,
                        "entry_price": current_price,
                        "tp": tp_price,
                        "sl": sl_price,
                        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }

                    print(f"\n🚀 Simulated {trade_direction.upper()} position opened!")
                    print(f"Size: {sol_qty} SOL | Entry: {current_price} | TP: {tp_price} | SL: {sl_price} | "
                          f"Time: {current_position['timestamp']}")
                    open_position = True
                else:
                    print("⚖️ Reversal detected but not aligned with large trend — skipping.")
            else:
                print("📉 No actionable signal.")

        except Exception as e:
            print(f"❌ Error in simulation loop: {e}")

        time.sleep(REFRESH_SEC)

if __name__ == "__main__":
    simulate_monitor_sar_and_trade()
