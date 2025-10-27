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
LIVE_MODE = False                # ⚠️ Set to False for demo (paper), True for live trades
USE_TOR = False                  # ✅ Toggle Tor proxy ON/OFF
USD_AMOUNT = 1000
LEVERAGE = 3
TPSL = True                       # ✅ Toggle Take Profit / Stop Loss ON/OFF
TP = 0.037                        # 4% Take Profit
SL = 0.018                        # 2% Stop Loss

SMALL_INTERVAL = "5m"         # can be 1m, 3m, 5m, 15m, 1H, 4H, 1D, etc.
LARGE_INTERVAL = "1H" 
REFRESH_SEC = 120        # how often to refresh
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
            take_profit_price = round(entry_price * (1 + TP/LEVERAGE), 4)  # +4% profit
            stop_loss_price   = round(entry_price * (1 - SL/LEVERAGE), 4)  # -2% loss
        else:
            take_profit_price = round(entry_price * (1 - TP/LEVERAGE), 4)  # +4% profit (short)
            stop_loss_price   = round(entry_price * (1 + SL/LEVERAGE), 4)  # -2% loss (short)

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
            "ordType": "conditional",  # "conditonal" or "oco" (One-Cancels-the-Other)
            "sz": pos_sz,   
            # "tpTriggerPx": str(take_profit_price),                   
            # "tpOrdPx": "-1",          # market order when triggered   
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
# ===        Close Function       ===  #
#======================================#
def close_position(instrument_id):
    # Fetch the open position details
    positions = okx_request(
        "GET",
        "/api/v5/account/positions",
        {"instId": instrument_id},  # Use the provided instrument ID
        auth=True
    )

    pos_data = positions.get("data", [])
    if not pos_data:
        raise Exception("❌ No open position found to close.")

    # Get position details
    pos = pos_data[0]  # Assuming you have only one position for the instrument
    pos_side = pos["posSide"]  # 'long' or 'short'
    pos_sz = pos["pos"]        # Size of the open position

    # Determine the opposite side of the position
    if pos_side == "long":
        opposite_side = "sell"
    else:
        opposite_side = "buy"

    # Prepare the close order (market order to close position)
    close_order = {
        "instId": instrument_id,   # Instrument ID to close
        "tdMode": "isolated",      # Isolated margin mode
        "side": opposite_side,     # 'buy' if short, 'sell' if long
        "posSide": pos_side,       # 'long' or 'short'
        "ordType": "market",       # Market order to close the position immediately
        "sz": pos_sz,              # Close the entire position size
    }

    # Make the request to close the position
    close_response = okx_request("POST", "/api/v5/trade/order", close_order, auth=True)

    # Print the response to confirm the position is closed
    print("\n✅ Close Position Response:")
    print(json.dumps(close_response, indent=2))
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
    """Detect SAR reversal, including within the first two bars of the new trend."""
    if len(df) < 3:
        return None

    prev2 = df.iloc[-3]   # two bars ago
    prev1 = df.iloc[-2]   # one bar ago
    curr = df.iloc[-1]    # current bar

    # Determine bullish/bearish state for each bar
    prev2_bullish = prev2["SAR"] < prev2["close"]
    prev1_bullish = prev1["SAR"] < prev1["close"]
    curr_bullish = curr["SAR"] < curr["close"]

    # Detect a reversal if:
    # - It occurred in the last two bars (prev2 vs prev1 or prev1 vs curr)
    # - And current bar is in the new direction (to avoid false mid-trend triggers)
    if prev2_bullish != prev1_bullish and curr_bullish == prev1_bullish:
        return "BULLISH" if curr_bullish else "BEARISH"
    elif prev1_bullish != curr_bullish:
        return "BULLISH" if curr_bullish else "BEARISH"

    return None
#======================================#


#======================================#
# ===       Logging Function        ===#
#======================================#
def log_event(event_type, message):
    """
    Save important trading events to a log file.
    Args:
        event_type (str): 'OPEN', 'CLOSE', 'STOPLOSS', 'ERROR', etc.
        message (str): Description of the event.
    """
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"trading_log_{datetime.now().strftime('%Y-%m-%d')}.log"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] [{event_type.upper()}] {message}\n"

    with open(log_file, "a") as f:
        f.write(log_line)

    print(f"📝 Logged: {event_type} → {message}")
#======================================#



#======================================#
# ===   SAR Monitoring + Trading    ===#
#======================================#
def monitor_sar_and_trade():
    """Monitor SAR signals on multiple timeframes and trade when aligned."""
    print("\n📡 Starting SAR monitor loop...")
    open_position = False
    current_position = None  # Store current position details (for stop loss)
    trade_direction = None   # 'long' or 'short'

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
                open_position = True
                print("⏳ Still in position, monitoring SAR reversal and stop loss...")

                # Check for small SAR reversal (close position if reversal detected)
                sar_reversal = (trade_direction == "long" and reversal == "BEARISH") or \
                               (trade_direction == "short" and reversal == "BULLISH")
                if sar_reversal: 
                    print(f"🔁 SAR reversal detected, closing position @ {current_price}")
                    close_position(INSTRUMENT_ID)
                    log_event("CLOSE", f"Closed {trade_direction.upper()} @ {current_price} due to SAR reversal")
                    open_position = False
                    current_position = None
                    continue

            else:
                if current_position:
                    print(f"❌ Stop loss hit for {current_position['trade_direction'].upper()}")
                    log_event("STOPLOSS", f"Position stop loss triggered for {current_position['trade_direction'].upper()} @ Entry: {current_position['entry_price']})")
                    open_position = False
                    current_position = None
                    continue

            # === Decision Logic ===
            if not has_open_pos and reversal:
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
                    place_order(INSTRUMENT_ID, side, pos_side, sol_qty, LEVERAGE, LIVE_MODE)
                    log_event("OPEN", f"Opened {trade_direction.upper()} position — {sol_qty} SOL @ {current_price}")
                    print(f"🕒 Trade opened at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    open_position = True

                    # Save current position details
                    current_position = {
                        "trade_direction": trade_direction,
                        "sol_qty": sol_qty,
                        "entry_price": current_price,
                        "entry_sar": small_sar,  # store SAR value at entry
                        "sl": round(current_price * (1 - SL/LEVERAGE) if trade_direction == "long" else (1 + SL/LEVERAGE), 4)
                    }

                else:
                    print("⚖️ Reversal detected but not aligned with large trend — skipping.")
            else:
                print("📉 No actionable signal.")

        except Exception as e:
            print(f"❌ Error in monitor loop: {e}")

        time.sleep(REFRESH_SEC)


def simulate_monitor_sar_and_trade():
    """Simulate SAR signals and trades without sending real orders; close on SAR reversal or SL."""
    print("\n📡 Starting SAR monitor simulation loop...")
    open_position = False
    current_position = None

    while True:
        try:
            # === Fetch candles ===
            small_df = fetch_candles(INSTRUMENT_ID, SMALL_INTERVAL, LIMIT)
            large_df = fetch_candles(INSTRUMENT_ID, LARGE_INTERVAL, LIMIT)

            # === Calculate SAR ===
            small_df = calc_sar(small_df)
            large_df = calc_sar(large_df)

            reversal = detect_reversal(small_df)
            current_price = float(small_df.iloc[-1]["close"])
            small_sar = float(small_df.iloc[-1]["SAR"])
            large_sar = float(large_df.iloc[-1]["SAR"])

            large_bullish = large_sar < current_price
            large_bearish = large_sar > current_price

            print(f"\n🕒 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                  f"Price: {current_price:.4f} | Small SAR: {small_sar:.2f} | Large SAR: {large_sar:.2f}")
            print(f"Reversal: {reversal} | Large Trend: {'BULLISH' if large_bullish else 'BEARISH'}")

            # === Check open simulated position ===
            if open_position:
                trade_direction = current_position['trade_direction']
                entry_price = current_position['entry_price']

                # Check SL
                sl_hit = (trade_direction == "long" and current_price <= entry_price * (1 - SL/LEVERAGE)) or \
                         (trade_direction == "short" and current_price >= entry_price * (1 + SL/LEVERAGE))
                if sl_hit:
                    print(f"❌ SL hit for {trade_direction.upper()} @ {current_price}, closing simulated position...")
                    log_event("STOPLOSS", f"Postion stop loss triggered for {current_position['trade_direction'].upper()} @ Entry: {current_position['entry_price']})")
                    open_position = False
                    current_position = None
                    time.sleep(REFRESH_SEC)
                    continue

                # Check SAR reversal
                sar_reversal = (trade_direction == "long" and reversal == "BEARISH") or \
                               (trade_direction == "short" and reversal == "BULLISH")
                if sar_reversal:
                    print(f"🔄 SAR reversed, closing {trade_direction.upper()} simulated position @ {current_price}...")
                    log_event("CLOSE", f"Closed {trade_direction.upper()} @ {current_price} due to SAR reversal")
                    open_position = False
                    current_position = None
                    time.sleep(REFRESH_SEC)
                    continue

                print("⏳ Position still open, monitoring SL and SAR...")
                time.sleep(REFRESH_SEC)
                continue

            # === No open position → look for new trades ===
            if not open_position and reversal:
                if reversal == "BULLISH" and large_bullish:
                    trade_direction = "long"
                elif reversal == "BEARISH" and large_bearish:
                    trade_direction = "short"
                else:
                    trade_direction = None

                if trade_direction:
                    sol_qty = round((USD_AMOUNT * LEVERAGE) / current_price, 3)
                    current_position = {
                        "trade_direction": trade_direction,
                        "sol_qty": sol_qty,
                        "entry_price": current_price,
                        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    print(f"\n🚀 Simulated {trade_direction.upper()} position opened!")
                    log_event("OPEN", f"Opened {trade_direction.upper()} position — {sol_qty} SOL @ {current_price}")
                    print(f"Size: {sol_qty} SOL | Entry: {current_price} | Time: {current_position['timestamp']}")
                    open_position = True
                else:
                    print("⚖️ Reversal detected but not aligned with large trend — skipping.")
            else:
                print("📉 No actionable signal.")

        except Exception as e:
            print(f"❌ Error in simulation loop: {e}")

        time.sleep(REFRESH_SEC)


if __name__ == "__main__":
    monitor_sar_and_trade()

