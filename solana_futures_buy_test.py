import requests
import certifi
import json
import base64
import hmac
import hashlib
from datetime import datetime, timezone
from dotenv import load_dotenv
from pathlib import Path
import os
import time

# === Define the instrument ID first ===
INSTRUMENT_ID = "SOL-USDT-SWAP"  # Solana perpetual futures
import json

proxies = {
    'http': 'socks5h://127.0.0.1:9150',
    'https': 'socks5h://127.0.0.1:9150'
}

def okx_request(method, path, body=None):
    url = "https://www.okx.com" + path
    headers = {
        "Content-Type": "application/json",
        # add your auth headers if needed
    }

    if method.upper() == "GET":
        r = requests.get(url, headers=headers, proxies=proxies, verify=certifi.where(), timeout=30)
    elif method.upper() == "POST":
        r = requests.post(url, headers=headers, json=body, proxies=proxies, verify=certifi.where(), timeout=30)
    else:
        raise ValueError("Unsupported HTTP method")

    return r.json()
    return response_json

# Then call it
ticker = okx_request("GET", f"/api/v5/market/ticker?instId={INSTRUMENT_ID}")
sol_price = float(ticker["data"][0]["last"])
print(f"Current SOL price: {sol_price} USDT")

# === CONFIG ===
LIVE_MODE = False  # ⚠️ Set to False for demo (paper), True for live trades
usd_quantity = 3000      # ~₱500
LEVERAGE = 3
ORDER_SIZE = "0.3"    # ~0.3 SOL (~$25 position at 3x)
sol_qty = round(usd_quantity / sol_price, 2)

# === Load API credentials ===
env_path = Path(__file__).parent / "demo_keys.env"
load_dotenv(dotenv_path=env_path)

API_KEY = os.getenv("API_KEY_DEMO")
SECRET_KEY = os.getenv("SECRET_KEY_DEMO")
PASSPHRASE = os.getenv("PASSPHRASE_DEMO")

# === Tor Proxy (default port 9150) ===
PROXIES = {
    'http': 'socks5h://127.0.0.1:9150',
    'https': 'socks5h://127.0.0.1:9150'
}

BASE_URL = "https://www.okx.com"

# === Helper: signature ===
def okx_sign(timestamp, method, request_path, body=""):
    message = f"{timestamp}{method.upper()}{request_path}{body}"
    signature = base64.b64encode(
        hmac.new(SECRET_KEY.encode('utf-8'), message.encode('utf-8'), hashlib.sha256).digest()
    ).decode()
    return signature

# === Helper: send request ===
def okx_request(method, path, body=None):
    url = BASE_URL + path
    timestamp = datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace("+00:00", "Z")
    body_str = json.dumps(body) if body else ""

    headers = {
        "OK-ACCESS-KEY": API_KEY,
        "OK-ACCESS-SIGN": okx_sign(timestamp, method, path, body_str),
        "OK-ACCESS-TIMESTAMP": timestamp,
        "OK-ACCESS-PASSPHRASE": PASSPHRASE,
        "Content-Type": "application/json",
    }

    if not LIVE_MODE:
        headers["x-simulated-trading"] = "1"

    try:
        resp = requests.request(
            method,
            url,
            headers=headers,
            data=body_str if body else None,
            proxies=PROXIES,
            timeout=45
        )
        print(f"📡 {method} {path} → {resp.status_code}")
        return resp.json()
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return None

# === Step 1: Check market info ===
print("🔍 Checking Solana Futures market info...")
info = okx_request("GET", f"/api/v5/public/instruments?instType=SWAP&instId={INSTRUMENT_ID}")
print(json.dumps(info, indent=2))

# === Step 2: Place BUY order ===
mode_label = "LIVE" if LIVE_MODE else "DEMO"
print(f"\n🚀 Placing {mode_label} BUY order on {INSTRUMENT_ID}...")

order_body = {
    "instId": INSTRUMENT_ID,
    "tdMode": "isolated",
    "side": "buy",
    "ordType": "market",
    "sz": sol_qty,
    "lever": str(LEVERAGE),
    "posSide": "long"  # <--- ADD THIS
}

response = okx_request("POST", "/api/v5/trade/order", order_body)
print("\n✅ Order Response:")
print(json.dumps(response, indent=2))

# === Step 3: Check open positions ===
time.sleep(2)
positions = okx_request("GET", f"/api/v5/account/positions?instType=SWAP&instId={INSTRUMENT_ID}")
print("\n📊 Open Positions:")
print(json.dumps(positions, indent=2))
