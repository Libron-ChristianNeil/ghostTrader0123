#!/usr/bin/env python3
"""
Simple worker: buy ETH perpetual futures every 10 minutes until terminated.
- Safe defaults: DRY_RUN=1 (no real orders). Set DRY_RUN=0 to enable live orders.
- Config via environment variables:
    API_KEY, SECRET_KEY, PASSPHRASE  (OKX credentials)
    INST_ID (default ETH-USDT-SWAP)
    USD_PER_TRADE (default 10)
    LEVERAGE (default 3)
    TD_MODE (isolated/cross; default isolated)
    INTERVAL_SECONDS (default 600)
    DRY_RUN (1/0 default 1)
- Uses OKX REST signing (HMAC SHA256 + base64).
"""

import os
import time
import json
import hmac
import base64
import hashlib
from datetime import datetime, timezone
import logging
import requests
from dotenv import load_dotenv

# -------------------------
# Load secrets from .env
# -------------------------
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)
API_KEY = os.getenv("API_KEY_DEMO")
SECRET_KEY = os.getenv("SECRET_KEY-DEMO")
PASSPHRASE = os.getenv("PASSPHRASE-DEMO")

if not (API_KEY and SECRET_KEY and PASSPHRASE):
    raise SystemExit("❌ Missing API credentials in .env")

# -------------------------
# User Config (safe to edit)
# -------------------------
BASE_URL = "https://www.okx.com"
INST_ID = "ETH-USDT-SWAP"   # Instrument
USD_PER_TRADE = 100           # USD per trade (before leverage)
LEVERAGE = 3                 # 3x leverage
TD_MODE = "isolated"         # "isolated" or "cross"
INTERVAL_SECONDS = 600       # 10 minutes
DRY_RUN = True               # Set to False for live trading
QTY_DECIMALS = 4             # Round quantity to this many decimals

# -------------------------
# Logging setup
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("buy_eth")

# -------------------------
# Helper functions
# -------------------------
def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

def _sign(ts: str, method: str, path: str, body: str = "") -> str:
    msg = ts + method.upper() + path + body
    mac = hmac.new(SECRET_KEY.encode(), msg.encode(), hashlib.sha256)
    return base64.b64encode(mac.digest()).decode()

def okx_request(method, path, data=None, params=None, auth=True):
    url = BASE_URL + path
    body_str = json.dumps(data) if data else ""
    headers = {"Content-Type": "application/json"}
    if auth:
        ts = _timestamp()
        headers.update({
            "OK-ACCESS-KEY": API_KEY,
            "OK-ACCESS-SIGN": _sign(ts, method, path, body_str),
            "OK-ACCESS-TIMESTAMP": ts,
            "OK-ACCESS-PASSPHRASE": PASSPHRASE,
        })
    try:
        if method == "GET":
            r = requests.get(url, headers=headers, params=params, timeout=15)
        else:
            r = requests.post(url, headers=headers, json=data, timeout=15)
        j = r.json()
        if j.get("code") != "0":
            log.warning("OKX error: %s", j)
        return j
    except Exception as e:
        log.error("Request failed: %s", e)
        return None

def get_price():
    r = okx_request("GET", "/api/v5/market/ticker", params={"instId": INST_ID}, auth=False)
    try:
        return float(r["data"][0]["last"])
    except Exception:
        return None

def place_order(qty):
    data = {
        "instId": INST_ID,
        "tdMode": TD_MODE,
        "side": "buy",
        "ordType": "market",
        "sz": str(qty),
        "posSide": "long"
    }
    return okx_request("POST", "/api/v5/trade/order", data=data, auth=True)

def calc_qty(price):
    effective = USD_PER_TRADE * LEVERAGE
    return round(effective / price, QTY_DECIMALS)

# -------------------------
# Main loop
# -------------------------
def main():
    log.info("Starting ETH futures auto-buyer (every %d seconds)", INTERVAL_SECONDS)
    log.info("Mode: %s", "DRY-RUN" if DRY_RUN else "LIVE")

    while True:
        price = get_price()
        if not price:
            log.warning("Could not fetch price; retrying in 30s")
            time.sleep(30)
            continue

        qty = calc_qty(price)
        log.info(f"Price: {price:.2f}, Buy Qty: {qty} ETH (x{LEVERAGE} leverage)")

        if DRY_RUN:
            log.info("DRY_RUN=True → no order sent.")
        else:
            res = place_order(qty)
            log.info("Order response: %s", json.dumps(res))

        log.info("Sleeping for %d seconds...", INTERVAL_SECONDS)
        try:
            time.sleep(INTERVAL_SECONDS)
        except KeyboardInterrupt:
            log.info("Terminated by user.")
            break

if __name__ == "__main__":
    main()



