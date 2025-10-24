import requests
import json
import base64
import hmac
import hashlib
from datetime import datetime, timezone
from dotenv import load_dotenv
from pathlib import Path
import os

# Load .env
env_path = Path(__file__).parent / "demo_keys.env"
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("API_KEY_DEMO")
secret_key = os.getenv("SECRET_KEY_DEMO")
passphrase = os.getenv("PASSPHRASE_DEMO")

# Tor proxy configuration (default Tor Browser port)
proxies = {
    'http': 'socks5h://127.0.0.1:9150',
    'https': 'socks5h://127.0.0.1:9150'
}

print("🔍 Testing Tor connection to OKX...")

# First, test basic connectivity
try:
    test_response = requests.get('https://www.okx.com', proxies=proxies, timeout=30)
    print(f"Tor Connection Test - Status: {test_response.status_code}")
    
    if 'prohibitedaccess' not in test_response.text:
        print("✅ Tor is bypassing PLDT block!")
    else:
        print("❌ Still getting blocking page through Tor")
        exit()
        
except Exception as e:
    print(f"❌ Tor connection failed: {e}")
    print("Make sure Tor Browser is running and connected")
    exit()

# Now try the OKX API through Tor
def get_okx_balance_tor():
    url = "https://www.okx.com/api/v5/account/balance"
    
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
    method = "GET"
    request_path = "/api/v5/account/balance"
    
    print(f"\n📡 Making authenticated request through Tor...")
    print(f"Timestamp: {timestamp}")
    
    message = timestamp + method + request_path
    signature = base64.b64encode(
        hmac.new(
            secret_key.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).digest()
    ).decode('utf-8')
    
    headers = {
        'OK-ACCESS-KEY': api_key,
        'OK-ACCESS-SIGN': signature,
        'OK-ACCESS-TIMESTAMP': timestamp,
        'OK-ACCESS-PASSPHRASE': passphrase,
        'X-Simulated-Trading': '1',   # 👈 Required for demo accounts
        'Content-Type': 'application/json'
    }
    
    response = requests.get(url, headers=headers, proxies=proxies, timeout=30)
    print(f"HTTP Status: {response.status_code}")
    
    try:
        return response.json()
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error: {e}")
        print(f"Response: {response.text[:500]}")
        return None

# Get balance through Tor
balance = get_okx_balance_tor()

if balance:
    print("\n✅ SUCCESS! Account Balance through Tor:")
    print(json.dumps(balance, indent=2))
else:
    print("\n❌ Failed to get balance through Tor")