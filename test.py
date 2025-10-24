from okx import Account
import os

# Configure proxy
os.environ['HTTP_PROXY'] = 'http://your-proxy:port'
os.environ['HTTPS_PROXY'] = 'http://your-proxy:port'

# Or configure directly in code
proxies = {
    'http': 'http://your-proxy:port',
    'https': 'http://your-proxy:port'
}