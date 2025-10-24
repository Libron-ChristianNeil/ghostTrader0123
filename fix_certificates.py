import certifi, ssl, os
from pathlib import Path

# Output merged CA path
merged_path = Path(certifi.where()).parent / "merged_cacerts.pem"

# Load certifi bundle
with open(certifi.where(), "rb") as f:
    certifi_certs = f.read()

# Load Windows root certificates
win_certs = ssl.enum_certificates("ROOT")
combined_certs = b"".join([c[0] for c in win_certs])

# Merge and save
with open(merged_path, "wb") as f:
    f.write(certifi_certs + combined_certs)

print("✅ Merged certificate bundle created at:", merged_path)
