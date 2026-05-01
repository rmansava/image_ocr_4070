"""Polls http://localhost:8000/health until the vLLM server is ready."""
import sys
import time
import urllib.request

TIMEOUT = 1200  # 20 min
INTERVAL = 15

deadline = time.monotonic() + TIMEOUT
while time.monotonic() < deadline:
    try:
        urllib.request.urlopen("http://localhost:8000/health", timeout=5)
        print("\nvLLM server is ready!")
        sys.exit(0)
    except Exception:
        elapsed = int(time.monotonic() % 3600)
        print(f"  Waiting for vLLM server... {elapsed}s elapsed", end="\r")
        time.sleep(INTERVAL)

print("\nERROR: vLLM server did not become ready within 20 minutes.")
sys.exit(1)
