"""Test OCR on a real comic image with the new Vulkan backend."""
import base64
import io
import json
import time
import urllib.request
from pathlib import Path
from PIL import Image

img_path = Path("test_real/0000.jpg")
orig = Image.open(img_path)
print(f"Image: {img_path} ({img_path.stat().st_size // 1024} KB, {orig.size})")

# Prepare at 1024px
max_dim = 1024
img = Image.open(img_path)
w, h = img.size
ratio = max_dim / max(w, h)
img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
if img.mode != "RGB":
    img = img.convert("RGB")
buf = io.BytesIO()
img.save(buf, format="JPEG", quality=90)
b64 = base64.b64encode(buf.getvalue()).decode()

print(f"Prepared: {img.size}, payload {len(b64) // 1024} KB")

payload = json.dumps({
    "model": "ocr-api",
    "prompt": "Extract all text from this image. Output only the text, nothing else.",
    "images": [b64],
    "stream": False,
}).encode()

req = urllib.request.Request(
    "http://localhost:11434/api/generate",
    data=payload,
    headers={"Content-Type": "application/json"},
)

# First request (includes model load)
print("\nFirst request (includes model loading)...")
t0 = time.perf_counter()
with urllib.request.urlopen(req, timeout=600) as resp:
    data = json.loads(resp.read())
elapsed = time.perf_counter() - t0
text = data.get("response", "")
print(f"  Time: {elapsed:.1f}s | Output: {len(text)} chars")
print(f"  Text: {text[:300]}")

# Check processor
ps_req = urllib.request.Request("http://localhost:11434/api/ps")
with urllib.request.urlopen(ps_req) as resp:
    ps = json.loads(resp.read())
for m in ps.get("models", []):
    sz = m.get("size", 0)
    vr = m.get("size_vram", 0)
    pct = 100 * vr / sz if sz > 0 else 0
    print(f"\nModel: {m.get('name')} | Size: {sz/1e9:.1f}GB | VRAM: {vr/1e9:.1f}GB ({pct:.0f}%)")

# Second request (warm)
print("\nSecond request (warm model)...")
t0 = time.perf_counter()
with urllib.request.urlopen(req, timeout=600) as resp:
    data = json.loads(resp.read())
elapsed = time.perf_counter() - t0
text = data.get("response", "")
print(f"  Time: {elapsed:.1f}s | Output: {len(text)} chars")
print(f"  Text: {text[:300]}")
