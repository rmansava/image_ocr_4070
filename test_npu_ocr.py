"""Test Gemma3 4B on NPU vs GLM-OCR on GPU for comic OCR quality."""
import base64
import io
import json
import time
import urllib.request
from pathlib import Path
from PIL import Image

# Test images
IMAGES = sorted(Path(r"C:\Users\rick\image_ocr\test_real").glob("*.jpg"))[:5]
print(f"Testing {len(IMAGES)} comic images\n")


def prepare_image(img_path, max_dim=1024):
    img = Image.open(img_path)
    w, h = img.size
    if max(w, h) > max_dim:
        ratio = max_dim / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def test_flm_gemma3(img_path, b64):
    """Test via FastFlowLM OpenAI-compatible API (NPU)."""
    payload = json.dumps({
        "model": "gemma3:4b",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract all text from this image. Output only the text, nothing else."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                ]
            }
        ],
        "max_tokens": 2048,
        "temperature": 0,
    }).encode()

    req = urllib.request.Request(
        "http://127.0.0.1:52625/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read())
    elapsed = time.perf_counter() - t0
    text = data["choices"][0]["message"]["content"].strip()
    return text, elapsed


def test_ollama_glm(img_path, b64):
    """Test via Ollama GLM-OCR (GPU)."""
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
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read())
    elapsed = time.perf_counter() - t0
    text = data.get("response", "").strip()
    return text, elapsed


# Warm up both
print("Warming up Gemma3 4B (NPU)...")
b64_warm = prepare_image(IMAGES[0])
try:
    _, t = test_flm_gemma3(IMAGES[0], b64_warm)
    print(f"  Gemma3 warm-up: {t:.1f}s")
except Exception as e:
    print(f"  Gemma3 ERROR: {e}")

print("Warming up GLM-OCR (GPU)...")
try:
    _, t = test_ollama_glm(IMAGES[0], b64_warm)
    print(f"  GLM-OCR warm-up: {t:.1f}s")
except Exception as e:
    print(f"  GLM-OCR ERROR: {e}")

print("\n" + "=" * 80)

# Test each image
for i, img_path in enumerate(IMAGES):
    print(f"\n{'='*80}")
    print(f"Image {i}: {img_path.name}")
    print(f"{'='*80}")

    b64 = prepare_image(img_path)

    # Gemma3 on NPU
    print(f"\n--- Gemma3 4B (NPU) ---")
    try:
        text_g, time_g = test_flm_gemma3(img_path, b64)
        print(f"Time: {time_g:.1f}s | Chars: {len(text_g)}")
        print(text_g[:500])
    except Exception as e:
        print(f"ERROR: {e}")
        text_g, time_g = "", 0

    # GLM-OCR on GPU
    print(f"\n--- GLM-OCR (GPU) ---")
    try:
        text_o, time_o = test_ollama_glm(img_path, b64)
        print(f"Time: {time_o:.1f}s | Chars: {len(text_o)}")
        print(text_o[:500])
    except Exception as e:
        print(f"ERROR: {e}")
        text_o, time_o = "", 0

print("\n\nDone!")
