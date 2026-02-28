"""OCR engine using GLM-OCR via Ollama."""

import base64
import io
from pathlib import Path

import aiohttp
from PIL import Image

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MAX_DIM = 1280  # Max width or height for OCR input


def _prepare_image(image_path: Path, max_dim: int = DEFAULT_MAX_DIM) -> str:
    """Read image, resize if needed, return base64 encoded JPEG."""
    # Fast path: if already JPEG and within max_dim, skip decode/re-encode
    suffix = image_path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        with Image.open(image_path) as img:
            w, h = img.size
            if max(w, h) <= max_dim and img.mode in ("RGB", "L"):
                return base64.b64encode(image_path.read_bytes()).decode("utf-8")

    # Slow path: need to resize or re-encode
    with Image.open(image_path) as img:
        w, h = img.size

        # Resize if either dimension exceeds max_dim (preserve aspect ratio)
        if max(w, h) > max_dim:
            ratio = max_dim / max(w, h)
            new_w = int(w * ratio)
            new_h = int(h * ratio)
            img = img.resize((new_w, new_h), Image.LANCZOS)

        # Convert to RGB if needed (RGBA, palette, etc.)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        # Encode as JPEG for smaller payload (faster transfer)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


async def ocr_image(
    session: aiohttp.ClientSession,
    image_path: Path,
    model: str = "ocr-api",
    ollama_url: str = DEFAULT_OLLAMA_URL,
    max_dim: int = DEFAULT_MAX_DIM,
) -> str:
    """Send an image to Ollama for OCR and return the extracted text."""
    image_b64 = _prepare_image(image_path, max_dim)

    payload = {
        "model": model,
        "prompt": "Extract all text from this image. Output only the text, nothing else.",
        "images": [image_b64],
        "stream": False,
    }

    url = f"{ollama_url}/api/generate"
    async with session.post(url, json=payload) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(
                f"Ollama returned {resp.status} for {image_path.name}: {body}"
            )
        data = await resp.json()
        return data.get("response", "").strip()


async def check_ollama(ollama_url: str = DEFAULT_OLLAMA_URL) -> bool:
    """Check if Ollama is running."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{ollama_url}/api/tags",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                return resp.status == 200
    except (aiohttp.ClientError, OSError):
        return False


async def check_model(model: str, ollama_url: str = DEFAULT_OLLAMA_URL) -> bool:
    """Check if a model is available in Ollama."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{ollama_url}/api/tags",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status != 200:
                    return False
                data = await resp.json()
                models = [m.get("name", "") for m in data.get("models", [])]
                return any(
                    m == model or m.startswith(f"{model}:") for m in models
                )
    except (aiohttp.ClientError, OSError):
        return False
