"""vLLM inference engine — sends images to a local vLLM OpenAI-compatible server.

The vLLM server must be running in WSL before starting the pipeline:
  run_vllm.bat   (handles server startup automatically)

This engine matches the HFVisionEngine interface and adds infer_batch() so
the pipeline can feed an entire prefetch batch to vLLM simultaneously.
"""

import base64
import io
import json
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

VLLM_URL = "http://localhost:8000"
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_MAX_DIM = 1280
REP_PENALTY = 1.15
MAX_TOKENS = 4096
SERVER_READY_TIMEOUT = 1200  # 20 min — model load from /mnt/c takes ~10 min


class VLLMEngine:
    """HTTP client to a running vLLM OpenAI-compatible server."""

    def __init__(self, max_dim: int = DEFAULT_MAX_DIM, url: str = VLLM_URL):
        self.max_dim = max_dim
        self.url = url.rstrip("/")

    # ── Lifecycle (server manages its own lifecycle) ──────────────────────────

    def load(self):
        """Wait until the vLLM server is ready (polls /health)."""
        health = f"{self.url}/health"
        print(f"Waiting for vLLM server at {health} ...")
        deadline = time.monotonic() + SERVER_READY_TIMEOUT
        interval = 10
        while time.monotonic() < deadline:
            try:
                urllib.request.urlopen(health, timeout=5)
                print("vLLM server is ready.")
                return
            except Exception:
                remaining = int(deadline - time.monotonic())
                print(f"  Not ready — retrying (up to {remaining}s remaining)...", end="\r")
                time.sleep(interval)
        raise RuntimeError(
            f"vLLM server did not become ready within {SERVER_READY_TIMEOUT}s. "
            "Check that the server is running in WSL."
        )

    def unload(self):
        pass  # Server keeps running — shut it down manually when done

    # ── Image encoding ────────────────────────────────────────────────────────

    def _encode_image(self, image_path: Path) -> str:
        """Load, resize to max_dim, and return a base64-encoded JPEG string."""
        with Image.open(image_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            w, h = img.size
            if max(w, h) > self.max_dim:
                r = self.max_dim / max(w, h)
                img = img.resize((int(w * r), int(h * r)), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=92)
            return base64.b64encode(buf.getvalue()).decode("utf-8")

    # ── Single-image inference (pipeline compatibility) ───────────────────────

    def _call_server(self, b64_image: str, prompt: str) -> str:
        payload = {
            "model": MODEL_ID,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            "max_tokens": MAX_TOKENS,
            "temperature": 0,
            "repetition_penalty": REP_PENALTY,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.url}/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            result = json.loads(resp.read())
        return result["choices"][0]["message"]["content"].strip()

    def infer(self, image_path: Path, prompt: str) -> str:
        """Single-image inference — used by existing pipeline as fallback."""
        b64 = self._encode_image(image_path)
        return self._call_server(b64, prompt)

    # ── Batch inference — feeds the whole prefetch batch at once ─────────────

    def infer_batch(
        self, image_paths: list[Path], prompt: str
    ) -> list[tuple[str | None, str | None]]:
        """Send all images concurrently to vLLM. Returns (text, error) per image.

        vLLM's server batches concurrent requests internally, giving the same
        throughput as offline batch mode without needing WSL-side path access.
        """
        if not image_paths:
            return []

        def _run(idx: int, path: Path):
            try:
                b64 = self._encode_image(path)
                text = self._call_server(b64, prompt)
                return idx, text, None
            except Exception as exc:
                return idx, None, str(exc)

        results = [None] * len(image_paths)
        with ThreadPoolExecutor(max_workers=len(image_paths)) as pool:
            futures = [pool.submit(_run, i, p) for i, p in enumerate(image_paths)]
            for future in as_completed(futures):
                idx, text, err = future.result()
                results[idx] = (text, err)

        return results
