"""Test Ollama parallel config and find optimal throughput settings."""
import asyncio
import base64
import time
import aiohttp

OLLAMA_URL = "http://localhost:11434"
IMAGE_PATH = "test_images/clear_comic.png"


async def ocr_once(session, img_b64, model="ocr-api", num_predict=None):
    payload = {
        "model": model,
        "prompt": "Extract all text from this image. Output only the text, nothing else.",
        "images": [img_b64],
        "stream": False,
    }
    if num_predict:
        payload["options"] = {"num_predict": num_predict}
    async with session.post(f"{OLLAMA_URL}/api/generate", json=payload) as resp:
        data = await resp.json()
        return data.get("response", "").strip()


async def check_ps():
    """Check how many parallel slots Ollama reports."""
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{OLLAMA_URL}/api/ps") as resp:
            data = await resp.json()
            print(f"Running models: {data}")


async def bench(label, model, count, workers, img_b64, num_predict=None):
    sem = asyncio.Semaphore(workers)
    timeout = aiohttp.ClientTimeout(total=300)

    async def guarded(session):
        async with sem:
            return await ocr_once(session, img_b64, model, num_predict)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        t0 = time.perf_counter()
        tasks = [guarded(session) for _ in range(count)]
        results = await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - t0

    rate = count / elapsed
    print(f"{label}: {count} imgs in {elapsed:.1f}s ({rate:.2f} img/s) "
          f"=> 2M in {2_000_000/rate/3600:.0f}h | output len: {len(results[0])} chars")


async def main():
    with open(IMAGE_PATH, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    await check_ps()

    # Warm up
    print("\nWarming up...")
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        await ocr_once(session, img_b64)

    await check_ps()

    n = 10
    print(f"\n--- Benchmarks ({n} images each) ---")

    # Test: limit output tokens (faster if less text to generate)
    await bench("ocr-api, 4w, 512tok", "ocr-api", n, 4, img_b64, num_predict=512)
    await bench("ocr-api, 4w, 256tok", "ocr-api", n, 4, img_b64, num_predict=256)

    # Test: direct glm-ocr vs custom ocr-api modelfile
    await bench("glm-ocr direct, 4w", "glm-ocr", n, 4, img_b64)

    # Test: higher concurrency
    await bench("ocr-api, 8w", "ocr-api", n, 8, img_b64)
    await bench("ocr-api, 16w", "ocr-api", n, 16, img_b64)


asyncio.run(main())
