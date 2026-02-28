"""Steady-state throughput benchmark — model stays warm throughout."""
import asyncio
import base64
import time
import aiohttp

OLLAMA_URL = "http://localhost:11434"
IMAGE_PATH = "test_images/clear_comic.png"


async def ocr_once(session, img_b64):
    payload = {
        "model": "ocr-api",
        "prompt": "Extract all text from this image. Output only the text, nothing else.",
        "images": [img_b64],
        "stream": False,
    }
    async with session.post(f"{OLLAMA_URL}/api/generate", json=payload) as resp:
        data = await resp.json()
        return data.get("response", "").strip()


async def bench(n, workers, img_b64):
    sem = asyncio.Semaphore(workers)
    timeout = aiohttp.ClientTimeout(total=600)

    async def guarded(session):
        async with sem:
            return await ocr_once(session, img_b64)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        t0 = time.perf_counter()
        tasks = [guarded(session) for _ in range(n)]
        await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - t0

    rate = n / elapsed
    return elapsed, rate


async def main():
    with open(IMAGE_PATH, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    timeout = aiohttp.ClientTimeout(total=300)

    # Warm up — ensure model is fully loaded
    print("Warming up (3 requests)...")
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for _ in range(3):
            await ocr_once(session, img_b64)
    print("Model warm.\n")

    # Check model state
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{OLLAMA_URL}/api/ps") as resp:
            data = await resp.json()
            for m in data.get("models", []):
                vram = m.get("size_vram", 0)
                size = m.get("size", 0)
                proc = "GPU" if vram > 0 else "CPU"
                print(f"Model: {m['name']} | Size: {size/1e9:.1f}GB | VRAM: {vram/1e9:.1f}GB | Processor: {proc}")
    print()

    # Steady-state benchmark: 20 images at different worker counts
    n = 20
    for workers in [1, 2, 4, 8]:
        elapsed, rate = await bench(n, workers, img_b64)
        hours_2m = 2_000_000 / rate / 3600
        days_2m = hours_2m / 24
        print(f"Workers={workers:2d}: {n} imgs in {elapsed:5.1f}s "
              f"| {rate:.2f} img/s | {1/rate:.2f}s/img "
              f"| 2M: {hours_2m:.0f}h ({days_2m:.1f} days)")


asyncio.run(main())
