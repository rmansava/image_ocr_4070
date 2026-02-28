"""Benchmark GLM-OCR throughput at different worker counts."""
import asyncio
import base64
import time
import aiohttp

OLLAMA_URL = "http://localhost:11434"
IMAGE_PATH = "test_images/clear_comic.png"
RUNS_PER_TEST = 10


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


async def bench_workers(worker_count, img_b64):
    sem = asyncio.Semaphore(worker_count)
    timeout = aiohttp.ClientTimeout(total=300)

    async def guarded(session):
        async with sem:
            return await ocr_once(session, img_b64)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        t0 = time.perf_counter()
        tasks = [guarded(session) for _ in range(RUNS_PER_TEST)]
        results = await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - t0

    rate = RUNS_PER_TEST / elapsed
    return elapsed, rate, results


async def main():
    with open(IMAGE_PATH, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    # Warm up
    print("Warming up...")
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        await ocr_once(session, img_b64)
    print()

    for workers in [1, 2, 4, 8]:
        elapsed, rate, results = await bench_workers(workers, img_b64)
        print(f"Workers={workers:2d}: {RUNS_PER_TEST} images in {elapsed:.1f}s "
              f"({rate:.2f} img/s, {1/rate:.2f}s/img) "
              f"=> 2M images in {2_000_000/rate/3600:.0f} hours")


asyncio.run(main())
