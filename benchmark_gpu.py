"""Benchmark GPU throughput on real comic images — parallel workers."""
import asyncio
import time
import aiohttp
from pathlib import Path
from image_ocr.ocr_engine import ocr_image

IMAGES = sorted(Path("test_real").glob("*.jpg"))[:20]
print(f"Benchmarking with {len(IMAGES)} real comic images on Vulkan GPU")


async def bench(workers, max_dim=1024):
    sem = asyncio.Semaphore(workers)
    timeout = aiohttp.ClientTimeout(total=600)
    n = len(IMAGES)
    errors = 0

    async def guarded(session, img):
        nonlocal errors
        async with sem:
            try:
                return await ocr_image(session, img, max_dim=max_dim)
            except Exception:
                errors += 1
                return ""

    async with aiohttp.ClientSession(timeout=timeout) as session:
        t0 = time.perf_counter()
        tasks = [guarded(session, img) for img in IMAGES]
        results = await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - t0

    ok = n - errors
    rate = ok / elapsed if elapsed > 0 else 0
    hours_2m = 2_000_000 / rate / 3600 if rate > 0 else 99999
    days_2m = hours_2m / 24
    print(f"  Workers={workers:2d}: {ok}/{n} ok in {elapsed:5.1f}s "
          f"| {rate:.2f} img/s | 2M: {hours_2m:.0f}h ({days_2m:.1f} days)")


async def main():
    timeout = aiohttp.ClientTimeout(total=600)

    # Warm up
    print("Warming up...")
    async with aiohttp.ClientSession(timeout=timeout) as session:
        await ocr_image(session, IMAGES[0], max_dim=1024)
    print("Model warm.\n")

    print("--- Parallel benchmark (max_dim=1024) ---")
    for workers in [1, 2, 4]:
        await bench(workers)

    print("\n--- Parallel benchmark (max_dim=768) ---")
    for workers in [1, 2, 4]:
        await bench(workers, max_dim=768)


asyncio.run(main())
