"""Benchmark GLM-OCR via Ollama vs DeepSeek-OCR-2 results."""
import asyncio
import base64
import time
import aiohttp

OLLAMA_URL = "http://localhost:11434"
IMAGE_PATH = "test_images/clear_comic.png"

GROUND_TRUTH = [
    "Hello World!",
    "Testing OCR now",
    "Meanwhile, across town",
    "I can read this!",
    "Panel two bubble",
    "BOOM",
    "The end of the comic page",
    "Credits: Artist Name, Writer Name",
]


async def ocr_once(session, model, prompt):
    with open(IMAGE_PATH, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,
    }
    async with session.post(f"{OLLAMA_URL}/api/generate", json=payload) as resp:
        data = await resp.json()
        return data.get("response", "").strip()


async def main():
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Warm up (loads model into GPU memory)
        print("--- Warming up GLM-OCR (ocr-api) ---")
        t0 = time.perf_counter()
        result = await ocr_once(session, "ocr-api", "Extract all text from this image.")
        warmup = time.perf_counter() - t0
        print(f"Warm-up: {warmup:.2f}s")
        print(f"Output preview: {result[:100]}...\n")

        # Speed benchmark: 10 runs
        print("--- Speed benchmark (10 runs) ---")
        times = []
        for i in range(10):
            t0 = time.perf_counter()
            result = await ocr_once(session, "ocr-api", "Extract all text from this image.")
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            print(f"  Run {i+1}: {elapsed:.2f}s")

        avg = sum(times) / len(times)
        print(f"\n  Average: {avg:.2f}s/image")
        print(f"  Throughput: {1/avg:.2f} img/s")
        print(f"  Projected for 2M images: {2_000_000 * avg / 3600:.0f} hours")

        # Accuracy check
        print("\n--- Accuracy check ---")
        result = await ocr_once(session, "ocr-api", "Extract all text from this image. Output only the text, nothing else.")
        print(f"OCR output:\n{result}\n")

        found = 0
        missed = []
        for phrase in GROUND_TRUTH:
            if phrase.lower() in result.lower():
                found += 1
            else:
                missed.append(phrase)

        print(f"Phrases found: {found}/{len(GROUND_TRUTH)} ({100*found/len(GROUND_TRUTH):.0f}%)")
        if missed:
            print(f"Missed: {missed}")

asyncio.run(main())
