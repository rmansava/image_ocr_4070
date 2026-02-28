"""Benchmark DeepSeek-OCR-2 speed and accuracy on the test image."""
import transformers.integrations.fsdp as _fsdp_module
_fsdp_module.is_fsdp_managed_module = lambda *args, **kwargs: False

import time
from pathlib import Path
from image_ocr.ocr_engine import OCREngine

# Known ground truth for clear_comic.png
GROUND_TRUTH = [
    "Hello World!",
    "Testing OCR now",
    "Meanwhile, across town...",
    "I can read this!",
    "Panel two bubble",
    "BOOM!",
    "The end of the comic page.",
    "Credits: Artist Name, Writer Name",
]

engine = OCREngine()

image = Path("test_images/clear_comic.png")

# Warm up (first inference is slower due to GPU kernel compilation)
print("\n--- Warm-up run ---")
t0 = time.perf_counter()
result = engine.ocr_image(image, mode="comic")
warmup_time = time.perf_counter() - t0
print(f"Warm-up: {warmup_time:.2f}s")

# Benchmark: 5 runs
print("\n--- Speed benchmark (5 runs, comic mode) ---")
times = []
for i in range(5):
    t0 = time.perf_counter()
    result = engine.ocr_image(image, mode="comic")
    elapsed = time.perf_counter() - t0
    times.append(elapsed)
    print(f"  Run {i+1}: {elapsed:.2f}s")

avg = sum(times) / len(times)
print(f"\n  Average: {avg:.2f}s/image")
print(f"  Throughput: {1/avg:.2f} img/s")
print(f"  Projected for 2M images: {2_000_000 * avg / 3600:.0f} hours")

# General mode (faster, no grounding boxes)
print("\n--- Speed benchmark (5 runs, general mode) ---")
times_gen = []
for i in range(5):
    t0 = time.perf_counter()
    result_gen = engine.ocr_image(image, mode="general")
    elapsed = time.perf_counter() - t0
    times_gen.append(elapsed)
    print(f"  Run {i+1}: {elapsed:.2f}s")

avg_gen = sum(times_gen) / len(times_gen)
print(f"\n  Average: {avg_gen:.2f}s/image")
print(f"  Throughput: {1/avg_gen:.2f} img/s")
print(f"  Projected for 2M images: {2_000_000 * avg_gen / 3600:.0f} hours")

# Accuracy check
print("\n--- Accuracy check (comic mode) ---")
result = engine.ocr_image(image, mode="comic")
print(f"OCR output:\n{result}\n")

found = 0
missed = []
for phrase in GROUND_TRUTH:
    clean_phrase = phrase.replace("...", "")
    if clean_phrase.lower().rstrip(".") in result.lower():
        found += 1
    else:
        missed.append(phrase)

print(f"Phrases found: {found}/{len(GROUND_TRUTH)} ({100*found/len(GROUND_TRUTH):.0f}%)")
if missed:
    print(f"Missed: {missed}")
