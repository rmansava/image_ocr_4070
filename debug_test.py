"""Debug test to get full traceback."""
import sys
import traceback

# Try to mock distributed before anything else
try:
    import torch.distributed
except ImportError:
    import types
    sys.modules["torch.distributed"] = types.ModuleType("torch.distributed")
    sys.modules["torch.distributed"].is_initialized = lambda: False
    sys.modules["torch.distributed"].get_rank = lambda: 0
    sys.modules["torch.distributed"].get_world_size = lambda: 1

from image_ocr.ocr_engine import OCREngine
from pathlib import Path

engine = OCREngine()

try:
    result = engine.ocr_image(Path("test_images/comic_test.png"), mode="comic")
    print(f"SUCCESS: {result[:200]}")
except Exception:
    traceback.print_exc()
