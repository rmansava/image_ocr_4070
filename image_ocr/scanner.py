"""Recursive image discovery with progress tracking."""

import json
import os
from pathlib import Path
from typing import Generator

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}

PROGRESS_FILE = "ocr_progress.json"


def iter_all_images(
    input_path: Path,
    completed_folders: set[str],
    extensions: set[str] | None = None,
) -> Generator[tuple[Path, str, bool], None, None]:
    """Yield (image_path, folder_key, is_last_in_folder) across all folders.

    Skips completed folders entirely. Walks in sorted order for determinism.
    """
    exts = extensions or IMAGE_EXTENSIONS

    if input_path.is_file():
        if input_path.suffix.lower() in exts:
            yield input_path, str(input_path.parent), True
        return

    for root, dirs, files in os.walk(input_path):
        dirs.sort()
        folder = Path(root)
        folder_key = str(folder)
        if folder_key in completed_folders:
            continue
        images = sorted(
            folder / f for f in files if Path(f).suffix.lower() in exts
        )
        if not images:
            continue
        for i, img in enumerate(images):
            yield img, folder_key, (i == len(images) - 1)


def load_progress(progress_dir: Path) -> dict:
    """Load progress from local tracking file."""
    pf = progress_dir / PROGRESS_FILE
    if pf.exists():
        try:
            return json.loads(pf.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"completed_folders": [], "lifetime_total": 0}


def save_progress(progress_dir: Path, progress: dict):
    """Save progress to local tracking file."""
    pf = progress_dir / PROGRESS_FILE
    pf.write_text(json.dumps(progress, default=str), encoding="utf-8")


def txt_path_for(image_path: Path) -> Path:
    """Return the .txt output path corresponding to an image."""
    return image_path.with_suffix(".txt")
