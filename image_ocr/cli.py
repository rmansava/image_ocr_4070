"""CLI entry point for image_ocr."""

import argparse
import asyncio
import sys
from pathlib import Path

DEFAULT_MODEL_TAG = "glm-ocr"
DEFAULT_OLLAMA_MODEL = "ocr-api"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="image_ocr",
        description="Batch OCR for comics, album covers, and more using GLM-OCR via Ollama.",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Image file or directory to process recursively.",
    )
    parser.add_argument(
        "--model-tag",
        default=DEFAULT_MODEL_TAG,
        help=f"Tag written to output files (default: {DEFAULT_MODEL_TAG}).",
    )
    parser.add_argument(
        "--ollama-model",
        default=DEFAULT_OLLAMA_MODEL,
        help=f"Ollama model name (default: {DEFAULT_OLLAMA_MODEL}).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Concurrent Ollama requests (default: 4).",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama API URL (default: http://localhost:11434).",
    )
    parser.add_argument(
        "--stage-dir",
        type=Path,
        default=None,
        help="Local SSD directory to stage images before processing.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Images per staging batch (default: 500).",
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=1280,
        help="Max image dimension (default: 1280).",
    )
    parser.add_argument(
        "--ext",
        nargs="+",
        help="Image extensions to include (e.g. --ext jpg png).",
    )
    parser.add_argument(
        "--reset-progress",
        action="store_true",
        help="Ignore saved progress and start from the beginning.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    input_path = args.input.resolve()
    if not input_path.exists():
        print(f"ERROR: Path does not exist: {input_path}")
        return 1

    extensions = None
    if args.ext:
        extensions = {f".{e.lstrip('.')}" for e in args.ext}

    progress_dir = Path(__file__).parent.parent

    if args.reset_progress:
        pf = progress_dir / "ocr_progress.json"
        if pf.exists():
            pf.unlink()
            print("Progress reset.")

    from .pipeline import run_pipeline

    print(f"OCR: {input_path}")
    print(f"Model: {args.ollama_model} [{args.model_tag}] | Workers: {args.workers} | Max dim: {args.max_dim}")
    if args.stage_dir:
        print(f"Staging: {args.stage_dir} (batch {args.batch_size}, 2-ahead buffer)")
    print("Progress saved after each batch. Ctrl+C to stop, re-run to resume.\n")

    errors = asyncio.run(
        run_pipeline(
            input_path=input_path,
            model_tag=args.model_tag,
            model=args.ollama_model,
            workers=args.workers,
            ollama_url=args.ollama_url,
            stage_dir=args.stage_dir,
            batch_size=args.batch_size,
            max_dim=args.max_dim,
            extensions=extensions,
            progress_dir=progress_dir,
        )
    )

    return 1 if errors else 0
