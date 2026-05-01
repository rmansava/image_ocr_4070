"""CLI entry point for image_ocr."""

import argparse
from pathlib import Path

from .hf_engine import DEFAULT_MODEL
DEFAULT_PROMPT = (
    "1. List all visible text and brand names from this image, including fine print.\n"
    "2. Describe the image in detail: all subjects, objects, their positions and spatial "
    "relationships, any actions or interactions, and notable visual elements. "
    "Be specific and complete."
)
VLLM_PROMPT = DEFAULT_PROMPT  # same prompt, vllm engine handles rep_penalty internally


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="image_ocr",
        description="Batch OCR / dense captioning for images using vision models.",
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=None,
        help="Image file or directory to process.",
    )

    # Engine / model selection
    parser.add_argument(
        "--engine",
        choices=["hf", "vllm"],
        default="hf",
        help="Inference engine: 'hf' = HuggingFace (default), 'vllm' = vLLM server (fast batched).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model short name for HF engine (default: {DEFAULT_MODEL}). Ignored when --engine vllm.",
    )
    parser.add_argument(
        "--model-tag",
        default=None,
        help="Tag written to output files. Defaults to model name.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt sent to the model.",
    )

    # Performance
    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="Torch dtype (default: bf16).",
    )
    parser.add_argument(
        "--quantize",
        choices=["4bit"],
        default=None,
        help="Quantization (requires bitsandbytes).",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile (use if you get graph break errors).",
    )
    parser.add_argument(
        "--buffer-dir",
        type=Path,
        default=None,
        help=r"Local SSD directory for rolling image buffer (default: C:\ocrbuffer\<category>).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Images per batch (default: 50).",
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
        "--rescan",
        action="store_true",
        help="Force a fresh scan of the input directory (rebuilds the DB index).",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help=r"Path to log file (default: ocr_scan.log next to this package). Pass empty string to disable.",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable file logging (log to console only).",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.list_models:
        from .hf_engine import list_models, MODEL_REGISTRY
        print("Available models:")
        for name in list_models():
            info = MODEL_REGISTRY[name]
            print(f"  {name:25s} -> {info['hf_id']}")
        return 0

    if args.input is None:
        print("ERROR: input path is required.")
        return 1

    input_path = args.input.resolve()
    if not input_path.exists():
        print(f"ERROR: Path does not exist: {input_path}")
        return 1

    extensions = None
    if args.ext:
        extensions = {f".{e.lstrip('.').lower()}" for e in args.ext}

    engine = args.engine
    model = "vllm" if engine == "vllm" else args.model
    model_tag = args.model_tag or ("qwen3-vl-8b-vllm" if engine == "vllm" else model)
    compile_model = not args.no_compile
    compile_str = "on" if compile_model else "off"
    buffer_dir = args.buffer_dir.resolve() if args.buffer_dir else None

    from .pipeline import run_pipeline, DEFAULT_LOG_FILE

    if args.no_log:
        log_file = None
    elif args.log_file is not None:
        log_file = args.log_file.resolve()
    else:
        log_file = DEFAULT_LOG_FILE

    print(f"Input: {input_path}")
    if engine == "vllm":
        print(f"Engine: vLLM (batched) | Tag: {model_tag}")
    else:
        print(f"Engine: HuggingFace | Model: {model} [{model_tag}]")
        print(f"Dtype: {args.dtype} | Quantize: {args.quantize or 'none'} | Compile: {compile_str}")
    print(f"Prompt: {args.prompt[:80]}{'...' if len(args.prompt) > 80 else ''}")
    if log_file:
        print(f"Log file: {log_file}")
    print("Progress saved after each batch. Ctrl+C to stop, re-run to resume.\n")

    errors = run_pipeline(
        input_path=input_path,
        model_tag=model_tag,
        model=model,
        batch_size=args.batch_size,
        max_dim=args.max_dim,
        extensions=extensions,
        prompt=args.prompt,
        dtype=args.dtype,
        quantize=args.quantize,
        compile_model=compile_model,
        buffer_dir=buffer_dir,
        rescan=args.rescan,
        log_file=log_file,
    )

    return 1 if errors else 0
