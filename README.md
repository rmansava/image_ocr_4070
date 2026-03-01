# image_ocr

Batch OCR and dense captioning pipeline for large image collections. Processes images using Qwen3-VL-8B-Instruct via HuggingFace transformers with SDPA Flash attention, writing tagged `.txt` results to a mirrored archive directory.

Built for an RTX 4070 Ti SUPER 16GB processing ~2.4M images across albums, comics, print ads, and more.

## How it works

1. **Scan** — Walks the input directory, indexes images into MS SQL Server. Detects which images are new (no `.txt`) and which need updating (`.txt` exists but is missing the current model's tag).
2. **Process** — 3-thread rolling buffer pipeline:
   - **Prefetch thread**: Copies images from NAS to local SSD buffer (stays 5 batches ahead)
   - **GPU thread**: Runs inference (never waits on disk I/O)
   - **Flush thread**: Writes `.txt` results to `T:\archive` while GPU works on next batch
3. **Resume** — Progress is saved to SQL after each batch. Ctrl+C to stop, re-run to resume.

## Output format

Results are written as `.txt` files to a mirrored path under `T:\archive`. Multiple models can append to the same file using tagged blocks:

```
[qwen3-vl-8b]
OCR: ABBEY ROAD / THE BEATLES
DESCRIPTION: four men walking across a zebra crossing on a tree-lined street
---
```

## Path mapping

| Source | Destination |
|---|---|
| `T:\archiverelated\albums\...` | `T:\archive\albums\...` |
| `T:\archiverelated\comics\...` | `T:\archive\comics\...` |
| `T:\archiverelated\books\pdf-images\...` | `T:\archive\book images\...` |

## Usage

```bash
# Process a directory
python -m image_ocr T:\archiverelated\albums

# Use a specific model
python -m image_ocr T:\archiverelated\comics --model qwen3-vl-4b

# Custom model tag (written to txt files)
python -m image_ocr T:\archiverelated\albums --model-tag my-run-1

# 4-bit quantization (lower VRAM)
python -m image_ocr T:\archiverelated\albums --quantize 4bit

# Force re-scan of the directory
python -m image_ocr T:\archiverelated\albums --rescan

# List available models
python -m image_ocr --list-models
```

Or use the batch launcher which auto-restarts on errors:

```
run_ocr.bat
```

## Available models

| Name | HuggingFace ID |
|---|---|
| `qwen3-vl-8b` (default) | `Qwen/Qwen3-VL-8B-Instruct` |
| `qwen3-vl-4b` | `Qwen/Qwen3-VL-4B-Instruct` |
| `qwen2.5-vl-7b` | `Qwen/Qwen2.5-VL-7B-Instruct` |

## CLI options

```
python -m image_ocr [input] [options]

positional:
  input                 Image file or directory to process

options:
  --model MODEL         Model short name (default: qwen3-vl-8b)
  --model-tag TAG       Tag written to output files (defaults to model name)
  --prompt PROMPT       Prompt sent to the model
  --dtype {bf16,fp16,fp32}  Torch dtype (default: bf16)
  --quantize {4bit}     Quantization (requires bitsandbytes)
  --no-compile          Disable torch.compile
  --buffer-dir PATH     Local SSD directory for rolling image buffer
  --batch-size N        Images per batch (default: 500)
  --max-dim N           Max image dimension (default: 1280)
  --ext EXT [EXT ...]   Image extensions to include (e.g. --ext jpg png)
  --rescan              Force a fresh scan of the input directory
  --list-models         List available models and exit
```

## Requirements

- Python 3.12+
- CUDA GPU (tested on RTX 4070 Ti SUPER 16GB)
- MS SQL Server (connection: `RMDESK/Trivia`, Windows auth)
- ODBC Driver 17 for SQL Server

```bash
pip install -r requirements.txt
```

## Project structure

```
image_ocr/
  __main__.py       # python -m image_ocr entry point
  cli.py            # Argument parsing and main()
  pipeline.py       # 3-thread prefetch/GPU/flush pipeline
  hf_engine.py      # Vision model loading and inference (SDPA)
  scan_db.py        # MS SQL image index (pyodbc)
  scanner.py        # Path mapping (archiverelated -> archive)
  text_formatter.py # Tagged text block formatting
```

## Database

Two tables in `RMDESK.Trivia`:

- **`ocr_images`** — One row per image, keyed by `(image_path, input_root)`. Tracks `pass_num` (1=new, 2=update), `processed` flag, and errors.
- **`ocr_scan_meta`** — Key-value metadata per input root (model tag, scan time, etc.).

Multiple input roots (albums, comics, print ads) coexist in the same tables.
