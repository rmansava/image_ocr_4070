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
  --buffer-dir PATH     Local SSD directory for rolling image buffer (default: C:\ocrbuffer\<category>)
  --batch-size N        Images per batch (default: 50)
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

## Architecture

### Pipeline threading model

```
  Prefetch thread          GPU thread            Flush thread
  ──────────────          ──────────            ────────────
  SQL index query   ──>   prefetch_q  ──>       flush_q  ──>
  Copy to SSD buffer      (bounded,5)           (bounded,5)
                          Run inference          Write .txt to T:\archive
                                                 Mark processed in SQL
                                                 Clean buffer dir
```

- All three threads run concurrently. Prefetch stays up to 20 batches (1,000 images) ahead of GPU.
- Queues are bounded (`maxsize=20`) for backpressure — prefetch blocks when GPU falls behind.
- Each thread propagates errors to a shared list checked by the main thread at exit.

### Error handling design

- **Per-file resilience**: A bad image (corrupt, unreadable, inference failure) is logged and skipped. It does not halt the batch or the pipeline.
- **Circuit breaker**: 10 consecutive inference errors abort the current pass immediately. Prevents OOM cascades from silently marking millions of images as permanently errored.
- **Transient vs permanent errors**: Copy errors (NAS I/O) and flush I/O errors are transient — images are NOT marked processed and will be retried next run. Inference errors (corrupt image, model failure) are permanent — marked processed with error string.
- **VRAM check**: After model load, warns if <2 GB VRAM free and suggests `--quantize 4bit` or `--model qwen3-vl-4b`.
- **Prefetch fatal errors** (DB down, network failure): Logged, propagated to main thread via `prefetch_errors` list, pipeline exits non-zero.
- **Flush errors**: Caught per-batch. I/O errors are transient (retry next run). Other errors mark images as permanently errored.
- **Atomic writes**: Archive `.txt` files are written to `.txt.tmp` first, then `os.replace()` to the final path. No partial writes on crash.
- **Crash-safe rescan**: `scan_time` metadata is cleared before DELETE and only set after the full scan succeeds. An interrupted `--rescan` always re-triggers on the next run.

### Concurrency assumptions

- **Single process per category**. No file locking on archive `.txt` writes. Running two processes on the same input root simultaneously will race on read-modify-write of `.txt` files.
- Different categories (albums, comics, etc.) can run concurrently — they use separate SQL rows and write to different archive paths.

### Skip / resume logic

The `[model-tag]` block inside each `.txt` file is the source of truth for what's been processed:

1. **Scan phase**: For each image, check if `T:\archive\...\image.txt` exists and contains `[model-tag]`. If yes, skip. If no txt exists → pass 1 (new). If txt exists but tag missing → pass 2 (update).
2. **Processing phase**: After inference, the flush thread reads the archive `.txt` again before writing (in case another run completed it). If `[model-tag]` is already present, skip the write.
3. **SQL index**: `processed` flag tracks what's been done in the current run. Re-running with the same model tag and input path reuses the index (no re-walk). Changing model tag or using `--rescan` rebuilds the index.

### Key constants

| Constant | Location | Value | Notes |
|---|---|---|---|
| `DEFAULT_MODEL` | `hf_engine.py` | `qwen3-vl-8b` | Single source of truth, imported by `cli.py` |
| `DEFAULT_PROMPT` | `cli.py` | OCR + description prompt | Single source of truth, imported by `pipeline.py` |
| `DEFAULT_MAX_DIM` | `hf_engine.py` | `1280` | Max image dimension before resize |
| `PREFETCH_DEPTH` | `pipeline.py` | `20` | Batches buffered ahead of GPU |
| `MAX_CONSECUTIVE_ERRORS` | `pipeline.py` | `10` | Circuit breaker threshold |
| `CONN_STR` | `scan_db.py` | `RMDESK/Trivia` | Windows auth, ODBC 17 |
| `SOURCE_PREFIX` | `scanner.py` | `T:\archiverelated` | Source root for path mapping |
| `DEST_PREFIX` | `scanner.py` | `T:\archive` | Destination root for path mapping |

## Project structure

```
image_ocr/
  __main__.py       # python -m image_ocr entry point
  cli.py            # Argument parsing, DEFAULT_PROMPT, main()
  pipeline.py       # 3-thread prefetch/GPU/flush pipeline
  hf_engine.py      # Vision model loading and inference (SDPA), DEFAULT_MODEL
  scan_db.py        # MS SQL image index (pyodbc), connection string
  scanner.py        # Path mapping (archiverelated -> archive), exceptions dict
  text_formatter.py # Tagged text block formatting
run_ocr.bat         # Auto-restart launcher
requirements.txt    # Pillow, pyodbc, torch, transformers, accelerate, qwen-vl-utils
```

## Database

Two tables in `RMDESK.Trivia`:

- **`ocr_images`** — One row per image, keyed by `(image_path, input_root)`. Tracks `pass_num` (1=new, 2=update), `processed` flag, and errors. Indexed on `(input_root, pass_num, processed)` for keyset pagination.
- **`ocr_scan_meta`** — Key-value metadata per input root (model tag, scan time, total scanned, input path). `scan_time` being empty signals an incomplete scan.

Multiple input roots (albums, comics, print ads) coexist in the same tables. Each `scan_to_db` call only DELETEs/INSERTs rows for its own `input_root`.

## Other repo

The Strix Halo version (Ollama backend) lives at [github.com/rmansava/image_ocr](https://github.com/rmansava/image_ocr). This repo is the RTX 4070 Ti SUPER version using HuggingFace transformers + SDPA. Do not cross-push.
