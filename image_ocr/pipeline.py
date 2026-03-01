"""Batch OCR pipeline — Qwen3-VL via HuggingFace transformers with SDPA Flash.

3-thread async rolling buffer pipeline:
  - Prefetch thread: copies images from T: to local SSD (stays 5 batches ahead)
  - GPU thread: runs inference on buffered images (never waits on disk I/O)
  - Flush thread: writes .txt results to T:\\archive while GPU works on next batch
"""

import shutil
import sys
import threading
import time
from pathlib import Path
from queue import Queue

from .scan_db import (
    init_db, scan_to_db, iter_unprocessed, mark_processed,
    get_stats, get_scan_meta, _input_root_key,
)

DEFAULT_PROMPT = (
    "OCR: <extract all text and brands>. "
    "DESCRIPTION: <Describe subjects and their actions or spatial relationships. "
    "Example: '3 cows on top of a hill', 'dog riding a sled', or "
    "'bottle of coke sitting on a table'. No colors, no styles, no flowery language.>"
)
DEFAULT_BUFFER_DIR = Path(__file__).parent.parent / ".buffer"
DEFAULT_MAX_DIM = 1280
PREFETCH_DEPTH = 5  # batches to keep buffered ahead of GPU


def _status(msg: str):
    sys.stdout.write(f"\r\033[K{msg}")
    sys.stdout.flush()


def _log(msg: str):
    sys.stdout.write(f"\r\033[K{msg}\n")
    sys.stdout.flush()


class _Counter:
    def __init__(self, pass1_total: int = 0, pass2_total: int = 0):
        self.done = 0
        self.pass1_done = 0
        self.pass2_done = 0
        self.pass1_total = pass1_total
        self.pass2_total = pass2_total
        self.errors = 0
        self.current_pass = 1
        self.start = time.perf_counter()

    @property
    def total(self):
        return self.pass1_total + self.pass2_total

    def _elapsed_str(self):
        s = int(time.perf_counter() - self.start)
        h, s = divmod(s, 3600)
        m, s = divmod(s, 60)
        if h:
            return f"{h}h{m:02d}m"
        return f"{m}m{s:02d}s"

    def _eta_str(self):
        if self.done == 0:
            return "?"
        rate = (time.perf_counter() - self.start) / self.done
        remaining = self.total - self.done
        secs = int(rate * remaining)
        h, secs = divmod(secs, 3600)
        m, secs = divmod(secs, 60)
        if h:
            return f"{h}h{m:02d}m"
        return f"{m}m{secs:02d}s"

    def show(self, activity: str):
        rate = (time.perf_counter() - self.start) / self.done if self.done else 0
        err = f" ({self.errors} err)" if self.errors else ""
        p1 = f"new {self.pass1_done:,}/{self.pass1_total:,}"
        p2 = f"upd {self.pass2_done:,}/{self.pass2_total:,}"
        pct = f"{self.done / self.total * 100:.1f}%" if self.total else "0%"
        _status(
            f"  {self.done:,}/{self.total:,} [{pct}] | {p1} | {p2}{err} "
            f"| {rate:.1f}s/img | {self._elapsed_str()} eta {self._eta_str()} | {activity}"
        )

    def tick(self, name: str, error: bool = False):
        self.done += 1
        if self.current_pass == 1:
            self.pass1_done += 1
        else:
            self.pass2_done += 1
        if error:
            self.errors += 1
        self.show(name)


# ─── Local buffer ──────────────────────────────────────────────

def _copy_to_buffer(
    batch: list[Path], buffer_dir: Path,
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, str]]]:
    """Copy batch images to local SSD buffer.

    Returns (pairs, copy_errors) where pairs is list of (local_path, original_path)
    and copy_errors is list of (original_path, error_message) for unreadable files.
    """
    buffer_dir.mkdir(parents=True, exist_ok=True)
    pairs = []
    copy_errors = []
    for i, img in enumerate(batch):
        local = buffer_dir / f"{i:06d}_{img.name}"
        try:
            shutil.copy2(img, local)
            pairs.append((local, img))
        except OSError as exc:
            copy_errors.append((img, str(exc)))
    return pairs, copy_errors


def _clear_buffer(buffer_dir: Path):
    """Delete all files in the buffer directory."""
    if not buffer_dir.exists():
        return
    for f in buffer_dir.iterdir():
        if f.is_file():
            try:
                f.unlink()
            except OSError:
                pass
    try:
        buffer_dir.rmdir()
    except OSError:
        pass


def _flush_results(results: list[tuple[Path, str]], model_tag: str, batch_dir: Path):
    """Write results to local SSD first, then copy .txt to T:\\archive.

    Image is in T:\\archiverelated\\albums\\Artist\\cover.jpg
    Txt goes to  T:\\archive\\albums\\Artist\\cover.txt
    """
    from .text_formatter import format_result
    from .scanner import txt_path_for

    for idx, (original_img, text) in enumerate(results):
        archive_txt = txt_path_for(original_img)
        block = format_result(text, model_tag)

        existing = ""
        if archive_txt.exists():
            try:
                existing = archive_txt.read_text(encoding="utf-8")
            except OSError:
                pass

        if f"[{model_tag}]" in existing:
            continue

        if existing and not existing.endswith("\n\n"):
            block = "\n" + block
        content = existing + block

        # Prefix with per-batch index to avoid stem collisions inside a batch.
        local_txt = batch_dir / f"{idx:06d}_{original_img.stem}.txt"
        local_txt.write_text(content, encoding="utf-8")

        archive_txt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_txt, archive_txt)


# ─── Async prefetch / flush ───────────────────────────────────

def _prefetch_worker(input_root, pass_num, batch_size, buffer_dir, prefetch_q):
    """Background thread: reads batches from SQL index, copies images to local SSD.

    Stays PREFETCH_DEPTH batches ahead of the GPU. Blocks when queue is full.
    Sends None when no more images. Copy errors for individual files are
    forwarded as part of the batch so the GPU thread can mark them in the DB.
    """
    batch_num = 0
    try:
        for batch in iter_unprocessed(input_root, pass_num, batch_size):
            image_paths = [img for img, _ in batch]
            batch_dir = buffer_dir / f"b{batch_num:04d}"
            pairs, copy_errors = _copy_to_buffer(image_paths, batch_dir)

            # Build lookup from original path to archive_txt
            archive_map = {img: atxt for img, atxt in batch}

            enriched = [
                (local, original, archive_map[original])
                for local, original in pairs
            ]
            if copy_errors:
                _log(f"  Prefetch batch {batch_num}: {len(copy_errors)} copy errors (skipped)")

            _log(f"  Prefetched batch {batch_num} ({len(enriched)} images) -> {batch_dir.name}")
            prefetch_q.put((enriched, batch_dir, copy_errors))
            batch_num += 1
    except Exception as exc:
        _log(f"  Prefetch error: {exc}")
    finally:
        prefetch_q.put(None)


def _flush_worker(flush_q, input_root):
    """Background thread: writes .txt locally then copies to T:\\archive, cleans buffer.

    Marks images as processed in SQL. Runs while GPU works on next batch.
    """
    while True:
        item = flush_q.get()
        if item is None:
            break
        results, errors, model_tag, batch_dir = item
        try:
            _flush_results(results, model_tag, batch_dir)
            if results:
                mark_processed(input_root, [img for img, _ in results])
            for img, err in errors:
                mark_processed(input_root, [img], error=err)
        except Exception as exc:
            _log(f"  Flush error: {exc}")
            # Mark all as errored so they don't block reruns
            for img, _ in results:
                try:
                    mark_processed(input_root, [img], error=f"flush: {exc}")
                except Exception:
                    pass
            for img, _ in errors:
                try:
                    mark_processed(input_root, [img], error=str(exc))
                except Exception:
                    pass
        finally:
            _clear_buffer(batch_dir)
            flush_q.task_done()


# ─── GPU processing ──────────────────────────────────────────────

def _run_pass(
    input_root, model_tag, engine, batch_size, prompt, buffer_dir, counter, pass_num,
):
    """Run one pass of the pipeline from the SQL index."""
    all_errors = []

    prefetch_q = Queue(maxsize=PREFETCH_DEPTH)
    flush_q = Queue()

    prefetch_t = threading.Thread(
        target=_prefetch_worker,
        args=(input_root, pass_num, batch_size, buffer_dir, prefetch_q),
        daemon=True,
    )
    flush_t = threading.Thread(target=_flush_worker, args=(flush_q, input_root), daemon=True)
    prefetch_t.start()
    flush_t.start()

    while True:
        item = prefetch_q.get()
        if item is None:
            break
        enriched, batch_dir, copy_errors = item

        results = []
        errors = []

        # Record copy errors from prefetch
        for img, err in copy_errors:
            errors.append((img, f"copy: {err}"))
            counter.tick(img.name, error=True)

        for local, original, _archive_txt in enriched:
            try:
                text = engine.infer(local, prompt)
                results.append((original, text))
                counter.tick(original.name)
            except Exception as exc:
                errors.append((original, str(exc)))
                counter.tick(original.name, error=True)

        all_errors.extend(errors)
        flush_q.put((results, errors, model_tag, batch_dir))

    flush_q.put(None)
    flush_t.join()
    prefetch_t.join()

    return all_errors


# ─── Main entry point ───────────────────────────────────────────

def run_pipeline(
    input_path: Path,
    model_tag: str,
    model: str,
    batch_size: int = 500,
    max_dim: int = DEFAULT_MAX_DIM,
    extensions: set[str] | None = None,
    prompt: str = DEFAULT_PROMPT,
    dtype: str = "bf16",
    quantize: str | None = None,
    compile_model: bool = True,
    buffer_dir: Path | None = None,
    rescan: bool = False,
) -> list[tuple[Path, str]]:
    if buffer_dir is None:
        buffer_dir = DEFAULT_BUFFER_DIR
    buffer_dir.mkdir(parents=True, exist_ok=True)

    # Clean leftover buffer dirs from previous crash
    for d in buffer_dir.iterdir():
        if d.is_dir():
            _clear_buffer(d)

    # ── Step 1: Scan and index ──
    input_root = _input_root_key(input_path)
    init_db()

    prev_tag = get_scan_meta(input_root, "model_tag")

    needs_scan = (
        rescan
        or prev_tag != model_tag
        or get_scan_meta(input_root, "scan_time") is None
    )

    if needs_scan:
        _log(f"Scanning {input_path} for images...")
        scan_to_db(input_path, model_tag, extensions, _log)
    else:
        stats = get_stats(input_root)
        _log(f"Using existing scan index ({stats[1]['remaining']:,} new, {stats[2]['remaining']:,} updates remaining)")

    stats = get_stats(input_root)
    total_work = stats[1]["remaining"] + stats[2]["remaining"]
    _log(f"Input root: {input_root}")
    _log(f"Local buffer: {buffer_dir} (prefetch depth: {PREFETCH_DEPTH})")
    _log(f"Work: {stats[1]['remaining']:,} new + {stats[2]['remaining']:,} updates = {total_work:,} total")
    _log("")

    if total_work == 0:
        _log("Nothing to process — all images up to date.")
        return []

    # ── Step 2: Load model ──
    from .hf_engine import HFVisionEngine

    engine = HFVisionEngine(
        model_name=model, dtype=dtype, quantize=quantize,
        max_dim=max_dim, compile_model=compile_model,
    )
    engine.load()

    # ── Step 3: Process ──
    counter = _Counter(pass1_total=stats[1]["remaining"], pass2_total=stats[2]["remaining"])
    all_errors = []

    if stats[1]["remaining"] > 0:
        counter.current_pass = 1
        _log(f"── Pass 1: new images ({stats[1]['remaining']:,} to process) ──")
        errors1 = _run_pass(
            input_root, model_tag, engine, batch_size, prompt, buffer_dir, counter, pass_num=1,
        )
        all_errors.extend(errors1)
    else:
        _log("── Pass 1: no new images ──")

    if stats[2]["remaining"] > 0:
        counter.current_pass = 2
        _log(f"\n── Pass 2: updating existing ({stats[2]['remaining']:,} to process) ──")
        errors2 = _run_pass(
            input_root, model_tag, engine, batch_size, prompt, buffer_dir, counter, pass_num=2,
        )
        all_errors.extend(errors2)
    else:
        _log("── Pass 2: no updates needed ──")

    if not counter.done:
        _log("No images to process.")

    engine.unload()

    # Final cleanup
    for d in buffer_dir.iterdir():
        if d.is_dir():
            _clear_buffer(d)

    # Final stats from DB
    final_stats = get_stats(input_root)
    elapsed = time.perf_counter() - counter.start
    rate = counter.done / elapsed if elapsed > 0 else 0
    _log("")
    _log(f"Done: {counter.done:,} processed, {counter.errors} errors in {elapsed:.1f}s ({rate:.2f} img/s)")
    _log(f"  Pass 1 (new):    {final_stats[1]['done']:,} / {final_stats[1]['total']:,}")
    _log(f"  Pass 2 (update): {final_stats[2]['done']:,} / {final_stats[2]['total']:,}")

    if all_errors:
        _log(f"\n{len(all_errors)} errors:")
        for path, err in all_errors[:20]:
            _log(f"  {path.name}: {err}")
        if len(all_errors) > 20:
            _log(f"  ... and {len(all_errors) - 20} more")

    return all_errors
