"""Batch OCR pipeline — flat image stream with double-buffered staging."""

import asyncio
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import aiohttp

from .ocr_engine import ocr_image, check_ollama, check_model, DEFAULT_OLLAMA_URL, DEFAULT_MAX_DIM
from .scanner import iter_all_images, load_progress, save_progress
from .text_formatter import write_result, format_result


def _status(msg: str):
    sys.stdout.write(f"\r\033[K{msg}")
    sys.stdout.flush()


def _log(msg: str):
    sys.stdout.write(f"\r\033[K{msg}\n")
    sys.stdout.flush()


class _Counter:
    def __init__(self, lifetime_total: int = 0):
        self.done = 0
        self.skipped = 0
        self.lifetime_base = lifetime_total
        self.errors = 0
        self.start = time.perf_counter()
        self.buffer_count = 0

    @property
    def lifetime(self):
        return self.lifetime_base + self.done

    def _elapsed_str(self):
        s = int(time.perf_counter() - self.start)
        h, s = divmod(s, 3600)
        m, s = divmod(s, 60)
        if h:
            return f"{h}h{m:02d}m"
        return f"{m}m{s:02d}s"

    def show(self, activity: str):
        rate = (time.perf_counter() - self.start) / self.done if self.done else 0
        err = f" ({self.errors} err)" if self.errors else ""
        _status(f"  {self.lifetime:,} done{err} | buf: {self.buffer_count} | {rate:.1f}s/img | {self._elapsed_str()} | {activity}")

    def skip(self):
        self.skipped += 1

    def tick(self, name: str, error: bool = False):
        self.done += 1
        self.buffer_count = max(0, self.buffer_count - 1)
        if error:
            self.errors += 1
        self.show(name)


async def _process_one(
    sem: asyncio.Semaphore,
    session: aiohttp.ClientSession,
    image_path: Path,
    original_path: Path,
    model: str,
    model_tag: str,
    ollama_url: str,
    max_dim: int,
    counter: _Counter,
    errors: list,
    results: list,
    stage_dir: Path | None = None,
):
    async with sem:
        try:
            text = await ocr_image(session, image_path, model, ollama_url, max_dim)
            if stage_dir:
                # Write txt locally first, queue for async NAS copy
                local_txt = stage_dir / f"{original_path.stem}.txt"
                block = format_result(text, model_tag)
                local_txt.write_text(block, encoding="utf-8")
                results.append((local_txt, original_path, block))
            else:
                write_result(original_path, text, model_tag)
            counter.tick(original_path.name)
        except Exception as exc:
            errors.append((original_path, str(exc)))
            counter.tick(original_path.name, error=True)


def _flush_results_to_nas(results: list[tuple[Path, Path, str]], counter: _Counter):
    """Write OCR results from local SSD to NAS in bulk. Runs in a thread."""
    for local_txt, original_path, block in results:
        try:
            nas_txt = original_path.with_suffix(".txt")
            if nas_txt.exists():
                existing = nas_txt.read_text(encoding="utf-8")
                if existing and not existing.endswith("\n\n"):
                    block = "\n" + block
                nas_txt.write_text(existing + block, encoding="utf-8")
            else:
                nas_txt.write_text(block, encoding="utf-8")
        except OSError:
            pass
        finally:
            local_txt.unlink(missing_ok=True)
    counter.show(f"flushed {len(results)} results to NAS")


STAGE_MAP_FILE = "stage_map.json"


def _save_stage_map(stage_dir: Path, all_pairs: list[list[tuple[Path, Path]]]):
    """Save staged->original mapping so we can resume after interruption."""
    mapping = {}
    for pairs in all_pairs:
        for staged, original in pairs:
            mapping[str(staged)] = str(original)
    (stage_dir / STAGE_MAP_FILE).write_text(
        json.dumps(mapping), encoding="utf-8"
    )


def _load_stage_map(stage_dir: Path) -> list[tuple[Path, Path]]:
    """Load staged->original mapping from a previous interrupted run."""
    mf = stage_dir / STAGE_MAP_FILE
    if not mf.exists():
        return []
    try:
        mapping = json.loads(mf.read_text(encoding="utf-8"))
        pairs = []
        for staged_str, original_str in mapping.items():
            staged = Path(staged_str)
            if staged.exists():
                pairs.append((staged, Path(original_str)))
        return pairs
    except (json.JSONDecodeError, OSError):
        return []


def _clear_stage_map(stage_dir: Path):
    (stage_dir / STAGE_MAP_FILE).unlink(missing_ok=True)


def _stage_files(files: list[Path], dest_dir: Path, counter: _Counter, label: str = "") -> list[tuple[Path, Path]]:
    pairs = []
    total = len(files)
    for i, img in enumerate(files, 1):
        staged_path = dest_dir / f"{i:05d}_{img.parent.name}_{img.name}"
        shutil.copy2(img, staged_path)
        pairs.append((staged_path, img))
        counter.show(f"staging{label} {i}/{total}")
    counter.buffer_count += total
    return pairs


def _cleanup_files(pairs: list[tuple[Path, Path]], counter: _Counter):
    for staged, _ in pairs:
        try:
            staged.unlink(missing_ok=True)
        except PermissionError:
            pass  # file still locked, will be cleaned on next startup


async def run_pipeline(
    input_path: Path,
    model_tag: str = "glm-ocr",
    model: str = "ocr-api",
    workers: int = 4,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    stage_dir: Path | None = None,
    batch_size: int = 500,
    max_dim: int = DEFAULT_MAX_DIM,
    extensions: set[str] | None = None,
    progress_dir: Path | None = None,
) -> list[tuple[Path, str]]:
    if not await check_ollama(ollama_url):
        print("ERROR: Ollama is not running. Start it with: ollama serve")
        return [()]

    if not await check_model(model, ollama_url):
        print(f"ERROR: Model '{model}' not found in Ollama.")
        return [()]

    if progress_dir is None:
        progress_dir = Path(__file__).parent.parent
    progress = load_progress(progress_dir)
    completed_folders = set(progress.get("completed_folders", []))
    lifetime_total = progress.get("lifetime_total", 0)

    counter = _Counter(lifetime_total)
    all_errors: list[tuple[Path, str]] = []
    sem = asyncio.Semaphore(workers)
    timeout = aiohttp.ClientTimeout(total=300)
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=3) if stage_dir else None

    resume_pairs = []
    if stage_dir:
        stage_dir = Path(stage_dir)
        stage_dir.mkdir(parents=True, exist_ok=True)
        # Check for leftover staged files from interrupted run
        resume_pairs = _load_stage_map(stage_dir)
        if resume_pairs:
            _log(f"Resuming {len(resume_pairs)} staged files from previous run...")
            counter.buffer_count = len(resume_pairs)
        else:
            # No valid mapping — clean any orphaned files
            for f in stage_dir.iterdir():
                if f.name != STAGE_MAP_FILE:
                    try:
                        f.unlink(missing_ok=True)
                    except PermissionError:
                        pass
        _clear_stage_map(stage_dir)

    # Flat image stream across all folders
    image_stream = iter_all_images(input_path, completed_folders, extensions)

    def _already_done(img: Path) -> bool:
        txt = img.with_suffix(".txt")
        if not txt.exists():
            return False
        try:
            return f"[{model_tag}]" in txt.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return False

    def collect_batch():
        """Collect up to batch_size unprocessed images from the stream."""
        batch = []
        meta = []  # (folder_key, is_last_in_folder)
        for img, folder_key, is_last in image_stream:
            if _already_done(img):
                counter.skip()
                # Still track folder completion for skipped last-in-folder
                if is_last:
                    meta.append((folder_key, True))
                continue
            batch.append(img)
            meta.append((folder_key, is_last))
            if len(batch) >= batch_size:
                break
        return batch, meta

    async with aiohttp.ClientSession(timeout=timeout) as session:
        if stage_dir and executor:
            staged_queue = []

            # Resume leftover staged files from previous run
            if resume_pairs:
                staged_queue.append((resume_pairs, []))

            # Fill up to 2 batches in the queue
            while len(staged_queue) < 2:
                b, m = collect_batch()
                if not b:
                    break
                pairs = await loop.run_in_executor(
                    executor, _stage_files, b, stage_dir, counter,
                    f" [{len(b)}]",
                )
                staged_queue.append((pairs, m))

            if not staged_queue:
                _log("No images to process.")
                return []

            # Save stage map so we can resume if interrupted
            _save_stage_map(stage_dir, [p for p, _ in staged_queue])

            # Process loop
            while staged_queue:
                pairs, meta = staged_queue.pop(0)
                errors: list[tuple[Path, str]] = []
                results: list[tuple[Path, Path, str]] = []

                # Start fetching next batch while we process
                fetch_fut = None
                next_batch, next_meta = collect_batch()
                if next_batch:
                    fetch_fut = loop.run_in_executor(
                        executor, _stage_files, next_batch, stage_dir, counter,
                        f" [{len(next_batch)}]",
                    )

                # Process current batch — writes txt to local SSD
                tasks = [
                    _process_one(
                        sem, session, staged, original,
                        model, model_tag, ollama_url, max_dim, counter, errors,
                        results, stage_dir,
                    )
                    for staged, original in pairs
                ]
                await asyncio.gather(*tasks)

                # Flush results to NAS + cleanup staged images in background
                loop.run_in_executor(executor, _flush_results_to_nas, results, counter)
                loop.run_in_executor(executor, _cleanup_files, pairs, counter)

                # Mark completed folders
                for folder_key, is_last in meta:
                    if is_last:
                        completed_folders.add(folder_key)

                all_errors.extend(errors)

                # Save progress + update stage map
                progress["completed_folders"] = list(completed_folders)
                progress["lifetime_total"] = counter.lifetime
                save_progress(progress_dir, progress)

                # Queue the next staged batch
                if fetch_fut:
                    next_pairs = await fetch_fut
                    staged_queue.append((next_pairs, next_meta))

                # Update stage map with current queue
                _save_stage_map(stage_dir, [p for p, _ in staged_queue])

        else:
            # No staging — process directly from NAS
            while True:
                batch, meta = collect_batch()
                if not batch:
                    break
                errors: list[tuple[Path, str]] = []
                results: list[tuple[Path, Path, str]] = []
                tasks = [
                    _process_one(
                        sem, session, img, img,
                        model, model_tag, ollama_url, max_dim, counter, errors,
                        results,
                    )
                    for img in batch
                ]
                await asyncio.gather(*tasks)

                for folder_key, is_last in meta:
                    if is_last:
                        completed_folders.add(folder_key)

                all_errors.extend(errors)
                progress["completed_folders"] = list(completed_folders)
                progress["lifetime_total"] = counter.lifetime
                save_progress(progress_dir, progress)

    if executor:
        executor.shutdown(wait=True)

    elapsed = time.perf_counter() - counter.start
    rate = counter.done / elapsed if elapsed > 0 else 0
    _log("")
    _log(f"Done: {counter.done} processed, {counter.errors} errors in {elapsed:.1f}s ({rate:.2f} img/s)")

    if all_errors:
        _log(f"\n{len(all_errors)} errors:")
        for path, err in all_errors[:20]:
            _log(f"  {path.name}: {err}")
        if len(all_errors) > 20:
            _log(f"  ... and {len(all_errors) - 20} more")

    return all_errors
