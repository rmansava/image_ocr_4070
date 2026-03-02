"""MS SQL index of images and their processing status.

Scan once, persist to DB, process from the index. No re-walking
the NAS on every restart. Multiple input roots (albums, comics, etc.)
coexist in the same table.
"""

import os
import time
from pathlib import Path

import pyodbc

from .scanner import IMAGE_EXTENSIONS, map_to_archive

CONN_STR = (
    "Driver={ODBC Driver 17 for SQL Server};"
    "Server=RMDESK;"
    "Database=Trivia;"
    "Trusted_Connection=Yes;"
)


def _connect() -> pyodbc.Connection:
    return pyodbc.connect(CONN_STR, autocommit=False)


def _input_root_key(input_path: Path) -> str:
    """Derive a short key from the input path (e.g. 'albums', 'comics')."""
    return str(input_path).replace("\\", "/").rstrip("/")


def init_db():
    """Create tables and indexes if they don't exist."""
    conn = _connect()
    cur = conn.cursor()
    cur.execute("""
        IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'ocr_images')
        CREATE TABLE ocr_images (
            image_path   NVARCHAR(450) NOT NULL,
            input_root   NVARCHAR(450) NOT NULL,
            archive_txt  NVARCHAR(900) NOT NULL,
            pass_num     INT NOT NULL,
            processed    BIT DEFAULT 0,
            error        NVARCHAR(MAX),
            CONSTRAINT PK_ocr_images PRIMARY KEY (image_path, input_root)
        )
    """)
    cur.execute("""
        IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_ocr_root_pass_proc')
        CREATE INDEX idx_ocr_root_pass_proc
            ON ocr_images(input_root, pass_num, processed)
    """)
    cur.execute("""
        IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'ocr_scan_meta')
        CREATE TABLE ocr_scan_meta (
            input_root  NVARCHAR(450) NOT NULL,
            [key]       NVARCHAR(100) NOT NULL,
            [value]     NVARCHAR(MAX),
            CONSTRAINT PK_ocr_scan_meta PRIMARY KEY (input_root, [key])
        )
    """)
    conn.commit()
    conn.close()


def get_scan_meta(input_root: str, key: str) -> str | None:
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT [value] FROM ocr_scan_meta WHERE input_root=? AND [key]=?",
        (input_root, key),
    )
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None


def set_scan_meta(input_root: str, key: str, value: str):
    conn = _connect()
    cur = conn.cursor()
    cur.execute("""
        MERGE ocr_scan_meta AS t
        USING (SELECT ? AS input_root, ? AS [key], ? AS [value]) AS s
        ON t.input_root = s.input_root AND t.[key] = s.[key]
        WHEN MATCHED THEN UPDATE SET [value] = s.[value]
        WHEN NOT MATCHED THEN INSERT (input_root, [key], [value])
            VALUES (s.input_root, s.[key], s.[value]);
    """, (input_root, key, value))
    conn.commit()
    conn.close()


SCAN_THREADS = 8  # parallel workers for NAS I/O during scan


def scan_to_db(
    input_path: Path,
    model_tag: str,
    extensions: set[str] | None = None,
    log_fn=None,
    force_full: bool = False,
) -> tuple[int, int]:
    """Walk the input directory and populate the SQL index.

    Only touches rows for this input_root — other roots are untouched.
    If force_full is True (--rescan or model tag changed), deletes all rows
    and re-scans from scratch. Otherwise, resumes incrementally — images
    already in the index are skipped.

    Archive checking is parallelized: instead of one stat() per image, we
    list each archive directory once (one NAS call per directory) and check
    8 directories in parallel via a thread pool.

    Returns (new_count, update_count).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if log_fn is None:
        log_fn = print

    exts = extensions or IMAGE_EXTENSIONS
    input_root = _input_root_key(input_path)

    # Clear scan_time BEFORE any changes. If we crash mid-scan, the next
    # run will see no scan_time and resume the incremental scan.
    set_scan_meta(input_root, "scan_time", "")
    set_scan_meta(input_root, "model_tag", model_tag)

    conn = _connect()
    cur = conn.cursor()
    cur.fast_executemany = True

    existing_paths = set()
    if force_full:
        cur.execute("DELETE FROM ocr_images WHERE input_root=?", (input_root,))
        conn.commit()
        log_fn(f"  Full rescan — cleared previous index")
    else:
        # Incremental: load already-indexed paths so we can skip them
        cur.execute("SELECT image_path FROM ocr_images WHERE input_root=?", (input_root,))
        existing_paths = {row[0] for row in cur.fetchall()}
        if existing_paths:
            log_fn(f"  Resuming scan ({len(existing_paths):,} images already indexed)")

    new_count = 0
    update_count = 0
    scan_count = 0
    skip_count = len(existing_paths)
    batch = []
    start = time.perf_counter()
    last_log = 0

    def _classify_dir(folder, image_names, archive_dir, archive_stems):
        """Classify images in one source directory against the archive.

        archive_stems is a pre-built dict of {stem_lower: Path} for .txt files
        in the archive directory (built from the Phase 2 walk — no NAS I/O here
        for existence checks). Only reads .txt content for tag checking.

        Returns (classified, dir_total) where classified is a list of
        (img_str, archive_txt_str, pass_num) tuples.
        """
        classified = []
        for f in image_names:
            img = folder / f
            img_str = str(img)
            if img_str in existing_paths:
                continue
            stem = Path(f).stem
            archive_txt = archive_dir / f"{stem}.txt"
            stem_lower = stem.lower()
            if stem_lower not in archive_stems:
                classified.append((img_str, str(archive_txt), 1))
            else:
                txt_path = archive_stems[stem_lower]
                try:
                    content = txt_path.read_text(encoding="utf-8", errors="ignore")
                    if f"[{model_tag}]" not in content:
                        classified.append((img_str, str(archive_txt), 2))
                except OSError:
                    classified.append((img_str, str(archive_txt), 2))
        return classified, len(image_names)

    def _flush_batch():
        nonlocal batch
        if not batch:
            return
        cur.executemany(
            "INSERT INTO ocr_images(image_path, input_root, archive_txt, pass_num, processed, error) "
            "VALUES(?, ?, ?, ?, ?, ?)",
            batch,
        )
        conn.commit()
        batch = []

    def _collect(classified, dir_total):
        """Collect results from one directory into counters and batch."""
        nonlocal scan_count, new_count, update_count, last_log
        scan_count += dir_total
        for img_str, archive_txt_str, pass_num in classified:
            batch.append((img_str, input_root, archive_txt_str, pass_num, False, None))
            if pass_num == 1:
                new_count += 1
            else:
                update_count += 1
        if scan_count - last_log >= 1000:
            elapsed = time.perf_counter() - start
            log_fn(
                f"  Scanned {scan_count:,} ({skip_count:,} existing, "
                f"{new_count:,} new, {update_count:,} update) [{elapsed:.0f}s]"
            )
            last_log = scan_count
        if len(batch) >= 5000:
            _flush_batch()

    if input_path.is_file():
        # Single file — no threading needed
        archive_dir = map_to_archive(input_path.parent)
        archive_stems = {}
        try:
            if archive_dir.is_dir():
                for p in archive_dir.iterdir():
                    if p.suffix.lower() == ".txt":
                        archive_stems[p.stem.lower()] = p
        except OSError:
            pass
        if input_path.suffix.lower() in exts:
            classified, dt = _classify_dir(
                input_path.parent, [input_path.name], archive_dir, archive_stems,
            )
            _collect(classified, dt)
    else:
        # Phase 1: walk source directories (just directory listings)
        dir_work = []
        for root, dirs, files in os.walk(input_path):
            dirs.sort()
            folder = Path(root)
            image_names = sorted(f for f in files if Path(f).suffix.lower() in exts)
            if not image_names:
                continue
            dir_work.append((folder, image_names, map_to_archive(folder)))

        total_images = sum(len(imgs) for _, imgs, _ in dir_work)
        elapsed = time.perf_counter() - start
        log_fn(
            f"  Found {total_images:,} images in {len(dir_work):,} directories [{elapsed:.0f}s]"
        )

        # Phase 2: walk archive to index existing .txt files
        archive_root = map_to_archive(input_path)
        archive_index = {}  # str(archive_dir) -> {stem_lower: Path}
        txt_count = 0
        if archive_root.is_dir():
            for root, dirs, files in os.walk(archive_root):
                dirs.sort()
                folder = Path(root)
                txt_files = {}
                for f in files:
                    if Path(f).suffix.lower() == ".txt":
                        txt_files[Path(f).stem.lower()] = folder / f
                if txt_files:
                    archive_index[str(folder)] = txt_files
                    txt_count += len(txt_files)

        elapsed = time.perf_counter() - start
        log_fn(
            f"  Found {txt_count:,} archive .txt files in "
            f"{len(archive_index):,} directories [{elapsed:.0f}s]"
        )

        # Phase 3: classify in parallel (tag reads only — existence is in-memory)
        with ThreadPoolExecutor(max_workers=SCAN_THREADS) as pool:
            futures = {
                pool.submit(
                    _classify_dir, folder, names, adir,
                    archive_index.get(str(adir), {}),
                ): None
                for folder, names, adir in dir_work
            }
            for future in as_completed(futures):
                classified, dir_total = future.result()
                _collect(classified, dir_total)

    _flush_batch()

    elapsed = time.perf_counter() - start
    log_fn(
        f"  Scan complete: {scan_count:,} images ({skip_count:,} existing, "
        f"{new_count:,} new, {update_count:,} update) [{elapsed:.0f}s]"
    )

    conn.close()

    # Set scan_time LAST — this marks the scan as complete. If we crashed
    # before this point, scan_time is empty and the next run resumes incrementally.
    set_scan_meta(input_root, "input_path", str(input_path))
    set_scan_meta(input_root, "total_scanned", str(scan_count))
    set_scan_meta(input_root, "scan_time", str(time.time()))

    return new_count, update_count


def iter_unprocessed(input_root: str, pass_num: int, batch_size: int = 500):
    """Yield batches of unprocessed image paths for the given pass and root.

    Uses keyset pagination so the prefetch thread always moves forward
    even if the flush thread hasn't marked previous batches yet.
    Each yield is a list of (image_path, archive_txt) tuples.
    """
    conn = _connect()
    try:
        cur = conn.cursor()
        last_path = ""
        while True:
            cur.execute(
                "SELECT TOP (?) image_path, archive_txt FROM ocr_images "
                "WHERE input_root=? AND pass_num=? AND processed=0 AND image_path > ? "
                "ORDER BY image_path",
                (batch_size, input_root, pass_num, last_path),
            )
            rows = cur.fetchall()
            if not rows:
                break
            last_path = rows[-1][0]
            yield [(Path(r[0]), Path(r[1])) for r in rows]
    finally:
        conn.close()


def mark_processed(input_root: str, image_paths: list[Path], error: str | None = None):
    """Mark images as processed in the database."""
    if not image_paths:
        return
    conn = _connect()
    cur = conn.cursor()
    cur.fast_executemany = True
    if error:
        cur.executemany(
            "UPDATE ocr_images SET processed=1, error=? WHERE image_path=? AND input_root=?",
            [(error, str(p), input_root) for p in image_paths],
        )
    else:
        cur.executemany(
            "UPDATE ocr_images SET processed=1 WHERE image_path=? AND input_root=?",
            [(str(p), input_root) for p in image_paths],
        )
    conn.commit()
    conn.close()


def get_stats(input_root: str) -> dict:
    """Get processing statistics for a specific input root."""
    conn = _connect()
    cur = conn.cursor()
    stats = {}
    for pass_num in (1, 2):
        cur.execute(
            "SELECT COUNT(*) FROM ocr_images WHERE input_root=? AND pass_num=?",
            (input_root, pass_num),
        )
        total = cur.fetchone()[0]
        cur.execute(
            "SELECT COUNT(*) FROM ocr_images WHERE input_root=? AND pass_num=? AND processed=1",
            (input_root, pass_num),
        )
        done = cur.fetchone()[0]
        cur.execute(
            "SELECT COUNT(*) FROM ocr_images WHERE input_root=? AND pass_num=? AND processed=1 AND error IS NOT NULL",
            (input_root, pass_num),
        )
        errors = cur.fetchone()[0]
        stats[pass_num] = {"total": total, "done": done, "remaining": total - done, "errors": errors}
    conn.close()
    return stats
