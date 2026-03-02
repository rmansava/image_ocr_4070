"""MS SQL index of images and their processing status.

Scan once, persist to DB, process from the index. No re-walking
the NAS on every restart. Multiple input roots (albums, comics, etc.)
coexist in the same table.
"""

import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import pyodbc

from .scanner import IMAGE_EXTENSIONS, map_to_archive


def _status(msg: str):
    """Overwrite current line (no newline) for live progress."""
    sys.stdout.write(f"\r\033[K{msg}")
    sys.stdout.flush()

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
    """Create tables and indexes if they don't exist.

    Uses an identity PK with a non-clustered unique index on image_path
    to avoid the 900-byte clustered index limit (long NAS paths).
    """
    conn = _connect()
    cur = conn.cursor()

    # Migrate: drop old table if it has the old composite PK schema
    cur.execute("""
        IF EXISTS (
            SELECT 1 FROM sys.indexes
            WHERE name = 'PK_ocr_images' AND object_id = OBJECT_ID('ocr_images')
              AND EXISTS (
                SELECT 1 FROM sys.index_columns ic
                JOIN sys.columns c ON ic.object_id = c.object_id AND ic.column_id = c.column_id
                WHERE ic.index_id = (SELECT index_id FROM sys.indexes WHERE name = 'PK_ocr_images' AND object_id = OBJECT_ID('ocr_images'))
                  AND ic.object_id = OBJECT_ID('ocr_images')
                  AND c.name = 'input_root'
            )
        )
        BEGIN
            DROP TABLE ocr_images
            -- Clear scan metadata so all roots get rescanned
            IF EXISTS (SELECT * FROM sys.tables WHERE name = 'ocr_scan_meta')
                DELETE FROM ocr_scan_meta
        END
    """)

    cur.execute("""
        IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'ocr_images')
        CREATE TABLE ocr_images (
            id           INT IDENTITY(1,1) NOT NULL,
            image_path   NVARCHAR(800) NOT NULL,
            input_root   NVARCHAR(200) NOT NULL,
            archive_txt  NVARCHAR(MAX) NOT NULL,
            pass_num     INT NOT NULL,
            processed    BIT DEFAULT 0,
            error        NVARCHAR(MAX),
            CONSTRAINT PK_ocr_images PRIMARY KEY CLUSTERED (id)
        )
    """)
    cur.execute("""
        IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'UQ_ocr_image_path')
        CREATE UNIQUE NONCLUSTERED INDEX UQ_ocr_image_path
            ON ocr_images(image_path)
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


def _scan_via_api(
    input_path, exts, existing_paths, model_tag,
    _classify_dir, _collect, _flush_batch, _write_phase1, log_fn, start, total_images,
):
    """Phase 1+2+3 using Synology FileStation API (fast NAS-side search)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from .nas_api import search_files, _connect as nas_connect

    # Reuse a single NAS session for all searches
    nas = nas_connect()

    # Phase 1: search NAS for images
    log_fn(f"  Phase 1: searching NAS for images (via API)...")
    ext_list = [e.lstrip(".") for e in exts]
    image_paths = search_files(input_path, ext_list, log_fn, label="images", nas=nas)

    # Group by directory
    dir_images = defaultdict(list)
    for unc_path in image_paths:
        p = Path(unc_path)
        dir_images[p.parent].append(p.name)

    walk_images = len(image_paths)
    elapsed = time.perf_counter() - start
    log_fn(
        f"  Phase 1: {walk_images:,} images in "
        f"{len(dir_images):,} directories [{elapsed:.0f}s]"
    )

    # Checkpoint after Phase 1: write all image paths to DB with pass_num=0
    # so a restart can skip Phase 1 entirely.
    _write_phase1(image_paths, dir_images)

    # Phase 2: search NAS for archive .txt files
    # NOTE: must use extension param, not pattern — Synology API quirk where
    # pattern="*.txt" + recursive=True silently returns 0 results.
    archive_root = map_to_archive(input_path)
    log_fn(f"  Phase 2: searching NAS for archive .txt files (via API)...")
    txt_paths = search_files(
        archive_root, ["txt"], log_fn, label=".txt files",
        use_extension_param=True, nas=nas,
    )

    # Done with NAS API — logout to free the session
    nas.logout()

    # Build archive index: {str(archive_dir): {stem_lower: Path}}
    archive_index = defaultdict(dict)
    for unc_path in txt_paths:
        p = Path(unc_path)
        archive_index[str(p.parent)][p.stem.lower()] = p

    elapsed = time.perf_counter() - start
    log_fn(
        f"  Phase 2: {len(txt_paths):,} archive .txt files in "
        f"{len(archive_index):,} directories [{elapsed:.0f}s]"
    )

    # Phase 3: classify in parallel (tag reads only — existence is in-memory)
    dir_work = []
    for folder, names in dir_images.items():
        adir = map_to_archive(folder)
        dir_work.append((folder, sorted(names), adir))

    total_images[0] = walk_images
    log_fn(f"  Phase 3: classifying {walk_images:,} images ({SCAN_THREADS} threads)...")
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


def _scan_via_walk(
    input_path, exts, existing_paths, model_tag,
    _classify_dir, _collect, _flush_batch, _write_phase1, log_fn, start, total_images,
):
    """Phase 1+2+3 using os.walk (fallback for local paths)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Phase 1: walk source directories (just directory listings)
    log_fn(f"  Phase 1: walking source directories...")
    dir_work = []
    dir_images = defaultdict(list)
    walk_images = 0
    walk_last_log = 0
    all_image_paths = []
    for root, dirs, files in os.walk(input_path):
        dirs.sort()
        folder = Path(root)
        image_names = sorted(f for f in files if Path(f).suffix.lower() in exts)
        if not image_names:
            continue
        adir = map_to_archive(folder)
        dir_work.append((folder, image_names, adir))
        for name in image_names:
            p = folder / name
            dir_images[folder].append(name)
            all_image_paths.append(str(p))
        walk_images += len(image_names)
        if walk_images - walk_last_log >= 10000:
            _status(f"  Phase 1: walking source... {walk_images:,} images, {len(dir_work):,} dirs")
            walk_last_log = walk_images

    elapsed = time.perf_counter() - start
    log_fn(
        f"  Phase 1: {walk_images:,} images in {len(dir_work):,} directories [{elapsed:.0f}s]"
    )

    # Checkpoint after Phase 1
    _write_phase1(all_image_paths, dir_images)

    # Phase 2: walk archive to index existing .txt files
    log_fn(f"  Phase 2: walking archive directories...")
    archive_root = map_to_archive(input_path)
    archive_index = {}  # str(archive_dir) -> {stem_lower: Path}
    txt_count = 0
    archive_dirs = 0
    if archive_root.is_dir():
        walk_last_log = 0
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
            archive_dirs += 1
            if txt_count - walk_last_log >= 10000:
                _status(f"  Phase 2: walking archive... {txt_count:,} .txt files, {archive_dirs:,} dirs")
                walk_last_log = txt_count

    elapsed = time.perf_counter() - start
    log_fn(
        f"  Phase 2: {txt_count:,} archive .txt files in "
        f"{len(archive_index):,} directories [{elapsed:.0f}s]"
    )

    # Phase 3: classify in parallel (tag reads only — existence is in-memory)
    total_images[0] = walk_images
    log_fn(f"  Phase 3: classifying {walk_images:,} images ({SCAN_THREADS} threads)...")
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


def _scan_phase2_3_api(
    input_path, model_tag, image_paths, dir_images,
    _classify_dir, _collect, _flush_batch, log_fn, start, total_images,
):
    """Phase 2+3 only (used when Phase 1 checkpoint exists): NAS API path."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from .nas_api import search_files, _connect as nas_connect

    nas = nas_connect()
    archive_root = map_to_archive(input_path)
    log_fn(f"  Phase 2: searching NAS for archive .txt files (via API)...")
    txt_paths = search_files(
        archive_root, ["txt"], log_fn, label=".txt files",
        use_extension_param=True, nas=nas,
    )
    nas.logout()

    archive_index = defaultdict(dict)
    for unc_path in txt_paths:
        p = Path(unc_path)
        archive_index[str(p.parent)][p.stem.lower()] = p

    elapsed = time.perf_counter() - start
    log_fn(
        f"  Phase 2: {len(txt_paths):,} archive .txt files in "
        f"{len(archive_index):,} directories [{elapsed:.0f}s]"
    )

    dir_work = [(folder, sorted(names), map_to_archive(folder)) for folder, names in dir_images.items()]
    walk_images = total_images[0]
    log_fn(f"  Phase 3: classifying {walk_images:,} images ({SCAN_THREADS} threads)...")
    with ThreadPoolExecutor(max_workers=SCAN_THREADS) as pool:
        futures = {
            pool.submit(_classify_dir, folder, names, adir, archive_index.get(str(adir), {})): None
            for folder, names, adir in dir_work
        }
        for future in as_completed(futures):
            _collect(*future.result())


def _scan_phase2_3_walk(
    input_path, model_tag, dir_images,
    _classify_dir, _collect, _flush_batch, log_fn, start, total_images,
):
    """Phase 2+3 only (used when Phase 1 checkpoint exists): local walk path."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    archive_root = map_to_archive(input_path)
    archive_index = {}
    txt_count = 0
    log_fn(f"  Phase 2: walking archive directories...")
    if archive_root.is_dir():
        for root, dirs, files in os.walk(archive_root):
            dirs.sort()
            folder = Path(root)
            txt_files = {Path(f).stem.lower(): folder / f for f in files if Path(f).suffix.lower() == ".txt"}
            if txt_files:
                archive_index[str(folder)] = txt_files
                txt_count += len(txt_files)

    elapsed = time.perf_counter() - start
    log_fn(f"  Phase 2: {txt_count:,} archive .txt files [{elapsed:.0f}s]")

    dir_work = [(folder, sorted(names), map_to_archive(folder)) for folder, names in dir_images.items()]
    walk_images = total_images[0]
    log_fn(f"  Phase 3: classifying {walk_images:,} images ({SCAN_THREADS} threads)...")
    with ThreadPoolExecutor(max_workers=SCAN_THREADS) as pool:
        futures = {
            pool.submit(_classify_dir, folder, names, adir, archive_index.get(str(adir), {})): None
            for folder, names, adir in dir_work
        }
        for future in as_completed(futures):
            _collect(*future.result())


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
    if log_fn is None:
        log_fn = print

    exts = extensions or IMAGE_EXTENSIONS
    input_root = _input_root_key(input_path)

    conn = _connect()
    cur = conn.cursor()
    cur.fast_executemany = True

    # Check for Phase 1 checkpoint: images already in DB from a previous run
    # that was interrupted after Phase 1 but before Phase 3 completed.
    prev_scan_time = get_scan_meta(input_root, "scan_time")
    phase1_resume = (not force_full) and prev_scan_time == "phase1_complete"

    if force_full:
        cur.execute("DELETE FROM ocr_images WHERE input_root=?", (input_root,))
        conn.commit()
        log_fn(f"  Full rescan — cleared previous index")
        set_scan_meta(input_root, "scan_time", "")

    set_scan_meta(input_root, "model_tag", model_tag)

    existing_paths = set()
    if not force_full:
        # Load already-classified paths (pass_num 1 or 2) — skip in Phase 3.
        # pass_num=0 rows are Phase 1 placeholders that still need classification.
        cur.execute(
            "SELECT image_path FROM ocr_images WHERE input_root=? AND pass_num > 0",
            (input_root,),
        )
        existing_paths = {row[0] for row in cur.fetchall()}
        if existing_paths:
            log_fn(f"  Resuming scan ({len(existing_paths):,} images already classified)")

    # phase1_paths: set of image_path strings written to DB with pass_num=0.
    # Phase 3 UPDATEs these rows to final pass_num instead of INSERTing.
    phase1_paths = set()

    new_count = 0
    update_count = 0
    scan_count = 0
    dirs_done = 0
    total_images = [0]  # mutable so _scan_via_* can set it before Phase 3
    skip_count = len(existing_paths)
    batch = []
    start = time.perf_counter()
    last_log = 0

    def _write_phase1(image_paths, dir_images):
        """Bulk-write all image paths to DB with pass_num=0 as a checkpoint.

        After this, a restart can skip Phase 1 (NAS image search) entirely.
        Rows already in existing_paths are skipped.
        """
        nonlocal phase1_paths
        rows = []
        for img_str in image_paths:
            if img_str in existing_paths:
                continue
            p = Path(img_str)
            archive_txt = str(map_to_archive(p).with_suffix(".txt"))
            rows.append((img_str, input_root, archive_txt, 0, False, None))
            phase1_paths.add(img_str)
        if rows:
            cur.fast_executemany = True
            # Use MERGE to skip rows already indexed under a different input_root
            cur.executemany(
                "MERGE ocr_images AS t "
                "USING (SELECT ? AS ip, ? AS ir, ? AS at, ? AS pn, ? AS pr, ? AS er) AS s "
                "ON t.image_path = s.ip "
                "WHEN NOT MATCHED THEN INSERT (image_path, input_root, archive_txt, pass_num, processed, error) "
                "VALUES(s.ip, s.ir, s.at, s.pn, s.pr, s.er);",
                rows,
            )
            conn.commit()
        set_scan_meta(input_root, "scan_time", "phase1_complete")
        log_fn(f"  Phase 1 checkpoint: {len(phase1_paths):,} new images written to DB")

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
        # Rows already written in Phase 1 get UPDATEd; new rows get INSERTed.
        updates = [(r[3], r[2], r[0]) for r in batch if r[0] in phase1_paths]
        inserts = [r for r in batch if r[0] not in phase1_paths]
        if updates:
            cur.executemany(
                "UPDATE ocr_images SET pass_num=?, archive_txt=? WHERE image_path=? AND input_root='" + input_root + "'",
                updates,
            )
        if inserts:
            cur.executemany(
                "INSERT INTO ocr_images(image_path, input_root, archive_txt, pass_num, processed, error) "
                "VALUES(?, ?, ?, ?, ?, ?)",
                inserts,
            )
        conn.commit()
        batch = []

    def _collect(classified, dir_total):
        """Collect results from one directory into counters and batch."""
        nonlocal scan_count, new_count, update_count, dirs_done, last_log
        scan_count += dir_total
        dirs_done += 1
        for img_str, archive_txt_str, pass_num in classified:
            batch.append((img_str, input_root, archive_txt_str, pass_num, False, None))
            if pass_num == 1:
                new_count += 1
            else:
                update_count += 1
        elapsed = time.perf_counter() - start
        rate = scan_count / elapsed if elapsed > 0 else 0
        tot = total_images[0]
        pct = f"{100 * scan_count / tot:.1f}%" if tot else ""
        _status(
            f"  Phase 3: {scan_count:,}/{tot:,} {pct} "
            f"({new_count:,} new, {update_count:,} update) "
            f"@ {rate:.0f} img/s [{elapsed:.0f}s]"
        )
        if scan_count - last_log >= 10000:
            log_fn(
                f"  Phase 3: {scan_count:,}/{tot:,} {pct} "
                f"({new_count:,} new, {update_count:,} update) "
                f"@ {rate:.0f} img/s [{elapsed:.0f}s]"
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
        # Detect whether this is a NAS path (UNC or T: drive)
        input_str = str(input_path)
        is_nas = (
            input_str.startswith("\\\\")
            or input_str.lower().startswith("t:\\")
            or input_str.lower().startswith("t:/")
        )

        if phase1_resume:
            # Phase 1 already done — load the pass_num=0 placeholder rows and
            # go straight to Phase 2+3 to classify them.
            cur.execute(
                "SELECT image_path, archive_txt FROM ocr_images "
                "WHERE input_root=? AND pass_num=0",
                (input_root,),
            )
            p1_rows = cur.fetchall()
            phase1_paths.update(row[0] for row in p1_rows)
            log_fn(f"  Skipping Phase 1 ({len(phase1_paths):,} images from checkpoint)")
            # Rebuild dir_images from the checkpoint rows
            dir_images = defaultdict(list)
            for img_str, _ in p1_rows:
                p = Path(img_str)
                dir_images[p.parent].append(p.name)
            image_paths = [row[0] for row in p1_rows]
            total_images[0] = len(image_paths)
            # Phase 2+3 only
            if is_nas:
                _scan_phase2_3_api(
                    input_path, model_tag, image_paths, dir_images,
                    _classify_dir, _collect, _flush_batch, log_fn, start, total_images,
                )
            else:
                _scan_phase2_3_walk(
                    input_path, model_tag, dir_images,
                    _classify_dir, _collect, _flush_batch, log_fn, start, total_images,
                )
        elif is_nas:
            _scan_via_api(
                input_path, exts, existing_paths, model_tag,
                _classify_dir, _collect, _flush_batch, _write_phase1, log_fn, start, total_images,
            )
        else:
            _scan_via_walk(
                input_path, exts, existing_paths, model_tag,
                _classify_dir, _collect, _flush_batch, _write_phase1, log_fn, start, total_images,
            )

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
