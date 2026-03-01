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


def scan_to_db(
    input_path: Path,
    model_tag: str,
    extensions: set[str] | None = None,
    log_fn=None,
) -> tuple[int, int]:
    """Walk the input directory and populate the SQL index.

    Only touches rows for this input_root — other roots are untouched.
    Returns (new_count, update_count).
    """
    if log_fn is None:
        log_fn = print

    exts = extensions or IMAGE_EXTENSIONS
    input_root = _input_root_key(input_path)

    conn = _connect()
    cur = conn.cursor()

    # Clear previous scan data for THIS root only
    cur.execute("DELETE FROM ocr_images WHERE input_root=?", (input_root,))
    conn.commit()

    new_count = 0
    update_count = 0
    scan_count = 0
    batch = []
    start = time.perf_counter()

    def _iter_images():
        """Yield image paths one at a time — no full list in memory."""
        if input_path.is_file():
            if input_path.suffix.lower() in exts:
                yield input_path
            return
        for root, dirs, files in os.walk(input_path):
            dirs.sort()
            folder = Path(root)
            for f in sorted(files):
                if Path(f).suffix.lower() in exts:
                    yield folder / f

    for img in _iter_images():
        scan_count += 1
        archive_txt = map_to_archive(img).with_suffix(".txt")

        if not archive_txt.exists():
            batch.append((str(img), input_root, str(archive_txt), 1, False, None))
            new_count += 1
        else:
            try:
                content = archive_txt.read_text(encoding="utf-8", errors="ignore")
                if f"[{model_tag}]" not in content:
                    batch.append((str(img), input_root, str(archive_txt), 2, False, None))
                    update_count += 1
            except OSError:
                batch.append((str(img), input_root, str(archive_txt), 2, False, None))
                update_count += 1

        if len(batch) >= 5000:
            cur.fast_executemany = True
            cur.executemany(
                "INSERT INTO ocr_images(image_path, input_root, archive_txt, pass_num, processed, error) "
                "VALUES(?, ?, ?, ?, ?, ?)",
                batch,
            )
            conn.commit()
            elapsed = time.perf_counter() - start
            log_fn(f"  Scanned {scan_count:,} images ({new_count:,} new, {update_count:,} update) [{elapsed:.0f}s]")
            batch = []

    if batch:
        cur.fast_executemany = True
        cur.executemany(
            "INSERT INTO ocr_images(image_path, input_root, archive_txt, pass_num, processed, error) "
            "VALUES(?, ?, ?, ?, ?, ?)",
            batch,
        )
        conn.commit()

    elapsed = time.perf_counter() - start
    log_fn(f"  Scan complete: {scan_count:,} images ({new_count:,} new, {update_count:,} update) [{elapsed:.0f}s]")

    conn.close()

    set_scan_meta(input_root, "model_tag", model_tag)
    set_scan_meta(input_root, "input_path", str(input_path))
    set_scan_meta(input_root, "scan_time", str(time.time()))
    set_scan_meta(input_root, "total_scanned", str(scan_count))

    return new_count, update_count


def iter_unprocessed(input_root: str, pass_num: int, batch_size: int = 500):
    """Yield batches of unprocessed image paths for the given pass and root.

    Uses keyset pagination so the prefetch thread always moves forward
    even if the flush thread hasn't marked previous batches yet.
    Each yield is a list of (image_path, archive_txt) tuples.
    """
    conn = _connect()
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
    conn.close()


def mark_processed(input_root: str, image_paths: list[Path], error: str | None = None):
    """Mark images as processed in the database."""
    conn = _connect()
    cur = conn.cursor()
    if error:
        for p in image_paths:
            cur.execute(
                "UPDATE ocr_images SET processed=1, error=? WHERE image_path=? AND input_root=?",
                (error, str(p), input_root),
            )
    else:
        for p in image_paths:
            cur.execute(
                "UPDATE ocr_images SET processed=1 WHERE image_path=? AND input_root=?",
                (str(p), input_root),
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
