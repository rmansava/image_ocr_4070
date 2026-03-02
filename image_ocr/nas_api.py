"""Synology FileStation API helpers for fast NAS file discovery.

Uses the Synology HTTP API to search for files on the NAS itself,
avoiding slow SMB directory walks over the network. The NAS does the
traversal locally and returns results via HTTP — ~6x faster than os.walk.
"""

import time
from pathlib import Path, PureWindowsPath

from synology_api.filestation import FileStation

# NAS connection settings
NAS_IP = "10.10.10.2"
NAS_PORT = "5000"
NAS_USER = "larry"
NAS_PASS = "olliegussywalter3C"

# UNC prefix for converting between API paths and Windows paths
# \\10.10.10.2\trivia\archiverelated\... <-> /trivia/archiverelated/...
UNC_PREFIX = f"\\\\{NAS_IP}"


def _connect() -> FileStation:
    """Create a new FileStation API connection."""
    return FileStation(
        ip_address=NAS_IP,
        port=NAS_PORT,
        username=NAS_USER,
        password=NAS_PASS,
        secure=False,
        cert_verify=False,
        dsm_version=7,
        debug=False,
        interactive_output=False,
    )


def unc_to_api(unc_path: Path) -> str:
    """Convert a UNC path to a Synology API path.

    \\\\10.10.10.2\\trivia\\archiverelated\\board games
    -> /trivia/archiverelated/board games
    """
    s = str(unc_path).replace("\\", "/")
    # Strip \\10.10.10.2 prefix
    prefix = UNC_PREFIX.replace("\\", "/")
    if s.startswith(prefix):
        return s[len(prefix):]
    # If it's a T: drive path, convert via known mapping
    # T:\ -> /trivia/
    s_lower = s.lower()
    if s_lower.startswith("t:/") or s_lower.startswith("t:\\"):
        return "/trivia/" + s[3:]
    raise ValueError(f"Cannot convert path to API format: {unc_path}")


def api_to_unc(api_path: str) -> str:
    """Convert a Synology API path back to a UNC path string.

    /trivia/archiverelated/board games/Die Macher/foo.jpg
    -> \\\\10.10.10.2\\trivia\\archiverelated\\board games\\Die Macher\\foo.jpg
    """
    return UNC_PREFIX + api_path.replace("/", "\\")


def search_files(
    folder_path: Path,
    extensions: list[str],
    log_fn=None,
    label: str = "files",
) -> list[str]:
    """Search for files with given extensions under folder_path using the NAS API.

    Runs parallel search tasks (one per extension) on the NAS itself.
    Returns a list of UNC path strings for all matching files.

    Args:
        folder_path: UNC or T: path to search under
        extensions: list of extensions without dots, e.g. ["jpg", "png"]
        log_fn: optional logging function
        label: label for progress messages
    """
    if log_fn is None:
        log_fn = print

    api_folder = unc_to_api(folder_path)
    fs = _connect()

    # Start parallel searches — one per extension
    tasks = {}
    for ext in extensions:
        result = fs.search_start(
            folder_path=api_folder,
            recursive=True,
            pattern=f"*.{ext}",
        )
        task_id = result.get("taskid") if isinstance(result, dict) else None
        if task_id:
            tasks[ext] = task_id

    if not tasks:
        log_fn(f"  No search tasks started")
        return []

    log_fn(f"  Searching NAS for {label} ({len(tasks)} extension searches)...")

    # Poll all tasks until all are finished
    all_paths = []
    finished_tasks = set()
    start = time.perf_counter()
    last_log_count = 0

    while len(finished_tasks) < len(tasks):
        time.sleep(2)
        total_found = 0
        all_done = True

        for ext, task_id in tasks.items():
            if ext in finished_tasks:
                # Already collected this one
                continue

            status = fs.get_search_list(task_id=task_id, offset=0, limit=1)
            data = status.get("data", {}) if isinstance(status, dict) else {}
            count = data.get("total", 0)
            finished = data.get("finished", False)
            total_found += count

            if finished:
                finished_tasks.add(ext)

        # Count already-collected paths too
        total_found += len(all_paths)
        elapsed = time.perf_counter() - start

        if total_found - last_log_count >= 10000 or len(finished_tasks) == len(tasks):
            log_fn(
                f"  NAS search: {total_found:,} {label} found "
                f"({len(finished_tasks)}/{len(tasks)} extensions done) [{elapsed:.0f}s]"
            )
            last_log_count = total_found

    # Collect all results via pagination
    log_fn(f"  Collecting search results...")
    for ext, task_id in tasks.items():
        offset = 0
        page_size = 5000
        while True:
            batch = fs.get_search_list(
                task_id=task_id, offset=offset, limit=page_size,
            )
            data = batch.get("data", {}) if isinstance(batch, dict) else {}
            files = data.get("files", [])
            if not files:
                break
            for f in files:
                api_path = f.get("path", "")
                if api_path:
                    all_paths.append(api_to_unc(api_path))
            offset += len(files)
            if len(files) < page_size:
                break

    fs.stop_all_search_task()

    elapsed = time.perf_counter() - start
    log_fn(f"  NAS search complete: {len(all_paths):,} {label} [{elapsed:.0f}s]")
    return all_paths
