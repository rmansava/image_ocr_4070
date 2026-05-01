"""Synology FileStation API helpers for fast NAS file discovery.

Uses the Synology HTTP API to search for files on the NAS itself,
avoiding slow SMB directory walks over the network. The NAS does the
traversal locally and returns results via HTTP — ~6x faster than os.walk.

Uses raw HTTP requests instead of the synology_api library to avoid
session management issues that cause the library's search to silently
return 0 results after multiple connections.

IMPORTANT Synology API quirks:
  - Only ONE search task at a time per session (sequential, not parallel).
  - pattern="*.txt" + recursive=True silently returns 0. Use extension="txt".
"""

import time
from pathlib import Path

import requests

# NAS connection settings
NAS_IP = "192.168.1.53"
NAS_PORT = "5000"
NAS_USER = "larry"
NAS_PASS = "olliegussywalter3C"
NAS_BASE = f"http://{NAS_IP}:{NAS_PORT}/webapi"

# UNC prefix for converting between API paths and Windows paths
# \\10.10.10.2\trivia\archiverelated\... <-> /trivia/archiverelated/...
UNC_PREFIX = f"\\\\{NAS_IP}"


class NasSession:
    """Lightweight Synology FileStation session using raw HTTP."""

    def __init__(self):
        self._session = requests.Session()
        r = self._session.get(f"{NAS_BASE}/auth.cgi", params={
            "api": "SYNO.API.Auth",
            "version": 6,
            "method": "login",
            "account": NAS_USER,
            "passwd": NAS_PASS,
            "session": "FileStation",
            "format": "sid",
        })
        data = r.json()
        if not data.get("success"):
            raise ConnectionError(f"NAS login failed: {data}")
        self._sid = data["data"]["sid"]

    def _api(self, api: str, method: str, **params) -> dict:
        """Make an API call and return the JSON response."""
        params["api"] = api
        params["method"] = method
        params["_sid"] = self._sid
        if "version" not in params:
            params["version"] = 2
        cgi = "auth.cgi" if api == "SYNO.API.Auth" else "entry.cgi"
        r = self._session.get(f"{NAS_BASE}/{cgi}", params=params)
        return r.json()

    def search_start(self, folder_path: str, **kwargs) -> str:
        """Start a search and return the task ID."""
        resp = self._api(
            "SYNO.FileStation.Search", "start",
            folder_path=folder_path, **kwargs,
        )
        if not resp.get("success"):
            raise RuntimeError(f"search_start failed: {resp}")
        return resp["data"]["taskid"]

    def search_list(self, task_id: str, offset: int = 0, limit: int = 5000) -> dict:
        """Get search results. Returns the 'data' dict."""
        resp = self._api(
            "SYNO.FileStation.Search", "list",
            taskid=f'"{task_id}"', offset=offset, limit=limit,
        )
        return resp.get("data", {})

    def search_stop(self, task_id: str):
        """Stop a search task."""
        self._api(
            "SYNO.FileStation.Search", "stop",
            taskid=f'"{task_id}"',
        )

    def logout(self):
        """End the session."""
        try:
            self._api("SYNO.API.Auth", "logout", version=6, session="FileStation")
        except Exception:
            pass


def _connect() -> NasSession:
    """Create a new NAS API session."""
    return NasSession()


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


def _search_one(
    nas: NasSession,
    api_folder: str,
    ext: str,
    use_extension_param: bool,
    log_fn,
    label: str,
) -> list[str]:
    """Run a single search task and return list of UNC paths.

    Starts, polls, collects, and stops — one extension at a time.
    """
    kwargs = {"recursive": "true"}
    if use_extension_param:
        kwargs["extension"] = ext
    else:
        kwargs["pattern"] = f"*.{ext}"

    task_id = nas.search_start(api_folder, **kwargs)

    # Poll until finished (with progress logging for large searches)
    last_log = 0
    start = time.perf_counter()
    while True:
        time.sleep(2)
        data = nas.search_list(task_id, offset=0, limit=1)
        total = data.get("total", 0)
        finished = data.get("finished", False)

        if total - last_log >= 10000:
            elapsed = time.perf_counter() - start
            log_fn(f"    .{ext}: {total:,} {label} found so far [{elapsed:.0f}s]")
            last_log = total

        if finished:
            break

    # Collect all results via pagination
    unc_paths = []
    offset = 0
    page_size = 5000
    while True:
        data = nas.search_list(task_id, offset=offset, limit=page_size)
        files = data.get("files", [])
        if not files:
            break
        for f in files:
            api_path = f.get("path", "")
            if api_path:
                unc_paths.append(api_to_unc(api_path))
        offset += len(files)
        if len(files) < page_size:
            break

    nas.search_stop(task_id)
    return unc_paths


def search_files(
    folder_path: Path,
    extensions: list[str],
    log_fn=None,
    label: str = "files",
    use_extension_param: bool = False,
    nas: NasSession | None = None,
) -> list[str]:
    """Search for files with given extensions under folder_path using the NAS API.

    Runs searches sequentially (one extension at a time) because the Synology
    API can only handle one search task at a time per session. Returns a list
    of UNC path strings for all matching files.

    Args:
        folder_path: UNC or T: path to search under
        extensions: list of extensions without dots, e.g. ["jpg", "png"]
        log_fn: optional logging function
        label: label for progress messages
        use_extension_param: if True, use the 'extension' API parameter instead
            of 'pattern'. Required for .txt files where pattern="*.txt" with
            recursive=True silently returns 0 results (Synology API quirk).
        nas: optional existing NasSession. Reusing one session across multiple
            searches avoids orphaned sessions.
    """
    if log_fn is None:
        log_fn = print

    api_folder = unc_to_api(folder_path)
    owns_connection = nas is None
    if nas is None:
        nas = _connect()

    log_fn(f"  Searching NAS for {label} ({len(extensions)} extensions)...")
    start = time.perf_counter()
    all_paths = []

    for i, ext in enumerate(extensions):
        log_fn(f"  Searching .{ext} ({i + 1}/{len(extensions)})...")
        unc_paths = _search_one(nas, api_folder, ext, use_extension_param, log_fn, label)
        all_paths.extend(unc_paths)
        elapsed = time.perf_counter() - start
        log_fn(
            f"  .{ext}: {len(unc_paths):,} found — "
            f"total {len(all_paths):,} {label} [{elapsed:.0f}s]"
        )

    if owns_connection:
        nas.logout()

    elapsed = time.perf_counter() - start
    log_fn(f"  NAS search complete: {len(all_paths):,} {label} [{elapsed:.0f}s]")
    return all_paths
