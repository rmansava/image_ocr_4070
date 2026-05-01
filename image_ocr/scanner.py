"""Image path helpers and archive destination mapping."""

from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}


def txt_path_for(image_path: Path) -> Path:
    """Return the .txt output path corresponding to an image."""
    return map_to_archive(image_path).with_suffix(".txt")


# ─── Path mapping (archiverelated → archive) ──────────────────

SOURCE_PREFIX = Path(r"T:\archiverelated")
DEST_PREFIX = Path(r"T:\archive")

# Exceptions: source prefix -> destination prefix
PATH_EXCEPTIONS: dict[Path, Path] = {
    Path(r"T:\archiverelated\books\pdf-images"): Path(r"T:\archive\book images"),
}

# Precomputed lowercase strings for fast matching — avoids per-call .resolve()
# which does network I/O on UNC/NAS paths.
_EXCEPTION_STRS: list[tuple[str, Path]] = sorted(
    [(str(src).lower().rstrip("\\"), dst) for src, dst in PATH_EXCEPTIONS.items()],
    key=lambda x: len(x[0]),
    reverse=True,
)
_ARCHIVE_MARKER = "\\archiverelated\\"
_ARCHIVE_MARKER_LEN = len(_ARCHIVE_MARKER)


def map_to_archive(source_path: Path) -> Path:
    """Map T:\\archiverelated\\... → T:\\archive\\... (or UNC equivalent).

    Fast: does NOT call source_path.resolve() (which triggers network I/O for
    UNC/NAS paths). Uses string-based prefix matching instead.
    Handles both T: mapped drive paths and \\\\server\\share UNC paths.
    """
    s = str(source_path)
    s_lower = s.lower()

    # Check exception prefixes first (T: path based)
    for src_lower, dest_prefix in _EXCEPTION_STRS:
        if s_lower.startswith(src_lower):
            rest = s[len(src_lower):].lstrip("\\")
            return dest_prefix / rest if rest else dest_prefix

    # Default: find \archiverelated\ and replace with \archive\
    # Handles T:\archiverelated\... and \\server\trivia\archiverelated\...
    idx = s_lower.find(_ARCHIVE_MARKER)
    if idx >= 0:
        return Path(s[:idx] + "\\archive\\" + s[idx + _ARCHIVE_MARKER_LEN:])

    # Fallback: not under archiverelated — return as-is
    return source_path
