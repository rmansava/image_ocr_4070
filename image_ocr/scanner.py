"""Image path helpers and archive destination mapping."""

from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}


def txt_path_for(image_path: Path) -> Path:
    """Return the .txt output path corresponding to an image.

    Maps from T:\\archiverelated\\... to T:\\archive\\... so txt files
    go directly to the archive, not next to the source image.
    """
    return map_to_archive(image_path).with_suffix(".txt")


# ─── Path mapping (archiverelated → archive) ──────────────────

SOURCE_PREFIX = Path(r"T:\archiverelated")
DEST_PREFIX = Path(r"T:\archive")

# Exceptions: source prefix -> destination prefix
PATH_EXCEPTIONS: dict[Path, Path] = {
    Path(r"T:\archiverelated\books\pdf-images"): Path(r"T:\archive\book images"),
}


def map_to_archive(source_path: Path) -> Path:
    """Map a path from T:\\archiverelated\\... to T:\\archive\\...

    Checks exception mappings (longest prefix first), then falls back
    to the default swap of archiverelated -> archive.
    """
    resolved = source_path.resolve()

    # Check exceptions (longest prefix first)
    for src_prefix, dest_prefix in sorted(
        PATH_EXCEPTIONS.items(), key=lambda kv: len(str(kv[0])), reverse=True
    ):
        src_resolved = src_prefix.resolve()
        try:
            relative = resolved.relative_to(src_resolved)
            return dest_prefix / relative
        except ValueError:
            continue

    # Default: swap archiverelated for archive
    src_resolved = SOURCE_PREFIX.resolve()
    try:
        relative = resolved.relative_to(src_resolved)
        return DEST_PREFIX / relative
    except ValueError:
        # Not under archiverelated — write next to the image (fallback)
        return source_path
