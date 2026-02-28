"""Format OCR output and manage text file writing with model tags."""

from pathlib import Path

from .scanner import txt_path_for


def format_result(text: str, model_tag: str) -> str:
    """Wrap OCR text with a model tag header and trailing separator.

    Example output:
        [deepseek-ocr-2]
        Some extracted text...
        ---
    """
    return f"[{model_tag}]\n{text}\n---\n"


def write_result(image_path: Path, text: str, model_tag: str) -> Path:
    """Write or append an OCR result to the corresponding .txt file.

    If the file doesn't exist, it is created.
    If it exists, the new result is appended with a blank line separator.

    Returns:
        Path to the written text file.
    """
    txt = txt_path_for(image_path)
    block = format_result(text, model_tag)

    if txt.exists():
        existing = txt.read_text(encoding="utf-8")
        # Add a blank line between entries if the file doesn't end with one
        if existing and not existing.endswith("\n\n"):
            block = "\n" + block
        txt.write_text(existing + block, encoding="utf-8")
    else:
        txt.write_text(block, encoding="utf-8")

    return txt
