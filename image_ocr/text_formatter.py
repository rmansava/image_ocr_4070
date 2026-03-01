"""Format OCR output into tagged appendable blocks."""


def format_result(text: str, model_tag: str) -> str:
    """Wrap OCR text with a model tag header and trailing separator.

    Example output:
        [deepseek-ocr-2]
        Some extracted text...
        ---
    """
    return f"[{model_tag}]\n{text}\n---\n"
