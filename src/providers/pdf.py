"""PDF text extraction (pypdf).

Not a registered provider — it is invoked directly by the read pipeline when a
url is detected to be a PDF (by Content-Type, ``.pdf`` suffix, or a ``%PDF``
magic prefix). Pure byte → text, so it is trivially unit-testable.
"""

from __future__ import annotations

import io

from pypdf import PdfReader

from src.providers.base import ProviderError

# A PDF file always starts with this magic marker.
PDF_MAGIC = b"%PDF"

NO_TEXT_LAYER_NOTICE = (
    "В PDF нет извлекаемого текстового слоя (вероятно, скан из картинок). "
    "OCR не выполняется."
)


def looks_like_pdf(url: str, content_type: str | None, head: bytes) -> bool:
    """Heuristic PDF detection: url suffix, Content-Type, or magic bytes."""
    if url.split("?", 1)[0].rstrip().lower().endswith(".pdf"):
        return True
    if content_type and "application/pdf" in content_type.lower():
        return True
    return head[:4] == PDF_MAGIC


def extract_pdf_text(data: bytes) -> str:
    """Extract the text layer from raw PDF bytes.

    Returns a Russian notice when there is no extractable text layer (scanned
    PDF). Raises ``ProviderError`` if the bytes are not a parseable PDF.
    """
    try:
        reader = PdfReader(io.BytesIO(data))
        parts = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:  # noqa: BLE001 — pypdf raises a variety of errors
        raise ProviderError(f"pdf: could not parse PDF: {exc}") from exc
    text = "\n\n".join(part.strip() for part in parts if part.strip()).strip()
    return text or NO_TEXT_LAYER_NOTICE
