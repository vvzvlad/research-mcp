"""PDF extraction helper tests (pure, no network)."""

from pathlib import Path

import pytest

from src.providers.base import ProviderError
from src.providers.pdf import (
    NO_TEXT_LAYER_NOTICE,
    extract_pdf_text,
    looks_like_pdf,
)

SAMPLE_PDF = (Path(__file__).parent / "fixtures" / "sample.pdf").read_bytes()


def test_extract_real_pdf():
    assert "Hello PDF research-mcp" in extract_pdf_text(SAMPLE_PDF)


def test_extract_garbage_raises():
    with pytest.raises(ProviderError):
        extract_pdf_text(b"not a pdf at all")


def test_no_text_layer_notice_mentions_ocr_without_promising_it_was_skipped():
    # Pipeline.read hands this notice back only after the read chain — which on
    # a keyed .pdf url includes jina's OCR tier — has failed too, so the text
    # must mention OCR (the caller should know recognition is a thing here)
    # while claiming neither that it ran nor that it was skipped.
    assert "OCR" in NO_TEXT_LAYER_NOTICE
    assert "не выполняется" not in NO_TEXT_LAYER_NOTICE


def test_looks_like_pdf_by_suffix():
    assert looks_like_pdf("https://x.test/a.pdf", None, b"")
    assert looks_like_pdf("https://x.test/a.pdf?token=1", None, b"")


def test_looks_like_pdf_by_content_type():
    assert looks_like_pdf("https://x.test/a", "application/pdf; charset=binary", b"")


def test_looks_like_pdf_by_magic():
    assert looks_like_pdf("https://x.test/a", "application/octet-stream", b"%PDF-1.7")


def test_not_pdf():
    assert not looks_like_pdf("https://x.test/a", "text/html", b"<html>")
