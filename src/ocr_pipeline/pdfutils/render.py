"""Render a PDF page to a PNG image (bytes) for the OCR engines."""
from __future__ import annotations


def render_page_png(page, dpi: int = 200) -> bytes:
    pix = page.get_pixmap(dpi=dpi)
    return pix.tobytes("png")
