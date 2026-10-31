"""Classify each PDF page so the pipeline knows what to do with it.

kind:
  - "text_only": reliable text layer, no significant visuals -> no OCR needed
  - "visual":    images / tables / dense vector drawings -> render + OCR path
  - "empty":     nothing extractable at all
"""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class PageInfo:
    page_no: int
    n_chars: int
    n_images: int          # raster images larger than the area threshold
    n_tables: int          # tables found by PyMuPDF's ruling-line detector
    n_drawings: int        # vector drawing ops (flowcharts/diagrams are dense)
    has_text: bool
    has_image: bool
    has_table: bool
    has_dense_drawings: bool
    kind: str              # text_only | visual | empty

    def to_dict(self) -> dict:
        return asdict(self)


def _large_image_count(page, area_ratio: float) -> int:
    page_area = abs(page.rect.width * page.rect.height) or 1.0
    count = 0
    for img in page.get_images(full=True):
        try:
            rects = page.get_image_rects(img[0])
        except Exception:
            rects = []
        for r in rects:
            if abs(r.width * r.height) / page_area >= area_ratio:
                count += 1
                break
    return count


def _ruling_line_count(page) -> int:
    """Fallback table heuristic: many long straight strokes = ruled table."""
    n = 0
    try:
        for d in page.get_drawings():
            for item in d["items"]:
                if item[0] == "l":  # line
                    p1, p2 = item[1], item[2]
                    if abs(p1.x - p2.x) < 1 or abs(p1.y - p2.y) < 1:  # axis-aligned
                        n += 1
    except Exception:
        pass
    return n


def analyze_page(page, page_no: int, min_text_chars: int = 20,
                 dense_drawings: int = 40, large_image_area_ratio: float = 0.01) -> PageInfo:
    text = page.get_text("text") or ""
    n_images = _large_image_count(page, large_image_area_ratio)
    try:
        n_tables = len(page.find_tables().tables)
    except Exception:
        n_tables = 1 if _ruling_line_count(page) > 6 else 0
    try:
        n_drawings = len(page.get_drawings())
    except Exception:
        n_drawings = 0

    has_text = len(text.strip()) >= min_text_chars
    has_image = n_images > 0
    has_table = n_tables > 0
    has_dense = n_drawings >= dense_drawings

    if not (has_text or has_image or has_table or has_dense):
        kind = "empty"
    elif has_text and not (has_image or has_table or has_dense):
        kind = "text_only"
    else:
        kind = "visual"

    return PageInfo(
        page_no=page_no,
        n_chars=len(text.strip()),
        n_images=n_images,
        n_tables=n_tables,
        n_drawings=n_drawings,
        has_text=has_text,
        has_image=has_image,
        has_table=has_table,
        has_dense_drawings=has_dense,
        kind=kind,
    )
