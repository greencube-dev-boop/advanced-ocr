"""Adaptive chunk planning for the OXR API.

The free OXR API accepts at most 20 pages and 10MB per request, so pending
pages are grouped into contiguous chunks that satisfy both limits. Pure
planning functions here (a fake `measure` makes them unit-testable); the
pipeline supplies the real measure by serializing PDF chunks with PyMuPDF.
"""
from __future__ import annotations


def paginate_range(start: int, end: int, max_pages: int) -> list[tuple[int, int]]:
    """Split an inclusive page range into consecutive runs of <= max_pages."""
    if max_pages < 1:
        raise ValueError("max_pages must be >= 1")
    return [(a, min(a + max_pages - 1, end)) for a in range(start, end + 1, max_pages)]


def split_range(start: int, end: int, measure, max_bytes: int) -> list[tuple[int, int]]:
    """Recursively halve [start, end] (inclusive) until each piece measures
    <= max_bytes. A single page is never split further — the caller decides
    how to handle an oversized single page (e.g. render to PNG)."""
    if start == end:
        return [(start, end)]
    if measure((start, end)) <= max_bytes:
        return [(start, end)]
    mid = (start + end) // 2
    return (split_range(start, mid, measure, max_bytes)
            + split_range(mid + 1, end, measure, max_bytes))
