"""Native text-layer extraction with block-level reading order."""
from __future__ import annotations


def extract_native_text(page) -> str:
    try:
        blocks = [b for b in page.get_text("blocks") if b[6] == 0]  # text blocks only
        blocks.sort(key=lambda b: (round(b[1], 1), b[0]))             # top-to-bottom, left-to-right
        parts = [b[4].strip() for b in blocks if b[4] and b[4].strip()]
        if parts:
            return "\n\n".join(parts)
    except Exception:
        pass
    return page.get_text("text") or ""
