"""Combine kept per-page markdown files into one PURE training file.

The output is training data for a scientific LLM: body content only — no
header block, no title line, no page markers, no separators. All provenance
lives in the ledger and the sidecar record.
"""
from __future__ import annotations

from pathlib import Path


def assemble_book(book_temp_dir: Path, out_path: Path) -> Path:
    pages = sorted(book_temp_dir.glob("page-*.final.md"))
    chunks = [p.read_text(encoding="utf-8").strip() for p in pages]
    text = "\n\n".join(c for c in chunks if c).strip() + "\n"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(out_path)  # atomic
    return out_path
