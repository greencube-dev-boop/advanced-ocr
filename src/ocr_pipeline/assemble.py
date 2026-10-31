"""Combine per-page refined markdown files into one book-level .md file."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path


def assemble_book(book_temp_dir: Path, out_path: Path, title: str, meta: dict,
                  add_page_markers: bool = True, page_separator: str = "---") -> Path:
    pages = sorted(book_temp_dir.glob("page-*.refined.md"))
    sep = f"\n\n{page_separator}\n\n"
    chunks: list[str] = []
    for p in pages:
        m = re.match(r"page-(\d+)\.refined\.md$", p.name)
        no = int(m.group(1)) if m else 0
        body = p.read_text(encoding="utf-8").strip()
        if not body:
            continue
        chunks.append(f"<!-- page {no} -->\n\n{body}" if add_page_markers else body)

    counts = meta.get("counts", {})
    header = "\n".join([
        f"# {title}",
        "",
        f"- source PDF: `{meta.get('source', '')}`",
        f"- sha256: `{meta.get('sha256', '')}`",
        f"- pages processed: {len(pages)} (state: {counts})",
        f"- generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "- pipeline: advanced-ocr",
        "",
    ])
    text = header + sep.join(chunks) + "\n"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(out_path)  # atomic
    return out_path
