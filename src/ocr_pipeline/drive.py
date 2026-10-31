"""Google Drive layout for the OCR Engine.

<drive-root>/            e.g. /content/drive/MyDrive/OCR Engine
  raw/       raw PDF files, in any folder depth (e.g. raw/math/algebra/book.pdf)
  config/    pipeline.json (settings) + books.json (resume manifest)
  refined/   final combined .md per book — mirrors the raw/ folder structure
  temp/      per-book working dir: page PNGs + intermediate files, same mirroring

A book's identity is its path relative to raw/ (posix, e.g. "math/algebra/book.pdf"),
so same-named PDFs in different folders never collide.
"""
from __future__ import annotations

from pathlib import Path


class DriveLayout:
    def __init__(self, root):
        self.root = Path(root)

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def config(self) -> Path:
        return self.root / "config"

    @property
    def refined(self) -> Path:
        return self.root / "refined"

    @property
    def temp(self) -> Path:
        return self.root / "temp"

    @property
    def books_json(self) -> Path:
        return self.config / "books.json"

    @property
    def pipeline_json(self) -> Path:
        return self.config / "pipeline.json"

    def book_temp(self, stem: str) -> Path:
        return self.temp / stem

    def ensure(self) -> None:
        for d in (self.raw, self.config, self.refined, self.temp):
            d.mkdir(parents=True, exist_ok=True)

    def list_pdfs(self) -> list[Path]:
        """All PDFs under raw/ at any folder depth (extension match is case-insensitive)."""
        return sorted(
            p for p in self.raw.rglob("*")
            if p.is_file() and p.suffix.lower() == ".pdf"
        )

    def rel_id(self, pdf_path: Path) -> str:
        """Book identity: path relative to raw/ in posix form ('sub/dir/book.pdf')."""
        return pdf_path.relative_to(self.raw).as_posix()

    def rel_stem(self, pdf_path: Path) -> str:
        """Relative id without the .pdf suffix ('sub/dir/book') for temp/refined paths."""
        return pdf_path.relative_to(self.raw).with_suffix("").as_posix()

    def display_name(self, pdf_path: Path) -> str:
        """Leaf file name without suffix, used as the Markdown H1 title."""
        return pdf_path.stem
