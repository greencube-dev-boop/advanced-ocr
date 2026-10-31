"""Google Drive layout for the OCR Engine.

<drive-root>/            e.g. /content/drive/MyDrive/OCR Engine
  raw/       raw PDF files (drop them here; the pipeline picks them up)
  config/    pipeline.json (settings) + books.json (resume manifest)
  refined/   final combined .md per book
  temp/      per-book working dir: page PNGs + intermediate .md files
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
        return sorted(p for p in self.raw.glob("*.pdf") if p.is_file())
