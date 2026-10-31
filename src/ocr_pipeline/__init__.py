"""Advanced OCR pipeline v2: PDF -> per-page native text (text-only pages) /
OXR-1.0 parsing (visual pages) -> one combined Markdown file per book.

OXR (SoMarkAI/OXR-1.0) handles tables (Markdown), formulas (LaTeX), chemical
equations (mhchem), chemical structures (SMILES) and code blocks natively, so
the old Tesseract / nemotron-3.5-parse / LLM-refiner stack is gone.

The Colab notebook is only a runner; all code lives in this repo and all data
lives in Google Drive under "OCR Engine/".
"""

__version__ = "0.2.0"
