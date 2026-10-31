"""Advanced OCR pipeline: PDF -> per-page native text / Tesseract -> nemotron-3.5-parse
escalation -> LLM refinement -> one combined Markdown file per book.

The Colab notebook is only a runner; all code lives in this repo and all data
lives in Google Drive under "OCR Engine/".
"""

__version__ = "0.1.0"
