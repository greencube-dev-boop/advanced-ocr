"""Advanced OCR pipeline v3: PDF -> 100% OXR-1.0 page parsing -> one combined
Markdown file per book.

Every page of every PDF is sent to SoMarkAI/OXR-1.0 (the native text layer is
not used), so the whole book gets uniform OXR structure: headings, tables
(Markdown), formulas ($$…$$ LaTeX), chemical equations (mhchem), chemical
structures (SMILES) and code blocks. raw/ is scanned recursively, so books can
live in any subfolder system; refined/ and temp/ mirror that structure.

The Colab notebook is only a runner; all code lives in this repo and all data
lives in Google Drive under "OCR Engine/".
"""

__version__ = "0.4.0"
