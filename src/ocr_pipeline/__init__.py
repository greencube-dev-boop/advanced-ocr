"""Advanced OCR pipeline v0.6.0 — curated training-corpus builder.

Every page of every PDF (raw/ scanned recursively) is parsed by
SoMarkAI/OXR-1.0 (no keys; free API by default). A deterministic curation pass
then keeps only body content — front matter, ToC pages, acknowledgements,
preface/foreword, references/bibliography, index, appendices, blank and
figure-only pages are dropped — and each book assembles into one PURE
Markdown file for training a scientific LLM.

Provenance: books.json is a permanent ledger (sha256, DOI, ISBN, drop
breakdown) with a sha pre-flight that skips duplicate uploads; each refined
file gets a sidecar <book>.record.json. Lifecycle: completed books have their
temp dir pruned and source PDF archived to raw/_done/; books that end
incomplete on `lifecycle.failed_run_limit` runs are quarantined to
raw/_failed/.

The Colab notebook is only a runner; all code lives in this repo and all data
lives in Google Drive under "OCR Engine/".
"""

__version__ = "0.6.0"
