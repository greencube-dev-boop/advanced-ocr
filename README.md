# Advanced OCR (v0.6.0 — curated training-corpus builder)

Turn a bulk archive of PDF books into **clean per-book Markdown files for training a scientific LLM**. Runs on **Google Colab (CPU is enough)**, keeps **all code in this GitHub repo** and **all data in Google Drive**, and is fully **resumable** — stop anytime, rerun, continue where it left off.

Every page of every PDF is parsed by **[SoMarkAI/OXR-1.0](https://github.com/SoMarkAI/OXR)** (tables → Markdown, formulas → LaTeX, chemical equations → mhchem, structures → SMILES, code → fences). A deterministic **curation pass** then keeps only body content, each book assembles into **one pure-content `.md` file**, and a **permanent ledger** records exactly what was done. **No API keys anywhere.**

## Architecture

```
raw/<any subfolders>/*.pdf
   │  intake gate: sha256 checked against the ledger — duplicates skipped & archived
   ▼
100% of pages → chunks (≤20 pages, ≤9 MiB; oversized page → rendered PNG)
   → POST {oxr}/parse/sync (output_formats=json), up to max_concurrent in flight
   → OXR JSON blocks → per-page Markdown (temp/<book>/page-XXXX.oxr.md)
   ▼
curation pass (deterministic, config-editable)
   DROP: title/copyright/dedication pages · printed ToC pages · acknowledgements ·
         preface/foreword · references/bibliography · index · appendices ·
         blank pages · figure-only pages · captions of dropped figures
   KEEP: body chapters only
   ▼
refined/<same relative path>/<book>.md   ← PURE content: no header, no markers
<book>.record.json (sidecar)             ← provenance next to every file
books.json (permanent ledger)            ← sha256, DOI, ISBN, drop breakdown
   ▼  lifecycle (only when 100% of pages are converted or dropped)
temp/<book>/ pruned · source PDF → raw/_done/<same relative path>
(books failing 5 runs → raw/_failed/)
```

## Curation

The refined file is **training data, not a readable book** — anything that is not body content is dropped silently and completely:

- **Section ranges**: a top-level heading matching a drop pattern ("References", "Appendix A", "Preface", …) drops everything until the next kept top-level heading — dense bibliography pages would never survive a per-page check.
- **Page rules**: dot-leader ToC pages, front-matter boilerplate (copyright/ISBN/publisher), blank pages, figure-only pages (figures themselves are always omitted — OXR's API does no image understanding).
- **Front-matter boundary**: pages before the book's first kept heading are dropped (guarded — the rule only fires when such a heading actually exists early enough).
- Everything is **config-editable** (`curation.*` in the Drive config) and **auditable**: each page's verdict lands in its `meta.json` and the ledger's drop breakdown (reason → page numbers). `reset <book>` re-does a book if a rule ever misfires.

## Ledger & lifecycle

- **`config/books.json` is a permanent ledger**: for every processed book it keeps sha256, extracted **DOI** and **ISBN** (regex-parsed from the pages — front-matter pages are parsed before being dropped), detected title, pages kept/dropped with reasons, output path, timestamps. Done entries are never pruned (except by an explicit `reset`).
- **Sidecar**: `refined/<…>/<book>.record.json` mirrors the record next to each output file, so dataset assembly doesn't need to parse the ledger.
- **Intake gate**: a PDF whose sha256 already exists in the ledger (as done or failed) is skipped automatically and its file archived — re-uploaded duplicates never cost API calls.
- **Lifecycle**: when 100% of a book's pages are converted *or* dropped → `temp/<book>/` is pruned and the source PDF moves to `raw/_done/<same relative path>`. Books that end incomplete on `lifecycle.failed_run_limit` (default 5) runs are quarantined to `raw/_failed/`. Partial books simply stay in `raw/` and resume.

## Drive layout

```
OCR Engine/
  raw/           intake queue, scanned recursively (any subfolder system)
    _done/       archived source PDFs of completed books (mirrors raw/ structure)
    _failed/     quarantined sources of books that kept failing
  config/
    pipeline.json   settings (written by `init-config`, editable in Drive)
    books.json      permanent ledger — do not edit by hand
  refined/        output, mirroring raw/: book.md + book.record.json per source
  temp/<book>/    working files, pruned automatically on completion:
    p0001-0020.oxr.json/.oxr.md   OXR chunk outputs (debug)
    page-XXXX.oxr.md              parsed page (pre-curation)
    page-XXXX.final.md            kept page (post-curation)
    page-XXXX.meta.json           engine, chunk, verdict, drop reason
```

## Usage on Colab

Open `notebooks/colab_runner.ipynb` from this repo in Colab (**CPU runtime is enough** — the OXR model runs on SoMark's servers) and run the cells top to bottom: install deps → mount Drive → sync repo (`git pull`, so pushing updates to GitHub is enough) → `doctor` → `run`. Drop PDFs into **any subfolder of `Drive/OCR Engine/raw/`**.

```bash
python -m ocr_pipeline doctor      --drive-root "/content/drive/MyDrive/OCR Engine"
python -m ocr_pipeline init-config --drive-root ...   # once: write default settings to Drive
python -m ocr_pipeline status      --drive-root ...   # ledger summary (kept/dropped/DOI per book)
python -m ocr_pipeline run         --drive-root ...   # process all pending PDFs (resumable)
python -m ocr_pipeline run-book math/algebra/book.pdf --drive-root ...   # path relative to raw/
python -m ocr_pipeline reset math/algebra/book.pdf    --drive-root ...   # forget state+ledger entry, reprocess
```

Note on `reset`: it clears the book's ledger entry and temp dir. If the source was already archived to `raw/_done/`, move it back into `raw/` first.

## OXR modes (`oxr.mode`)

> **Runtime note: a CPU-only Colab runtime is enough.** In `free_api` mode the OXR model runs on SoMark's servers — your runtime only slices PDFs and makes HTTP calls. A GPU is used **only** in `oxr.mode=local` (self-hosted, ≥24GB BF16 NVIDIA GPU, Colab Pro L4/A100-class).

- **`free_api`** (default): `https://free-oxr.somark.ai/api/v1` — no key, no sign-up. Limits: 10MB / 20 pages per request (the chunker respects both), queued and best-effort. *Contingency:* the free API is explicitly not-for-production; if it ever degrades, the paths are (a) self-hosted OXR per the OXR repo, or (b) a paid SoMark key — the client already speaks that API, only `oxr.base_url`/`api_key` change.
- **`local`**: point `oxr.local_base_url` at your own OXR service (`http://127.0.0.1:8000/v1`), no limits.

## Config reference

See `config/pipeline.example.json` (= what `init-config` writes to Drive). Groups: `oxr.*` (mode/URL/formats/sampling), `chunking.*` (max_pages, max_bytes, max_concurrent, render_dpi, chunk_retries), `curation.*` (enabled, section_drop, section_drop_prefix, page_drop, front-matter boundary + ToC thresholds), `lifecycle.*` (prune_temp, done_folder, failed_folder, delete_source, failed_run_limit), `output.figure_marker`.

## Development

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v     # stdlib-only unit tests
python -m ocr_pipeline doctor --drive-root "/path/to/OCR Engine"
```
