# Advanced OCR (v2 — OXR-1.0)

PDF book → clean, single Markdown file. Runs on **Google Colab**, keeps **all code in this GitHub repo** and **all data in Google Drive**, and is fully **resumable** — stop anytime, rerun, continue where it left off.

v2 replaces the v1 engine stack (Tesseract + nemotron-3.5-parse + LLM refiner) with **[SoMarkAI/OXR-1.0](https://github.com/SoMarkAI/OXR)**, a 1.3B document-parsing model that natively emits Markdown — tables (Markdown/HTML), formulas (LaTeX), chemical equations (mhchem), chemical structures (SMILES), code blocks, titles and reading order. **No API keys are needed at all.**

## Architecture

```
raw PDF (Drive: OCR Engine/raw/*.pdf)
   │  page-by-page analysis (PyMuPDF)
   ▼
page classification
   ├─ text_only ──► native text layer (exact, free, no OCR) ──────────────┐
   ├─ empty ──────► blank page ───────────────────────────────────────────┤
   └─ visual ─────► contiguous pending pages → adaptive chunks
                    (≤20 pages AND ≤9 MiB per chunk; oversized single page
                     → rendered 200dpi PNG)
                    → POST {oxr}/parse/sync (output_formats=json)
                    → OXR JSON blocks → per-page Markdown ─────────────────┤
                                                                           ▼
                                                refined/<book>.md  (one combined Markdown)
```

**OXR element handling** (per page, from the `json` output):

| OXR element | Goes into the Markdown as |
| --- | --- |
| Title | `#`-heading (uses OXR's `title_level`) |
| Text / Caption / Footnote / Stamp | plain paragraph |
| Table | Markdown table (element_formats `table=markdown`) |
| Formula | `$$…$$` LaTeX |
| Chemical Equation | `$$\ce{…}$$` mhchem LaTeX |
| Chemical Structure | SMILES inline code |
| Code Block | fenced code block |
| Picture | dropped (OXR's API has no image understanding; non-textifiable images are ignored) — or `*[figure omitted]*` marker with `output.figure_marker: true` |
| Page header/footer | dropped (`keep_header_footer: false`) |

> Note: unlike the v1 Gemini-refiner path, flowcharts/diagrams that are pure graphics cannot be converted to text any more — OXR's API does not support image understanding. Everything with textual content (tables, formulas, chemistry, code) is captured as text.

## OXR modes (`oxr.mode` in the Drive config)

- **`free_api`** (default): `https://free-oxr.somark.ai/api/v1` — **no key, no sign-up**. Limits per request: 10 MB / 20 pages (our chunker respects both; single oversized pages are sent as rendered PNGs), requests are queued and best-effort. Works on any Colab runtime, CPU included.
- **`local`** (self-hosted OXR): point `base_url` (or `oxr.local_base_url`) at your own OXR service, e.g. `http://127.0.0.1:8000/v1` — same SoMark-compatible `/parse/sync` contract, no limits. Requirements per the [OXR repo](https://github.com/SoMarkAI/OXR): Linux, **NVIDIA GPU ≥24 GB with BF16** (a free Colab T4 does NOT qualify; Colab Pro A100/L4-class does), CUDA 13.0 driver, Python 3.12. Deployment: Docker all-in-one, or the three-service source setup (vLLM + out-of-tree plugin on :8080, layout model on :8081, `oxr serve` on :8000).
- `oxr.base_url` overrides whichever mode default; `oxr.api_key` stays `dummy` unless you aim it at SoMark production.

## Drive layout

```
OCR Engine/
  raw/       drop raw PDFs here            (e.g. raw/mybook.pdf)
  config/
    pipeline.json   runtime settings (written by `init-config`, then editable in Drive)
    books.json      resume manifest — do not edit by hand
  refined/     final output: mybook.md
  temp/        per-book working files:
    mybook/
      p0001-0020.oxr.json     OXR JSON output (per-chunk source of truth)
      p0001-0020.oxr.md       OXR markdown output (debug reference)
      page-0001.final.md      final per-page Markdown (from native text or OXR)
      page-0001.meta.json     decision trail (kind, engine, chunk, chars)
```

## Usage on Colab

Open `notebooks/colab_runner.ipynb` from this repo in Colab and run the cells top to bottom: install deps → mount Drive → sync repo (`git pull`, so pushing updates to GitHub is enough — never re-upload the notebook) → `doctor` → `run`. Drop PDFs into **Drive → `OCR Engine/raw/`** first.

The notebook cells call the CLI:

```bash
python -m ocr_pipeline doctor        --drive-root "/content/drive/MyDrive/OCR Engine"
python -m ocr_pipeline init-config   --drive-root ...   # once: write default settings to Drive
python -m ocr_pipeline status        --drive-root ...   # show resume state
python -m ocr_pipeline run           --drive-root ...   # process all pending PDFs (resumable)
python -m ocr_pipeline run-book mybook.pdf --drive-root ...
python -m ocr_pipeline reset mybook.pdf    --drive-root ...   # forget state + temp, reprocess from scratch
```

- **Resume**: state lives in `OCR Engine/config/books.json`. A page is only marked `done` after its final markdown is on disk, so rerunning redoes only the unfinished pages. Failed OXR chunks are retried on the next run (up to `chunking.chunk_retries`, then the pages are marked `error`). If a PDF changes on disk (sha256 mismatch), its page states reset.
- **Upgrading from v1**: books processed by v1 keep their old states. To reprocess a book with OXR, run `reset <book>` and rerun.

## Config reference

See `config/pipeline.example.json` (same content as what `init-config` writes to Drive). Notable knobs:

- `oxr.mode`, `oxr.base_url` — free API vs self-hosted (see above)
- `oxr.element_formats` — `image: none|base64|url`, `formula: latex`, `table: markdown|html`, `cs: smiles|image`
- `oxr.extra` — OXR-only sampling: `temperature` / `top_p` / `repetition_penalty`
- `chunking.max_pages` (free-API ceiling 20), `chunking.max_bytes` (ceiling 10 MB), `chunking.render_dpi`, `chunking.chunk_retries`
- `page.*` — text_only / visual / empty classification thresholds
- `output.figure_marker` — emit `*[figure omitted]*` where a picture was dropped

## Local development

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v     # stdlib-only unit tests
python -m ocr_pipeline doctor --drive-root "/path/to/OCR Engine"
```
