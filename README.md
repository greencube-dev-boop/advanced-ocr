# Advanced OCR (v3 — 100% OXR-1.0)

PDF books → clean, individual Markdown files. Runs on **Google Colab**, keeps **all code in this GitHub repo** and **all data in Google Drive**, and is fully **resumable** — stop anytime, rerun, continue where it left off. `raw/` is scanned **recursively**, so you can organize PDFs in any subfolder system; `refined/` mirrors that structure.

v3 uses **[SoMarkAI/OXR-1.0](https://github.com/SoMarkAI/OXR)**, a 1.3B document-parsing model, as the **only** engine: **every page of every PDF is parsed by OXR** (the native PDF text layer is not used at all), so the whole book gets uniform OXR structure — tables (Markdown), formulas (LaTeX `$$…$$`), chemical equations (mhchem), chemical structures (SMILES), code blocks, titles and reading order. **No API keys are needed at all.**

## Architecture

```
raw PDF (Drive: OCR Engine/raw/*.pdf)
   │  100% of pages go to OXR (page-level resume in books.json)
   ▼
pending pages → contiguous chunks (≤20 pages AND ≤9 MiB per chunk;
                an oversized single page → rendered 200dpi PNG)
   → POST {oxr}/parse/sync (output_formats=json)
   → OXR JSON blocks → per-page Markdown (temp/<book>/page-XXXX.final.md)
   → refined/<book>.md   (one combined Markdown per book)
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

> **Runtime note: a CPU-only Colab runtime is enough.** In `free_api` mode the OXR model runs on
> SoMark's servers — your runtime only slices PDFs and makes HTTP calls, so GPU VRAM stays at zero.
> A GPU is used **only** in `oxr.mode=local` (self-hosted), which needs a ≥24GB BF16 NVIDIA GPU.

- **`free_api`** (default): `https://free-oxr.somark.ai/api/v1` — **no key, no sign-up**. Limits per request: 10 MB / 20 pages (our chunker respects both; single oversized pages are sent as rendered PNGs), requests are queued and best-effort. Works on any Colab runtime, CPU included.
- **`local`** (self-hosted OXR): point `base_url` (or `oxr.local_base_url`) at your own OXR service, e.g. `http://127.0.0.1:8000/v1` — same SoMark-compatible `/parse/sync` contract, no limits. Requirements per the [OXR repo](https://github.com/SoMarkAI/OXR): Linux, **NVIDIA GPU ≥24 GB with BF16** (a free Colab T4 does NOT qualify; Colab Pro A100/L4-class does), CUDA 13.0 driver, Python 3.12. Deployment: Docker all-in-one, or the three-service source setup (vLLM + out-of-tree plugin on :8080, layout model on :8081, `oxr serve` on :8000).
- `oxr.base_url` overrides whichever mode default; `oxr.api_key` stays `dummy` unless you aim it at SoMark production.

## Drive layout

```
OCR Engine/
  raw/       raw PDF files — scanned recursively, so any subfolder system works:
             raw/mybook.pdf, raw/math/algebra/book.pdf, ...
  config/
    pipeline.json   runtime settings (written by `init-config`, then editable in Drive)
    books.json      resume manifest — do not edit by hand
  refined/     final output, mirroring the raw/ structure:
               raw/math/algebra/book.pdf -> refined/math/algebra/book.md
  temp/        per-book working files, mirrored too (temp/math/algebra/book/):
    page-0001.final.md      final per-page Markdown (OXR JSON blocks -> Markdown)
    p0001-0020.oxr.json     OXR JSON output (per-chunk source of truth)
    p0001-0020.oxr.md       OXR markdown output (debug reference)
    page-0001.meta.json     decision trail (engine, chunk, chars)
```

A book's identity is its path relative to `raw/` (e.g. `math/algebra/book.pdf`), so same-named PDFs in different subfolders never collide.

## Usage on Colab

Open `notebooks/colab_runner.ipynb` from this repo in Colab and run the cells top to bottom: install deps → mount Drive → sync repo (`git pull`, so pushing updates to GitHub is enough — never re-upload the notebook) → `doctor` → `run`. Drop PDFs into **Drive → `OCR Engine/raw/`** first.

The notebook cells call the CLI:

```bash
python -m ocr_pipeline doctor        --drive-root "/content/drive/MyDrive/OCR Engine"
python -m ocr_pipeline init-config   --drive-root ...   # once: write default settings to Drive
python -m ocr_pipeline status        --drive-root ...   # show resume state
python -m ocr_pipeline run           --drive-root ...   # process all pending PDFs in raw/ (resumable, recursive)
python -m ocr_pipeline run-book math/algebra/book.pdf --drive-root ...   # path relative to raw/
python -m ocr_pipeline reset math/algebra/book.pdf    --drive-root ...   # forget state + temp, reprocess
```

- **Resume**: state lives in `OCR Engine/config/books.json`. A page is only marked `done` after its final markdown is on disk, so rerunning redoes only the unfinished pages. Failed OXR chunks are retried on the next run (up to `chunking.chunk_retries`, then the pages are marked `error`). If a PDF changes on disk (sha256 mismatch), its page states reset.
- **Upgrading from v1/v2**: books processed by earlier versions keep their old states (v2 pages extracted natively stay native). To re-convert a book 100% with OXR, run `reset <book>` and rerun.

## Config reference

See `config/pipeline.example.json` (same content as what `init-config` writes to Drive). Notable knobs:

- `oxr.mode`, `oxr.base_url` — free API vs self-hosted (see above)
- `oxr.element_formats` — `image: none|base64|url`, `formula: latex`, `table: markdown|html`, `cs: smiles|image`
- `oxr.extra` — OXR-only sampling: `temperature` / `top_p` / `repetition_penalty`
- `chunking.max_pages` (free-API ceiling 20), `chunking.max_bytes` (ceiling 10 MB), `chunking.render_dpi`, `chunking.chunk_retries`
- `chunking.max_concurrent` — parallel chunk uploads (default 2). Each request is independent and resume is page-level, so raising it to 3–4 can cut wall time further — but the free API is queued and best-effort, so raise it gently and back off if you see 429s.
- `output.figure_marker` — emit `*[figure omitted]*` where a picture was dropped

## Local development

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v     # stdlib-only unit tests
python -m ocr_pipeline doctor --drive-root "/path/to/OCR Engine"
```
