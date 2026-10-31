# Advanced OCR

PDF book → clean, single Markdown file. Runs on **Google Colab** (free GPU), keeps **all code in this GitHub repo** and **all data in Google Drive**, and is fully **resumable** — stop anytime, rerun, continue where it left off.

## Architecture

```
raw PDF (Drive: OCR Engine/raw/*.pdf)
   │  page-by-page loop (PyMuPDF)
   ▼
page classification ── text_only ──► native text layer ─────────────┐
   │ visual (images/tables/flowcharts)                              │
   ▼                                                               ▼
render page PNG ─► Tesseract OCR ─► suspicion check ──clean──► native text
                                        │ suspicious / table
                                        ▼
                              nemotron-3.5-parse (NVIDIA API, escalation)
                                        │
                                        ▼
                              LLM refinement (keyless Gemma 3 on Colab GPU by default,
                              or Gemini API when a key is filled in)
                                        │
                                        ▼
                              temp/<book>/page-XXXX.refined.md
                                        │
                                        ▼
                              refined/<book>.md   (one combined Markdown per book)
```

- **Text-only pages** never touch OCR — the PDF text layer is exact.
- **Visual pages**: native text is kept, the page is rendered to PNG, Tesseract runs as a cross-check. If Tesseract's output looks suspicious (low confidence, garbage characters, big length mismatch with the native text) **or a table/diagram is detected**, the page is escalated to **nemotron-3.5-parse**, whose output (tables come back as LaTeX `tabular`, converted to Markdown tables) wins.
- **Refinement** turns the raw per-page content into clean Markdown: tables → Markdown tables, equations → LaTeX (`$...$` / `$$...$$`), flowcharts → numbered steps or `mermaid` blocks, OCR noise removed. Figures without extractable text are ignored. Original language is preserved.
- **Resume**: state lives in `OCR Engine/config/books.json`. A page is only marked `done` after its refined file is on disk, so rerunning the notebook simply redoes the unfinished pages. If a PDF changes on disk (sha256 mismatch), its page states reset.

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
      page-0001.native.md     PDF text layer
      page-0001.png           rendered page image
      page-0001.ocr.md        Tesseract output
      page-0001.nemotron.md   nemotron-3.5-parse output (post-processed)
      page-0001.refined.md    final per-page Markdown
      page-0001.meta.json     decision trail (kind, engines, suspicion reasons)
```

## Setup (once)

1. Open `notebooks/colab_runner.ipynb` from this repo in Colab (*File → Open notebook → GitHub*, or "Open in Colab" badge).
2. Run the **install** cell and the **mount Drive** cell.
3. Run the **sync repo** cell. The repo is private — either paste a GitHub PAT (read access) into the `GITHUB_TOKEN` cell variable, or make the repo public.
4. Run the **api.txt** cell and upload your `api.txt`:
   ```
   nvapi-...            <- line 1: NVIDIA key (nemotron-3.5-parse)
   GEMINI_API_KEY=...   <- optional, only if you later want the Gemini backend
   HF_TOKEN=...         <- optional, only if your HF account hasn't accepted the Gemma license
   ```
   The file is used only in the Colab runtime — it is never written to Drive or git.
5. Put PDFs into **Drive → `OCR Engine/raw/`**.
6. Run `doctor` to verify everything, then `run`.

## Usage on Colab

The notebook cells call the CLI (all code comes from the repo via `git pull`, so pushing updates to GitHub is enough — never re-upload the notebook):

```bash
python -m ocr_pipeline doctor        --drive-root "/content/drive/MyDrive/OCR Engine" --api-file /content/api.txt
python -m ocr_pipeline init-config   --drive-root ...   # once: write default settings to Drive
python -m ocr_pipeline status        --drive-root ...   # show resume state
python -m ocr_pipeline run           --drive-root ...   # process all pending PDFs (resumable)
python -m ocr_pipeline run-book mybook.pdf --drive-root ...
```

Outputs:

- combined book: `Drive/OCR Engine/refined/<book>.md`
- per-page trail: `Drive/OCR Engine/temp/<book>/`
- state: `Drive/OCR Engine/config/books.json`

To **resume after a disconnect**: just rerun the same notebook cells — finished pages are skipped.

## Refiner backends (no keyless Gemini exists)

Google's Gemini API strictly requires an API key (even the free tier), and Hugging Face inference is token-gated too. The default backend is therefore:

- **`gemma_local`** (default, keyless): `google/gemma-3-4b-it` in 4-bit (~3 GB VRAM) on the Colab T4 GPU via `transformers`. ~1–2 min/page, unattended overnight runs are fine thanks to resume. No key of any kind.
- **`gemini_api`**: implemented and ready — set `refiner.backend = "gemini_api"` in `config/pipeline.json` and put `GEMINI_API_KEY=...` in api.txt (or `api_key` in the config). The key slot is intentionally left empty for now. Free tier ≈ 15 RPM; the client self-throttles via `rpm`.

Switch backends by editing `OCR Engine/config/pipeline.json` on Drive — no code changes needed.

## OCR escalation order (per page)

`native text → Tesseract → nemotron-3.5-parse (only if tables/diagrams detected or Tesseract output suspicious)`

Suspicion triggers are configurable under `suspicion` in `pipeline.json` (min word confidence, garbage-char ratio, native-vs-OCR length mismatch). Tesseract language is configurable (`tesseract.lang`, e.g. `"eng+ben"`).

## Config reference

See `config/pipeline.example.json` — the same content is written to Drive by `init-config`. Notable knobs: `pdf_render_dpi`, `nemotron.max_output_tokens` (API ceiling 9000), `nemotron.table_format` (`markdown` | `latex`), `refiner.*`, `output.*`.

## Local development

```bash
pip install -r requirements.txt   # plus: apt-get install tesseract-ocr
python -m unittest discover -s tests -v     # stdlib-only unit tests
python -m ocr_pipeline doctor --drive-root "/path/to/OCR Engine" --api-file api.txt
```
