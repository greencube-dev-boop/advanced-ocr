"""Per-book processing pipeline with resume support.

Per page:
  1. classify the page (text_only / visual / empty)
  2. extract the native text layer          -> temp/<book>/page-XXXX.native.md
  3. text_only: native text is the candidate, done
     visual:    render PNG -> Tesseract OCR  -> temp/.../page-XXXX.ocr.md
                suspicion check (confidence / garbage / mismatch / table)
                if suspicious -> nemotron-3.5-parse escalation -> page-XXXX.nemotron.md
  4. refine the candidate with the LLM      -> temp/.../page-XXXX.refined.md
  5. mark the page done in books.json
After all pages: assemble refined/<book>.md.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from . import hashing
from .assemble import assemble_book
from .ocr import suspicion
from .ocr.nemotron import NemotronClient, NemotronError
from .ocr.nemotron_postprocess import postprocess as nemotron_postprocess
from .ocr.tesseract_ocr import ocr_image, tesseract_available
from .pdfutils.analyze import analyze_page
from .pdfutils.render import render_page_png
from .pdfutils.text import extract_native_text
from .refine.refiner import Refiner
from .state import StateStore

log = logging.getLogger("ocr_pipeline.pipeline")


class Pipeline:
    def __init__(self, layout, cfg, keys, state: StateStore, refiner: Refiner):
        self.layout = layout
        self.cfg = cfg
        self.keys = keys
        self.state = state
        self.refiner = refiner
        self.nemotron: NemotronClient | None = None
        nk = cfg.section("nemotron")
        if keys.nvidia:
            self.nemotron = NemotronClient(
                api_key=keys.nvidia,
                endpoint=nk.get("endpoint",
                                default="https://ai.api.nvidia.com/v1/retrieval/nvidia/nemotron-3.5-parse/parse"),
                max_output_tokens=nk.get("max_output_tokens", default=4096),
                timeout=nk.get("timeout_sec", default=180),
                max_retries=nk.get("max_retries", default=4),
                backoff=nk.get("backoff_sec", default=5.0),
            )
        else:
            log.warning("No NVIDIA_API_KEY in api.txt — nemotron escalation is disabled")

    # ------------------------------------------------------------------
    def run_all(self) -> dict:
        pdfs = self.layout.list_pdfs()
        if not pdfs:
            log.info("No PDFs in %s — drop raw PDFs into raw/ and rerun.", self.layout.raw)
            return {}
        summary = {}
        for pdf in pdfs:
            summary[pdf.name] = self.run_book(pdf)
        return summary

    def run_book(self, pdf_path: Path) -> dict:
        import fitz  # PyMuPDF — imported lazily so pure modules stay testable

        name = pdf_path.name
        stem = pdf_path.stem
        sha = hashing.sha256_file(pdf_path)
        doc = fitz.open(str(pdf_path))
        total = doc.page_count
        self.state.register_book(name, sha, total)
        log.info("Book %s: %d pages, sha=%s..., resume state %s", name, total, sha[:12],
                 self.state.counts(name))

        btemp = self.layout.book_temp(stem)
        btemp.mkdir(parents=True, exist_ok=True)

        tess_ok = self.cfg.get("tesseract", "enabled", default=True) and tesseract_available()
        if not tess_ok:
            log.warning("Tesseract unavailable — visual pages will go native-text -> nemotron only")
        dpi = self.cfg.get("pdf_render_dpi", default=200)

        for page_no in range(1, total + 1):
            if self.state.page(name, page_no).get("status") == "done":
                continue  # resume: skip finished pages
            try:
                self._process_page(doc, name, btemp, page_no, total, tess_ok, dpi)
            except Exception as e:
                log.exception("Page %d of %s failed: %s", page_no, name, e)
                self.state.set_page(name, page_no, "error", error=str(e))
        doc.close()

        counts = self.state.counts(name)
        out = self.layout.refined / f"{stem}.md"
        assemble_book(btemp, out, stem,
                      {"source": name, "sha256": sha, "counts": counts},
                      add_page_markers=self.cfg.get("output", "add_page_markers", default=True),
                      page_separator=self.cfg.get("output", "page_separator", default="---"))
        status = "done" if counts.get("done", 0) == total else "partial"
        self.state.set_book_status(name, status, output=str(out))
        log.info("Book %s -> %s [%s] %s", name, out, status, counts)
        return {"book": name, "output": str(out), "status": status, "counts": counts}

    # ------------------------------------------------------------------
    def _process_page(self, doc, book: str, btemp: Path, page_no: int, total: int,
                      tess_ok: bool, dpi: int) -> None:
        page = doc[page_no - 1]
        pbase = btemp / f"page-{page_no:04d}"
        pcfg = self.cfg.section("page")

        info = analyze_page(
            page, page_no,
            min_text_chars=pcfg.get("min_text_chars", default=20),
            dense_drawings=pcfg.get("dense_drawings", default=40),
            large_image_area_ratio=pcfg.get("large_image_area_ratio", default=0.01),
        )
        native = extract_native_text(page)
        pbase.with_suffix(".native.md").write_text(native, encoding="utf-8")

        engines: list[str] = []
        candidate = native
        ocr_text, mean_conf = "", None
        susp_reasons: list[str] = []
        nemo_md = ""

        if info.kind == "text_only":
            engines.append("native")
        else:
            png = render_page_png(page, dpi=dpi)
            pbase.with_suffix(".png").write_bytes(png)
            if tess_ok:
                tcfg = self.cfg.section("tesseract")
                res = ocr_image(png, lang=tcfg.get("lang", default="eng"),
                                psm=tcfg.get("psm", default=3))
                ocr_text, mean_conf = res.text, res.mean_conf
                pbase.with_suffix(".ocr.md").write_text(ocr_text, encoding="utf-8")
                engines.append("tesseract")

            suspicious, susp_reasons = suspicion.evaluate(
                native, ocr_text, mean_conf,
                has_table=info.has_table or info.has_dense_drawings,
                cfg=self.cfg,
            )
            if suspicious and self.nemotron is not None:
                try:
                    raw = self.nemotron.parse_page(png, mime="image/png")
                    nemo_md = nemotron_postprocess(
                        raw.get("content", ""),
                        table_format=self.cfg.get("nemotron", "table_format", default="markdown"),
                    )
                    pbase.with_suffix(".nemotron.md").write_text(nemo_md, encoding="utf-8")
                    engines.append("nemotron")
                except NemotronError as e:
                    log.warning("page %d: nemotron failed (%s) — falling back", page_no, e)
                    susp_reasons.append(f"nemotron-error:{e}")

            if nemo_md:
                candidate = nemo_md                       # vision parse wins on hard pages
            elif suspicious:
                candidate = ocr_text or native            # tesseract is the fallback
            elif native.strip():
                candidate = native                        # text layer was reliable
            else:
                candidate = ocr_text                      # scanned page, no text layer
            if "nemotron" not in engines and info.kind == "visual":
                engines.append("native" if native.strip() else "ocr-only")

        # ---- refine ---------------------------------------------------
        refined, refiner_engine = candidate, "none"
        if self.cfg.get("refiner", "enabled", default=True) and self.refiner is not None:
            if self.refiner.available():
                try:
                    refined, refiner_engine = self.refiner.refine_page(candidate, page_no, total)
                except Exception as e:
                    log.warning("page %d: refinement failed (%s); keeping pre-refine text",
                                page_no, e)
                    refined, refiner_engine = candidate, "refine-failed"
            elif self.cfg.get("refiner", "skip_if_unavailable", default=True):
                log.warning("page %d: refiner backend unavailable; keeping pre-refine text", page_no)

        pbase.with_suffix(".refined.md").write_text(refined, encoding="utf-8")
        meta = {
            "page": page_no,
            "kind": info.kind,
            "info": info.to_dict(),
            "engines": engines,
            "suspicion": susp_reasons,
            "refiner": refiner_engine,
            "chars_in": len(candidate),
            "chars_out": len(refined),
        }
        pbase.with_suffix(".meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        self.state.set_page(book, page_no, "done", engine="+".join(engines),
                            extra={"kind": info.kind, "refiner": refiner_engine,
                                   "suspicion": susp_reasons})
        log.info("page %d/%d [%s] engines=%s refiner=%s suspicion=%s",
                 page_no, total, info.kind, "+".join(engines), refiner_engine, susp_reasons)
