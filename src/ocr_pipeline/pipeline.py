"""Per-book processing pipeline v2 (OXR-only) with resume support.

Per book:
  1. analyze every page (text_only / visual / empty)
  2. text_only pages -> native PDF text layer is the final page markdown
     empty pages     -> blank page markdown
  3. visual pages -> grouped into contiguous chunks (<=20 pages AND <=9MiB;
     an oversized single page is rendered to PNG) -> POST {oxr}/parse/sync
     -> OXR JSON blocks -> per-page markdown
  4. every finished page lands in temp/<book>/page-XXXX.final.md and books.json
  5. assemble refined/<book>.md
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from . import hashing
from .assemble import assemble_book
from .ocr.oxr_blocks import pages_to_markdown
from .ocr.oxr_client import OxrClient, OxrError
from .pdfutils.analyze import analyze_page
from .pdfutils.chunking import paginate_range, split_range
from .pdfutils.render import render_page_png
from .pdfutils.text import extract_native_text
from .state import StateStore

log = logging.getLogger("ocr_pipeline.pipeline")


class Pipeline:
    def __init__(self, layout, cfg, state: StateStore):
        self.layout = layout
        self.cfg = cfg
        self.state = state
        o = cfg.section("oxr")
        self.mode = o.get("mode", default="free_api")
        base = (o.get("base_url", default="")
                or o.get("free_api_base_url" if self.mode == "free_api" else "local_base_url",
                         default="https://free-oxr.somark.ai/api/v1"))
        self.base_url = base
        self.oxr = OxrClient(
            base_url=base,
            api_key=o.get("api_key", default=""),
            timeout=o.get("timeout_sec", default=600),
            max_retries=o.get("max_retries", default=5),
            backoff=o.get("backoff_sec", default=10.0),
            element_formats=o.get("element_formats", default=None),
            feature_config=o.get("feature_config", default=None),
            extra=o.get("extra", default=None),
        )
        c = cfg.section("chunking")
        self.max_pages = int(c.get("max_pages", default=20))
        self.max_bytes = int(c.get("max_bytes", default=9437184))
        self.render_dpi = int(c.get("render_dpi", default=200))
        self.chunk_retries = int(c.get("chunk_retries", default=3))
        self.figure_marker = bool(cfg.get("output", "figure_marker", default=False))

    # ------------------------------------------------------------------
    def run_all(self) -> dict:
        pdfs = self.layout.list_pdfs()
        if not pdfs:
            log.info("No PDFs in %s — drop raw PDFs into raw/ and rerun.", self.layout.raw)
            return {}
        return {pdf.name: self.run_book(pdf) for pdf in pdfs}

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

        pcfg = self.cfg.section("page")
        infos = [
            analyze_page(doc[i], i + 1,
                         min_text_chars=pcfg.get("min_text_chars", default=20),
                         dense_drawings=pcfg.get("dense_drawings", default=40),
                         large_image_area_ratio=pcfg.get("large_image_area_ratio", default=0.01))
            for i in range(total)
        ]

        # phase 1: text-only pages need no OXR at all (exact native text layer)
        for page_no, info in enumerate(infos, start=1):
            if self.state.page(name, page_no).get("status") == "done":
                continue
            if info.kind == "text_only":
                self._finish_native(doc[page_no - 1], name, btemp, page_no, total, info)
            elif info.kind == "empty":
                self._finish_blank(name, btemp, page_no, total, info)

        # phase 2: remaining (visual) pages go through OXR in adaptive chunks
        pending = [p for p in range(1, total + 1)
                   if self.state.page(name, p).get("status") != "done"]
        for a0, b0 in _contiguous(pending):        # inclusive 0-based
            for ca, cb, payload, kind in self._build_chunks(doc, a0, b0):
                self._process_chunk(name, btemp, ca, cb, payload, kind, total)

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
    def _finish_native(self, page, book: str, btemp: Path, page_no: int, total: int, info) -> None:
        native = extract_native_text(page)
        pbase = btemp / f"page-{page_no:04d}"
        pbase.with_suffix(".final.md").write_text(native, encoding="utf-8")
        self._write_meta(pbase, {"page": page_no, "kind": info.kind, "engine": "native",
                                 "chars": len(native)})
        self.state.set_page(book, page_no, "done", engine="native", extra={"kind": info.kind})
        log.info("page %d/%d [%s] engine=native", page_no, total, info.kind)

    def _finish_blank(self, book: str, btemp: Path, page_no: int, total: int, info) -> None:
        pbase = btemp / f"page-{page_no:04d}"
        pbase.with_suffix(".final.md").write_text("", encoding="utf-8")
        self._write_meta(pbase, {"page": page_no, "kind": info.kind, "engine": "none", "chars": 0})
        self.state.set_page(book, page_no, "done", engine="none", extra={"kind": info.kind})
        log.info("page %d/%d [%s] engine=none (blank)", page_no, total, info.kind)

    # ---- chunk building ------------------------------------------------
    def _serialize(self, doc, a: int, b: int) -> bytes:
        import fitz

        out = fitz.open()
        try:
            out.insert_pdf(doc, from_page=a, to_page=b)
            return out.tobytes(deflate=True, garbage=3)
        finally:
            out.close()

    def _build_chunks(self, doc, a: int, b: int):
        """Yield (ca, cb, payload, kind) pieces covering [a, b] (inclusive 0-based)."""
        for ca, cb in paginate_range(a, b, self.max_pages):
            for piece in self._size_split(doc, ca, cb):
                yield piece

    def _size_split(self, doc, a: int, b: int):
        data = self._serialize(doc, a, b)
        if len(data) <= self.max_bytes:
            return [(a, b, data, "pdf")]
        if a == b:
            # a single page too big for the free-API limit -> send a rendered PNG
            log.warning("page %d exceeds %d bytes as PDF; sending rendered PNG instead",
                        a + 1, self.max_bytes)
            return [(a, b, render_page_png(doc[a], dpi=self.render_dpi), "image")]
        mid = (a + b) // 2
        return self._size_split(doc, a, mid) + self._size_split(doc, mid + 1, b)

    # ---- chunk processing ----------------------------------------------
    def _process_chunk(self, book: str, btemp: Path, ca: int, cb: int,
                       payload: bytes, kind: str, total: int) -> None:
        chunk_key = f"p{ca + 1:04d}-{cb + 1:04d}"
        if self.state.chunk_attempts(book, chunk_key) >= self.chunk_retries:
            for p in range(ca + 1, cb + 2):
                if self.state.page(book, p).get("status") != "done":
                    self.state.set_page(book, p, "error",
                                        error=f"chunk {chunk_key} failed {self.chunk_retries}+ times")
            return
        try:
            if kind == "pdf":
                resp = self.oxr.parse(payload, filename=f"{chunk_key}.pdf", mime="application/pdf")
            else:
                resp = self.oxr.parse(payload, filename=f"{chunk_key}.png", mime="image/png")
            outputs = ((resp.get("data") or {}).get("result") or {}).get("outputs") or {}
            pages_json = (outputs.get("json") or {}).get("pages")
            if not pages_json:
                raise OxrError("response carries no JSON pages output")
            (btemp / f"{chunk_key}.oxr.json").write_text(
                json.dumps(outputs.get("json"), ensure_ascii=False), encoding="utf-8")
            if outputs.get("markdown"):
                (btemp / f"{chunk_key}.oxr.md").write_text(
                    str(outputs["markdown"]), encoding="utf-8")

            md_by_page = pages_to_markdown(pages_json, page_base=ca)
            for p in range(ca + 1, cb + 2):
                if p not in md_by_page:
                    log.warning("chunk %s: OXR returned no blocks for page %d — left pending",
                                chunk_key, p)
                    continue
                pbase = btemp / f"page-{p:04d}"
                pbase.with_suffix(".final.md").write_text(md_by_page[p], encoding="utf-8")
                self._write_meta(pbase, {"page": p, "kind": "visual", "engine": f"oxr:{self.mode}",
                                         "chunk": chunk_key, "chars": len(md_by_page[p])})
                self.state.set_page(book, p, "done", engine=f"oxr:{self.mode}",
                                    extra={"chunk": chunk_key})
            self.state.record_chunk(book, chunk_key, error=None)
            log.info("chunk %s ok (engine=oxr:%s, %d pages)", chunk_key, self.mode, cb - ca + 1)
        except Exception as e:
            self.state.record_chunk(book, chunk_key, error=str(e))
            log.warning("chunk %s failed: %s", chunk_key, e)

    @staticmethod
    def _write_meta(pbase: Path, meta: dict) -> None:
        pbase.with_suffix(".meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def _contiguous(pages: list[int]) -> list[tuple[int, int]]:
    """[3,4,5,8] (1-based) -> [(2,4), (7,7)] inclusive 0-based ranges."""
    if not pages:
        return []
    ranges: list[tuple[int, int]] = []
    start = prev = pages[0] - 1
    for p in pages[1:]:
        p0 = p - 1
        if p0 == prev + 1:
            prev = p0
        else:
            ranges.append((start, prev))
            start = prev = p0
    ranges.append((start, prev))
    return ranges
