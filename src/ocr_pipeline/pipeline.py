"""Per-book processing pipeline (OXR-only, 100% coverage) with resume support.

Every page of every PDF is parsed by OXR-1.0 — the native text layer is not
used at all, so the whole book gets uniform OXR structure (headings, tables,
LaTeX formulas, mhchem, SMILES, code fences).

Per book:
  1. all not-yet-done pages are grouped into contiguous chunks
     (<=20 pages AND <=9MiB per chunk; an oversized single page is rendered
     to PNG) -> POST {oxr}/parse/sync -> OXR JSON blocks -> per-page markdown
  2. every finished page lands in temp/<book>/page-XXXX.final.md and books.json
  3. assemble refined/<book>.md
"""
from __future__ import annotations

import json
import logging

from . import hashing
from .assemble import assemble_book
from .ocr.oxr_blocks import pages_to_markdown
from .ocr.oxr_client import OxrClient, OxrError
from .pdfutils.chunking import paginate_range, split_range
from .pdfutils.render import render_page_png
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

    def run_book(self, pdf_path) -> dict:
        import fitz  # PyMuPDF — imported lazily so pure modules stay testable

        # folder-aware identity: "sub/dir/book.pdf" — same-named PDFs in
        # different raw/ subfolders never collide
        book_key = self.layout.rel_id(pdf_path)
        rel_stem = self.layout.rel_stem(pdf_path)
        title = self.layout.display_name(pdf_path)
        sha = hashing.sha256_file(pdf_path)
        doc = fitz.open(str(pdf_path))
        total = doc.page_count
        self.state.register_book(book_key, sha, total)
        log.info("Book %s: %d pages, sha=%s..., resume state %s", book_key, total, sha[:12],
                 self.state.counts(book_key))

        btemp = self.layout.book_temp(rel_stem)
        btemp.mkdir(parents=True, exist_ok=True)

        # every not-yet-done page goes through OXR — no native-text fast path
        pending = [p for p in range(1, total + 1)
                   if self.state.page(book_key, p).get("status") != "done"]
        for a0, b0 in _contiguous(pending):        # inclusive 0-based
            for ca, cb, payload, kind in self._build_chunks(doc, a0, b0):
                self._process_chunk(book_key, btemp, ca, cb, payload, kind)

        doc.close()
        counts = self.state.counts(book_key)
        out = self.layout.refined / f"{rel_stem}.md"
        assemble_book(btemp, out, title,
                      {"source": book_key, "sha256": sha, "counts": counts},
                      add_page_markers=self.cfg.get("output", "add_page_markers", default=True),
                      page_separator=self.cfg.get("output", "page_separator", default="---"))
        status = "done" if counts.get("done", 0) == total else "partial"
        self.state.set_book_status(book_key, status, output=str(out))
        log.info("Book %s -> %s [%s] %s", book_key, out, status, counts)
        return {"book": book_key, "output": str(out), "status": status, "counts": counts}

    # ------------------------------------------------------------------
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
                       payload: bytes, kind: str) -> None:
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
                self._write_meta(pbase, {"page": p, "engine": f"oxr:{self.mode}",
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
