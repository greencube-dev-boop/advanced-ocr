"""Per-book processing pipeline (OXR-only, curated training corpus).

Every page of every PDF is parsed by OXR-1.0; the output is training data for
a scientific LLM, so a curation pass drops all non-body content (front matter,
ToC pages, acknowledgements, preface/foreword, references/bibliography, index,
appendices, blank and figure-only pages) and the assembled Markdown file is
pure content.

Per book:
  1. sha256 pre-flight against the ledger — duplicates are skipped and their
     source archived
  2. all not-yet-done pages are grouped into contiguous chunks
     (<=20 pages AND <=9MiB; oversized single page -> rendered PNG) and
     uploaded with up to `chunking.max_concurrent` parallel requests
  3. curation pass classifies every parsed page keep/drop (see curation.py)
  4. refined/<book>.md = pure concatenation of kept pages, plus a sidecar
     <book>.record.json with full provenance
  5. ledger (books.json) keeps sha256/DOI/ISBN/drop breakdown permanently;
     on completion the temp dir is pruned and the source PDF moves to
     raw/_done/; books that end incomplete on `lifecycle.failed_run_limit`
     runs are quarantined to raw/_failed/
"""
from __future__ import annotations

import json
import logging
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from . import hashing
from .assemble import assemble_book
from .curation import curate
from .ocr.oxr_blocks import pages_picture_counts, pages_to_markdown
from .ocr.oxr_client import OxrClient, OxrError
from .pdfutils.chunking import paginate_range
from .pdfutils.render import render_page_png
from .state import StateStore

log = logging.getLogger("ocr_pipeline.pipeline")

_DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s\"<>]+", re.I)
_ISBN_RE = re.compile(r"ISBN(?:-1[03])?[:\s]*([0-9Xx][0-9Xx\-\s]{8,16}\d|[0-9Xx\-]{9,13}[0-9Xx])")
_TITLE_RE = re.compile(r"^#\s+(.+?)\s*$", re.M)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _clean_doi(candidate: str) -> str:
    return candidate.rstrip(".,;)]}'\"")


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
        self.max_concurrent = max(1, int(c.get("max_concurrent", default=2)))
        self.render_dpi = int(c.get("render_dpi", default=200))
        self.chunk_retries = int(c.get("chunk_retries", default=3))
        self.figure_marker = bool(cfg.get("output", "figure_marker", default=False))

        cu = cfg.section("curation")
        self.curation_enabled = bool(cu.get("enabled", default=True))
        self.curation_kwargs = dict(
            section_drop=cu.get("section_drop", default=[]),
            section_drop_prefix=cu.get("section_drop_prefix", default=[]),
            page_drop=cu.get("page_drop", default=[]),
            drop_before_first_heading=cu.get("drop_before_first_heading", default=True),
            first_heading_max_page=cu.get("first_heading_max_page", default=60),
            toc_min_lines=cu.get("toc_min_lines", default=8),
            toc_min_ratio=cu.get("toc_min_ratio", default=0.5),
            front_matter_max_chars=cu.get("front_matter_max_chars", default=2000),
        )

        lc = cfg.section("lifecycle")
        self.prune_temp = bool(lc.get("prune_temp", default=True))
        self.done_folder = str(lc.get("done_folder", default="_done"))
        self.failed_folder = str(lc.get("failed_folder", default="_failed"))
        self.delete_source = bool(lc.get("delete_source", default=False))
        self.failed_run_limit = int(lc.get("failed_run_limit", default=5))

    # ------------------------------------------------------------------
    def run_all(self) -> dict:
        pdfs = self.layout.list_pdfs()
        if not pdfs:
            log.info("No PDFs in %s — drop raw PDFs into raw/ and rerun.", self.layout.raw)
            return {}
        return {pdf.name: self.run_book(pdf) for pdf in pdfs}

    def run_book(self, pdf_path) -> dict:
        import fitz  # PyMuPDF — imported lazily so pure modules stay testable

        book_key = self.layout.rel_id(pdf_path)
        rel_stem = self.layout.rel_stem(pdf_path)
        sha = hashing.sha256_file(pdf_path)

        # ledger pre-flight: same content already processed (or quarantined)?
        prior = self.state.find_by_sha(sha, exclude_book=book_key)
        if prior is not None:
            folder = self.done_folder if prior.get("status") == "done" else self.failed_folder
            dest = self._dispose_source(pdf_path, rel_stem, folder)
            log.warning("Book %s skipped: sha256 already processed as '%s' [%s] — source -> %s",
                        book_key, prior["book"], prior.get("status"), dest)
            return {"book": book_key, "status": "skipped",
                    "reason": f"duplicate of {prior['book']} ({prior.get('status')})"}

        doc = fitz.open(str(pdf_path))
        total = doc.page_count
        self.state.register_book(book_key, sha, total)
        if self.state.book(book_key).get("archived"):
            doc.close()
            log.info("Book %s already completed and archived — nothing to do", book_key)
            return {"book": book_key, "status": self.state.book(book_key).get("status", "done"),
                    "skipped": "already archived"}
        log.info("Book %s: %d pages, sha=%s..., resume state %s", book_key, total, sha[:12],
                 self.state.counts(book_key))

        btemp = self.layout.book_temp(rel_stem)
        btemp.mkdir(parents=True, exist_ok=True)

        # every not-yet-processed page goes through OXR
        pending = [p for p in range(1, total + 1)
                   if self.state.page(book_key, p).get("status") not in ("done", "dropped")]
        if self.max_concurrent == 1:
            for a0, b0 in _contiguous(pending):        # inclusive 0-based
                for ca, cb, payload, kind in self._build_chunks(doc, a0, b0):
                    self._process_chunk(book_key, btemp, ca, cb, payload, kind)
        else:
            self._run_chunks_parallel(doc, book_key, btemp, pending,
                                      workers=self.max_concurrent)
        doc.close()

        # curation pass over every parsed page
        verdicts = self._curation_pass(book_key, btemp, total)

        out = self.layout.refined / f"{rel_stem}.md"
        assemble_book(btemp, out)
        counts = self.state.counts(book_key)
        complete = all(self.state.page(book_key, p).get("status") in ("done", "dropped")
                       for p in range(1, total + 1))

        if complete:
            self._finalize(book_key, rel_stem, btemp, total, sha, out, pdf_path, verdicts)
            status = "done"
        else:
            failed_runs = self.state.bump_failed_runs(book_key)
            if failed_runs >= self.failed_run_limit:
                self.state.set_book_status(book_key, "failed")
                self.state.set_record_fields(book_key, {"archived": True,
                                                        "quarantined_at": _now()})
                self._dispose_source(pdf_path, rel_stem, self.failed_folder)
                log.error("Book %s quarantined to %s/ after %d incomplete runs",
                          book_key, self.failed_folder, failed_runs)
                status = "failed"
            else:
                self.state.set_book_status(book_key, "partial")
                log.info("Book %s incomplete (%s) — failed run %d/%d",
                         book_key, counts, failed_runs, self.failed_run_limit)
                status = "partial"
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
            log.warning("page %d exceeds %d bytes as PDF; sending rendered PNG instead",
                        a + 1, self.max_bytes)
            return [(a, b, render_page_png(doc[a], dpi=self.render_dpi), "image")]
        mid = (a + b) // 2
        return self._size_split(doc, a, mid) + self._size_split(doc, mid + 1, b)

    def _chunk_generator(self, doc, pending: list[int]):
        for a0, b0 in _contiguous(pending):
            yield from self._build_chunks(doc, a0, b0)

    def _run_chunks_parallel(self, doc, book_key: str, btemp: Path,
                             pending: list[int], workers: int) -> None:
        """Upload up to `workers` chunks concurrently.

        Chunk payloads are produced lazily in the MAIN thread (fitz documents are
        not thread-safe) and only the HTTP work runs in the worker pool. At most
        workers*2 chunks are held in memory at once.
        """
        from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

        def drain(in_flight: dict) -> None:
            done = wait(list(in_flight), return_when=FIRST_COMPLETED).done
            for fut in done:
                in_flight.pop(fut, None)
                try:
                    fut.result()
                except Exception:
                    log.exception("chunk worker crashed unexpectedly")

        gen = self._chunk_generator(doc, pending)
        log.info("uploading chunks with up to %d parallel requests", workers)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            in_flight: dict = {}
            more = True
            while more or in_flight:
                while more and len(in_flight) < workers * 2:
                    try:
                        ca, cb, payload, kind = next(gen)
                    except StopIteration:
                        more = False
                        break
                    fut = pool.submit(self._process_chunk, book_key, btemp, ca, cb, payload, kind)
                    in_flight[fut] = (ca, cb)
                if not in_flight:
                    break
                drain(in_flight)

    # ---- chunk processing ----------------------------------------------
    def _process_chunk(self, book: str, btemp: Path, ca: int, cb: int,
                       payload: bytes, kind: str) -> None:
        chunk_key = f"p{ca + 1:04d}-{cb + 1:04d}"
        if self.state.chunk_attempts(book, chunk_key) >= self.chunk_retries:
            for p in range(ca + 1, cb + 2):
                if self.state.page(book, p).get("status") not in ("done", "dropped"):
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
            pics_by_page = pages_picture_counts(pages_json, page_base=ca)
            for p in range(ca + 1, cb + 2):
                if p not in md_by_page:
                    log.warning("chunk %s: OXR returned no blocks for page %d — left pending",
                                chunk_key, p)
                    continue
                pbase = btemp / f"page-{p:04d}"
                pbase.with_suffix(".oxr.md").write_text(md_by_page[p], encoding="utf-8")
                self._write_meta(pbase, {"page": p, "engine": f"oxr:{self.mode}",
                                         "chunk": chunk_key, "chars": len(md_by_page[p]),
                                         "n_pictures": pics_by_page.get(p, 0)})
                self.state.set_page(book, p, "done", engine=f"oxr:{self.mode}",
                                    extra={"chunk": chunk_key})
            self.state.record_chunk(book, chunk_key, error=None)
            log.info("chunk %s ok (engine=oxr:%s, %d pages)", chunk_key, self.mode, cb - ca + 1)
        except Exception as e:
            self.state.record_chunk(book, chunk_key, error=str(e))
            log.warning("chunk %s failed: %s", chunk_key, e)

    # ---- curation --------------------------------------------------------
    def _curation_pass(self, book_key: str, btemp: Path, total: int) -> list[dict] | None:
        page_infos = []
        for p in range(1, total + 1):
            st = self.state.page(book_key, p).get("status")
            if st not in ("done", "dropped"):
                continue  # un-parsed pages are not classifiable this run
            oxr_md = btemp / f"page-{p:04d}.oxr.md"
            if not oxr_md.exists():
                continue
            try:
                meta = json.loads((btemp / f"page-{p:04d}.meta.json").read_text(encoding="utf-8"))
            except Exception:
                meta = {}
            page_infos.append({"page_no": p, "md": oxr_md.read_text(encoding="utf-8"),
                               "n_pictures": int(meta.get("n_pictures", 0))})
        if not page_infos:
            return None

        if not self.curation_enabled:
            verdicts = [{"page_no": pg["page_no"], "verdict": "keep", "reason": None}
                        for pg in page_infos]
        else:
            verdicts = curate(page_infos, **self.curation_kwargs)

        for v in verdicts:
            p = v["page_no"]
            pbase = btemp / f"page-{p:04d}"
            if v["verdict"] == "keep":
                text = (btemp / f"page-{p:04d}.oxr.md").read_text(encoding="utf-8")
                pbase.with_suffix(".final.md").write_text(text, encoding="utf-8")
                self.state.set_page(book_key, p, "done", extra={"verdict": "keep"})
            else:
                pbase.with_suffix(".final.md").unlink(missing_ok=True)
                self.state.set_page(book_key, p, "dropped", extra={"drop_reason": v["reason"]})
            self._update_meta(pbase, {"verdict": v["verdict"], "drop_reason": v["reason"]})
        dropped = sum(1 for v in verdicts if v["verdict"] == "drop")
        log.info("curation: %d parsed pages -> %d kept, %d dropped",
                 len(verdicts), len(verdicts) - dropped, dropped)
        return verdicts

    @staticmethod
    def _write_meta(pbase: Path, meta: dict) -> None:
        pbase.with_suffix(".meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _update_meta(pbase: Path, extra: dict) -> None:
        meta_path = pbase.with_suffix(".meta.json")
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
        meta.update(extra)
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- finalization & lifecycle -----------------------------------------
    def _finalize(self, book_key: str, rel_stem: str, btemp: Path, total: int,
                  sha: str, out: Path, pdf_path, verdicts: list[dict] | None) -> None:
        # provenance: DOI / ISBN / title, extracted from ALL parsed pages
        all_text, doi, isbn, title = [], None, None, rel_stem
        dropped_map: dict[str, list[int]] = {}
        kept_pages: list[int] = []
        for v in (verdicts or []):
            if v["verdict"] == "drop":
                dropped_map.setdefault(v["reason"], []).append(v["page_no"])
            else:
                kept_pages.append(v["page_no"])
        for p in range(1, total + 1):
            f = btemp / f"page-{p:04d}.oxr.md"
            if not f.exists():
                continue
            text = f.read_text(encoding="utf-8")
            all_text.append(text)
            if doi is None:
                m = _DOI_RE.search(text)
                if m:
                    doi = _clean_doi(m.group(0))
            if isbn is None:
                m = _ISBN_RE.search(text)
                if m:
                    isbn = re.sub(r"[^0-9Xx]", "", m.group(1))
            if title == rel_stem:
                m = _TITLE_RE.search(text)
                if m:
                    title = m.group(1)

        out.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "book": book_key,
            "sha256": sha,
            "title": title,
            "doi": doi,
            "isbn": isbn,
            "pages_total": total,
            "pages_kept": len(kept_pages),
            "pages_dropped": total - len(kept_pages),
            "dropped": dropped_map,
            "engine": f"oxr:{self.mode}",
            "pipeline_version": self._version(),
            "output": str(out),
            "status": "done",
            "completed_at": _now(),
        }
        sidecar = out.with_name(out.stem + ".record.json")
        sidecar.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

        self.state.set_book_status(book_key, "done", output=str(out))
        self.state.set_record_fields(book_key, {
            "title": title, "doi": doi, "isbn": isbn,
            "pages_kept": len(kept_pages),
            "pages_dropped": total - len(kept_pages),
            "dropped": dropped_map,
            "archived": True, "completed_at": record["completed_at"],
        })

        if self.prune_temp:
            import time
            shutil.rmtree(btemp, ignore_errors=True)
            if btemp.exists():  # Windows can transiently hold a just-emptied dir
                time.sleep(0.5)
                shutil.rmtree(btemp, ignore_errors=True)
                if btemp.exists():
                    log.warning("could not fully prune %s (harmless leftover)", btemp)
        if len(kept_pages) == 0:
            log.warning("Book %s: curation kept 0 of %d pages — archived anyway, "
                        "review the record if this looks wrong", book_key, total)
        self._dispose_source(pdf_path, rel_stem, self.done_folder)
        log.info("Book %s finalized: kept %d/%d pages, doi=%s, isbn=%s",
                 book_key, len(kept_pages), total, doi, isbn)

    def _dispose_source(self, pdf_path, rel_stem: str, folder: str) -> Path | None:
        """Move the source PDF into raw/<folder>/<rel> (or delete if configured)."""
        if self.delete_source:
            try:
                pdf_path.unlink()
                log.info("source deleted: %s", pdf_path)
            except OSError:
                return None
            return None
        dest = self.layout.raw / folder / f"{rel_stem}.pdf"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            dest.unlink()
        shutil.move(str(pdf_path), str(dest))
        return dest

    @staticmethod
    def _version() -> str:
        from . import __version__
        return __version__


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
