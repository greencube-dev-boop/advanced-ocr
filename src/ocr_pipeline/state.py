"""books.json — the resume manifest.

Everything needed to restart the project mid-book lives here:
per-book sha256 + page count, and per-page status/engines/attempts/errors.
A page only becomes "done" after its refined markdown is on disk, so a
crashed run simply redoes the pages that are not done yet.

All mutating methods take a reentrant lock, so parallel chunk workers can
update page state without corrupting books.json (writes go through a temp
file + atomic replace).
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class StateStore:
    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.RLock()
        self.data: dict = {"version": 1, "books": {}}
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self.data = loaded
            except Exception:
                pass  # corrupted manifest -> start fresh rather than crash
        self.data.setdefault("books", {})

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.path)  # atomic: a kill mid-write never corrupts books.json

    # ---- books -------------------------------------------------------
    def register_book(self, name: str, sha256: str, num_pages: int) -> dict:
        with self._lock:
            return self._register_book(name, sha256, num_pages)

    def _register_book(self, name: str, sha256: str, num_pages: int) -> dict:
        books = self.data["books"]
        entry = books.get(name)
        if entry is None:
            entry = {
                "sha256": sha256,
                "pages_total": num_pages,
                "status": "in_progress",
                "added_at": _now(),
                "updated_at": _now(),
                "pages": {},
                "output": None,
            }
            books[name] = entry
        else:
            if entry.get("sha256") != sha256:
                # the PDF on disk changed -> page states no longer apply
                entry["sha256"] = sha256
                entry["pages"] = {}
                entry["status"] = "in_progress"
                entry["output"] = None
            entry["pages_total"] = num_pages
            entry["updated_at"] = _now()
        self.save()
        return entry

    def book(self, name: str) -> dict:
        return self.data["books"].get(name, {})

    def set_book_status(self, name: str, status: str, output: str | None = None) -> None:
        with self._lock:
            entry = self.data["books"].get(name)
            if entry is None:
                return
            entry["status"] = status
            if output is not None:
                entry["output"] = output
            entry["updated_at"] = _now()
            self.save()

    # ---- pages -------------------------------------------------------
    def page(self, book: str, page_no: int) -> dict:
        return self.book(book).get("pages", {}).get(str(page_no), {})

    def set_page(self, book: str, page_no: int, status: str, engine: str | None = None,
                 error: str | None = None, extra: dict | None = None) -> None:
        with self._lock:
            entry = self.data["books"].get(book)
            if entry is None:
                return
            p = entry.setdefault("pages", {}).setdefault(str(page_no), {})
            p["status"] = status
            p["attempts"] = p.get("attempts", 0) + 1
            p["updated_at"] = _now()
            if engine is not None:
                p["engine"] = engine
            if error is not None:
                p["error"] = error
            if extra:
                p.update(extra)
            entry["updated_at"] = _now()
            self.save()

    def counts(self, book: str) -> dict:
        tally: dict[str, int] = {}
        for p in self.book(book).get("pages", {}).values():
            s = p.get("status", "pending")
            tally[s] = tally.get(s, 0) + 1
        return tally

    # ---- OXR chunks (one /parse/sync request covering a page range) ----
    def record_chunk(self, book: str, key: str, error: str | None = None) -> None:
        with self._lock:
            entry = self.data["books"].get(book)
            if entry is None:
                return
            c = entry.setdefault("chunks", {}).setdefault(key, {"attempts": 0, "last_error": None})
            c["attempts"] = c.get("attempts", 0) + 1
            c["last_error"] = error
            c["updated_at"] = _now()
            c["ok"] = error is None
            entry["updated_at"] = _now()
            self.save()

    def chunk_attempts(self, book: str, key: str) -> int:
        return int(self.book(book).get("chunks", {}).get(key, {}).get("attempts", 0))

    # ---- ledger ---------------------------------------------------------
    def find_by_sha(self, sha256: str, exclude_book: str | None = None) -> dict | None:
        """First completed/failed record with this sha256 (duplicate pre-flight)."""
        with self._lock:
            for name, entry in self.data["books"].items():
                if name == exclude_book:
                    continue
                if entry.get("sha256") == sha256 and entry.get("status") in ("done", "failed"):
                    return {"book": name, "status": entry.get("status"),
                            "doi": entry.get("doi"), "output": entry.get("output")}
        return None

    def bump_failed_runs(self, book: str) -> int:
        with self._lock:
            entry = self.data["books"].get(book)
            if entry is None:
                return 0
            entry["failed_runs"] = entry.get("failed_runs", 0) + 1
            entry["updated_at"] = _now()
            self.save()
            return entry["failed_runs"]

    def set_record_fields(self, book: str, fields: dict) -> None:
        """Attach ledger fields (doi, isbn, title, kept/dropped breakdown, ...)."""
        with self._lock:
            entry = self.data["books"].get(book)
            if entry is None:
                return
            entry.update(fields)
            entry["updated_at"] = _now()
            self.save()

    # ---- reset ---------------------------------------------------------
    def forget(self, book: str) -> None:
        """Drop a book's whole state (used by `ocr_pipeline reset`)."""
        with self._lock:
            self.data["books"].pop(book, None)
            self.save()
