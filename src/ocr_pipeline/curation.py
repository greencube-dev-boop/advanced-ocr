"""Content curation: decide which parsed pages belong in the training corpus.

The output of this pipeline is training data for a scientific LLM, so anything
that is not body content is dropped: title/copyright/dedication pages, printed
table-of-contents pages, acknowledgements, preface/foreword, references,
bibliography, index, appendices, blank pages and figure-only pages.

Classification is deterministic and section-aware:
  - page-level rules: blank/figure-only, ToC dot-leader pages, front-matter
    pages (copyright/ISBN/publisher boilerplate)
  - section ranges: when a top-level heading matches a drop pattern
    ("References", "Appendix A", ...), every page until the next top-level
    heading that does NOT match is dropped — dense bibliography pages would
    never survive a per-page keyword check
  - front-matter boundary: everything before the book's first kept top-level
    heading is front matter (guarded: the rule only fires when such a heading
    actually exists within `first_heading_max_page` pages, so a book whose
    headings OXR failed to detect keeps everything)

Pure functions (stdlib only); all patterns come from config so they can be
tuned without touching code. Every verdict is recorded by the pipeline.
"""
from __future__ import annotations

import re

_TOP_HEADING_RE = re.compile(r"^(#{1,2})\s+(.+?)\s*$", re.M)
_TOC_LINE_RE = re.compile(r"^.+?[.·…_―—–-]{2,}\s*\d{1,4}\s*$")
# strips "12.", "iv.", "Chapter 1:" style prefixes; the lookaheads stop the roman/digit
# classes from eating the first letters of ordinary words ("Contents" -> "ontents")
_NUM_PREFIX_RE = re.compile(
    r"^(?:chapter|part|section)?\s*(?:\d{1,3}(?!\d)|[ivxlcdm]{1,7}(?![a-z]))\s*[.:)\]-]*\s*")
_PUNCT_RE = re.compile(r"[^\w\s]")


def _norm_heading(text: str) -> str:
    t = text.strip().lower()
    t = _NUM_PREFIX_RE.sub("", t)
    t = _PUNCT_RE.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def _matches(text: str, exact: set[str], prefix: list[str]) -> bool:
    h = _norm_heading(text)
    if not h:
        return False
    if h in exact:
        return True
    return any(h.startswith(p) for p in prefix)


def _top_headings(md: str) -> list[tuple[int, str]]:
    return [(len(m.group(1)), m.group(2)) for m in _TOP_HEADING_RE.finditer(md)]


def _looks_like_toc(md: str, min_lines: int, min_ratio: float) -> bool:
    lines = [ln.strip() for ln in md.splitlines() if ln.strip()]
    if len(lines) < min_lines:
        return False
    hits = sum(1 for ln in lines if _TOC_LINE_RE.match(ln))
    return hits / len(lines) >= min_ratio


def _looks_like_front_matter(md: str, page_drop: list[str], max_chars: int) -> bool:
    low = md.lower()
    hits = sum(1 for p in page_drop if p in low)
    if hits == 0:
        return False
    return len(md) <= max_chars or hits >= 3


def curate(pages: list[dict], *, section_drop: list[str] = (),
           section_drop_prefix: list[str] = (), page_drop: list[str] = (),
           drop_before_first_heading: bool = True, first_heading_max_page: int = 60,
           toc_min_lines: int = 8, toc_min_ratio: float = 0.5,
           front_matter_max_chars: int = 2000) -> list[dict]:
    """Classify parsed pages as keep/drop.

    pages: [{"page_no": int, "md": str, "n_pictures": int}, ...] in book order.
    Returns [{"page_no", "verdict": "keep"|"drop", "reason": str|None}, ...].
    """
    exact = {_norm_heading(p) for p in section_drop}
    prefix = [p.strip().lower() for p in section_drop_prefix]

    # pass A — front-matter boundary: first page carrying a kept top-level heading.
    # Pages that themselves look like front matter / ToC (title pages carry a real
    # heading — the book title) are skipped, so they can't become the boundary.
    first_head = None
    if drop_before_first_heading:
        for pg in pages:
            if pg["page_no"] > first_heading_max_page:
                break
            md = pg.get("md") or ""
            if (_looks_like_front_matter(md, list(page_drop), front_matter_max_chars)
                    or _looks_like_toc(md, toc_min_lines, toc_min_ratio)):
                continue
            if any(not _matches(t, exact, prefix) for _, t in _top_headings(md)):
                first_head = pg["page_no"]
                break

    verdicts: list[dict] = []
    section: str | None = None  # drop pattern currently governing a trash range
    for pg in pages:
        no = pg["page_no"]
        md = pg.get("md") or ""
        body = md.strip()

        def drop(reason: str) -> None:
            verdicts.append({"page_no": no, "verdict": "drop", "reason": reason})

        if not body:
            drop("figure-only" if pg.get("n_pictures") else "blank")
            continue

        heads = _top_headings(md)

        if section is not None:
            survivor = next((t for _, t in heads if not _matches(t, exact, prefix)), None)
            if survivor is None:
                drop(f"section:{section}")
                continue
            section = None  # this page opens a kept section again

        if _looks_like_toc(md, toc_min_lines, toc_min_ratio):
            drop("toc-page")
            continue

        if _looks_like_front_matter(md, list(page_drop), front_matter_max_chars) and (
                not heads or all(_matches(t, exact, prefix) for _, t in heads)):
            drop("front-matter")
            continue

        trigger = next((t for _, t in heads if _matches(t, exact, prefix)), None)
        if trigger:
            section = _norm_heading(trigger)
            drop(f"section:{section}")
            continue

        if first_head is not None and no < first_head:
            drop("front-matter")
            continue

        verdicts.append({"page_no": no, "verdict": "keep", "reason": None})
    return verdicts
