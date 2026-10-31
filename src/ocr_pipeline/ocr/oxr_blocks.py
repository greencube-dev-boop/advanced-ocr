"""Convert OXR JSON output (pages -> blocks) into per-page Markdown.

OXR element types (13) and how they map to Markdown:
  Title              -> '#'*title_level heading
  Text/Caption/Footnote/Stamp -> plain paragraph
  Table              -> content passed through (element_formats.table="markdown")
  Formula            -> LaTeX wrapped in $$...$$ (unless already delimited)
  Chemical Equation  -> mhchem LaTeX wrapped in $$...$$
  Chemical Structure -> SMILES as inline code
  Code Block         -> fenced code block (unless already fenced)
  Picture/Stamp(figure) -> dropped (OXR's API does no image understanding;
                           non-textifiable images are ignored per spec)
  Page-header/Footer -> dropped

Pure functions (stdlib only) so they are unit-testable anywhere.
"""
from __future__ import annotations

_DROPPED = ("page_header", "page_footer", "header", "footer")
_FIGURES = ("picture", "figure", "image", "figure_image")


def _norm_type(value) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def block_to_markdown(block: dict) -> str | None:
    btype = _norm_type(block.get("type"))
    content = str(block.get("content") or "").strip()
    if not content:
        return None
    if btype in _DROPPED or btype in _FIGURES:
        return None
    if btype == "title":
        try:
            level = int(block.get("title_level") or 1)
        except (TypeError, ValueError):
            level = 1
        level = min(max(level, 1), 6)
        return f"{'#' * level} {content}"
    if btype == "table":
        return content  # already Markdown (element_formats.table="markdown")
    if btype in ("formula", "chemical_equation", "chemistry"):
        return content if content.startswith("$") else f"$${content}$$"
    if btype == "chemical_structure":
        return f"`{content}`"  # SMILES as inline code
    if btype in ("code_block", "code"):
        return content if content.lstrip().startswith("```") else f"```\n{content}\n```"
    return content  # text / caption / footnote / unknown-but-textual


def blocks_to_markdown(blocks, figure_marker: bool = False) -> str:
    parts: list[str] = []
    prev_dropped_figure = False
    for block in blocks or []:
        btype = _norm_type(block.get("type"))
        md = block_to_markdown(block)
        if md:
            if btype == "caption" and prev_dropped_figure:
                prev_dropped_figure = False
                continue  # caption belongs to the dropped figure -> drop it too
            parts.append(md)
            prev_dropped_figure = False
        elif btype in _FIGURES:
            prev_dropped_figure = True
            if figure_marker:
                parts.append("*[figure omitted]*")
    return "\n\n".join(parts).strip()


def pages_to_markdown(pages, page_base: int = 0) -> dict[int, str]:
    """Map OXR pages[] -> {absolute_page_no: markdown}.

    page_base is the 0-based absolute index of the chunk's first page; the
    response's page_num is 0-based relative to the submitted chunk.
    """
    out: dict[int, str] = {}
    for i, page in enumerate(pages or []):
        try:
            num = int(page.get("page_num", i))
        except (TypeError, ValueError):
            num = i
        out[page_base + num + 1] = blocks_to_markdown(page.get("blocks"))
    return out


def pages_picture_counts(pages, page_base: int = 0) -> dict[int, int]:
    """Map OXR pages[] -> {absolute_page_no: number of figure/picture blocks}."""
    out: dict[int, int] = {}
    for i, page in enumerate(pages or []):
        try:
            num = int(page.get("page_num", i))
        except (TypeError, ValueError):
            num = i
        out[page_base + num + 1] = sum(
            1 for b in page.get("blocks") or [] if _norm_type(b.get("type")) in _FIGURES)
    return out
