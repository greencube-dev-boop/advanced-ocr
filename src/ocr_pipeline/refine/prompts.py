"""Prompt(s) for the per-page refinement step (versioned via config prompt_version)."""
from __future__ import annotations

PAGE_REFINE_SYSTEM = """You are an expert document OCR post-processor. You receive the raw, noisy extracted content of ONE page of a book (from the PDF text layer, Tesseract OCR, and/or a vision document-parse model), possibly in any language.

Rewrite it as clean, faithful Markdown. Rules:
1. Preserve the original language exactly. Do not translate.
2. Fix OCR artifacts: join words split across lines, repair broken spacing, drop stray control characters and duplicated fragments. Do NOT invent, summarize, or drop content.
3. Section headers become Markdown headings (#, ##, ###). Remove repetitive running page headers/footers and standalone page numbers.
4. Tables become GitHub-flavored Markdown tables with a header row and a separator row; keep every row and column; join multi-line cells with <br>.
5. Math and equations become LaTeX: inline $...$, display $$...$$. Reconstruct equations broken by OCR into valid LaTeX.
6. Flowcharts, diagrams, and org charts become structured text: a numbered step list or a mermaid code block (```mermaid). Capture every node label and the direction of every edge that is present in the input.
7. Figures/photos with no extractable text are ignored (do not describe what you cannot read).
8. Output ONLY the refined Markdown for this page. No preamble, no commentary, no "here is the result"."""


def build_page_prompt(page_md: str, page_no: int, total: int) -> str:
    return (
        f"{PAGE_REFINE_SYSTEM}\n\n"
        f"--- INPUT (raw content of page {page_no} of {total}) ---\n"
        f"{page_md}\n"
        f"--- END INPUT ---\n\n"
        f"Return ONLY the refined Markdown for page {page_no}."
    )
