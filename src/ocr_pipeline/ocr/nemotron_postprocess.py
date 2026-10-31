"""Turn the raw nemotron-3.5-parse `content` string into usable Markdown-ish text.

The API returns one string interleaving normalized coordinate tokens
(<x_0.0684><y_0.0375>), layout class tokens (<class_Table>, <class_Picture>, ...)
and the actual text. Tables arrive as LaTeX \\begin{tabular}...\\end{tabular}
inside <class_Table>; figures arrive as <class_Picture>/<class_Chart> markers
(the text inside pictures is predicted inline when present).

Pure functions (stdlib only) so they are unit-testable anywhere.
"""
from __future__ import annotations

import re

COORD_RE = re.compile(r"<[xy]_[0-9.]+>")
CLASS_RE = re.compile(r"<class_([A-Za-z-]+)>")
TABULAR_RE = re.compile(r"\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}", re.S)
BOLD_RE = re.compile(r"\\(?:textbf|textit|emph|underline)\{(.*?)\}", re.S)


def tabular_to_markdown(body: str) -> str:
    """Convert the inside of a LaTeX tabular to a GitHub-flavored Markdown table."""
    rows: list[list[str]] = []
    for raw in body.split("\\\\"):  # LaTeX row terminator
        raw = raw.replace("\\hline", " ").replace("\\cline", " ").strip()
        if not raw:
            continue
        cells = [BOLD_RE.sub(r"\1", c).strip() for c in raw.split("&")]
        if any(cells):
            rows.append(cells)
    if not rows:
        return ""
    ncol = max(len(r) for r in rows)
    rows = [r + [""] * (ncol - len(r)) for r in rows]
    header, rest = rows[0], rows[1:]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join([" --- "] * ncol) + "|",
    ]
    lines += ["| " + " | ".join(r) + " |" for r in rest]
    return "\n".join(lines)


def postprocess(content: str, table_format: str = "markdown") -> str:
    """Strip coordinate tokens, convert tables, map class tokens to Markdown."""
    if not content:
        return ""
    s = content.replace("\r\n", "\n").replace("\r", "\n")

    # 1) tables first, while the class_Table context is still intact
    if table_format == "markdown":
        s = TABULAR_RE.sub(lambda m: "\n" + tabular_to_markdown(m.group(1)) + "\n", s)
    # (table_format == "latex": leave tabular blocks untouched for the refiner)

    # 2) line-wise token mapping
    out_lines: list[str] = []
    for line in s.split("\n"):
        line = COORD_RE.sub("", line)

        def cls_repl(m: re.Match) -> str:
            cls = m.group(1)
            if cls in ("Picture", "Chart"):
                return "\n**[figure]**\n"
            if cls == "Section-header":
                return "\n## "
            return ""  # Text, Page-header/-footer, Caption, Table, ... -> drop token only

        line = CLASS_RE.sub(cls_repl, line)
        out_lines.append(line)
    s = "\n".join(out_lines)

    s = re.sub(r"[ \t]{2,}", " ", s)          # collapse runs of spaces
    s = re.sub(r"\n{3,}", "\n\n", s)           # collapse blank runs
    return s.strip()
