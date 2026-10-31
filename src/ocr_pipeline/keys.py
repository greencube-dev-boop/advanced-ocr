"""Parsing of the api.txt key file (lives only in the Colab runtime, never in git/Drive).

Agreed format:
  - line 1: the bare NVIDIA API key (single line, no prefix)
  - optional further lines: KEY=value, e.g. GEMINI_API_KEY=..., HF_TOKEN=...
Blank lines and lines starting with '#' are ignored.
"""
from __future__ import annotations

from pathlib import Path


class Keys:
    def __init__(self, nvidia: str = "", extra: dict | None = None):
        self.nvidia = nvidia or ""
        self.extra = dict(extra or {})

    @property
    def gemini(self) -> str:
        return self.extra.get("GEMINI_API_KEY", "")

    @property
    def hf_token(self) -> str:
        return self.extra.get("HF_TOKEN", "")

    def get(self, name: str) -> str:
        return self.extra.get(name, "")

    def as_dict(self) -> dict:
        d = dict(self.extra)
        if self.nvidia:
            d["NVIDIA_API_KEY"] = self.nvidia
        return d


def parse_api_txt(text: str) -> Keys:
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith("#")]
    if not lines:
        return Keys()
    extra: dict[str, str] = {}
    first = lines[0]
    if "=" in first and first.split("=", 1)[0].strip().isupper():
        # tolerate KEY=value on line 1 as well
        k, v = first.split("=", 1)
        extra[k.strip()] = v.strip()
        nvidia = extra.pop("NVIDIA_API_KEY", "")
    else:
        nvidia = first
    for ln in lines[1:]:
        if "=" in ln:
            k, v = ln.split("=", 1)
            extra[k.strip()] = v.strip()
    return Keys(nvidia=nvidia, extra=extra)


def load_api_txt(path) -> Keys:
    p = Path(path)
    if not p.exists():
        return Keys()
    return parse_api_txt(p.read_text(encoding="utf-8", errors="replace"))
