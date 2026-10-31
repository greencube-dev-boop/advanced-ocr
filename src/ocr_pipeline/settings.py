"""Runtime configuration: packaged DEFAULTS merged with Drive config/pipeline.json.

Single source of truth for defaults is DEFAULTS below; `ocr_pipeline init-config`
copies it to <drive>/config/pipeline.json, and from then on the Drive copy wins
(deep-merged), so every experiment is recorded where resume state lives.
"""
from __future__ import annotations

import json
from pathlib import Path

DEFAULTS: dict = {
    "version": 1,
    "pdf_render_dpi": 200,
    "page": {
        "min_text_chars": 20,        # below this the native text layer counts as empty
        "dense_drawings": 40,        # vector drawings count that marks a flowchart/diagram page
        "large_image_area_ratio": 0.01,  # raster images bigger than 1% of the page count as visuals
    },
    "tesseract": {
        "enabled": True,
        "lang": "eng",               # add e.g. "+ben" for Bengali books
        "psm": 3,
    },
    "suspicion": {
        "min_confidence": 80.0,      # mean tesseract word confidence below this -> suspicious
        "max_len_mismatch": 0.5,     # |len_native - len_ocr| / max(len) above this -> suspicious
        "garbage_ratio": 0.08,       # fraction of control/format/symbol chars above this -> garbage
    },
    "nemotron": {
        "endpoint": "https://ai.api.nvidia.com/v1/retrieval/nvidia/nemotron-3.5-parse/parse",
        "max_output_tokens": 4096,   # API ceiling is 9000; keep headroom for dense pages
        "timeout_sec": 180,
        "max_retries": 4,
        "backoff_sec": 5.0,
        "table_format": "markdown",  # markdown | latex (post-processing of class_Table blocks)
    },
    "refiner": {
        "enabled": True,
        # gemma_local = keyless, runs on the Colab GPU (default).
        # gemini_api  = Google AI Studio key; leave api_key empty to fill later.
        "backend": "gemma_local",
        "skip_if_unavailable": True,
        "gemma_local": {
            "model_id": "google/gemma-3-4b-it",
            "quantization": "4bit",  # 4bit | 8bit | none
            "max_new_tokens": 2048,
            "temperature": 0.1,
            "hf_token": "",          # only needed if your HF account hasn't accepted the Gemma license
        },
        "gemini_api": {
            "model": "gemini-2.0-flash",
            "api_key": "",           # or put GEMINI_API_KEY=... in api.txt
            "timeout_sec": 120,
            "max_retries": 5,
            "rpm": 14,               # stay under the free-tier ~15 RPM
            "temperature": 0.1,
            "max_output_tokens": 8192,
        },
    },
    "output": {
        "add_page_markers": True,    # <!-- page N --> comments in the combined md
        "equations": "latex",
        "page_separator": "---",
    },
    "prompt_version": "page-refine-v1",
}


class Cfg:
    """Tiny nested-dict accessor: cfg.get('nemotron', 'endpoint', default=...)."""

    def __init__(self, data: dict):
        self.data = data or {}

    def get(self, *path, default=None):
        node = self.data
        for key in path:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def section(self, key: str) -> "Cfg":
        v = self.data.get(key, {})
        return Cfg(v if isinstance(v, dict) else {})

    def to_dict(self) -> dict:
        return dict(self.data)


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(drive_config_path) -> Cfg:
    data = dict(DEFAULTS)
    p = Path(drive_config_path)
    if p.exists():
        user = json.loads(p.read_text(encoding="utf-8"))
        data = deep_merge(data, user)
    return Cfg(data)


def save_config(cfg: Cfg, path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
