"""Runtime configuration: packaged DEFAULTS merged with Drive config/pipeline.json.

v2 (OXR-only): SoMarkAI/OXR-1.0 replaces Tesseract, nemotron-3.5-parse and the
LLM refiner. Defaults live here; `ocr_pipeline init-config` copies them to
<drive>/config/pipeline.json, and from then on the Drive copy wins (deep-merged).
"""
from __future__ import annotations

import json
from pathlib import Path

DEFAULTS: dict = {
    "version": 2,
    "oxr": {
        "mode": "free_api",      # free_api (no key, 10MB/20-pages per request) | local (self-hosted)
        "base_url": "",          # explicit override; empty -> per-mode default below
        "free_api_base_url": "https://free-oxr.somark.ai/api/v1",
        "local_base_url": "http://127.0.0.1:8000/v1",
        "api_key": "",           # empty -> "dummy" (the field is required but never validated)
        "timeout_sec": 600,
        "max_retries": 5,
        "backoff_sec": 10.0,
        # element output choices (all OXR-supported values; see ocr/oxr_blocks.py)
        "element_formats": {"image": "none", "formula": "latex", "table": "markdown", "cs": "smiles"},
        # OXR supports only keep_header_footer from feature_config
        "feature_config": {"keep_header_footer": False},
        # OXR-only sampling options
        "extra": {"temperature": 0.0},
    },
    "chunking": {
        "max_pages": 20,         # free-API ceiling per request
        "max_bytes": 9437184,    # 9 MiB, under the 10MB free-API ceiling
        "render_dpi": 200,       # fallback rendering when a single page exceeds max_bytes
        "chunk_retries": 3,      # failed chunks are retried on the next run, up to this
    },
    "page": {
        "min_text_chars": 20,        # below this the native text layer counts as empty
        "dense_drawings": 40,        # vector drawing count that marks a flowchart/diagram page
        "large_image_area_ratio": 0.01,  # raster images bigger than 1% of the page count as visuals
    },
    "output": {
        "add_page_markers": True,    # <!-- page N --> comments in the combined md
        "page_separator": "---",
        "figure_marker": False,      # true -> insert *[figure omitted]* where OXR found an image
    },
}


class Cfg:
    """Tiny nested-dict accessor: cfg.get('oxr', 'mode', default=...)."""

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
