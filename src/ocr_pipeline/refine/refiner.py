"""Backend selection and per-page refinement dispatch."""
from __future__ import annotations

import logging

from .gemini_api import GeminiApiRefiner
from .gemma_local import GemmaLocalRefiner
from .prompts import build_page_prompt

log = logging.getLogger("ocr_pipeline.refine")


class Refiner:
    def __init__(self, backend: str, cfg, keys):
        self.backend_name = backend
        self.cfg = cfg
        self.keys = keys
        self._impl = self._build()

    def _build(self):
        r = self.cfg.section("refiner")
        if self.backend_name == "gemini_api":
            g = r.section("gemini_api")
            return GeminiApiRefiner(
                api_key=self.keys.gemini or g.get("api_key", default=""),
                model=g.get("model", default="gemini-2.0-flash"),
                timeout=g.get("timeout_sec", default=120),
                max_retries=g.get("max_retries", default=5),
                rpm=g.get("rpm", default=14),
                temperature=g.get("temperature", default=0.1),
                max_output_tokens=g.get("max_output_tokens", default=8192),
            )
        if self.backend_name == "gemma_local":
            g = r.section("gemma_local")
            return GemmaLocalRefiner(
                model_id=g.get("model_id", default="google/gemma-3-4b-it"),
                quantization=g.get("quantization", default="4bit"),
                max_new_tokens=g.get("max_new_tokens", default=2048),
                temperature=g.get("temperature", default=0.1),
                hf_token=self.keys.hf_token or g.get("hf_token", default=""),
            )
        raise ValueError(f"unknown refiner backend: {self.backend_name}")

    def available(self) -> bool:
        return self._impl.available()

    def refine_page(self, page_md: str, page_no: int, total: int) -> tuple[str, str]:
        """Return (refined_markdown, engine_name)."""
        prompt = build_page_prompt(page_md, page_no, total)
        text = self._impl.refine(prompt)
        return text, self._impl.name


def build_refiner(cfg, keys) -> Refiner:
    backend = cfg.get("refiner", "backend", default="gemma_local")
    if backend == "gemini_api":
        has_key = bool(keys.gemini or cfg.get("refiner", "gemini_api", "api_key"))
        if not has_key:
            log.warning("refiner.backend=gemini_api but no GEMINI_API_KEY set — "
                        "falling back to keyless gemma_local")
            backend = "gemma_local"
    return Refiner(backend, cfg, keys)
