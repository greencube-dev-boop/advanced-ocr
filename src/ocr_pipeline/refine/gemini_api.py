"""Gemini refinement backend via the generativelanguage REST API (needs a key).

There is no keyless Gemini endpoint — an AI Studio key is required even on the
free tier. Leave the key empty and this backend simply reports unavailable; the
pipeline then uses the keyless gemma_local backend instead. Fill the key later
either here in config or as GEMINI_API_KEY=... in api.txt.
"""
from __future__ import annotations

import time

import requests


class GeminiApiRefiner:
    name = "gemini_api"

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash", timeout: int = 120,
                 max_retries: int = 5, rpm: int = 14, temperature: float = 0.1,
                 max_output_tokens: int = 8192):
        self.api_key = api_key or ""
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self._min_interval = (60.0 / rpm) if rpm else 0.0
        self._last_call = 0.0

    def available(self) -> bool:
        return bool(self.api_key)

    def _respect_rpm(self) -> None:
        if self._min_interval:
            wait = self._min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
        self._last_call = time.monotonic()

    def refine(self, prompt: str) -> str:
        if not self.available():
            raise RuntimeError("gemini_api: no API key configured")
        self._respect_rpm()
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        headers = {"Content-Type": "application/json", "x-goog-api-key": self.api_key}
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_output_tokens,
            },
        }
        last: str | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                r = requests.post(url, headers=headers, json=body, timeout=self.timeout)
                if r.status_code == 200:
                    data = r.json()
                    parts = data["candidates"][0]["content"]["parts"]
                    return "".join(p.get("text", "") for p in parts)
                if r.status_code in (429, 500, 502, 503):
                    last = f"HTTP {r.status_code}: {r.text[:200]}"
                else:
                    raise RuntimeError(f"gemini HTTP {r.status_code}: {r.text[:300]}")
            except requests.RequestException as e:
                last = str(e)
            if attempt < self.max_retries:
                time.sleep(5 * attempt)
        raise RuntimeError(f"gemini refine failed after {self.max_retries} attempts: {last}")
