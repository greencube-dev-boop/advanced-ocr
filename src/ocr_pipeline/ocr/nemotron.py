"""Client for NVIDIA nemotron-3.5-parse (POST /parse, multipart/form-data).

Request:  field "document" = one rendered page image (png/jpeg)
          field "options"  = {"decoding": {"max_output_tokens": N}} as JSON
Response: {"content": <one string with coordinate/class tokens + LaTeX tabular tables>,
           "finish_reason": ..., "usage": {...}}
The content string is cleaned up by nemotron_postprocess.postprocess().
"""
from __future__ import annotations

import json
import time

import requests


class NemotronError(RuntimeError):
    pass


class NemotronClient:
    def __init__(self, api_key: str, endpoint: str, max_output_tokens: int = 4096,
                 timeout: int = 180, max_retries: int = 4, backoff: float = 5.0,
                 session: requests.Session | None = None):
        if not api_key:
            raise NemotronError("NVIDIA_API_KEY is missing (api.txt line 1)")
        self.api_key = api_key
        self.endpoint = endpoint
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self.session = session or requests.Session()

    def parse_page(self, image_bytes: bytes, mime: str = "image/png") -> dict:
        files = {
            "document": ("page.png", image_bytes, mime),
            "options": (None, json.dumps({"decoding": {"max_output_tokens": self.max_output_tokens}}),
                        "application/json"),
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}
        last: str | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                r = self.session.post(self.endpoint, headers=headers, files=files,
                                      timeout=self.timeout)
                if r.status_code == 200:
                    return r.json()
                if r.status_code in (429, 500, 502, 503, 504):
                    last = f"HTTP {r.status_code} (throttled/server, retrying)"
                else:
                    # 4xx other than 429 is not retryable (bad image, bad options, ...)
                    raise NemotronError(f"HTTP {r.status_code}: {r.text[:300]}")
            except requests.RequestException as e:
                last = str(e)
            if attempt < self.max_retries:
                time.sleep(self.backoff * attempt)
        raise NemotronError(f"nemotron parse failed after {self.max_retries} attempts: {last}")
