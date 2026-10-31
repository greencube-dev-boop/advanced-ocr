"""OXR-1.0 client — SoMark-compatible POST {base}/parse/sync.

Works with both deployments of SoMarkAI/OXR-1.0:
  - free OXR API : base https://free-oxr.somark.ai/api/v1  (api_key=dummy, NO key needed;
                   limits: 10MB file / 20 pages per request, queued, best-effort)
  - self-hosted  : base http://<host>:8000/v1              (no limits; same contract)

Request: multipart/form-data with fields
  file            (binary: PDF or page image)
  api_key         (any value, e.g. "dummy" — the field is required but not validated)
  output_formats  ("json"; repeated fields supported)
  element_formats (JSON string: image/formula/table/cs output choices)
  feature_config  (JSON string; OXR supports only keep_header_footer)
  extra           (OXR-only sampling JSON: temperature / top_p / repetition_penalty)

Response: {"code": 0|error_code, "message", "warnings", "data": {"result": {"outputs": ...}}}
  code 1133 = parameter error (deterministic, not retried)
  code 1137 = parsing failure (retried)
"""
from __future__ import annotations

import json
import random
import threading
import time

import requests


class OxrError(RuntimeError):
    pass


RETRYABLE_STATUS = (429, 500, 502, 503, 504)
RETRYABLE_CODES = {1137}


class OxrClient:
    def __init__(self, base_url: str, api_key: str = "", timeout: int = 600,
                 max_retries: int = 5, backoff: float = 10.0,
                 element_formats: dict | None = None, feature_config: dict | None = None,
                 extra: dict | None = None, session: requests.Session | None = None):
        if not base_url:
            raise OxrError("OXR base_url is empty")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or "dummy"
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self.element_formats = element_formats
        self.feature_config = feature_config
        self.extra = extra
        self._shared_session = session
        self._local = threading.local()  # requests.Session is not guaranteed thread-safe

    @property
    def session(self) -> requests.Session:
        if self._shared_session is not None:
            return self._shared_session
        s = getattr(self._local, "session", None)
        if s is None:
            s = self._local.session = requests.Session()
        return s

    def parse(self, file_bytes: bytes, filename: str = "document.pdf",
              mime: str = "application/pdf") -> dict:
        """Parse one PDF chunk or page image; returns the full response dict."""
        files = {"file": (filename, file_bytes, mime)}
        data: list[tuple[str, str]] = [("api_key", self.api_key), ("output_formats", "json")]
        if self.element_formats:
            data.append(("element_formats", json.dumps(self.element_formats)))
        if self.feature_config:
            data.append(("feature_config", json.dumps(self.feature_config)))
        if self.extra:
            data.append(("extra", json.dumps(self.extra)))

        last: str | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                r = self.session.post(f"{self.base_url}/parse/sync", data=data,
                                      files=files, timeout=self.timeout)
                if r.status_code == 200:
                    body = r.json()
                    code = body.get("code")
                    if code in (None, 0):
                        return body
                    if code in RETRYABLE_CODES:
                        last = f"code {code}: {body.get('message')}"
                    else:
                        # e.g. 1133 parameter error — deterministic, retrying will not help
                        raise OxrError(f"OXR error code {code}: {body.get('message')} "
                                       f"warnings={body.get('warnings')}")
                elif r.status_code in RETRYABLE_STATUS:
                    last = f"HTTP {r.status_code} (throttled/server, retrying)"
                else:
                    raise OxrError(f"HTTP {r.status_code}: {r.text[:300]}")
            except requests.RequestException as e:
                last = str(e)
            if attempt < self.max_retries:
                # jittered exponential backoff — with parallel workers this also
                # staggers retries so they don't hammer the server in lockstep
                time.sleep(self.backoff * attempt + random.uniform(0.0, 2.0))
        raise OxrError(f"parse/sync failed after {self.max_retries} attempts: {last}")
