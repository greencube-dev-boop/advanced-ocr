"""Logging setup for the pipeline."""
from __future__ import annotations

import logging
import sys


def setup_logging(level: int = logging.INFO, log_file: str | None = None) -> logging.Logger:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger("ocr_pipeline")
