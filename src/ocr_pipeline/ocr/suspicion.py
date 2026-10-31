"""Decide whether the Tesseract pass looks untrustworthy and Nemotron should be called.

Pure functions (no third-party imports) so they are unit-testable anywhere.
"""
from __future__ import annotations

import unicodedata

# Control/format/surrogate/private/unassigned chars are garbage for OCR purposes.
# NOTE: combining marks (Mn/Mc/Me) are NOT flagged — Bengali and other scripts
# legitimately use them heavily.
_BAD_CATEGORIES = ("Cc", "Cf", "Cs", "Co", "Cn")


def garbage_ratio(text: str) -> float:
    if not text:
        return 0.0
    bad = 0
    for ch in text:
        if ch in "\n\t\r ":
            continue
        cat = unicodedata.category(ch)
        if cat in _BAD_CATEGORIES or cat == "So":  # So = emoji/pictographs
            bad += 1
    return bad / len(text)


def length_mismatch(a: str, b: str) -> float:
    la, lb = len(a.strip()), len(b.strip())
    m = max(la, lb)
    if m == 0:
        return 0.0
    return abs(la - lb) / m


def evaluate(native: str, ocr: str, mean_conf, has_table: bool, cfg) -> tuple[bool, list[str]]:
    """Return (escalate_to_nemotron, reasons)."""
    reasons: list[str] = []
    if has_table:
        reasons.append("table-or-diagram-detected")
    if mean_conf is not None and mean_conf < cfg.get("suspicion", "min_confidence", default=80.0):
        reasons.append(f"low-tesseract-confidence:{mean_conf:.1f}")
    g = garbage_ratio(ocr)
    if ocr.strip() and g > cfg.get("suspicion", "garbage_ratio", default=0.08):
        reasons.append(f"ocr-garbage-chars:{g:.2f}")
    if len(native.strip()) >= 40 and garbage_ratio(native) > cfg.get("suspicion", "garbage_ratio", default=0.08):
        reasons.append("native-text-layer-garbage")
    if len(native.strip()) >= 40 and len(ocr.strip()) >= 40:
        mm = length_mismatch(native, ocr)
        if mm > cfg.get("suspicion", "max_len_mismatch", default=0.5):
            reasons.append(f"native-vs-ocr-length-mismatch:{mm:.2f}")
    return (len(reasons) > 0), reasons
