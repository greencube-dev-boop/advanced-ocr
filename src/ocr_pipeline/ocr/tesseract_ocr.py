"""Tesseract OCR wrapper (pytesseract + the tesseract binary)."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO


@dataclass
class OcrResult:
    text: str
    mean_conf: float | None   # mean word confidence from image_to_data (-1 = unknown)
    n_words: int


def tesseract_available() -> bool:
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def ocr_image(png_bytes: bytes, lang: str = "eng", psm: int = 3) -> OcrResult:
    import pytesseract
    from PIL import Image

    img = Image.open(BytesIO(png_bytes))
    cfg_str = f"--psm {psm}"
    text = pytesseract.image_to_string(img, lang=lang, config=cfg_str) or ""
    mean_conf: float | None = None
    try:
        data = pytesseract.image_to_data(img, lang=lang, config=cfg_str,
                                         output_type=pytesseract.Output.DICT)
        vals = [float(c) for c, t in zip(data["conf"], data["text"])
                if str(t).strip() and float(c) >= 0]
        if vals:
            mean_conf = sum(vals) / len(vals)
    except Exception:
        pass
    n_words = len([w for w in text.split() if w])
    return OcrResult(text=text, mean_conf=mean_conf, n_words=n_words)
