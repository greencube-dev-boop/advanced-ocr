"""CLI entry point: python -m ocr_pipeline <command> [--drive-root ...] [--api-file ...]"""
from __future__ import annotations

import argparse
import importlib
import sys

from .drive import DriveLayout
from .keys import load_api_txt
from .logging_setup import setup_logging
from .pipeline import Pipeline
from .refine.refiner import build_refiner
from .settings import DEFAULTS, Cfg, load_config, save_config
from .state import StateStore


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="ocr_pipeline",
        description="Advanced OCR pipeline — the Colab notebook only calls this.")
    ap.add_argument("--drive-root", default="/content/drive/MyDrive/OCR Engine",
                    help="Path to the 'OCR Engine' folder on the mounted Drive")
    ap.add_argument("--api-file", default="api.txt",
                    help="api.txt: line 1 = NVIDIA key, optional KEY=value lines after")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("run", help="Process all pending PDFs in raw/ (resumable)")
    p_book = sub.add_parser("run-book", help="Process a single PDF from raw/")
    p_book.add_argument("name", help="file name, e.g. mybook.pdf")
    sub.add_parser("status", help="Show the resume state (books.json)")
    sub.add_parser("doctor", help="Check environment: deps, keys, drive, tesseract")
    sub.add_parser("init-config", help="Write the default pipeline.json into Drive config/")
    args = ap.parse_args(argv)

    log = setup_logging()
    layout = DriveLayout(args.drive_root)
    layout.ensure()

    if args.command == "init-config":
        p = layout.pipeline_json
        if p.exists():
            print(f"already exists: {p}")
        else:
            save_config(Cfg(DEFAULTS), p)
            print(f"wrote default config: {p} — edit it there to change settings")
        return 0

    cfg = load_config(layout.pipeline_json)
    keys = load_api_txt(args.api_file)
    state = StateStore(layout.books_json)

    if args.command == "status":
        if not state.data["books"]:
            print("no books registered yet")
        for name, entry in state.data["books"].items():
            print(f"{name}: status={entry.get('status')} pages_total={entry.get('pages_total')} "
                  f"counts={state.counts(name)} output={entry.get('output')}")
        return 0

    if args.command == "doctor":
        return _doctor(layout, cfg, keys)

    refiner = build_refiner(cfg, keys)
    pipe = Pipeline(layout, cfg, keys, state, refiner)
    if args.command == "run":
        pipe.run_all()
    elif args.command == "run-book":
        pdf = layout.raw / args.name
        if not pdf.exists():
            log.error("not found in raw/: %s", pdf)
            return 2
        pipe.run_book(pdf)
    return 0


def _doctor(layout, cfg, keys) -> int:
    print(f"drive root : {layout.root} (exists={layout.root.exists()})")
    print(f"raw pdfs   : {[p.name for p in layout.list_pdfs()]}")
    print(f"NVIDIA key : {'set' if keys.nvidia else 'MISSING'} "
          f"(nemotron escalation {'ON' if keys.nvidia else 'OFF'})")
    print(f"GEMINI key : {'set' if keys.gemini else 'empty (gemma_local keyless backend is default; fill later to use Gemini)'}")
    print(f"HF token   : {'set' if keys.hf_token else 'empty (only needed if Gemma license not yet accepted on your HF account)'}")
    for mod in ("fitz", "pytesseract", "PIL", "requests"):
        try:
            importlib.import_module(mod)
            print(f"module {mod:12s}: ok")
        except Exception as e:
            print(f"module {mod:12s}: MISSING ({e})")
    try:
        from .ocr.tesseract_ocr import tesseract_available
        print(f"tesseract binary: {'ok' if tesseract_available() else 'MISSING (apt-get install tesseract-ocr)'}")
    except Exception as e:
        print(f"tesseract binary: unknown ({e})")
    print(f"refiner backend: {cfg.get('refiner', 'backend', default='gemma_local')}")
    print(f"config source  : {layout.pipeline_json} (exists={layout.pipeline_json.exists()})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
