"""CLI entry point: python -m ocr_pipeline <command> [--drive-root ...]"""
from __future__ import annotations

import argparse
import importlib
import shutil
import sys
from pathlib import Path

from .drive import DriveLayout
from .logging_setup import setup_logging
from .pipeline import Pipeline
from .settings import DEFAULTS, Cfg, load_config, save_config
from .state import StateStore


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="ocr_pipeline",
        description="Advanced OCR pipeline v2 (OXR-1.0) — the Colab notebook only calls this.")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--drive-root", default="/content/drive/MyDrive/OCR Engine",
                        help="Path to the 'OCR Engine' folder on the mounted Drive")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("run", help="Process all pending PDFs in raw/ (resumable)", parents=[common])
    p_book = sub.add_parser("run-book", help="Process a single PDF from raw/", parents=[common])
    p_book.add_argument("name",
                        help="path relative to raw/, e.g. mybook.pdf or math/algebra/book.pdf")
    sub.add_parser("status", help="Show the resume state (books.json)", parents=[common])
    p_reset = sub.add_parser("reset", help="Forget a book's state and temp files (reprocess)",
                             parents=[common])
    p_reset.add_argument("name", help="path relative to raw/, e.g. math/algebra/book.pdf")
    sub.add_parser("doctor", help="Check environment: deps, drive, OXR reachability",
                   parents=[common])
    sub.add_parser("init-config", help="Write the default pipeline.json into Drive config/",
                   parents=[common])
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
    state = StateStore(layout.books_json)

    if args.command == "status":
        if not state.data["books"]:
            print("no books registered yet")
        for name, entry in state.data["books"].items():
            print(f"{name}: status={entry.get('status')} pages_total={entry.get('pages_total')} "
                  f"counts={state.counts(name)} output={entry.get('output')}")
        return 0

    if args.command == "reset":
        rel = Path(args.name)
        shutil.rmtree(layout.book_temp(rel.with_suffix("").as_posix()), ignore_errors=True)
        state.forget(rel.as_posix())
        print(f"reset: {rel.as_posix()} (temp cleared, state forgotten)")
        return 0

    if args.command == "doctor":
        return _doctor(layout, cfg)

    pipe = Pipeline(layout, cfg, state)
    if args.command == "run":
        pipe.run_all()
    elif args.command == "run-book":
        pdf = layout.raw / args.name
        if not pdf.exists():
            log.error("not found under raw/: %s (use the path relative to raw/, "
                      "see `doctor` for the discovered PDFs)", pdf)
            return 2
        pipe.run_book(pdf)
    return 0


def _doctor(layout, cfg) -> int:
    import requests

    print(f"drive root : {layout.root} (exists={layout.root.exists()})")
    print(f"raw pdfs   : {[p.name for p in layout.list_pdfs()]}")
    for mod in ("fitz", "requests"):
        try:
            importlib.import_module(mod)
            print(f"module {mod:10s}: ok")
        except Exception as e:
            print(f"module {mod:10s}: MISSING ({e})")

    o = cfg.section("oxr")
    mode = o.get("mode", default="free_api")
    base = (o.get("base_url", default="")
            or o.get("free_api_base_url" if mode == "free_api" else "local_base_url",
                     default="https://free-oxr.somark.ai/api/v1"))
    print(f"oxr mode   : {mode}")
    print(f"oxr base   : {base}")
    print(f"chunking   : max_pages={cfg.get('chunking', 'max_pages', default=20)} "
          f"max_bytes={cfg.get('chunking', 'max_bytes', default=9437184)}")
    try:
        r = requests.get(base, timeout=10)
        print(f"oxr reach  : HTTP {r.status_code} (reachable)")
    except Exception as e:
        hint = ("start the local OXR services, or set oxr.mode=free_api"
                if mode == "local" else "check network/proxy")
        print(f"oxr reach  : UNREACHABLE ({e}) — {hint}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
