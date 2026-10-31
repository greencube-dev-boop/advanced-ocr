"""Unit tests for the dependency-free modules (stdlib only).

Run:  python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ocr_pipeline.drive import DriveLayout
from ocr_pipeline.ocr.oxr_blocks import block_to_markdown, blocks_to_markdown, pages_to_markdown
from ocr_pipeline.pdfutils.chunking import paginate_range, split_range
from ocr_pipeline.settings import DEFAULTS, Cfg, deep_merge


class TestCfg(unittest.TestCase):
    def test_get_default(self):
        self.assertEqual(Cfg({}).get("oxr", "mode", default="X"), "X")

    def test_deep_merge_keeps_siblings(self):
        a = deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 9}})
        self.assertEqual(a, {"a": {"b": 9, "c": 2}})

    def test_defaults_have_v3_sections(self):
        for sec in ("oxr", "chunking", "output"):
            self.assertIn(sec, DEFAULTS)
        # v1 engines and the v2 native-text fast path are gone
        for gone in ("tesseract", "nemotron", "suspicion", "refiner", "page"):
            self.assertNotIn(gone, DEFAULTS)

    def test_parallel_upload_defaults_are_polite(self):
        self.assertEqual(DEFAULTS["chunking"]["max_concurrent"], 2)

    def test_default_mode_is_free_api(self):
        self.assertEqual(DEFAULTS["oxr"]["mode"], "free_api")
        self.assertEqual(DEFAULTS["oxr"]["element_formats"]["table"], "markdown")
        self.assertEqual(DEFAULTS["oxr"]["element_formats"]["formula"], "latex")


class TestOxrBlocks(unittest.TestCase):
    def test_title_becomes_heading(self):
        self.assertEqual(block_to_markdown({"type": "title", "content": "Intro",
                                            "title_level": 2}), "## Intro")

    def test_text_becomes_paragraph(self):
        self.assertEqual(block_to_markdown({"type": "text", "content": "hello"}), "hello")

    def test_table_passthrough(self):
        md = "| A | B |\n| --- | --- |\n| 1 | 2 |"
        self.assertEqual(block_to_markdown({"type": "table", "content": md}), md)

    def test_formula_wrapped_in_display_math(self):
        self.assertEqual(block_to_markdown({"type": "formula", "content": "E=mc^2"}),
                         "$$E=mc^2$$")
        already = "$$E=mc^2$$"
        self.assertEqual(block_to_markdown({"type": "formula", "content": already}), already)

    def test_chemical_equation(self):
        self.assertEqual(block_to_markdown({"type": "chemical_equation", "content": r"\ce{2H2 + O2 -> 2H2O}"}),
                         r"$$\ce{2H2 + O2 -> 2H2O}$$")

    def test_chemical_structure_inline_code(self):
        self.assertEqual(block_to_markdown({"type": "chemical_structure", "content": "CCO"}),
                         "`CCO`")

    def test_code_block_fenced(self):
        self.assertEqual(block_to_markdown({"type": "code_block", "content": "print(1)"}),
                         "```\nprint(1)\n```")

    def test_pictures_and_headers_dropped(self):
        self.assertIsNone(block_to_markdown({"type": "picture", "content": "http://x/y.png"}))
        self.assertIsNone(block_to_markdown({"type": "page_footer", "content": "12"}))
        self.assertIsNone(block_to_markdown({"type": "page_header", "content": "Ch. 1"}))

    def test_unknown_type_kept_as_text(self):
        self.assertEqual(block_to_markdown({"type": "stamp", "content": "APPROVED"}), "APPROVED")

    def test_blocks_joined(self):
        md = blocks_to_markdown([
            {"type": "title", "content": "T", "title_level": 1},
            {"type": "text", "content": "a"},
            {"type": "picture", "content": "x"},
        ], figure_marker=True)
        self.assertIn("# T", md)
        self.assertIn("a", md)
        self.assertIn("[figure omitted]", md)

    def test_caption_after_dropped_figure_is_dropped(self):
        blocks = [{"type": "text", "content": "before"},
                  {"type": "picture", "content": "http://x/fig.png"},
                  {"type": "caption", "content": "Figure 1: A result"},
                  {"type": "text", "content": "after"}]
        md = blocks_to_markdown(blocks)
        self.assertIn("before", md)
        self.assertIn("after", md)
        self.assertNotIn("Figure 1", md)

    def test_caption_alone_is_kept(self):
        self.assertIn("Table 1: B", blocks_to_markdown([{"type": "caption", "content": "Table 1: B"}]))

    def test_pages_picture_counts(self):
        from ocr_pipeline.ocr.oxr_blocks import pages_picture_counts
        pages = [{"page_num": 0, "blocks": [{"type": "Picture"}, {"type": "Text"}]},
                 {"page_num": 1, "blocks": [{"type": "Text"}]}]
        self.assertEqual(pages_picture_counts(pages, page_base=0), {1: 1, 2: 0})

    def test_pages_to_markdown_offsets(self):
        pages = [{"page_num": 0, "blocks": [{"type": "text", "content": "one"}]},
                 {"page_num": 1, "blocks": [{"type": "text", "content": "two"}]}]
        out = pages_to_markdown(pages, page_base=4)  # chunk starts at 0-based page 4
        self.assertEqual(out[5], "one")
        self.assertEqual(out[6], "two")


class TestDrive(unittest.TestCase):
    def _make_tree(self, root):
        raw = root / "raw"
        (raw / "math" / "algebra").mkdir(parents=True)
        (raw / "top.pdf").write_bytes(b"%PDF-1.4")
        (raw / "math" / "mid.pdf").write_bytes(b"%PDF-1.4")
        (raw / "math" / "algebra" / "deep.PDF").write_bytes(b"%PDF-1.4")
        (raw / "math" / "notes.txt").write_text("not a pdf")
        (root / "outside.pdf").write_bytes(b"%PDF-1.4")  # outside raw/ must be ignored

    def test_list_pdfs_scans_subfolders_case_insensitive(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._make_tree(root)
            rels = [p.relative_to(root / "raw").as_posix() for p in DriveLayout(root).list_pdfs()]
            self.assertEqual(rels, ["math/algebra/deep.PDF", "math/mid.pdf", "top.pdf"])

    def test_book_identity_and_paths(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._make_tree(root)
            layout = DriveLayout(root)
            pdf = root / "raw" / "math" / "algebra" / "deep.PDF"
            self.assertEqual(layout.rel_id(pdf), "math/algebra/deep.PDF")
            self.assertEqual(layout.rel_stem(pdf), "math/algebra/deep")
            self.assertEqual(layout.display_name(pdf), "deep")
            self.assertEqual(layout.book_temp(layout.rel_stem(pdf)),
                             root / "temp" / "math" / "algebra" / "deep")
            self.assertEqual(layout.refined / f"{layout.rel_stem(pdf)}.md",
                             root / "refined" / "math" / "algebra" / "deep.md")


class TestChunking(unittest.TestCase):
    def test_paginate_range(self):
        self.assertEqual(paginate_range(0, 4, 2), [(0, 1), (2, 3), (4, 4)])
        self.assertEqual(paginate_range(0, 19, 20), [(0, 19)])
        self.assertEqual(paginate_range(7, 7, 20), [(7, 7)])

    def test_paginate_range_rejects_zero(self):
        with self.assertRaises(ValueError):
            paginate_range(0, 1, 0)

    def test_split_range_no_split_needed(self):
        measure = lambda rng: (rng[1] - rng[0] + 1) * 100
        self.assertEqual(split_range(0, 2, measure, 300), [(0, 2)])

    def test_split_range_halves_until_fit(self):
        measure = lambda rng: (rng[1] - rng[0] + 1) * 100
        pieces = split_range(0, 9, measure, 250)
        # every piece <= 2 pages; pieces cover [0, 9] contiguously
        self.assertTrue(all(rng[1] - rng[0] + 1 <= 2 for rng in pieces))
        self.assertEqual(pieces[0][0], 0)
        self.assertEqual(pieces[-1][1], 9)
        for (_, b), (a2, _) in zip(pieces, pieces[1:]):
            self.assertEqual(b + 1, a2)

    def test_split_range_oversized_single_page_kept(self):
        measure = lambda rng: 10 ** 9
        self.assertEqual(split_range(5, 5, measure, 100), [(5, 5)])


class TestStateThreadSafety(unittest.TestCase):
    def test_concurrent_set_page_keeps_manifest_valid(self):
        import json as jsonlib
        import tempfile
        import threading

        from ocr_pipeline.state import StateStore

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "books.json"
            store = StateStore(path)
            store.register_book("b.pdf", "sha", 40)

            def worker(i: int) -> None:
                for p in range(i * 5 + 1, i * 5 + 6):  # 5 distinct pages per worker
                    store.set_page("b.pdf", p, "done", engine="oxr:free_api")

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            self.assertEqual(store.counts("b.pdf").get("done"), 40)
            data = jsonlib.loads(path.read_text(encoding="utf-8"))  # file must be valid JSON
            self.assertEqual(len(data["books"]["b.pdf"]["pages"]), 40)


class TestCuration(unittest.TestCase):
    BASE = dict(
        section_drop=["references", "bibliography", "index", "preface", "foreword",
                      "acknowledgements", "table of contents", "contents"],
        section_drop_prefix=["appendix", "acknowledg"],
        page_drop=["copyright", "all rights reserved", "isbn", "published by", "doi:"],
        drop_before_first_heading=True,
        first_heading_max_page=60,
        toc_min_lines=4,
        toc_min_ratio=0.5,
        front_matter_max_chars=2000,
    )

    @staticmethod
    def _pg(no, md, pics=0):
        return {"page_no": no, "md": md, "n_pictures": pics}

    def test_heading_normalization_and_matching(self):
        from ocr_pipeline.curation import _matches, _norm_heading
        self.assertEqual(_norm_heading("12. References"), "references")
        self.assertTrue(_matches("Appendix A: Data", set(), ["appendix"]))
        self.assertTrue(_matches("Contents", {"contents"}, []))
        # dominance guard: a body chapter about contents is NOT a ToC section
        self.assertFalse(_matches("Contents of the atmosphere", {"contents"}, []))

    def test_blank_and_figure_only(self):
        from ocr_pipeline.curation import curate
        v = curate([self._pg(1, ""), self._pg(2, "", pics=2)], **self.BASE)
        self.assertEqual(v[0]["reason"], "blank")
        self.assertEqual(v[1]["reason"], "figure-only")

    def test_toc_page_detected_by_dot_leaders(self):
        from ocr_pipeline.curation import curate
        toc = "\n".join(f"Chapter {i} Title {'.' * 8} {i}" for i in range(1, 7))
        v = curate([self._pg(1, toc)], **self.BASE)
        self.assertEqual(v[0]["reason"], "toc-page")

    def test_front_matter_page(self):
        from ocr_pipeline.curation import curate
        md = ("Smoke Test Book\n\nISBN 978-0-123456-78-9\nCopyright the author\n"
              "All rights reserved\nPublished by Example Press")
        v = curate([self._pg(1, md)], **self.BASE)
        self.assertEqual(v[0]["reason"], "front-matter")

    def test_section_ranges_drop_until_next_kept_heading(self):
        from ocr_pipeline.curation import curate
        pages = [
            self._pg(1, "# Chapter 1\n\nBody text one."),
            self._pg(2, "More body text."),
            self._pg(3, "# Acknowledgements\n\nThanks to everyone."),
            self._pg(4, "More thanks."),
            self._pg(5, "# References\n\n[1] A. Author, 2020."),
            self._pg(6, "[2] B. Author, 2021."),
            self._pg(7, "# Chapter 2\n\nBody resumes."),
        ]
        v = curate(pages, **self.BASE)
        got = {d["page_no"]: d["verdict"] for d in v}
        self.assertEqual(got, {1: "keep", 2: "keep", 3: "drop", 4: "drop",
                               5: "drop", 6: "drop", 7: "keep"})
        self.assertTrue(v[2]["reason"].startswith("section:"))
        self.assertEqual(v[6]["verdict"], "keep")

    def test_title_page_with_heading_is_front_matter(self):
        # regression: a title page carries a real heading (the book title) and must
        # not become the front-matter boundary
        from ocr_pipeline.curation import curate
        pages = [
            self._pg(1, "# Smoke Test Book\n\nISBN 978-0-123456-78-9\nCopyright (c) 2026\n"
                        "All rights reserved. Published by Example Press.\nDOI 10.1234/x"),
            self._pg(2, "# Chapter 1\n\nBody."),
        ]
        v = curate(pages, **self.BASE)
        self.assertEqual(v[0]["verdict"], "drop")
        self.assertEqual(v[0]["reason"], "front-matter")
        self.assertEqual(v[1]["verdict"], "keep")

    def test_drop_before_first_heading(self):
        from ocr_pipeline.curation import curate
        pages = [self._pg(1, "Dedication: for my mother."),
                 self._pg(2, "# Chapter 1\n\nBody.")]
        v = curate(pages, **self.BASE)
        self.assertEqual(v[0]["reason"], "front-matter")
        self.assertEqual(v[1]["verdict"], "keep")

    def test_drop_before_first_heading_guard(self):
        from ocr_pipeline.curation import curate
        # no kept heading inside the window -> the rule must not fire
        v = curate([self._pg(1, "Body text without any headings.")],
                   **dict(self.BASE, first_heading_max_page=1))
        self.assertEqual(v[0]["verdict"], "keep")


class TestAssemble(unittest.TestCase):
    def test_pure_join_no_headers_or_markers(self):
        import tempfile
        from ocr_pipeline.assemble import assemble_book

        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "page-0002.final.md").write_text("two", encoding="utf-8")
            (d / "page-0001.final.md").write_text("one", encoding="utf-8")
            (d / "page-0003.oxr.md").write_text("dropped page never assembles", encoding="utf-8")
            out = assemble_book(d, d / "book.md")
            self.assertEqual(out.read_text(encoding="utf-8"), "one\n\ntwo\n")


class TestLedger(unittest.TestCase):
    def test_find_by_sha_and_failed_runs(self):
        import tempfile

        from ocr_pipeline.state import StateStore

        with tempfile.TemporaryDirectory() as td:
            s = StateStore(Path(td) / "books.json")
            s.register_book("a.pdf", "SHA1", 3)
            s.set_book_status("a.pdf", "done")
            s.set_record_fields("a.pdf", {"doi": "10.1/x", "archived": True})
            self.assertIsNotNone(s.find_by_sha("SHA1"))
            self.assertIsNone(s.find_by_sha("SHA1", exclude_book="a.pdf"))
            self.assertIsNone(s.find_by_sha("SHA2"))
            self.assertEqual(s.bump_failed_runs("missing.pdf"), 0)
            s.register_book("b.pdf", "SHA2", 3)
            self.assertEqual(s.bump_failed_runs("b.pdf"), 1)
            self.assertEqual(s.bump_failed_runs("b.pdf"), 2)


if __name__ == "__main__":
    unittest.main()
