"""Unit tests for the dependency-free modules (stdlib only).

Run:  python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

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

    def test_pages_to_markdown_offsets(self):
        pages = [{"page_num": 0, "blocks": [{"type": "text", "content": "one"}]},
                 {"page_num": 1, "blocks": [{"type": "text", "content": "two"}]}]
        out = pages_to_markdown(pages, page_base=4)  # chunk starts at 0-based page 4
        self.assertEqual(out[5], "one")
        self.assertEqual(out[6], "two")


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


if __name__ == "__main__":
    unittest.main()
