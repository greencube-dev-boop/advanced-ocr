"""Unit tests for the dependency-free modules (stdlib only).

Run:  python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ocr_pipeline.keys import parse_api_txt
from ocr_pipeline.ocr import suspicion
from ocr_pipeline.ocr.nemotron_postprocess import postprocess, tabular_to_markdown
from ocr_pipeline.settings import DEFAULTS, Cfg, deep_merge


class TestKeys(unittest.TestCase):
    def test_single_line_nvidia(self):
        k = parse_api_txt("nvapi-abc123\n")
        self.assertEqual(k.nvidia, "nvapi-abc123")
        self.assertEqual(k.gemini, "")

    def test_extra_key_value_lines(self):
        k = parse_api_txt("nvapi-1\nGEMINI_API_KEY=AIza-x\nHF_TOKEN=hf_y\n# comment\n\n")
        self.assertEqual(k.nvidia, "nvapi-1")
        self.assertEqual(k.gemini, "AIza-x")
        self.assertEqual(k.hf_token, "hf_y")

    def test_empty(self):
        self.assertEqual(parse_api_txt("").nvidia, "")
        self.assertEqual(parse_api_txt("\n# only a comment\n").nvidia, "")


class TestCfg(unittest.TestCase):
    def test_get_default(self):
        self.assertEqual(Cfg({}).get("nemotron", "endpoint", default="X"), "X")

    def test_deep_merge_keeps_siblings(self):
        a = deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 9}})
        self.assertEqual(a, {"a": {"b": 9, "c": 2}})

    def test_defaults_have_all_sections(self):
        for sec in ("page", "tesseract", "suspicion", "nemotron", "refiner", "output"):
            self.assertIn(sec, DEFAULTS)


class TestNemotronPostprocess(unittest.TestCase):
    def test_strips_coordinate_and_class_tokens(self):
        s = "<x_0.1><y_0.2>Hello <class_Text>world<x_0.5><y_0.9>"
        out = postprocess(s)
        self.assertNotIn("<x_", out)
        self.assertNotIn("<class_", out)
        self.assertIn("Hello", out)
        self.assertIn("world", out)

    def test_tabular_becomes_markdown_table(self):
        content = "<class_Table>\\begin{tabular}{cc}\nCity & Pop \\\\\nDhaka & 22M \\\\\n\\end{tabular}"
        out = postprocess(content, table_format="markdown")
        self.assertIn("| City | Pop |", out)
        self.assertIn("| --- | --- |", out)
        self.assertIn("| Dhaka | 22M |", out)
        self.assertNotIn("tabular", out)

    def test_tabular_kept_as_latex_when_requested(self):
        content = "<class_Table>\\begin{tabular}{c}\nA \\\\\n\\end{tabular}"
        out = postprocess(content, table_format="latex")
        self.assertIn("\\begin{tabular}", out)

    def test_picture_marker(self):
        self.assertIn("figure", postprocess("<class_Picture>"))

    def test_section_header_becomes_heading(self):
        out = postprocess("Intro <class_Section-header>Chapter 1")
        self.assertIn("##", out)

    def test_tabular_to_markdown_directly(self):
        md = tabular_to_markdown("A & B \\\\\n1 & 2 \\\\")
        self.assertEqual(md.splitlines()[0], "| A | B |")


class TestSuspicion(unittest.TestCase):
    cfg = Cfg(DEFAULTS)

    def test_clean_match_is_not_suspicious(self):
        native = "This is a normal paragraph of text. " * 5
        suspicious, reasons = suspicion.evaluate(native, native, 95.0, False, self.cfg)
        self.assertFalse(suspicious)
        self.assertEqual(reasons, [])

    def test_low_confidence_is_suspicious(self):
        suspicious, reasons = suspicion.evaluate("some text here ok", "some text here ok", 40.0, False, self.cfg)
        self.assertTrue(suspicious)
        self.assertTrue(any("confidence" in r for r in reasons))

    def test_table_escalates_immediately(self):
        suspicious, reasons = suspicion.evaluate("abc", "abc", 99.0, True, self.cfg)
        self.assertTrue(suspicious)
        self.assertIn("table-or-diagram-detected", reasons)

    def test_garbage_ocr_is_suspicious(self):
        suspicious, reasons = suspicion.evaluate("", "\x00\x01\x02\x03" * 20, None, False, self.cfg)
        self.assertTrue(suspicious)
        self.assertTrue(any("garbage" in r for r in reasons))

    def test_length_mismatch_is_suspicious(self):
        suspicious, reasons = suspicion.evaluate("a" * 200, "b" * 60, 95.0, False, self.cfg)
        self.assertTrue(suspicious)
        self.assertTrue(any("mismatch" in r for r in reasons))

    def test_bengali_marks_are_not_garbage(self):
        bengali = "/books Bengali text with vowel signs like \u09bf \u09c7 \u09be/ " * 10
        self.assertLess(suspicion.garbage_ratio(bengali), 0.08)


if __name__ == "__main__":
    unittest.main()
