import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from check_citations import check


class CitationTests(unittest.TestCase):
    def setUp(self):
        self.sources = [
            {"n": 1, "url": "https://example.org/a"},
            {"n": 2, "url": "https://example.org/b"},
        ]
        self.refs = "\n## References\n[1] A. https://example.org/a\n[2] B. https://example.org/b\n"

    def test_valid_single_grouped_and_range_citations(self):
        for body in ("Claims [1][2].", "Claims [1, 2].", "Claims [1-2].", "Claims [1–2]."):
            with self.subTest(body=body):
                self.assertEqual(check(body + self.refs, self.sources), [])

    def test_ignores_code_and_markdown_links(self):
        body = "Claims [1][2].\n`[99]` ``[98]`` [97](https://example.org)\n"
        body += "```python\n[96]\n## References\n```\n~~~\n[95]\n~~~\n    [94]\n"
        self.assertEqual(check(body + self.refs, self.sources), [])

    def test_references_do_not_count_as_body_citations(self):
        self.assertTrue(any("never cited" in p for p in check(self.refs, self.sources)))

    def test_rejects_bad_sources(self):
        for sources in ([], {}, [None], [{"n": True, "url": "bad"}],
                        self.sources + [self.sources[0]],
                        [{"n": 1, "url": "https://example.org/a"},
                         {"n": 2, "url": "https://example.org/a"}]):
            with self.subTest(sources=sources):
                self.assertTrue(check("Claims [1][2]." + self.refs, sources))

    def test_rejects_missing_unknown_duplicate_and_mismatched_references(self):
        reports = [
            "Claims [1][2].",
            "Claims [1][2][3]." + self.refs,
            "Claims [1][2]." + self.refs.replace("[2] B. https://example.org/b\n", ""),
            "Claims [1][2]." + self.refs + "[1] A. https://example.org/a\n",
            "Claims [1][2]." + self.refs + "[3] C. https://example.org/c\n",
            "Claims [1][2]." + self.refs.replace("https://example.org/b", "https://example.org/wrong"),
            "Claims [1][2]." + self.refs.replace("[1] A.", "[1] https://example.org/extra A."),
            "Claims [2-1]." + self.refs,
        ]
        for report in reports:
            with self.subTest(report=report):
                self.assertTrue(check(report, self.sources))

    def test_cli_exit_codes(self):
        script = Path(__file__).resolve().parents[1] / "check_citations.py"
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.md"
            sources = Path(directory) / "sources.json"
            sources.write_text(json.dumps(self.sources), encoding="utf-8")
            for body, expected in (("Claims [1][2].", 0), ("Claims [3].", 1)):
                report.write_text(body + self.refs, encoding="utf-8")
                result = subprocess.run([sys.executable, str(script), str(report), str(sources)],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, expected)
                if expected == 0:
                    self.assertIn("OK: 2 sources, all citations resolve", result.stdout)


if __name__ == "__main__":
    unittest.main()
