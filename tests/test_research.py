import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from unittest.mock import Mock
from contextlib import contextmanager

from langchain_core.messages import AIMessage
import research


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.sources = [
            {"n": 1, "id": "2501.00001", "url": "https://arxiv.org/abs/2501.00001", "source": "arxiv"},
            {"n": 2, "id": "2501.00002", "url": "https://huggingface.co/papers/2501.00002", "source": "hf-search"},
            {"n": 3, "id": "", "url": "https://example.org", "source": "web"},
        ]
        self.report = "# Survey\nClaims [1][2][3].\n\n## References\n" + "".join(
            f"[{s['n']}] Source. {s['url']}\n" for s in self.sources)
        self.messages = [AIMessage(content="", tool_calls=[
            {"name": "task", "args": {}, "id": str(i)} for i in range(3)],
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})]

    def test_slug_and_metadata(self):
        self.assertEqual(research.slugify("../../x"), "x")
        self.assertEqual(research.slugify(""), "topic")
        self.assertLessEqual(len(research.slugify("a" * 100)), 60)
        meta = research.summarize(self.messages, 1.234, "test-model")
        self.assertEqual(meta["subagent_calls"], 3)
        self.assertEqual(meta["tokens"], {"input": 10, "output": 5})
        self.assertEqual(meta["elapsed_s"], 1.2)

    def test_save_preserves_sandbox_bytes(self):
        data = {research.REPORT_PATH: self.report.encode(),
                research.SOURCES_PATH: json.dumps(self.sources).encode()}
        with tempfile.TemporaryDirectory() as folder, patch("research.download", return_value=data):
            path = research.save_outputs(None, "Test", self.messages, 1, "model", folder)
            self.assertEqual(path.read_bytes(), data[research.REPORT_PATH])
            self.assertEqual(path.with_suffix(".sources.json").read_bytes(), data[research.SOURCES_PATH])
            meta = json.loads(path.with_suffix(".meta.json").read_text())
            self.assertEqual(meta["source_families"], ["arxiv", "hf-search", "web"])

    def test_invalid_outputs_write_nothing(self):
        cases = [{}, {research.REPORT_PATH: b"   ", research.SOURCES_PATH: b"[]"},
                 {research.REPORT_PATH: self.report.encode(), research.SOURCES_PATH: b"invalid"},
                 {research.REPORT_PATH: b"Claims [99]", research.SOURCES_PATH: json.dumps(self.sources).encode()}]
        for data in cases:
            with self.subTest(data=data), tempfile.TemporaryDirectory() as folder, patch("research.download", return_value=data):
                with self.assertRaises(RuntimeError):
                    research.save_outputs(None, "Test", self.messages, 1, "model", folder)
                self.assertEqual(list(Path(folder).iterdir()), [])

    def test_empty_topic_does_not_create_model(self):
        with patch("research.make_model") as model:
            self.assertEqual(research.main("  "), 2)
            model.assert_not_called()

    def test_failed_agent_leaves_context_and_does_not_save(self):
        cleaned = []
        backend = Mock()
        backend.execute.return_value = Mock(exit_code=0, output="[]")
        @contextmanager
        def sandbox():
            try:
                yield backend
            finally:
                cleaned.append(True)
        agent = Mock()
        agent.stream.side_effect = RuntimeError("model unavailable")
        with tempfile.TemporaryDirectory() as folder, patch("research.PROGRESS", Path(folder)), \
                patch("research.open_sandbox", sandbox), patch("research.make_model", return_value=Mock()), \
                patch("research.upload"), patch("research.build_lead_agent", return_value=agent), \
                patch("research.save_outputs") as save:
            self.assertEqual(research.main("test"), 1)
            save.assert_not_called()
        self.assertEqual(cleaned, [True])


if __name__ == "__main__":
    unittest.main()
