import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage
from agents import NOTES_DIR, RequestBudget
from progress import ProgressStore, allowed_path, combine_stats
import research


class MemorySandbox:
    def __init__(self):
        self.files = {}
        self.closed = False

    def execute(self, command):
        output = json.dumps([p for p in self.files if allowed_path(p)]) if command.startswith('python3 -c ') else 'OK'
        return SimpleNamespace(exit_code=0, output=output)

    def upload_files(self, files):
        self.files.update(files)
        return []

    def download_files(self, paths):
        return [SimpleNamespace(path=p, content=self.files.get(p)) for p in paths]


class ProgressTests(unittest.TestCase):
    def test_file_write_callback_saves_before_next_model_call(self):
        store = Mock()
        logger = research.ProgressLogger(store)
        logger.on_tool_start({'name': 'write_file'}, 'content not printed', run_id='write')
        logger.on_tool_end('written', run_id='write')
        store.save.assert_called_once()
        logger.on_tool_start({'name': 'web_fetch'}, 'url not printed', run_id='fetch')
        logger.on_tool_end('source', run_id='fetch')
        store.save.assert_called_once()

    def test_atomic_snapshot_restore_exact_bytes_and_stats(self):
        stats = {'subagent_calls': 3, 'elapsed_s': 1, 'tool_calls': {'task': 3},
                 'tokens': {'input': 10, 'output': 5}, 'model_requests_attempted': 4}
        original = MemorySandbox()
        path = NOTES_DIR + '/01.md'
        original.files[path] = '# Evidence\nA claim with Unicode: tiếng Việt.\n'.encode()
        with tempfile.TemporaryDirectory() as folder:
            store = ProgressStore(original, 'topic', lambda: stats, folder)
            store.save()
            restored = MemorySandbox()
            next_store = ProgressStore(restored, 'topic', lambda: {}, folder)
            self.assertEqual(next_store.restore(), [path])
            self.assertEqual(restored.files[path], original.files[path])
            self.assertEqual(next_store.previous, stats)
            next_store.save()
            saved = json.loads(next_store.path.read_text())
            self.assertEqual(saved['stats']['subagent_calls'], 3)
            self.assertEqual(list(Path(folder).glob('*.tmp')), [])

    def test_invalid_paths_and_secret_snapshots_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            store = ProgressStore(MemorySandbox(), 'topic', lambda: {}, folder)
            store.path.write_text(json.dumps({'version': 1, 'topic': 'topic', 'stats': {},
                                             'artifacts': {NOTES_DIR + '/../evil.md': 'bad'}}))
            with self.assertRaises(ValueError):
                store.restore()
            self.assertEqual(store.backend.files, {})
            store.backend.files[NOTES_DIR + '/01.md'] = b'secret-test-api-key'
            with patch.dict('os.environ', {'LAB_API_KEY': 'secret-test-api-key'}):
                with self.assertRaises(ValueError):
                    store.save()

    def test_fresh_and_closed_store(self):
        with tempfile.TemporaryDirectory() as folder:
            store = ProgressStore(MemorySandbox(), 'topic', lambda: {}, folder, fresh=True)
            store.path.write_text('invalid old snapshot')
            self.assertEqual(store.restore(), [])
            store.close()
            store.save()
            self.assertEqual(store.path.read_text(), 'invalid old snapshot')

    def test_budget_is_shared_under_concurrency(self):
        budget = RequestBudget(5)
        def attempt(_):
            try:
                budget.before_model({}, None)
                return True
            except RuntimeError:
                return False
        with ThreadPoolExecutor(max_workers=8) as pool:
            self.assertEqual(sum(pool.map(attempt, range(20))), 5)
        self.assertEqual(budget.used, 5)

    def test_fail_then_resume_and_save_validated_report(self):
        sandboxes = []
        @contextmanager
        def open_box():
            backend = MemorySandbox()
            sandboxes.append(backend)
            try:
                yield backend
            finally:
                backend.closed = True

        attempt = [0]
        notes = NOTES_DIR + '/01.md'
        sources = [
            {'n': 1, 'id': '2501.00001', 'url': 'https://arxiv.org/abs/2501.00001', 'source': 'arxiv'},
            {'n': 2, 'id': '2501.00002', 'url': 'https://huggingface.co/papers/2501.00002', 'source': 'hf-search'},
            {'n': 3, 'id': '', 'url': 'https://example.org', 'source': 'web'},
        ]
        report = '# Survey\nEvidence [1][2][3].\n## References\n' + ''.join(
            f"[{s['n']}] Source. {s['url']}\n" for s in sources)
        first_message = AIMessage(content='', tool_calls=[
            {'name': 'task', 'args': {}, 'id': str(i)} for i in range(3)])

        def build(backend, model, budget, *, resume=False):
            attempt[0] += 1
            current_attempt = attempt[0]
            def stream(inputs, **kwargs):
                budget.used = 2
                if current_attempt == 1:
                    backend.files[notes] = b'# Retrieved evidence\n'
                    yield {'messages': [first_message]}
                    raise RuntimeError('daily quota exhausted')
                self.assertEqual(backend.files[notes], b'# Retrieved evidence\n')
                self.assertIn('Saved-progress files', inputs['messages'][0]['content'])
                self.assertTrue(resume)
                self.assertIn('# Retrieved evidence', inputs['messages'][0]['content'])
                backend.files[research.REPORT_PATH] = report.encode()
                backend.files[research.SOURCES_PATH] = json.dumps(sources).encode()
                yield {'messages': [AIMessage(content='Complete')]}
            return SimpleNamespace(stream=stream)

        with tempfile.TemporaryDirectory() as folder, \
                patch('research.PROGRESS', Path(folder) / 'progress'), \
                patch('research.REPORTS', Path(folder) / 'reports'), \
                patch('research.open_sandbox', open_box), \
                patch('research.make_model', return_value=Mock()), \
                patch('research.build_lead_agent', side_effect=build):
            self.assertEqual(research.main('test topic'), 1)
            self.assertTrue(sandboxes[0].closed)
            self.assertFalse((Path(folder) / 'reports').exists())
            self.assertEqual(len(list((Path(folder) / 'progress').glob('*.json'))), 1)
            self.assertEqual(research.main('test topic'), 0)
            self.assertTrue(sandboxes[1].closed)
            output = Path(folder) / 'reports' / 'test-topic.md'
            self.assertEqual(output.read_bytes(), report.encode())
            meta = json.loads(output.with_suffix('.meta.json').read_text())
            self.assertEqual(meta['subagent_calls'], 3)
            self.assertEqual(meta['model_requests_attempted'], 4)
            self.assertEqual(list((Path(folder) / 'progress').glob('*.json')), [])


if __name__ == '__main__':
    unittest.main()
