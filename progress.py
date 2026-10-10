"""Local recovery snapshots, separate from validated submission reports."""
import hashlib
import json
import os
import shlex
import threading
from pathlib import Path, PurePosixPath

from agents import NOTES_DIR, REPORT_PATH, SOURCES_PATH
from sandbox import download, upload


def allowed_path(path):
    """Only restore research artifacts, never host paths or executable scripts."""
    if not isinstance(path, str) or '..' in PurePosixPath(path).parts:
        return False
    return path in {REPORT_PATH, SOURCES_PATH} or (
        path.startswith(NOTES_DIR + '/') and path.endswith('.md'))


def combine_stats(previous, current):
    """Accumulate actual lead activity across attempts without counting it twice."""
    result = dict(current)
    for key in ('elapsed_s', 'subagent_calls', 'model_requests_attempted'):
        result[key] = previous.get(key, 0) + current.get(key, 0)
    result['elapsed_s'] = round(result['elapsed_s'], 1)
    for key in ('tool_calls', 'tokens'):
        old, new = previous.get(key, {}), current.get(key, {})
        result[key] = {name: old.get(name, 0) + new.get(name, 0) for name in old.keys() | new.keys()}
    return result


class ProgressStore:
    def __init__(self, backend, topic, stats, root, *, fresh=False):
        self.backend, self.topic, self.stats = backend, topic, stats
        digest = hashlib.sha256(topic.encode()).hexdigest()[:16]
        self.path = Path(root) / f'{digest}.json'
        self.previous = {}
        self.restored = []
        self.restored_artifacts = {}
        self._lock = threading.RLock()
        self.active = True
        self.fresh = fresh

    def restore(self):
        if self.fresh or not self.path.exists():
            return []
        snapshot = json.loads(self.path.read_text(encoding='utf-8'))
        if snapshot.get('version') != 1 or snapshot.get('topic') != self.topic:
            raise ValueError('saved progress version/topic mismatch')
        artifacts = snapshot.get('artifacts')
        stats = snapshot.get('stats')
        if not isinstance(artifacts, dict) or not all(allowed_path(p) and isinstance(v, str) for p, v in artifacts.items()):
            raise ValueError('saved progress contains invalid artifact paths/content')
        if not isinstance(stats, dict):
            raise ValueError('saved progress has invalid statistics')
        for key in ('elapsed_s', 'subagent_calls', 'model_requests_attempted'):
            if not isinstance(stats.get(key, 0), (int, float)) or stats.get(key, 0) < 0:
                raise ValueError('saved progress has invalid counts')
        for key in ('tool_calls', 'tokens'):
            values = stats.get(key, {})
            if not isinstance(values, dict) or not all(isinstance(k, str) and type(v) is int and v >= 0 for k, v in values.items()):
                raise ValueError('saved progress has invalid counts')
        self._check_secrets(json.dumps(snapshot))
        upload(self.backend, {p: v.encode('utf-8') for p, v in artifacts.items()})
        if artifacts:
            actual = download(self.backend, list(artifacts))
            if any(actual.get(p) != v.encode('utf-8') for p, v in artifacts.items()):
                raise RuntimeError('could not restore all progress files into sandbox')
        self.previous = stats
        self.restored = sorted(artifacts)
        self.restored_artifacts = dict(artifacts)
        return self.restored

    def _check_secrets(self, text):
        for key in ('LAB_API_KEY', 'OPENAI_API_KEY', 'GOOGLE_API_KEY', 'ANTHROPIC_API_KEY', 'EXA_API_KEY', 'DAYTONA_API_KEY'):
            value = os.getenv(key, '')
            if value and value in text:
                raise ValueError('refusing to save/restore progress containing a configured API key')

    def cumulative_stats(self):
        return combine_stats(self.previous, self.stats())

    def save(self):
        with self._lock:
            if not self.active:
                return
            # The script is fixed, runs inside the sandbox, and does not read secrets.
            script = (
                'import json\nfrom pathlib import Path\n'
                f'root=Path({NOTES_DIR!r})\n'
                'paths=[str(p) for p in root.rglob("*.md") if p.is_file() and not p.is_symlink()]\n'
                f'paths += [p for p in [{REPORT_PATH!r}, {SOURCES_PATH!r}] if Path(p).is_file() and not Path(p).is_symlink()]\n'
                'print(json.dumps(paths))\n'
            )
            result = self.backend.execute('python3 -c ' + shlex.quote(script))
            if result.exit_code != 0:
                raise RuntimeError('could not list sandbox progress files')
            paths = json.loads(result.output)
            if not isinstance(paths, list) or not all(allowed_path(p) for p in paths):
                raise ValueError('sandbox returned invalid progress paths')
            files = download(self.backend, paths) if paths else {}
            artifacts = {}
            for path in paths:
                data = files.get(path)
                if data is None:
                    raise RuntimeError('sandbox progress file disappeared during backup')
                artifacts[path] = data.decode('utf-8')
            snapshot = {'version': 1, 'topic': self.topic, 'stats': self.cumulative_stats(), 'artifacts': artifacts}
            text = json.dumps(snapshot, ensure_ascii=False, indent=2) + '\n'
            self._check_secrets(text)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix('.tmp')
            try:
                temporary.write_text(text, encoding='utf-8')
                temporary.replace(self.path)
            finally:
                temporary.unlink(missing_ok=True)

    def close(self):
        with self._lock:
            self.active = False
