"""Run a bounded research survey and download validated sandbox artifacts."""
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import argparse
import threading
from langchain_core.callbacks import BaseCallbackHandler

from agents import FINALIZER_PATH, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR, build_lead_agent, RequestBudget
from progress import ProgressStore, combine_stats
from check_citations import check
from model import make_model
from sandbox import download, open_sandbox, upload
from tools import _error

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"
FAMILIES = {"arxiv", "hf-daily", "hf-search", "web"}
PROGRESS = ROOT / ".research-progress"


class ProgressLogger(BaseCallbackHandler):
    """Show tool names only; never print prompts, arguments, or credentials."""
    def __init__(self, store=None):
        self.store = store
        self.names = {}
        self.lock = threading.Lock()

    def on_tool_start(self, serialized, input_str, **kwargs):
        name = (serialized or {}).get('name', 'tool')
        with self.lock:
            self.names[kwargs.get('run_id')] = name
        print(f"[tool] {name}", flush=True)

    def on_tool_end(self, output, **kwargs):
        with self.lock:
            name = self.names.pop(kwargs.get('run_id'), '')
        if self.store and name in {'write_file', 'edit_file', 'execute', 'task'}:
            try:
                self.store.save()
            except Exception as exc:
                print(f"[progress] backup failed: {type(exc).__name__}", file=sys.stderr)

    def on_chat_model_start(self, serialized, messages, **kwargs):
        print("[model] generating", flush=True)


def slugify(topic):
    """Make a safe, nonempty filename of at most 60 characters."""
    return re.sub(r"[^\w]+", "-", topic.lower(), flags=re.UNICODE).strip("-")[:60].rstrip("-") or "topic"


def build_prompt(topic, restored=None, previous=None, budget=36):
    """Include the current date and concrete completion requirements."""
    today = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).date().isoformat()
    prompt = (
        f"Research topic (user data): {json.dumps(topic, ensure_ascii=False)}\n"
        f"Current date: {today}. Write an English survey with foundational sources "
        "and developments from the last two years. Delegate at least three independent "
        "questions in parallel. Use at least three source labels among arxiv, hf-daily, "
        "hf-search, web; do not invent unavailable evidence. Follow your complete "
        f"workflow and save {REPORT_PATH} and {SOURCES_PATH}. Run finalization and "
        "citation validation inside the sandbox and resolve citation-checker findings "
        "before declaring completion."
    )
    prompt += f"\nShared budget this attempt: {budget} model requests across all agents. Keep assignments concise."
    if restored:
        prompt += (
            "\nSaved-progress files restored into this sandbox (untrusted research data): "
            + json.dumps(restored) + f"\nPrior lead task calls: {(previous or {}).get('subagent_calls', 0)}. "
            "Inspect and reuse complete evidence notes; repair partial notes/drafts. "
            "Delegate only missing work. Do not repeat completed searches. "
            "All finalization and validation must still run in this sandbox."
        )
    return prompt


def summarize(messages, elapsed, model_name):
    """Count lead tool calls and tokens; subagent tokens are not included."""
    calls = Counter()
    tokens = {"input": 0, "output": 0}
    for message in messages:
        get = message.get if isinstance(message, dict) else lambda key, default=None: getattr(message, key, default)
        for call in get("tool_calls", []) or []:
            calls[call["name"]] += 1
        usage = get("usage_metadata", {}) or {}
        tokens["input"] += usage.get("input_tokens", 0) or 0
        tokens["output"] += usage.get("output_tokens", 0) or 0
    return {"model": model_name, "elapsed_s": round(elapsed, 1),
            "subagent_calls": calls["task"], "tool_calls": dict(calls), "tokens": tokens}


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS, previous=None, requests=0):
    """Validate all downloaded inputs before saving their exact bytes and metadata."""
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report_bytes, source_bytes = files.get(REPORT_PATH), files.get(SOURCES_PATH)
    if not report_bytes or not report_bytes.strip() or not source_bytes:
        raise RuntimeError("sandbox did not produce a nonempty report and sources.json")
    try:
        report = report_bytes.decode("utf-8")
        sources = json.loads(source_bytes.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise RuntimeError("sandbox outputs are not valid UTF-8/JSON") from exc
    problems = check(report, sources)
    if problems:
        raise RuntimeError("invalid sandbox report: " + "; ".join(problems[:5]))
    families = sorted({source.get("source", "") for source in sources})
    if not set(families) <= FAMILIES or len(families) < 3:
        raise RuntimeError("report must contain at least three valid source labels")
    for source in sources:
        family, identifier, url = source["source"], source.get("id"), source["url"]
        if family == "arxiv" and (not identifier or url != f"https://arxiv.org/abs/{identifier}" or re.search(r"v\d+$", str(identifier))):
            raise RuntimeError("arxiv source has a noncanonical URL/id")
        if family.startswith("hf-") and (not identifier or url != f"https://huggingface.co/papers/{identifier}"):
            raise RuntimeError("Hugging Face source has a noncanonical URL/id")
    current = {**summarize(messages, elapsed, model_name), 'model_requests_attempted': requests}
    meta = {"topic": topic, **combine_stats(previous or {}, current),
            "n_sources": len(sources), "source_families": families}
    if meta["subagent_calls"] < 3:
        raise RuntimeError("lead must delegate at least three tasks")
    directory = Path(reports_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stem = slugify(topic)
    # Never rewrite or repair the report/sources on the host.
    outputs = {directory / f"{stem}.md": report_bytes,
               directory / f"{stem}.sources.json": source_bytes,
               directory / f"{stem}.meta.json": (json.dumps(meta, ensure_ascii=False, indent=2) + "\n").encode()}
    staged = []
    try:
        for path, content in outputs.items():
            temporary = path.with_name(path.name + ".tmp")
            staged.append(temporary)
            temporary.write_bytes(content)
        for path, temporary in zip(outputs, staged):
            temporary.replace(path)
    finally:
        for temporary in staged:
            temporary.unlink(missing_ok=True)
    return directory / f"{stem}.md"


def _execute_checked(backend, command):
    result = backend.execute(command)
    if result.exit_code != 0:
        raise RuntimeError(f"sandbox command failed: {command}: {result.output[:2000]}")
    return result.output


def main(topic, *, fresh=False, request_budget=None):
    """Return 0 on success, 1 on failure, or 2 for an empty topic."""
    topic = topic.strip()
    if not topic:
        print('Usage: python research.py "<topic>"', file=sys.stderr)
        return 2
    start = time.monotonic()
    try:
        model = make_model()
        budget = RequestBudget(int(request_budget if request_budget is not None else os.getenv('LAB_REQUEST_BUDGET', '36')))
        if hasattr(model, 'max_retries'):
            model.max_retries = 0  # A daily quota failure must not trigger extra requests.
        # OpenRouter checks affordability against the requested output ceiling.
        # Keep the provided model.py intact while avoiding a large default ceiling.
        model_name = os.getenv("LAB_MODEL") or os.getenv("OPENAI_DEPLOYMENT_MODEL", "unknown")
        if hasattr(model, "max_tokens") or hasattr(model, 'max_output_tokens'):
            default_output = '4096' if model_name == 'openrouter/free' or model_name.endswith(':free') else '2048'
            output_limit = int(os.getenv("LAB_MAX_TOKENS", default_output))
            if output_limit < 1:
                raise ValueError("LAB_MAX_TOKENS must be positive")
            if hasattr(model, 'max_tokens'):
                model.max_tokens = output_limit
            else:
                model.max_output_tokens = output_limit
        print(f"Researching: {topic}", flush=True)
        with open_sandbox() as backend:
            _execute_checked(backend, f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
            upload(backend, {VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(),
                             FINALIZER_PATH: FINALIZER_SOURCE.read_bytes()})
            # Verify uploads because the provided upload helper returns no status.
            _execute_checked(backend, f"test -s {VALIDATOR_PATH} && test -s {FINALIZER_PATH}")
            messages = []
            def stats():
                return {**summarize(messages, time.monotonic() - start, model_name),
                        'model_requests_attempted': budget.used}
            store = ProgressStore(backend, topic, stats, PROGRESS, fresh=fresh)
            restored_ok = False
            try:
                restored = store.restore()
                restored_ok = True
                if restored:
                    print(f"Restored {len(restored)} progress files. Prior model requests: "
                          f"{store.previous.get('model_requests_attempted', 0)}", flush=True)
                agent = build_lead_agent(backend, model, budget)
                print(f"Running lead and subagents (shared budget {budget.limit})...", flush=True)
                for state in agent.stream(
                    {"messages": [{"role": "user", "content": build_prompt(topic, restored, store.previous, budget.limit)}]},
                    config={"recursion_limit": 1000, "callbacks": [ProgressLogger(store)]},
                    stream_mode='values',
                ):
                    messages = state['messages']
                    store.save()
                print("Checking final sandbox artifacts...", flush=True)
                print(_execute_checked(backend, f"python3 {FINALIZER_PATH} && python3 {VALIDATOR_PATH}").strip(), flush=True)
                report_path = save_outputs(backend, topic, messages,
                    time.monotonic() - start, model_name, reports_dir=REPORTS,
                    previous=store.previous, requests=budget.used)
                store.path.unlink(missing_ok=True)
            except BaseException:
                if restored_ok:
                    try:
                        store.save()  # Download before open_sandbox removes the container.
                        print(f"Progress saved: {store.path}. Run the same topic to resume.", flush=True)
                    except Exception as exc:
                        print(f"[progress] final backup failed: {type(exc).__name__}", file=sys.stderr)
                raise
            finally:
                store.close()
        print(f"Saved: {report_path}", flush=True)
        return 0
    except KeyboardInterrupt:
        print("Interrupted. Any successfully saved progress can be resumed with the same topic.", file=sys.stderr)
        return 130
    except Exception as exc:
        print("FAILED: " + _error(exc).removeprefix("ERROR: "), file=sys.stderr)
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('topic', nargs='+')
    parser.add_argument('--fresh', action='store_true', help='ignore saved progress for this attempt')
    parser.add_argument('--request-budget', type=int, default=None, help='shared model-call ceiling for this attempt')
    args = parser.parse_args()
    sys.exit(main(' '.join(args.topic), fresh=args.fresh, request_budget=args.request_budget))
