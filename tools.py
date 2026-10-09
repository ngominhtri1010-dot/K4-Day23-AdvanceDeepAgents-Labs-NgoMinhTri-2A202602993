"""tools.py - STUDENT IMPLEMENTS.  Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import json
import os
import time
import random
import re
import threading
from urllib.parse import quote
import xml.etree.ElementTree

import httpx
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"
_ARXIV_LOCK = threading.Lock()
_LAST_ARXIV_CALL = None
_ARXIV_UNAVAILABLE_UNTIL = 0.0


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


# ---- Part 1: retry helper ----
def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Call fn(); when it raises RetryableError, wait and call it again.

    PSEUDO-CODE:
      for attempt in 0 .. attempts-1:
          try: return fn()
          except RetryableError as e:
              if this was the last attempt: raise
              delay = e.retry_after if the server told us, else exponential backoff base * 2**attempt
              cap the delay at `cap` seconds; add random jitter to the exponential case
              sleep(delay)
    Use it to wrap EVERY network call below. Also treat these as retryable: HTTP 429/500/502/503/504,
    httpx.TransportError (timeouts, connection resets). Read the Retry-After header when present.
    """
    if attempts < 1 or base < 0 or cap < 0:
        raise ValueError("attempts must be positive; base and cap must be nonnegative")
    for attempt in range(attempts):
        try:
            return fn()
        except RetryableError as exc:
            if attempt == attempts - 1:
                raise
            if exc.retry_after is not None:
                delay = min(cap, max(0.0, float(exc.retry_after)))
            else:
                delay = min(cap, base * 2 ** attempt + random.uniform(0, base))
            time.sleep(delay)


def _request(method, url, **kwargs):
    """One network attempt; callers wrap this in with_retry."""
    try:
        response = httpx.request(method, url, timeout=45, follow_redirects=True, **kwargs)
    except httpx.TransportError as exc:
        raise RetryableError(str(exc)) from exc
    if response.status_code in {429, 500, 502, 503, 504}:
        try:
            retry_after = float(response.headers.get("Retry-After", ""))
        except ValueError:
            retry_after = None
        raise RetryableError(f"HTTP {response.status_code}", retry_after)
    response.raise_for_status()
    return response


def _error(exc):
    text = f"{type(exc).__name__}: {exc}"
    for name in ("EXA_API_KEY", "LAB_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
        key = os.getenv(name, "")
        if key:
            text = text.replace(key, "[REDACTED]").replace(quote(key, safe=""), "[REDACTED]")
    text = re.sub(r"(?i)(exaApiKey=)[^\s&\"']+", r"\1[REDACTED]", text)
    return "ERROR: " + text


def _clean(value):
    return " ".join(str(value or "").split())


def _records(records):
    return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"


def _hf_records(items, *, prefer_ai=False, keyword=""):
    if not isinstance(items, list):
        raise ValueError("Expected a list of Hugging Face papers")
    records = []
    for item in items:
        paper = item.get("paper") if isinstance(item, dict) else None
        if not isinstance(paper, dict) or not paper.get("id"):
            continue
        def value(key):
            return paper.get(key) or item.get(key)
        summary = (value("ai_summary") if prefer_ai else None) or value("summary")
        if keyword and keyword.casefold() not in (_clean(value("title")) + " " + _clean(summary)).casefold():
            continue
        records.append({
            "id": str(paper["id"]), "url": f"https://huggingface.co/papers/{paper['id']}",
            "published": _clean(value("publishedAt"))[:10], "title": _clean(value("title")),
            "summary": _clean(summary)[:600], "upvotes": value("upvotes") or 0,
            "github": value("githubRepo") or "", "stars": value("githubStars") or 0,
        })
    return records


def _mcp_message(response):
    if response.text.lstrip().startswith("{"):
        return response.json()
    # SSE events can have multiple data lines and include unrelated notifications.
    for event in re.split(r"\r?\n\r?\n", response.text):
        data = "\n".join(line[5:].lstrip() for line in event.splitlines() if line.startswith("data:"))
        if data and data != "[DONE]":
            message = json.loads(data)
            if "result" in message or "error" in message:
                return message
    raise ValueError("Exa response contains no JSON-RPC result")


def _rate_limited(meta):
    if not isinstance(meta, dict):
        return False
    for key, value in meta.items():
        normalized = re.sub(r"[^a-z]", "", key.lower())
        if ("ratelimit" in normalized or "throttl" in normalized) and value:
            return True
        if _rate_limited(value):
            return True
    return False


def _exa(name, arguments):
    def call():
        key = os.getenv("EXA_API_KEY", "").strip()
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        if key:
            headers["x-api-key"] = key
        message = _mcp_message(_request("POST", EXA_URL, headers=headers,
            params={"exaApiKey": key} if key else {},
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                  "params": {"name": name, "arguments": arguments}}))
        if "error" in message:
            raise ValueError(f"Exa JSON-RPC error: {message['error']}")
        result = message["result"]
        text = "\n".join(part.get("text", "") for part in result.get("content", [])
                         if part.get("type") == "text").strip()
        if _rate_limited(result.get("_meta", {})) or re.search(
                r"rate[ -]?limit(?:ed| exceeded| reached)|too many requests", text, re.I):
            raise RetryableError("Exa rate limit reached", retry_after=20)
        if result.get("isError"):
            raise ValueError(text or "Exa tool failed")
        return text or "NO RESULTS"
    return with_retry(call, attempts=8, base=3, cap=60)


# ---- Part 2: arXiv ----
@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by keywords, newest first. Returns a JSON list of {id, url, published, title, summary}."""
    try:
        global _ARXIV_UNAVAILABLE_UNTIL
        terms = [term for term in re.findall(r"[^\W_]+(?:-[^\W_]+)*", query)
                 if term.lower() not in {"and", "or", "not", "all", "ti", "au", "abs"}]
        if not terms:
            return "NO RESULTS"
        if time.monotonic() < _ARXIV_UNAVAILABLE_UNTIL:
            return "ERROR: arXiv temporarily unavailable after retries; use another source"
        def call():
            global _LAST_ARXIV_CALL
            with _ARXIV_LOCK:
                now = time.monotonic()
                if _LAST_ARXIV_CALL is not None:
                    time.sleep(max(0, 3 - (now - _LAST_ARXIV_CALL)))
                _LAST_ARXIV_CALL = time.monotonic()
                return _request("GET", ARXIV_URL, params={
                    "search_query": " AND ".join(f"all:{term}" for term in terms),
                    "sortBy": "submittedDate", "sortOrder": "descending",
                    "max_results": max(1, min(30, max_results)), "start": 0})
        try:
            response = with_retry(call, attempts=5, base=3, cap=60)
        except RetryableError:
            _ARXIV_UNAVAILABLE_UNTIL = time.monotonic() + 600
            raise
        root = xml.etree.ElementTree.fromstring(response.text)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        records = []
        for entry in root.findall("a:entry", ns):
            identifier = re.sub(r"v\d+$", "", (entry.findtext("a:id", "", ns)).split("/abs/")[-1])
            if not identifier:
                continue
            records.append({"id": identifier, "url": f"https://arxiv.org/abs/{identifier}",
                "published": entry.findtext("a:published", "", ns)[:10],
                "title": _clean(entry.findtext("a:title", "", ns)),
                "summary": _clean(entry.findtext("a:summary", "", ns))[:600]})
        return _records(records)
    except Exception as exc:
        return _error(exc)


# ---- Part 3: Hugging Face ----
@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic)."""
    try:
        params = {"limit": max(1, min(100, limit))}
        if date:
            params["date"] = date
        response = with_retry(lambda: _request("GET", HF_DAILY_URL, params=params))
        records = _hf_records(response.json(), keyword=keyword)
        records.sort(key=lambda r: r["upvotes"], reverse=True)
        return _records(records)
    except Exception as exc:
        return _error(exc)


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars}."""
    try:
        if not query.strip():
            return "NO RESULTS"
        response = with_retry(lambda: _request("GET", HF_SEARCH_URL,
            params={"q": query, "limit": max(1, min(50, limit))}))
        return _records(_hf_records(response.json(), prefer_ai=True))
    except Exception as exc:
        return _error(exc)


# ---- Part 4: web search / fetch through the Exa MCP endpoint ----
@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa). Describe the ideal page in natural language. Returns clean text of the top results with URLs."""
    try:
        if not query.strip():
            return "NO RESULTS"
        return _exa("web_search_exa", {"query": query,
            "objective": objective.strip() or f"Find reliable sources about {query}",
            "numResults": max(1, min(10, num_results))})
    except Exception as exc:
        return _error(exc)


@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page) as markdown. Long pages are truncated."""
    try:
        if not url.startswith(("https://", "http://")):
            raise ValueError("url must start with http:// or https://")
        return _exa("web_fetch_exa", {"urls": [url]})[:12000]
    except Exception as exc:
        return _error(exc)


# ---- Part 5: registry (the researcher subagent gets exactly these) ----
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        try:
            print(f"== {name}\n{fn.invoke(args)[:400]}\n")
        except NotImplementedError as exc:
            print(f"== {name}: not implemented yet ({exc})\n")
