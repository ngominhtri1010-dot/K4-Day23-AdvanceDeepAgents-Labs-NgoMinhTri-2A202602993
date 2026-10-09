import json
import unittest
from unittest.mock import Mock, patch

import httpx
import tools


def response(text="", status=200, headers=None):
    return httpx.Response(status, text=text, headers=headers,
                          request=httpx.Request("GET", "https://example.org"))


class ToolTests(unittest.TestCase):
    def test_retry_backoff_jitter_cap_and_no_final_sleep(self):
        call = Mock(side_effect=tools.RetryableError("busy"))
        with patch("tools.time.sleep") as sleep, patch("tools.random.uniform", return_value=0.5):
            with self.assertRaises(tools.RetryableError):
                tools.with_retry(call, attempts=4, base=1, cap=3)
        self.assertEqual(call.call_count, 4)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [1.5, 2.5, 3])

    def test_retry_after_and_non_retryable_error(self):
        call = Mock(side_effect=[tools.RetryableError("busy", 7), "ok"])
        with patch("tools.time.sleep") as sleep:
            self.assertEqual(tools.with_retry(call), "ok")
            sleep.assert_called_once_with(7)
        call = Mock(side_effect=ValueError("bad"))
        with patch("tools.time.sleep") as sleep, self.assertRaises(ValueError):
            tools.with_retry(call)
        sleep.assert_not_called()
        self.assertEqual(call.call_count, 1)

    def test_http_and_transport_errors(self):
        with patch("tools.httpx.request", return_value=response(status=429, headers={"Retry-After": "4"})):
            with self.assertRaises(tools.RetryableError) as error:
                tools._request("GET", "https://example.org")
            self.assertEqual(error.exception.retry_after, 4)
        with patch("tools.httpx.request", side_effect=httpx.ConnectError("offline")):
            with self.assertRaises(tools.RetryableError):
                tools._request("GET", "https://example.org")
        with patch("tools.httpx.request", return_value=response(status=401)):
            with self.assertRaises(httpx.HTTPStatusError):
                tools._request("GET", "https://example.org")

    def test_arxiv_normalization_and_empty_query(self):
        xml = '<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/2501.00001v2</id><published>2025-01-02T00:00:00Z</published><title> A\n B </title><summary> Summary </summary></entry></feed>'
        with patch("tools._request", return_value=response(xml)) as request, patch("tools.time.sleep"), patch.object(tools, "_LAST_ARXIV_CALL", None):
            record = json.loads(tools.arxiv_search.invoke({"query": 'all:"world" AND model OR', "max_results": 99}))[0]
            self.assertEqual(record["id"], "2501.00001")
            self.assertEqual(record["url"], "https://arxiv.org/abs/2501.00001")
            self.assertEqual(record["title"], "A B")
            self.assertEqual(request.call_args.kwargs["params"]["search_query"], "all:world AND all:model")
            self.assertEqual(request.call_args.kwargs["params"]["max_results"], 30)
        with patch("tools._request") as request:
            self.assertEqual(tools.arxiv_search.invoke({"query": '" : AND OR'}), "NO RESULTS")
            request.assert_not_called()

    def test_arxiv_spacing(self):
        with patch.object(tools, "_LAST_ARXIV_CALL", 10), patch("tools.time.monotonic", return_value=11), patch("tools.time.sleep") as sleep, patch("tools._request", return_value=response('<feed xmlns="http://www.w3.org/2005/Atom"/>')):
            self.assertEqual(tools.arxiv_search.invoke({"query": "test"}), "NO RESULTS")
            sleep.assert_called_once_with(2)

    def test_hf_mapping_filter_sort_and_ai_summary(self):
        items = [{"paper": {"id": "1", "title": "World model", "summary": "full", "upvotes": 1}},
                 {"paper": {"id": "2", "title": "WORLD", "summary": "full", "ai_summary": "AI short", "upvotes": 4}},
                 {"paper": {"title": "skip"}}]
        with patch("tools._request", return_value=response(json.dumps(items))):
            daily = json.loads(tools.hf_daily_papers.invoke({"keyword": "world"}))
            self.assertEqual([r["id"] for r in daily], ["2", "1"])
            search = json.loads(tools.hf_search_papers.invoke({"query": "world"}))
            self.assertEqual(search[1]["summary"], "AI short")

    def test_exa_sse_rate_limit_retry_and_arguments(self):
        limited = {"result": {"_meta": {"rateLimited": True}, "content": []}}
        good = {"result": {"content": [{"type": "text", "text": "real source"}]}}
        replies = [response("data: " + json.dumps(limited) + "\n\n"), response(json.dumps(good))]
        with patch("tools._request", side_effect=replies) as request, patch("tools.time.sleep") as sleep:
            self.assertEqual(tools.web_search.invoke({"query": "world"}), "real source")
            sleep.assert_called_once_with(20)
            arguments = request.call_args.kwargs["json"]["params"]["arguments"]
            self.assertTrue(arguments["objective"])
        with patch("tools._exa", return_value="x" * 15000) as exa:
            self.assertEqual(len(tools.web_fetch.invoke({"url": "https://example.org"})), 12000)
            exa.assert_called_once_with("web_fetch_exa", {"urls": ["https://example.org"]})

    def test_errors_are_returned_and_secrets_redacted(self):
        with patch.dict("tools.os.environ", {"EXA_API_KEY": "secret/key"}), patch("tools._request", side_effect=ValueError("secret/key secret%2Fkey")):
            for source_tool, args in [(tools.arxiv_search, {"query": "world"}),
                                      (tools.hf_daily_papers, {}),
                                      (tools.hf_search_papers, {"query": "world"}),
                                      (tools.web_search, {"query": "world"}),
                                      (tools.web_fetch, {"url": "https://example.org"})]:
                with patch("tools.time.sleep"):
                    result = source_tool.invoke(args)
                self.assertTrue(result.startswith("ERROR:"))
                self.assertNotIn("secret", result)


if __name__ == "__main__":
    unittest.main()
