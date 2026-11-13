import contextlib
import http.server
import io
import json
import os
import tempfile
import threading
import unittest
from unittest import mock

import spark as cl


class Prompts(unittest.TestCase):
    def test_ask_messages(self):
        m = cl.ask_messages("hi")
        self.assertEqual(m[0]["role"], "system")
        self.assertIn(cl.PERSONA, m[0]["content"])
        self.assertEqual(m[1]["content"], "hi")

    def test_code_messages(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "x.py")
            with open(p, "w") as fh:
                fh.write("def f():\n    return 1\n")
            m = cl.code_messages("fix", p, "", "")
        self.assertIn("def f()", m[1]["content"])
        self.assertIn("```py", m[1]["content"])
        self.assertIn("fix", m[1]["content"].lower())

    def test_write_messages_style(self):
        m = cl.write_messages("cats", "concise")
        self.assertIn("tightly", m[1]["content"])
        self.assertIn("cats", m[1]["content"])

    def test_repo_context_ranking(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "a.py"), "w") as fh:
                fh.write("def calculate_invoice_total(items):\n    return sum(items)\n")
            with open(os.path.join(tmp, "b.md"), "w") as fh:
                fh.write("banana bread recipe with flour and sugar\n")
            ctx = cl.select_context(tmp, "invoice total", topk=2)
        self.assertTrue(ctx)
        self.assertEqual(ctx[0][0], "a.py")

    def test_repo_context_empty_query(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(cl.select_context(tmp, "ab"), [])


class MockBackend(unittest.TestCase):
    def _opts(self):
        import argparse
        return argparse.Namespace(backend="mock", model="m", base_url="", temperature=0.2,
                                  max_tokens=64, timeout=5)

    def test_mock_code(self):
        out = cl.complete(cl.ask_messages("write python code def f"), self._opts())
        self.assertIn("```python", out)
        self.assertIn("mock backend", out)

    def test_mock_prose(self):
        out = cl.complete(cl.ask_messages("hello there"), self._opts())
        self.assertIn("Spark", out)


class Handler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(n))
        self.server.last = (self.path, body)
        if self.path == "/api/chat":
            resp = {"message": {"content": "OLLAMA_OK"}}
        else:
            resp = {"choices": [{"message": {"content": "OPENAI_OK"}}]}
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


class HttpBackends(unittest.TestCase):
    def setUp(self):
        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.srv.serve_forever, daemon=True)
        self.thread.start()
        self.url = "http://127.0.0.1:%d" % self.srv.server_address[1]

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()

    def _opts(self, backend):
        import argparse
        return argparse.Namespace(backend=backend, model="test-model", base_url=self.url,
                                  temperature=0.3, max_tokens=32, timeout=5)

    def test_ollama(self):
        out = cl.chat_ollama(cl.ask_messages("hi"), self._opts("ollama"))
        self.assertEqual(out, "OLLAMA_OK")
        path, body = self.srv.last
        self.assertEqual(path, "/api/chat")
        self.assertEqual(body["model"], "test-model")
        self.assertFalse(body["stream"])

    def test_openai(self):
        out = cl.chat_openai(cl.ask_messages("hi"), self._opts("openai"))
        self.assertEqual(out, "OPENAI_OK")
        path, body = self.srv.last
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(body["max_tokens"], 32)

    def test_unreachable(self):
        import argparse
        opts = argparse.Namespace(backend="ollama", model="m", base_url="http://127.0.0.1:1",
                                  temperature=0.2, max_tokens=8, timeout=0.5)
        with self.assertRaises(cl.BackendError):
            cl.chat_ollama(cl.ask_messages("hi"), opts)


class Cli(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, args):
        with contextlib.redirect_stdout(io.StringIO()) as out, \
                contextlib.redirect_stderr(io.StringIO()):
            rc = cl.main(args)
        return rc, out.getvalue()

    def test_ask_mock(self):
        rc, out = self._run(["--backend", "mock", "ask", "hello"])
        self.assertEqual(rc, 0)
        self.assertIn("Spark", out)

    def test_code_mock(self):
        p = os.path.join(self.root, "x.py")
        with open(p, "w") as fh:
            fh.write("def f():\n    return 1\n")
        rc, out = self._run(["--backend", "mock", "code", "explain", "--file", p])
        self.assertEqual(rc, 0)
        self.assertIn("```python", out)

    def test_code_bad_file(self):
        rc, _ = self._run(["--backend", "mock", "code", "fix", "--file",
                           os.path.join(self.root, "nope.py")])
        self.assertEqual(rc, 2)

    def test_repo_mock(self):
        with open(os.path.join(self.root, "a.py"), "w") as fh:
            fh.write("def invoice_total(x):\n    return x\n")
        rc, out = self._run(["--backend", "mock", "--json", "repo", "invoice", "--root", self.root])
        self.assertEqual(rc, 0)
        payload = json.loads(out)
        self.assertIn("a.py", payload["context"])

    def test_repo_bad_root(self):
        rc, _ = self._run(["--backend", "mock", "repo", "q", "--root",
                           os.path.join(self.root, "nope")])
        self.assertEqual(rc, 2)

    def test_write_mock(self):
        rc, out = self._run(["--backend", "mock", "write", "cats", "--style", "friendly"])
        self.assertEqual(rc, 0)

    def test_no_cmd(self):
        rc, _ = self._run(["--backend", "mock"])
        self.assertEqual(rc, 2)

    def test_backend_error_exit1(self):
        rc, _ = self._run(["--backend", "ollama", "--base-url", "http://127.0.0.1:1",
                           "--timeout", "0.5", "ask", "hi"])
        self.assertEqual(rc, 1)

    def test_json(self):
        rc, out = self._run(["--backend", "mock", "--json", "ask", "hi"])
        payload = json.loads(out)
        self.assertEqual(payload["persona"], cl.PERSONA)
        self.assertEqual(payload["backend"], "mock")

    def test_globals_after_subcommand(self):
        rc, out = self._run(["ask", "hello", "--backend", "mock"])
        self.assertEqual(rc, 0)
        self.assertIn("Spark", out)

    def test_globals_after_subcommand_json(self):
        rc, out = self._run(["ask", "hi", "--backend", "mock", "--json"])
        payload = json.loads(out)
        self.assertEqual(payload["backend"], "mock")


class StreamHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(n))
        self.server.last = (self.path, body)
        if self.path == "/api/chat":
            lines = [json.dumps({"message": {"content": "Hel"}}),
                     json.dumps({"message": {"content": "lo"}}),
                     json.dumps({"done": True})]
            data = ("\n".join(lines) + "\n").encode()
            ctype = "application/x-ndjson"
        else:
            lines = ['data: ' + json.dumps({"choices": [{"delta": {"content": "Hel"}}]}),
                     'data: ' + json.dumps({"choices": [{"delta": {"content": "lo"}}]}),
                     'data: [DONE]']
            data = ("\n\n".join(lines) + "\n\n").encode()
            ctype = "text/event-stream"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


class Streaming(unittest.TestCase):
    def setUp(self):
        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), StreamHandler)
        self.thread = threading.Thread(target=self.srv.serve_forever, daemon=True)
        self.thread.start()
        self.url = "http://127.0.0.1:%d" % self.srv.server_address[1]

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()

    def _opts(self, backend):
        import argparse
        return argparse.Namespace(backend=backend, model="m", base_url=self.url,
                                  temperature=0.2, max_tokens=32, timeout=5)

    def test_stream_ollama(self):
        toks = []
        out = cl.stream_ollama(cl.ask_messages("hi"), self._opts("ollama"), toks.append)
        self.assertEqual(out, "Hello")
        self.assertEqual(toks, ["Hel", "lo"])
        path, body = self.srv.last
        self.assertEqual(path, "/api/chat")
        self.assertTrue(body["stream"])

    def test_stream_openai(self):
        toks = []
        out = cl.stream_openai(cl.ask_messages("hi"), self._opts("openai"), toks.append)
        self.assertEqual(out, "Hello")
        self.assertEqual(toks, ["Hel", "lo"])
        path, body = self.srv.last
        self.assertEqual(path, "/v1/chat/completions")
        self.assertTrue(body["stream"])

    def test_complete_stream_mock(self):
        import argparse
        opts = argparse.Namespace(backend="mock", model="m", base_url="",
                                  temperature=0.2, max_tokens=32, timeout=5)
        toks = []
        out = cl.complete_stream(cl.ask_messages("hello"), opts, toks.append)
        self.assertEqual(len(toks), 1)
        self.assertEqual(out, toks[0])
        self.assertIn("Spark", out)

    def test_stream_unreachable(self):
        import argparse
        opts = argparse.Namespace(backend="ollama", model="m", base_url="http://127.0.0.1:1",
                                  temperature=0.2, max_tokens=8, timeout=0.5)
        with self.assertRaises(cl.BackendError):
            cl.stream_ollama(cl.ask_messages("hi"), opts, lambda t: None)


class SmarterMock(unittest.TestCase):
    def _opts(self):
        import argparse
        return argparse.Namespace(backend="mock", model="m", base_url="", temperature=0.2,
                                  max_tokens=64, timeout=5)

    def test_citations_surfaced(self):
        msg = ("Answer using excerpts.\n\n--- billing.py:1-3 ---\ndef invoice_total():\n"
               "    return 1\n\nQuestion: where is invoice total?")
        out = cl.complete([{"role": "user", "content": msg}], self._opts())
        self.assertIn("Retrieved context (BM25-ranked chunks)", out)
        self.assertIn("billing.py", out)

    def test_no_citations_no_block(self):
        out = cl.complete(cl.ask_messages("hello there"), self._opts())
        self.assertNotIn("Retrieved context", out)


class PromptScaffold(unittest.TestCase):
    def test_code_scaffold_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "x.py")
            with open(p, "w") as fh:
                fh.write("def f():\n    return 1\n")
            m = cl.code_messages("review", p, "", "")
        self.assertIn(cl.CODE_SCAFFOLD["review"], m[1]["content"])

    def test_repo_messages_cited(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "a.py"), "w") as fh:
                fh.write("def invoice_total(items):\n    return sum(items)\n")
            m, ctx = cl.repo_messages(tmp, "invoice total", 3)
        self.assertIn("--- a.py:", m[1]["content"])
        self.assertTrue(ctx)
        self.assertEqual(ctx[0][0], "a.py")

    def test_system_mentions_citations(self):
        self.assertIn("cite", cl.SYSTEM.lower())


class Memory(unittest.TestCase):
    def test_short_unchanged(self):
        msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]
        self.assertEqual(cl.trim_memory(msgs), msgs)

    def test_trims_old_turns(self):
        msgs = [{"role": "system", "content": "s"}]
        for i in range(40):
            msgs.append({"role": "user", "content": "x" * 400})
            msgs.append({"role": "assistant", "content": "y" * 400})
        out = cl.trim_memory(msgs, budget=500)
        self.assertEqual(out[0]["role"], "system")
        self.assertLess(len(out), len(msgs))
        self.assertGreaterEqual(len(out), 3)

    def test_keeps_recent(self):
        msgs = [{"role": "system", "content": "s"}]
        for i in range(20):
            msgs.append({"role": "user", "content": "msg%d " % i + "x" * 400})
        out = cl.trim_memory(msgs, budget=300)
        self.assertEqual(out[-1]["content"], msgs[-1]["content"])


class SearchIndexCli(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        with open(os.path.join(self.root, "billing.py"), "w") as fh:
            fh.write("def invoice_total(items):\n    return sum(items)\n")

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, args):
        with contextlib.redirect_stdout(io.StringIO()) as out, \
                contextlib.redirect_stderr(io.StringIO()):
            rc = cl.main(args)
        return rc, out.getvalue()

    def test_search_text(self):
        rc, out = self._run(["search", "invoice total", "--root", self.root])
        self.assertEqual(rc, 0)
        self.assertIn("billing.py:", out)
        self.assertIn("score", out)

    def test_search_json(self):
        rc, out = self._run(["--json", "search", "invoice_total", "--root", self.root])
        self.assertEqual(rc, 0)
        payload = json.loads(out)
        self.assertTrue(payload["results"])
        self.assertEqual(payload["results"][0]["relpath"], "billing.py")

    def test_search_no_match(self):
        rc, out = self._run(["search", "zzzznotpresent", "--root", self.root])
        self.assertEqual(rc, 0)
        self.assertIn("no matching chunks", out)

    def test_search_bad_root(self):
        rc, _ = self._run(["search", "q", "--root", os.path.join(self.root, "nope")])
        self.assertEqual(rc, 2)

    def test_index_text(self):
        rc, out = self._run(["index", "--root", self.root])
        self.assertEqual(rc, 0)
        self.assertIn("indexed", out)
        self.assertIn("cache:", out)

    def test_index_json(self):
        rc, out = self._run(["--json", "index", "--root", self.root])
        self.assertEqual(rc, 0)
        payload = json.loads(out)
        self.assertEqual(payload["files"], 1)
        self.assertIn("chunks", payload)

    def test_index_rebuild(self):
        self._run(["index", "--root", self.root])
        rc, out = self._run(["--json", "index", "--root", self.root, "--rebuild"])
        payload = json.loads(out)
        self.assertEqual(rc, 0)
        self.assertEqual(payload["reused"], 0)

    def test_index_bad_root(self):
        rc, _ = self._run(["index", "--root", os.path.join(self.root, "nope")])
        self.assertEqual(rc, 2)

    def test_search_non_utf8_console(self):
        # Windows cp1252 console must not crash on Cyrillic repo text
        with open(os.path.join(self.root, "ru.md"), "w", encoding="utf-8") as fh:
            fh.write("документация про invoice total\n")
        buf = io.BytesIO()
        wrapper = io.TextIOWrapper(buf, encoding="cp1252", errors="strict")
        with contextlib.redirect_stdout(wrapper), \
                contextlib.redirect_stderr(io.StringIO()):
            rc = cl.main(["search", "invoice total", "--root", self.root])
        wrapper.flush()
        self.assertEqual(rc, 0)
        self.assertTrue(buf.getvalue())


class ChatCommands(unittest.TestCase):
    def _opts(self):
        import argparse
        return argparse.Namespace(backend="mock", model="m", base_url="", temperature=0.2,
                                  max_tokens=64, timeout=5, stream=False, root="", json=False)

    def test_reset_and_context(self):
        inputs = iter(["hello", "/context", "/reset", "/exit"])
        with mock.patch("builtins.input", lambda *a: next(inputs)), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            rc = cl.run_chat(self._opts())
        text = out.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("tokens in context", text)
        self.assertIn("context cleared", text)

    def test_eof_exits_zero(self):
        with mock.patch("builtins.input", side_effect=EOFError), \
                contextlib.redirect_stdout(io.StringIO()):
            rc = cl.run_chat(self._opts())
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()