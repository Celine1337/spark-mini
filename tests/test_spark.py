import contextlib
import http.server
import io
import json
import os
import tempfile
import threading
import unittest

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


if __name__ == "__main__":
    unittest.main()