import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "finetune"))
import prepare_data as pd
import eval as ev
import train_lora as tl
import merge_lora as ml


class PrepareData(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_from_jsonl_pairs(self):
        src = os.path.join(self.root, "pairs.jsonl")
        with open(src, "w") as fh:
            fh.write(json.dumps({"instruction": "add", "input": "1,2", "output": "3"}) + "\n")
            fh.write(json.dumps({"instruction": "add", "input": "1,2", "output": "3"}) + "\n")
            fh.write(json.dumps({"instruction": "sub", "output": "1"}) + "\n")
        out = os.path.join(self.root, "data")
        with contextlib.redirect_stdout(io.StringIO()):
            rc = pd.main(["--jsonl", src, "--out", out, "--val-split", "0.5"])
        self.assertEqual(rc, 0)
        with open(os.path.join(out, "train.jsonl")) as fh:
            train = [json.loads(l) for l in fh]
        with open(os.path.join(out, "val.jsonl")) as fh:
            val = [json.loads(l) for l in fh]
        self.assertEqual(len(train) + len(val), 2)  # deduped
        self.assertEqual(train[0]["messages"][0]["role"], "system")

    def test_from_code(self):
        code = os.path.join(self.root, "src")
        os.makedirs(code)
        with open(os.path.join(code, "a.py"), "w") as fh:
            fh.write("def f(x):\n    return x * 2\n" * 20)
        out = os.path.join(self.root, "data")
        with contextlib.redirect_stdout(io.StringIO()):
            rc = pd.main(["--code", code, "--out", out])
        self.assertEqual(rc, 0)
        with open(os.path.join(out, "train.jsonl")) as fh:
            rows = [json.loads(l) for l in fh]
        self.assertTrue(rows)
        self.assertIn("Complete the following py", rows[0]["messages"][1]["content"])

    def test_empty_source(self):
        src = os.path.join(self.root, "empty.jsonl")
        open(src, "w").close()
        with contextlib.redirect_stderr(io.StringIO()):
            rc = pd.main(["--jsonl", src, "--out", os.path.join(self.root, "d")])
        self.assertEqual(rc, 2)


class Eval(unittest.TestCase):
    def test_score_code_pass(self):
        ok, _ = ev.score({"kind": "code"}, "```python\ndef f():\n    return 1\n```")
        self.assertTrue(ok)

    def test_score_code_syntax_fail(self):
        ok, _ = ev.score({"kind": "code"}, "```python\ndef f(:\n```")
        self.assertFalse(ok)

    def test_score_code_noblock(self):
        ok, _ = ev.score({"kind": "code"}, "no code here")
        self.assertFalse(ok)

    def test_score_text(self):
        ok, _ = ev.score({"kind": "text", "keyword": "test"}, "I love tests")
        self.assertTrue(ok)
        ok, _ = ev.score({"kind": "text", "keyword": "test"}, "nothing")
        self.assertFalse(ok)

    def test_eval_mock_all_pass(self):
        with contextlib.redirect_stdout(io.StringIO()):
            rc = ev.main(["--backend", "mock"])
        self.assertEqual(rc, 0)


class GpuScriptsGraceful(unittest.TestCase):
    def test_train_lora_no_data_or_deps(self):
        with contextlib.redirect_stderr(io.StringIO()):
            rc = tl.main(["--data", os.path.join(tempfile.gettempdir(), "nope-spark-data")])
        self.assertEqual(rc, 2)

    def test_merge_lora_no_adapter_or_deps(self):
        with contextlib.redirect_stderr(io.StringIO()):
            rc = ml.main(["--adapter", os.path.join(tempfile.gettempdir(), "nope-spark-adapter")])
        self.assertEqual(rc, 2)

    def test_load_config(self):
        cfg = tl.load_config(os.path.join(os.path.dirname(pd.__file__), "config.json"))
        self.assertIn("base_model", cfg)
        self.assertIn("lora", cfg)


if __name__ == "__main__":
    unittest.main()