#!/usr/bin/env python3
"""Tiny offline eval harness for Spark Mini (stdlib only).

Runs a fixed prompt suite against a backend and scores it: code answers must
contain a parseable Python block; prose answers must mention a keyword.
Default backend is `mock` so it runs with no model; use --backend ollama to
score a real local model.
    python finetune/eval.py --backend mock
"""
import argparse
import ast
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import spark

SUITE = [
    {"name": "py_func", "kind": "code",
     "prompt": "Write a python function add(a, b) that returns the sum."},
    {"name": "py_class", "kind": "code",
     "prompt": "Write a python class Stack with push and pop methods."},
    {"name": "prose_tests", "kind": "text", "keyword": "test",
     "prompt": "Write one sentence about why unit tests matter."},
]


def extract_python(text):
    m = re.search(r"```python\n(.*?)```", text, re.S)
    return m.group(1) if m else None


def score(task, response):
    if task["kind"] == "code":
        code = extract_python(response)
        if not code:
            return False, "no python block"
        try:
            ast.parse(code)
        except SyntaxError as exc:
            return False, "syntax error: %s" % exc
        return True, "parses"
    ok = task["keyword"] in response.lower()
    return ok, "keyword" if ok else "missing keyword"


def main(argv=None):
    p = argparse.ArgumentParser(prog="eval", description=__doc__)
    p.add_argument("--backend", choices=sorted(spark.BACKENDS), default="mock")
    p.add_argument("--model", default=spark.DEFAULT_MODEL)
    p.add_argument("--base-url", default="")
    p.add_argument("--timeout", type=float, default=120)
    opts = p.parse_args(argv)
    if not opts.base_url:
        opts.base_url = (spark.DEFAULT_OLLAMA if opts.backend == "ollama"
                         else spark.DEFAULT_OPENAI)
    ns = argparse.Namespace(backend=opts.backend, model=opts.model, base_url=opts.base_url,
                            temperature=0.0, max_tokens=512, timeout=opts.timeout)
    passed = 0
    for task in SUITE:
        try:
            resp = spark.complete(spark.ask_messages(task["prompt"]), ns)
        except spark.BackendError as exc:
            print("%-12s FAIL backend: %s" % (task["name"], exc))
            continue
        ok, why = score(task, resp)
        passed += ok
        print("%-12s %s  %s" % (task["name"], "PASS" if ok else "FAIL", why))
    print("score %d/%d" % (passed, len(SUITE)))
    return 0 if passed == len(SUITE) else 1


if __name__ == "__main__":
    sys.exit(main())