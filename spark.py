#!/usr/bin/env python3
"""Spark Mini - a local code & writing assistant over open-weight models.

The CLI is standard-library only: inference is delegated to a local Ollama or
OpenAI-compatible server (llama.cpp, vLLM, LM Studio), or to a built-in mock
backend for offline demos and tests. "Spark Mini" is the assistant persona
and tooling; the intelligence comes from the open base model you point it at
(default Qwen2.5-Coder). An optional LoRA fine-tune pipeline lives in finetune/.
"""

import argparse
import json
import math
import os
import re
import sys
import urllib.error
import urllib.request
from collections import Counter

__version__ = "1.0.0"
PERSONA = "Spark Mini"
DEFAULT_MODEL = "qwen2.5-coder:1.5b"
DEFAULT_OLLAMA = "http://localhost:11434"
DEFAULT_OPENAI = "http://localhost:8080"

SYSTEM = (
    "You are Spark Mini, a fast, precise and friendly coding and writing "
    "assistant. You write correct, idiomatic, minimal code and clear prose. "
    "For code: give complete runnable snippets, name edge cases, and prefer the "
    "standard library. For text: match the requested tone, stay concrete, avoid "
    "filler. If unsure, say so briefly instead of inventing facts."
)

CODE_EXT = {".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".h", ".cpp",
            ".hpp", ".cs", ".go", ".rs", ".rb", ".php", ".sh", ".sql", ".md",
            ".txt", ".json", ".yml", ".yaml", ".toml", ".html", ".css"}
SKIP_DIRS = {".git", "node_modules", "build", "dist", ".venv", "venv", "__pycache__",
             ".gradle", "target", ".idea", ".vscode"}
TEXT_STYLES = {
    "plain": "Write in clear, neutral prose.",
    "concise": "Write tightly: short sentences, no filler, bullet points where useful.",
    "friendly": "Write warm and conversational, like a helpful colleague.",
    "formal": "Write in a polished, professional register suitable for documentation.",
}
CODE_TASKS = {
    "write": "Write new code that satisfies the request.",
    "fix": "Find and fix bugs in the code; explain each fix briefly.",
    "explain": "Explain what the code does, step by step, for a competent developer.",
    "review": "Review the code for correctness, security and style; list concrete issues.",
    "test": "Write thorough unit tests for the code.",
}


class BackendError(Exception):
    pass


# ---------------------------------------------------------------- backends
def chat_mock(messages, opts):
    last = messages[-1]["content"] if messages else ""
    low = last.lower()
    prose_trig = ("sentence", "prose", "about why", "explain why", "draft")
    code_trig = ("```", "def ", "code", "function", "class", "python", "implement", "test")
    is_code = any(t in low for t in code_trig) and not any(t in low for t in prose_trig)
    if is_code:
        body = ("Here is a minimal, runnable example:\n\n"
                "```python\n"
                "def greet(name: str) -> str:\n"
                "    return f\"Hello, {name}!\"\n\n"
                "if __name__ == \"__main__\":\n"
                "    print(greet(\"world\"))\n"
                "```\n\n"
                "Edge cases: empty name returns 'Hello, !'; pass a stripped string "
                "if you want to normalise input.")
    else:
        body = ("Spark here. I write, fix, explain, review and test code, and draft "
                "clear prose. Unit tests matter because they catch regressions "
                "early. Point me at a file or a repository and ask away — I keep "
                "answers concrete and runnable.")
    return body + "\n\n(mock backend — no model loaded)"


def _post(url, payload, timeout):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError) as exc:
        raise BackendError("cannot reach backend at %s: %s" % (url, exc))


def chat_ollama(messages, opts):
    payload = {"model": opts.model, "messages": messages, "stream": False,
               "options": {"temperature": opts.temperature, "num_predict": opts.max_tokens}}
    data = _post(opts.base_url.rstrip("/") + "/api/chat", payload, opts.timeout)
    try:
        return data["message"]["content"]
    except (KeyError, TypeError):
        raise BackendError("unexpected Ollama response: %r" % data)


def chat_openai(messages, opts):
    payload = {"model": opts.model, "messages": messages,
               "temperature": opts.temperature, "max_tokens": opts.max_tokens}
    data = _post(opts.base_url.rstrip("/") + "/v1/chat/completions", payload, opts.timeout)
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise BackendError("unexpected OpenAI-compatible response: %r" % data)


BACKENDS = {"ollama": chat_ollama, "openai": chat_openai, "mock": chat_mock}


def complete(messages, opts):
    return BACKENDS[opts.backend](messages, opts)


# ---------------------------------------------------------------- repo RAG
def iter_files(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in sorted(filenames):
            if os.path.splitext(fn)[1].lower() in CODE_EXT:
                yield os.path.join(dirpath, fn)


def _terms(query):
    return [w for w in re.findall(r"[a-z0-9_]{3,}", query.lower())]


def select_context(root, query, topk=3, max_chars=1500):
    terms = _terms(query)
    if not terms:
        return []
    docs = []
    for path in iter_files(root):
        try:
            if os.path.getsize(path) > 200_000:
                continue
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            continue
        docs.append((path, text.lower()))
    if not docs:
        return []
    df = Counter()
    for _, text in docs:
        for t in set(terms):
            if t in text:
                df[t] += 1
    n = len(docs)
    scored = []
    for path, text in docs:
        score = 0.0
        for t in terms:
            if df[t]:
                score += text.count(t) * (math.log((n + 1) / (df[t] + 1)) + 1.0)
        if score > 0:
            scored.append((score, path))
    scored.sort(key=lambda x: (-x[0], x[1]))
    out = []
    for _, path in scored[:topk]:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            snippet = fh.read(max_chars)
        out.append((os.path.relpath(path, root), snippet))
    return out


# ---------------------------------------------------------------- prompts
def code_messages(task, path, lang, extra):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        source = fh.read()
    lang = lang or os.path.splitext(path)[1].lstrip(".") or "text"
    instruction = CODE_TASKS[task]
    if extra:
        instruction += " Additional request: " + extra
    user = ("%s\n\nFile: %s\n```%s\n%s\n```" % (instruction, path, lang, source))
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def repo_messages(root, question, topk):
    ctx = select_context(root, question, topk)
    parts = ["Answer the question using the repository excerpts below."]
    for rel, snippet in ctx:
        parts.append("\n--- %s ---\n%s" % (rel, snippet))
    parts.append("\nQuestion: " + question)
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "\n".join(parts)}], ctx


def write_messages(topic, style):
    user = "%s Topic: %s" % (TEXT_STYLES[style], topic)
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def ask_messages(prompt):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]


# ---------------------------------------------------------------- cli
def _globals(p, suppress=False):
    # Subparsers re-declare the globals with SUPPRESS defaults so flags work
    # both before and after the subcommand without clobbering earlier values.
    def d(v):
        return argparse.SUPPRESS if suppress else v
    p.add_argument("--backend", choices=sorted(BACKENDS), default=d(os.environ.get("SPARK_BACKEND", "ollama")))
    p.add_argument("--model", default=d(os.environ.get("SPARK_MODEL", DEFAULT_MODEL)))
    p.add_argument("--base-url", default=d(os.environ.get("SPARK_BASE_URL", "")))
    p.add_argument("--temperature", type=float, default=d(0.2))
    p.add_argument("--max-tokens", type=int, default=d(1024))
    p.add_argument("--timeout", type=float, default=d(120))
    p.add_argument("--json", action="store_true", default=d(False))
    p.add_argument("--no-color", action="store_true", default=d(False))


def build_parser():
    p = argparse.ArgumentParser(
        prog="spark", description="Spark Mini - local code & writing assistant.")
    _globals(p)
    p.add_argument("--version", action="version", version="spark " + __version__ + " (" + PERSONA + ")")
    sub = p.add_subparsers(dest="cmd")
    a = sub.add_parser("ask", help="one-shot question")
    _globals(a, True)
    a.add_argument("prompt")
    ch = sub.add_parser("chat", help="interactive session")
    _globals(ch, True)
    c = sub.add_parser("code", help="work on a single file")
    _globals(c, True)
    c.add_argument("task", choices=sorted(CODE_TASKS))
    c.add_argument("--file", required=True)
    c.add_argument("--lang", default="")
    c.add_argument("extra", nargs="?", default="")
    r = sub.add_parser("repo", help="question with repository context")
    _globals(r, True)
    r.add_argument("question")
    r.add_argument("--root", default=".")
    r.add_argument("--topk", type=int, default=3)
    w = sub.add_parser("write", help="draft prose")
    _globals(w, True)
    w.add_argument("topic")
    w.add_argument("--style", choices=sorted(TEXT_STYLES), default="plain")
    return p


def emit(opts, response, extra=None):
    if opts.json:
        payload = {"persona": PERSONA, "backend": opts.backend, "model": opts.model,
                   "response": response}
        if extra is not None:
            payload["context"] = extra
        print(json.dumps(payload, indent=2))
    else:
        print(response)


def run_chat(opts):
    messages = [{"role": "system", "content": SYSTEM}]
    print("%s (%s backend). Type /exit to quit." % (PERSONA, opts.backend))
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line in ("/exit", "/quit"):
            return 0
        messages.append({"role": "user", "content": line})
        try:
            reply = complete(messages, opts)
        except BackendError as exc:
            sys.stderr.write("spark: %s\n" % exc)
            return 1
        messages.append({"role": "assistant", "content": reply})
        print("spark> " + reply)


def main(argv=None):
    opts = build_parser().parse_args(argv)
    if not opts.base_url:
        opts.base_url = DEFAULT_OLLAMA if opts.backend == "ollama" else DEFAULT_OPENAI
    if opts.cmd is None:
        build_parser().print_help()
        return 2
    if opts.cmd == "chat":
        return run_chat(opts)
    if opts.cmd == "ask":
        messages = ask_messages(opts.prompt)
        ctx = None
    elif opts.cmd == "code":
        if not os.path.isfile(opts.file):
            sys.stderr.write("spark: file not found: %s\n" % opts.file)
            return 2
        messages = code_messages(opts.task, opts.file, opts.lang, opts.extra)
        ctx = None
    elif opts.cmd == "repo":
        if not os.path.isdir(opts.root):
            sys.stderr.write("spark: root not found: %s\n" % opts.root)
            return 2
        messages, ctx = repo_messages(opts.root, opts.question, opts.topk)
        ctx = [rel for rel, _ in ctx]
    else:  # write
        messages = write_messages(opts.topic, opts.style)
        ctx = None
    try:
        response = complete(messages, opts)
    except BackendError as exc:
        sys.stderr.write("spark: %s\n" % exc)
        return 1
    emit(opts, response, ctx)
    return 0


if __name__ == "__main__":
    sys.exit(main())