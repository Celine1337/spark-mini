#!/usr/bin/env python3
"""Spark Mini - a local code & writing assistant over open-weight models.

The CLI is standard-library only: inference is delegated to a local Ollama or
OpenAI-compatible server (llama.cpp, vLLM, LM Studio), or to a built-in mock
backend for offline demos and tests. "Spark Mini" is the assistant persona
and tooling; the intelligence comes from the open base model you point it at
(default Qwen2.5-Coder).

v1.1 makes the tooling itself much smarter: repository context is now selected
by a real BM25 retrieval engine over structure-aware chunks (see retrieval.py)
with symbol boosting, a persistent index cache and cited, token-budgeted
context — instead of the old whole-file TF-IDF + truncated file head. Adds
`search` and `index` commands, opt-in `--stream`, and chat memory management.
An optional LoRA fine-tune pipeline lives in finetune/.
"""

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request

from retrieval import (CODE_EXT, SKIP_DIRS, build_index, estimate_tokens,
                       pack_context, search, select_context)

__version__ = "1.1.0"
PERSONA = "Spark Mini"
DEFAULT_MODEL = "qwen2.5-coder:1.5b"
DEFAULT_OLLAMA = "http://localhost:11434"
DEFAULT_OPENAI = "http://localhost:8080"
CHAT_TOKEN_BUDGET = 6000

SYSTEM = (
    "You are Spark Mini, a fast, precise and friendly coding and writing "
    "assistant. You write correct, idiomatic, minimal code and clear prose. "
    "For code: give complete runnable snippets, name edge cases, and prefer the "
    "standard library. For text: match the requested tone, stay concrete, avoid "
    "filler. When repository excerpts are provided, ground your answer in them "
    "and cite the path:line ranges you rely on. If unsure, say so briefly "
    "instead of inventing facts."
)

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
CODE_SCAFFOLD = {
    "write": "Plan briefly, then give complete runnable code and name the edge cases.",
    "fix": "State the root cause, give the corrected code, then explain each fix.",
    "explain": "Walk through the logic step by step; note inputs, outputs and side effects.",
    "review": "List concrete issues by severity (correctness, security, style) with fixes.",
    "test": "Cover the happy path, edge cases and failure modes; keep tests deterministic.",
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
    cites = re.findall(r"^--- (.+?):\d+-\d+ ---$", last, re.M)
    if cites:
        files = list(dict.fromkeys(cites))
        body += ("\n\nRetrieved context (BM25-ranked chunks): "
                 + ", ".join(files[:5]) + ".")
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


# ---------------------------------------------------------------- streaming
def _stream_objects(url, payload, timeout):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST")
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError) as exc:
        raise BackendError("cannot reach backend at %s: %s" % (url, exc))
    with resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            if line.startswith("data:"):
                line = line[5:].strip()
            if line == "[DONE]":
                break
            try:
                yield json.loads(line)
            except ValueError:
                continue


def stream_ollama(messages, opts, on_token):
    payload = {"model": opts.model, "messages": messages, "stream": True,
               "options": {"temperature": opts.temperature, "num_predict": opts.max_tokens}}
    parts = []
    for obj in _stream_objects(opts.base_url.rstrip("/") + "/api/chat", payload, opts.timeout):
        tok = (obj.get("message") or {}).get("content", "")
        if tok:
            parts.append(tok)
            on_token(tok)
        if obj.get("done"):
            break
    return "".join(parts)


def stream_openai(messages, opts, on_token):
    payload = {"model": opts.model, "messages": messages, "stream": True,
               "temperature": opts.temperature, "max_tokens": opts.max_tokens}
    parts = []
    for obj in _stream_objects(opts.base_url.rstrip("/") + "/v1/chat/completions", payload, opts.timeout):
        choices = obj.get("choices") or []
        if choices:
            tok = (choices[0].get("delta") or {}).get("content", "")
            if tok:
                parts.append(tok)
                on_token(tok)
    return "".join(parts)


STREAMERS = {"ollama": stream_ollama, "openai": stream_openai}


def complete_stream(messages, opts, on_token):
    if opts.backend in STREAMERS:
        return STREAMERS[opts.backend](messages, opts, on_token)
    text = chat_mock(messages, opts)
    on_token(text)
    return text


# ---------------------------------------------------------------- prompts
def code_messages(task, path, lang, extra):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        source = fh.read()
    lang = lang or os.path.splitext(path)[1].lstrip(".") or "text"
    instruction = CODE_TASKS[task] + " " + CODE_SCAFFOLD[task]
    if extra:
        instruction += " Additional request: " + extra
    user = ("%s\n\nFile: %s\n```%s\n%s\n```" % (instruction, path, lang, source))
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def repo_messages(root, question, topk):
    results = search(root, question, topk=max(topk * 2, topk))
    ctx, seen = [], set()
    for r in results:
        if r["relpath"] not in seen:
            seen.add(r["relpath"])
            ctx.append((r["relpath"], "%s:%d-%d" % (r["relpath"], r["start"], r["end"])))
        if len(ctx) >= topk:
            break
    context = pack_context(results, max_chars=6000)
    parts = ["Answer the question using the repository excerpts below. Each excerpt "
             "is cited as path:start-end; cite the paths you rely on. If the excerpts "
             "are insufficient, say so rather than guessing."]
    if context:
        parts.append("\n" + context)
    parts.append("\nQuestion: " + question)
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "\n".join(parts)}], ctx


def write_messages(topic, style):
    user = "%s Topic: %s" % (TEXT_STYLES[style], topic)
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def ask_messages(prompt):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]


def trim_memory(messages, budget=CHAT_TOKEN_BUDGET):
    """Keep the system prompt plus the most recent turns within a token budget."""
    if len(messages) <= 2:
        return messages
    system, rest = messages[0], messages[1:]
    total = sum(estimate_tokens(m["content"]) for m in rest)
    while total > budget and len(rest) > 2:
        total -= estimate_tokens(rest.pop(0)["content"])
    return [system] + rest


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
    p.add_argument("--stream", action="store_true", default=d(False))


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
    ch.add_argument("--root", default="", help="augment chat with repo context")
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
    s = sub.add_parser("search", help="rank repo chunks for a query (no model needed)")
    _globals(s, True)
    s.add_argument("question")
    s.add_argument("--root", default=".")
    s.add_argument("--topk", type=int, default=5)
    ix = sub.add_parser("index", help="build / inspect the repo retrieval index")
    _globals(ix, True)
    ix.add_argument("--root", default=".")
    ix.add_argument("--rebuild", action="store_true", default=False)
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


def _print_search(opts, results):
    if opts.json:
        print(json.dumps({"persona": PERSONA, "query": opts.question,
                          "root": os.path.abspath(opts.root), "results": results}, indent=2))
        return
    if not results:
        print("no matching chunks")
        return
    for r in results:
        print("%s:%d-%d  [score %.3f]" % (r["relpath"], r["start"], r["end"], r["score"]))
        print(r["text"])
        print()


def _print_index(opts, meta):
    if opts.json:
        payload = {"persona": PERSONA, "root": os.path.abspath(opts.root)}
        payload.update(meta)
        print(json.dumps(payload, indent=2))
        return
    print("indexed %d files, %d chunks, %d terms, %d symbols (reused %d)"
          % (meta["files"], meta["chunks"], meta["terms"], meta["symbols"], meta["reused"]))
    print("cache: %s" % meta["cache"])


def run_chat(opts):
    messages = [{"role": "system", "content": SYSTEM}]
    root = getattr(opts, "root", "") or ""
    print("%s (%s backend). /exit quit, /reset clear, /context usage." % (PERSONA, opts.backend))
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
        if line == "/reset":
            messages = [{"role": "system", "content": SYSTEM}]
            print("spark> context cleared")
            continue
        if line == "/context":
            used = sum(estimate_tokens(m["content"]) for m in messages)
            print("spark> ~%d tokens in context (%d turns)" % (used, len(messages) - 1))
            continue
        user = line
        if root and os.path.isdir(root):
            res = search(root, line, topk=3)
            if res:
                user = line + "\n\nRepository context:\n" + pack_context(res, max_chars=3000)
        messages.append({"role": "user", "content": user})
        messages = trim_memory(messages)
        try:
            if getattr(opts, "stream", False) and opts.backend in STREAMERS:
                sys.stdout.write("spark> ")
                reply = complete_stream(messages, opts, lambda t: sys.stdout.write(t))
                sys.stdout.write("\n")
                sys.stdout.flush()
            else:
                reply = complete(messages, opts)
                print("spark> " + reply)
        except BackendError as exc:
            sys.stderr.write("spark: %s\n" % exc)
            return 1
        messages.append({"role": "assistant", "content": reply})


def main(argv=None):
    opts = build_parser().parse_args(argv)
    if not opts.base_url:
        opts.base_url = DEFAULT_OLLAMA if opts.backend == "ollama" else DEFAULT_OPENAI
    if opts.cmd is None:
        build_parser().print_help()
        return 2
    if opts.cmd == "search":
        if not os.path.isdir(opts.root):
            sys.stderr.write("spark: root not found: %s\n" % opts.root)
            return 2
        _print_search(opts, search(opts.root, opts.question, topk=opts.topk))
        return 0
    if opts.cmd == "index":
        if not os.path.isdir(opts.root):
            sys.stderr.write("spark: root not found: %s\n" % opts.root)
            return 2
        _, meta = build_index(opts.root, rebuild=opts.rebuild)
        _print_index(opts, meta)
        return 0
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
        if getattr(opts, "stream", False) and not opts.json and opts.backend in STREAMERS:
            response = complete_stream(messages, opts, lambda t: sys.stdout.write(t))
            sys.stdout.write("\n")
            sys.stdout.flush()
            return 0
        response = complete(messages, opts)
    except BackendError as exc:
        sys.stderr.write("spark: %s\n" % exc)
        return 1
    emit(opts, response, ctx)
    return 0


if __name__ == "__main__":
    sys.exit(main())