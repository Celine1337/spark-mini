#!/usr/bin/env python3
"""Regenerate docs/code.png and docs/repo.png from the offline mock backend.

Run: python tools/make_shots.py   (requires Pillow)
Every shot is labelled "mock backend" - no model is loaded, so the images are
reproducible in CI and never imply a real model produced them.
"""
import argparse
import os
import sys
import tempfile
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import spark
import render_terminal as rt

PINK = "\x1b[95m"
CYAN = "\x1b[36m"
DIM = "\x1b[2m"
BOLD = "\x1b[1m"
RESET = "\x1b[0m"


def opts():
    return argparse.Namespace(backend="mock", model=spark.DEFAULT_MODEL, base_url="",
                              temperature=0.2, max_tokens=1024, timeout=120)


def colorize(body):
    out, in_code = [], False
    for line in body.split("\n"):
        if line.strip().startswith("```"):
            in_code = not in_code
            out.append(CYAN + line + RESET)
        elif in_code:
            out.append(CYAN + line + RESET)
        elif line.startswith("(mock backend"):
            out.append(DIM + line + RESET)
        elif line.strip():
            out.append(textwrap.fill(line, width=96))
        else:
            out.append(line)
    return "\n".join(out)


def header(sub):
    return BOLD + PINK + spark.PERSONA + RESET + DIM + "  ·  " + sub + RESET


def render(name, ansi, cmd, title):
    rt.render(rt.parse_ansi(ansi), title, cmd, os.path.join(ROOT, "docs", name + ".png"))


def main():
    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
    o = opts()

    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "greet.py")
        with open(src, "w", encoding="utf-8") as fh:
            fh.write("def greet(name):\n    return 'Hello, ' + name\n")
        body = spark.complete(spark.code_messages("write", src, "py", ""), o)
    ansi = header("code write · mock backend") + "\n\n" + colorize(body)
    render("code", ansi, "python spark.py code write --file greet.py --backend mock",
           "spark — code")

    q = "explain why context selection matters"
    msgs, ctx = spark.repo_messages(ROOT, q, 3)
    body = spark.complete(msgs, o)
    files = ", ".join(rel.replace(os.sep, "/") for rel, _ in ctx) or "(no matches)"
    ansi = (header("repo · mock backend") + "\n" + DIM + "context: " + files + RESET
            + "\n\n" + colorize(body))
    render("repo", ansi, 'python spark.py repo "explain why context selection matters" --backend mock',
           "spark — repo")


if __name__ == "__main__":
    main()