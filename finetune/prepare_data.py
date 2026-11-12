#!/usr/bin/env python3
"""Build a chat-format LoRA training set for Spark Mini (stdlib only).

Two sources:
  --jsonl FILE   supervised pairs: {"instruction":..., "input":..., "output":...}
  --code DIR     self-supervised code completion: split each source file into
                 prefix (prompt) / suffix (target).

Emits train.jsonl / val.jsonl of {"messages":[system,user,assistant]}.
"""
import argparse
import hashlib
import json
import os
import random
import sys

SYSTEM = ("You are Spark Mini, a fast, precise and friendly coding and "
          "writing assistant. You write correct, idiomatic, minimal code and clear prose.")
CODE_EXT = {".py", ".js", ".ts", ".java", ".c", ".cpp", ".go", ".rs", ".rb", ".sh"}
SKIP_DIRS = {".git", "node_modules", "build", "dist", "__pycache__", ".venv", "venv"}


def pair_messages(instruction, inp, output):
    user = instruction if not inp else instruction + "\n\n" + inp
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": output}]


def from_jsonl(path):
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "messages" in d:
                yield d["messages"]
            elif d.get("instruction") and d.get("output"):
                yield pair_messages(d["instruction"], d.get("input", ""), d["output"])


def from_code(root, min_chars=200):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in sorted(filenames):
            ext = os.path.splitext(fn)[1].lower()
            if ext not in CODE_EXT:
                continue
            path = os.path.join(dirpath, fn)
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    text = fh.read()
            except OSError:
                continue
            if len(text) < min_chars:
                continue
            cut = int(len(text) * 0.6)
            prefix, suffix = text[:cut], text[cut:]
            if not suffix.strip():
                continue
            lang = ext.lstrip(".")
            user = ("Complete the following %s code. Return only the continuation.\n"
                    "```%s\n%s\n```" % (lang, lang, prefix))
            yield pair_messages(user, "", suffix)


def main(argv=None):
    p = argparse.ArgumentParser(prog="prepare_data", description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--jsonl")
    src.add_argument("--code")
    p.add_argument("--out", default="data")
    p.add_argument("--val-split", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=1337)
    opts = p.parse_args(argv)

    rows = list(from_jsonl(opts.jsonl) if opts.jsonl else from_code(opts.code))
    seen = set()
    uniq = []
    for msgs in rows:
        key = hashlib.sha256(json.dumps(msgs, sort_keys=True).encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            uniq.append(msgs)
    if not uniq:
        sys.stderr.write("prepare_data: no usable samples\n")
        return 2
    random.Random(opts.seed).shuffle(uniq)
    nval = max(1, int(len(uniq) * opts.val_split)) if len(uniq) > 1 else 0
    val, train = uniq[:nval], uniq[nval:]
    os.makedirs(opts.out, exist_ok=True)
    for name, subset in (("train", train), ("val", val)):
        with open(os.path.join(opts.out, name + ".jsonl"), "w", encoding="utf-8") as fh:
            for msgs in subset:
                fh.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
    print("train=%d val=%d -> %s" % (len(train), len(val), opts.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())