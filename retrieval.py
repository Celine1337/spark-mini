#!/usr/bin/env python3
"""Smart retrieval engine for Spark Mini — standard library only.

This replaces the original naive whole-file TF-IDF (which scored entire files by
substring counts and then injected only the *first* 1500 characters) with a real
lexical retrieval pipeline:

  * structure-aware chunking — code is split at definition boundaries (def /
    class / function / fn / func / impl ...), prose at paragraphs; every chunk
    keeps its 1-based line range so answers can cite ``path:start-end``.
  * identifier-aware tokenization — camelCase and snake_case are split into
    sub-tokens so ``invoiceTotal`` matches ``invoice_total``; stopwords removed.
  * Okapi BM25 ranking with document-length normalization (k1=1.5, b=0.75).
  * symbol extraction + boosting — chunks that *define* a symbol named in the
    query are boosted; filename mentions boost that file.
  * a persistent, incrementally invalidated index cache (``.spark/index.json``)
    keyed on file mtime+size, so repeated queries are fast and stay correct.
  * token-budgeted context packing with citations.

No embeddings, no vector DB, no third-party deps. The base model still does the
reasoning; this engine makes sure it sees the *right* code instead of a truncated
file head. Compatible with Python 3.9+ on Linux/macOS/Windows.
"""

import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict

INDEX_VERSION = 3
CACHE_DIRNAME = ".spark"
MAX_FILE_BYTES = 200_000
MAX_CHUNK_CHARS = 1400
MAX_CHUNK_LINES = 80
MIN_PARA_CHARS = 200

CODE_EXT = {".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".h", ".cpp",
            ".hpp", ".cs", ".go", ".rs", ".rb", ".php", ".sh", ".sql", ".md",
            ".txt", ".json", ".yml", ".yaml", ".toml", ".html", ".css"}
PROSE_EXT = {".md", ".txt", ".rst"}
SKIP_DIRS = {".git", "node_modules", "build", "dist", ".venv", "venv", "__pycache__",
             ".gradle", "target", ".idea", ".vscode", ".spark", ".mypy_cache",
             ".pytest_cache", ".ruff_cache"}

STOPWORDS = frozenset("""
a an and are as at be but by for from had has have he her his how i if in into is
it its of on or our she so than that the their them then there these they this to
was we were what when where which who why will with you your not no do does did
can could should would may might must shall about above after again all also am
any because before below between both during each few further here more most
other out over own same some such only too under until up very just now
""".split())

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")
_DEF = re.compile(
    r"^(\s*)(?:async\s+|pub\s+|export\s+|default\s+|static\s+|public\s+|private\s+|"
    r"protected\s+|abstract\s+|final\s+)*"
    r"(def|class|function|func|fn|impl|struct|interface|enum|trait|type|sub|proc)\b(.*)$")
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_FILE_REF = re.compile(r"[\w./\\-]+\.[A-Za-z0-9]+")


# ---------------------------------------------------------------- tokenizer
def split_identifier(tok):
    """Split a camelCase / snake_case identifier into lowercased sub-tokens."""
    parts = []
    for piece in tok.split("_"):
        if piece:
            parts.extend(p.lower() for p in _CAMEL.findall(piece))
    return parts


def tokenize(text):
    """Document tokens: words + identifier sub-tokens, stopwords removed."""
    out = []
    for m in _WORD.findall(text):
        low = m.lower()
        out.append(low)
        if "_" in m or any(c.isupper() for c in m):
            for sub in split_identifier(m):
                if sub != low:
                    out.append(sub)
    return [t for t in out if len(t) >= 2 and t not in STOPWORDS]


def query_terms(query):
    """Query tokens (min length 3 to keep noise like 'ab' from matching)."""
    seen = []
    for t in tokenize(query):
        if len(t) >= 3 and t not in seen:
            seen.append(t)
    return seen


def understand(query):
    """Lightweight query understanding: terms, identifiers, filename mentions."""
    identifiers = _WORD.findall(query)
    filenames = [os.path.basename(f.replace("\\", "/")) for f in _FILE_REF.findall(query)]
    return {"terms": query_terms(query), "identifiers": identifiers,
            "filenames": filenames}


def estimate_tokens(text):
    """Cheap token estimate (~4 chars/token) for context budgeting."""
    return max(1, len(text) // 4)


# ---------------------------------------------------------------- chunking
def _chunk(relpath, start, end, text, kind, symbols):
    return {"relpath": relpath, "start": start, "end": end,
            "text": text, "kind": kind, "symbols": symbols}


def _windows(lines, relpath, kind):
    """Char/line-bounded windows; prose also breaks at paragraph boundaries."""
    chunks, buf, start = [], [], 1
    for i, ln in enumerate(lines):
        if not buf:
            start = i + 1
        buf.append(ln)
        size = sum(len(x) + 1 for x in buf)
        para_break = (kind == "prose" and not ln.strip() and size >= MIN_PARA_CHARS)
        if size >= MAX_CHUNK_CHARS or len(buf) >= MAX_CHUNK_LINES or para_break:
            text = "\n".join(buf).strip()
            if text:
                chunks.append(_chunk(relpath, start, i + 1, text, kind, []))
            buf = []
    if buf:
        text = "\n".join(buf).strip()
        if text:
            chunks.append(_chunk(relpath, start, len(lines), text, kind, []))
    return chunks


def chunk_text(relpath, text, ext):
    """Split file text into ranked-ready chunks with line ranges + symbols."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    kind = "prose" if ext in PROSE_EXT else "code"
    if kind == "prose":
        return _windows(lines, relpath, "prose")
    anchors = []
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("#"):
            continue
        m = _DEF.match(ln)
        if m:
            indent = len(m.group(1).expandtabs(4))
            nm = _NAME.search(m.group(3))
            anchors.append((i, indent, nm.group(0) if nm else ""))
    if not anchors:
        return _windows(lines, relpath, "code")
    chunks = []
    first = anchors[0][0]
    if first > 0:
        head = "\n".join(lines[:first]).strip()
        if head:
            chunks.append(_chunk(relpath, 1, first, "\n".join(lines[:first]), "code", []))
    for j, (i, indent, name) in enumerate(anchors):
        end = len(lines)
        for k in range(j + 1, len(anchors)):
            if anchors[k][1] <= indent:
                end = anchors[k][0]
                break
        seg = lines[i:end]
        syms = [name] if name else []
        if len(seg) > MAX_CHUNK_LINES:
            for w in range(0, len(seg), MAX_CHUNK_LINES):
                sub = seg[w:w + MAX_CHUNK_LINES]
                chunks.append(_chunk(relpath, i + w + 1, i + w + len(sub),
                                     "\n".join(sub), "code", syms if w == 0 else []))
        else:
            chunks.append(_chunk(relpath, i + 1, i + len(seg), "\n".join(seg), "code", syms))
    return chunks


def iter_files(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in sorted(filenames):
            if os.path.splitext(fn)[1].lower() in CODE_EXT:
                yield os.path.join(dirpath, fn)


def _read_chunks(root, path):
    try:
        if os.path.getsize(path) > MAX_FILE_BYTES:
            return []
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return []
    relpath = os.path.relpath(path, root).replace("\\", "/")
    return chunk_text(relpath, text, os.path.splitext(path)[1].lower())


# ---------------------------------------------------------------- BM25 index
class Index:
    def __init__(self, root, chunks):
        self.root = root
        self.chunks = chunks
        self.n = len(chunks)
        self.df = Counter()
        self.tf = []
        self.dl = []
        self.symbols = defaultdict(set)
        for i, c in enumerate(chunks):
            cnt = Counter(tokenize(c["text"]))
            self.tf.append(cnt)
            self.dl.append(sum(cnt.values()))
            for t in cnt:
                self.df[t] += 1
            for s in c.get("symbols", []):
                self.symbols[s.lower()].add(i)
        self.avgdl = (sum(self.dl) / self.n) if self.n else 0.0
        self.inverted = defaultdict(list)
        for i, cnt in enumerate(self.tf):
            for t, f in cnt.items():
                self.inverted[t].append((i, f))

    def idf(self, term):
        df = self.df.get(term, 0)
        return math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))

    def score(self, terms, identifiers=(), filenames=()):
        scores = [0.0] * self.n
        k1, b = 1.5, 0.75
        avgdl = self.avgdl or 1.0
        for t in terms:
            idf = self.idf(t)
            for i, f in self.inverted.get(t, ()):
                dl = self.dl[i] or 1
                scores[i] += idf * (f * (k1 + 1.0)) / (f + k1 * (1.0 - b + b * dl / avgdl))
        # symbol-bonus: chunks defining a symbol named in the query
        for ident in identifiers:
            for key in {ident.lower(), *split_identifier(ident)}:
                for i in self.symbols.get(key, ()):
                    scores[i] += 1.5 * (self.idf(key) if self.df.get(key) else 1.0)
        # filename-bonus: query explicitly mentions a file
        if filenames:
            wanted = {f.lower() for f in filenames}
            for i, c in enumerate(self.chunks):
                if os.path.basename(c["relpath"]).lower() in wanted:
                    scores[i] += 2.0
        return scores

    def stats(self):
        return {"files": len({c["relpath"] for c in self.chunks}),
                "chunks": self.n, "terms": len(self.df),
                "symbols": len(self.symbols)}


# ---------------------------------------------------------------- index cache
def cache_path(root, cache_dir=None):
    return os.path.join(cache_dir or os.path.join(root, CACHE_DIRNAME), "index.json")


def _sig(path):
    st = os.stat(path)
    return [st.st_mtime, st.st_size]


def build_index(root, cache_dir=None, rebuild=False):
    """Build (or incrementally refresh) the BM25 index for ``root``.

    Reuses cached chunks for files whose mtime+size are unchanged; re-chunks
    changed/new files and drops deleted ones. Returns (Index, meta).
    """
    root = os.path.abspath(root)
    cpath = cache_path(root, cache_dir)
    cached = {}
    if not rebuild:
        try:
            with open(cpath, "r", encoding="utf-8") as fh:
                blob = json.load(fh)
            if blob.get("version") == INDEX_VERSION and blob.get("root") == root:
                cached = {f["relpath"]: f for f in blob.get("files", [])}
        except (OSError, ValueError, KeyError):
            cached = {}
    files, chunks, reused = [], [], 0
    for path in iter_files(root):
        relpath = os.path.relpath(path, root).replace("\\", "/")
        try:
            sig = _sig(path)
        except OSError:
            continue
        prev = cached.get(relpath)
        if prev and prev.get("sig") == sig:
            fchunks = prev["chunks"]
            reused += 1
        else:
            fchunks = _read_chunks(root, path)
        files.append({"relpath": relpath, "sig": sig, "chunks": fchunks})
        chunks.extend(fchunks)
    idx = Index(root, chunks)
    meta = idx.stats()
    meta["reused"] = reused
    meta["cache"] = cpath
    try:
        os.makedirs(os.path.dirname(cpath), exist_ok=True)
        tmp = cpath + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"version": INDEX_VERSION, "root": root, "files": files}, fh)
        os.replace(tmp, cpath)
    except OSError:
        meta["cache_written"] = False
    else:
        meta["cache_written"] = True
    return idx, meta


# ---------------------------------------------------------------- search API
def search(root, query, topk=5, cache_dir=None, rebuild=False):
    """Return up to ``topk`` ranked chunks (dicts) for ``query`` in ``root``."""
    u = understand(query)
    if not u["terms"]:
        return []
    idx, _ = build_index(root, cache_dir=cache_dir, rebuild=rebuild)
    if not idx.n:
        return []
    scores = idx.score(u["terms"], u["identifiers"], u["filenames"])
    order = sorted(range(idx.n),
                   key=lambda i: (-scores[i], idx.chunks[i]["relpath"], idx.chunks[i]["start"]))
    out = []
    for i in order[:topk]:
        if scores[i] <= 0:
            continue
        c = idx.chunks[i]
        out.append({"relpath": c["relpath"], "start": c["start"], "end": c["end"],
                    "score": round(scores[i], 4), "text": c["text"],
                    "symbols": c.get("symbols", []), "kind": c["kind"]})
    return out


def pack_context(results, max_chars=6000):
    """Pack ranked chunks into a citation-bearing context block within budget."""
    parts, used = [], 0
    for r in results:
        block = "--- %s:%d-%d ---\n%s" % (r["relpath"], r["start"], r["end"], r["text"])
        if parts and used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)


def select_context(root, query, topk=3, max_chars=1500, cache_dir=None):
    """Backward-compatible selector: top-k *files* as (relpath, snippet) pairs.

    Snippets are now the most relevant chunk(s) with a ``path:start-end`` header
    rather than a blind file head. Returns [] when the query has no usable terms.
    """
    results = search(root, query, topk=max(topk * 3, topk), cache_dir=cache_dir)
    if not results:
        return []
    out, by_file = [], {}
    for r in results:
        rp = r["relpath"]
        header = "%s:%d-%d" % (rp, r["start"], r["end"])
        if rp in by_file:
            pos = by_file[rp]
            rel, snip = out[pos]
            extra = "\n" + header + "\n" + r["text"]
            if len(snip) + len(extra) <= max_chars:
                out[pos] = (rel, snip + extra)
            continue
        snippet = (header + "\n" + r["text"])[:max_chars]
        by_file[rp] = len(out)
        out.append((rp, snippet))
        if len(out) >= topk:
            break
    return out