<p align="center"><img src="docs/style/hero.png" alt="Spark Mini — local code & writing assistant" width="100%"></p>

<p align="center">
  <a href="README.ru.md"><img src="docs/style/pill-ru.png" alt="README RU" height="42"></a>
  <a href="https://github.com/Celine1337/spark-mini/releases"><img src="docs/style/pill-release.png" alt="Releases" height="42"></a>
  <a href="https://github.com/Celine1337/spark-mini/actions/workflows/ci.yml"><img src="docs/style/pill-ci.png" alt="CI" height="42"></a>
  <a href="LICENSE"><img src="docs/style/pill-license.png" alt="MIT" height="42"></a>
</p>

<img src="docs/style/sec-about-en.png" alt="About" width="100%">

**Spark Mini** is a local code & writing assistant: a persona, a standard-library
CLI, retrieval over your repository, and an optional LoRA fine-tune recipe —
running on top of an **open-weight base model** (default
[Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct)).

What it is **not**: a from-scratch foundation model. No trained weights ship in
this repo, and nothing here claims to be "perfect". The intelligence comes from
the open base model you point it at; Spark Mini adds the persona, the prompts,
the repo context, the eval harness and the fine-tune tooling around it.

The CLI itself is standard-library only (Python 3.9+). It delegates inference to
a local [Ollama](https://ollama.com) or any OpenAI-compatible server (llama.cpp,
vLLM, LM Studio). A built-in `mock` backend produces deterministic placeholder
answers so demos, tests and CI run offline with **no model loaded** — every
screenshot below is labelled `mock` for that reason.

**v1.1 makes the tooling itself much smarter.** Repository context is no longer a
naive whole-file TF-IDF score plus a truncated file head. A real lexical retrieval
engine (`retrieval.py`) now splits code at definition boundaries and prose at
paragraphs, tokenizes identifiers (so `invoiceTotal` matches `invoice_total`),
ranks chunks with **Okapi BM25** plus symbol / filename boosting, caches the index
incrementally in `.spark/`, and packs **cited, token-budgeted** excerpts
(`path:start-end`) into the prompt. The base model still does the reasoning — it
just finally sees the *right* code instead of the first 1500 characters of a file.

<img src="docs/style/sec-features-en.png" alt="Features" width="100%">

- Seven modes: `ask`, interactive `chat`, `code` (write / fix / explain / review /
  test a single file), `repo` (question answered with retrieved repo context),
  `write` (prose in plain / concise / friendly / formal tone), plus two
  model-free retrieval commands: `search` (rank repo chunks) and `index`
  (build / inspect the retrieval index).
- **BM25 retrieval engine** (`retrieval.py`, stdlib only): structure-aware
  chunking (code at `def`/`class`/`fn`/… boundaries, prose at paragraphs) with
  1-based line ranges; identifier-aware tokenization (camelCase / snake_case
  sub-tokens, stopword removal); Okapi BM25 (k1=1.5, b=0.75) with document-length
  normalization; symbol- and filename-boosting; a persistent, incrementally
  invalidated `.spark/index.json` cache keyed on mtime+size; token-budgeted
  context packing with `path:start-end` citations. No embeddings, no vector DB.
- Three backends behind one interface: `ollama`, `openai` (compatible), `mock`.
  Switch with `--backend` or `SPARK_BACKEND`; point anywhere with `--base-url`.
- Opt-in `--stream` token streaming for `ollama` / `openai` (NDJSON / SSE);
  `chat` gains `--root` repo augmentation, `/reset` and `/context`, and automatic
  token-budgeted memory trimming so long sessions stay within context.
- `--json` machine-readable output (persona, backend, model, response, context;
  `search`/`index` emit ranked results / index stats); `--temperature`,
  `--max-tokens`, `--timeout` knobs; `--no-color` / `NO_COLOR`.
- Optional GPU fine-tune pipeline in `finetune/`: prepare a chat dataset, train a
  LoRA / QLoRA adapter, merge it, and score it with a tiny offline eval harness.
- Honest failure modes: unreachable backend → exit `1` with a clear message;
  missing file / root → exit `2`. No silent fallbacks, no invented answers.

<img src="docs/style/sec-checks-en.png" alt="Modes & pipeline" width="100%">

| Command | What it does |
|---------|--------------|
| `spark.py ask "…"` | one-shot question |
| `spark.py chat` | interactive session (`/exit`, `/reset`, `/context`; `--root` for repo context) |
| `spark.py code <task> --file F` | write / fix / explain / review / test one file |
| `spark.py repo "…" --root DIR` | answer using top-k retrieved, cited repo chunks |
| `spark.py write "…" --style S` | draft prose in the chosen tone |
| `spark.py search "…" --root DIR` | rank repo chunks by BM25 (no model needed) |
| `spark.py index --root DIR` | build / inspect the retrieval index (`--rebuild` to force) |

Fine-tune pipeline (needs your own GPU; weights are **not** included):

| Step | Script | Notes |
|------|--------|-------|
| 1 | `finetune/prepare_data.py` | stdlib; `--jsonl` pairs or `--code DIR` self-supervised completion; dedupe + train/val split |
| 2 | `finetune/train_lora.py` | LoRA / QLoRA (PEFT) on the base model; guarded imports, exits `2` with an install hint if deps/GPU absent |
| 3 | `finetune/merge_lora.py` | merge adapter into base; convert to GGUF via llama.cpp for a light local runtime |
| 4 | `finetune/eval.py` | stdlib scoring: code answers must parse, prose must hit a keyword; `--backend mock` runs offline |

<img src="docs/style/sec-stack-en.png" alt="Tech stack" width="100%">

<img src="docs/style/chips.png" alt="Python 3.9+, stdlib CLI, BM25 retrieval, Ollama / OpenAI API; LoRA / QLoRA (PEFT), unittest, Pillow" width="100%">

<img src="docs/style/sec-quick-en.png" alt="Quick start" width="100%">

Offline, no model needed (this is what CI runs):

```console
$ python spark.py ask "hi" --backend mock
$ python spark.py code explain --file spark.py --backend mock
$ python spark.py repo "how does context selection work?" --backend mock
$ python spark.py search "BM25 chunk ranking" --topk 3
$ python spark.py index
$ python finetune/eval.py --backend mock
```

With a real local model (Ollama):

```console
$ ollama pull qwen2.5-coder:1.5b
$ python spark.py code review --file app.py
$ python spark.py repo "where is auth handled?" --root .
$ python spark.py write "release notes for v1.1" --style concise
$ python spark.py ask "explain this stack trace" --stream
```

OpenAI-compatible server (llama.cpp / vLLM / LM Studio):

```console
$ python spark.py ask "explain this stack trace" --backend openai --base-url http://localhost:8080
```

Fine-tune on your GPU:

```console
$ pip install torch transformers peft datasets accelerate bitsandbytes
$ python finetune/prepare_data.py --code ./src
$ python finetune/train_lora.py --data data
$ python finetune/merge_lora.py --adapter spark-lora --out spark-merged
$ python finetune/eval.py --backend ollama --model spark-merged
```

Exit codes: `0` — success, `1` — backend unreachable / eval failed, `2` — bad
path, missing data, or missing GPU dependencies.

<img src="docs/style/sec-shots-en.png" alt="Screenshots" width="100%">

The `code` and `repo` shots use the offline `mock` backend — placeholder answers,
no model loaded. The `search` shot is pure retrieval (BM25 ranking, no model at all).

![code mode](docs/code.png)

![repo mode with retrieved, cited context](docs/repo.png)

![search mode: BM25-ranked chunks with scores](docs/search.png)

<img src="docs/style/sec-tests-en.png" alt="Tests" width="100%">

```console
$ python -m unittest discover -s tests -v
```

100 tests cover the prompt builders, all three backends (including a local fake
HTTP server asserting request paths and payloads, plus NDJSON/SSE streaming),
the BM25 retrieval engine (tokenization, chunking, symbol/filename boosting,
cache invalidation, budgeted packing), the `search`/`index`/`chat` commands, CLI
exit codes, the data preparer, the eval scorer, and graceful degradation of the
GPU scripts when dependencies are absent. Images are regenerable:

```console
$ pip install pillow
$ python tools/make_shots.py      # docs/code.png, docs/repo.png, docs/search.png
$ python tools/make_style.py tools/style.json docs/style
```

<img src="docs/style/stats.png" alt="100 tests, 3 OSes in CI, v1.1.0" width="100%">

[MIT](LICENSE) © Celine1337