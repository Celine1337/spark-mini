#!/usr/bin/env python3
"""Merge a trained Spark LoRA adapter into the base model (GPU-friendly, CPU ok).

Requires: pip install torch transformers peft
    python finetune/merge_lora.py --adapter spark-lora --out spark-merged
For a lightweight local runtime, convert the merged dir to GGUF with llama.cpp
(convert_hf_to_gguf.py) and quantize, then serve via Ollama / llama.cpp server.
"""
import argparse
import json
import os
import sys


def main(argv=None):
    p = argparse.ArgumentParser(prog="merge_lora", description=__doc__)
    p.add_argument("--adapter", default="spark-lora")
    p.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.json"))
    p.add_argument("--out", default="spark-merged")
    opts = p.parse_args(argv)
    with open(opts.config, "r", encoding="utf-8") as fh:
        base = json.load(fh)["base_model"]

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        sys.stderr.write("merge_lora: missing dependency (%s).\n"
                         "Install: pip install torch transformers peft\n" % exc)
        return 2
    if not os.path.isdir(opts.adapter):
        sys.stderr.write("merge_lora: adapter dir not found: %s\n" % opts.adapter)
        return 2

    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(model, opts.adapter)
    model = model.merge_and_unload()
    model.save_pretrained(opts.out)
    AutoTokenizer.from_pretrained(base).save_pretrained(opts.out)
    print("merged model saved to", opts.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())