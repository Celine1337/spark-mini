#!/usr/bin/env python3
"""LoRA / QLoRA fine-tune of the open base model into a Spark Mini adapter.

Requires a GPU and: pip install torch transformers peft datasets accelerate
(bitsandbytes for 4-bit QLoRA). Run prepare_data.py first, then:
    python finetune/train_lora.py --data data --config finetune/config.json
The adapter is saved to config.output_dir; merge with merge_lora.py.
"""
import argparse
import json
import os
import sys


def load_config(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def main(argv=None):
    p = argparse.ArgumentParser(prog="train_lora", description=__doc__)
    p.add_argument("--data", default="data")
    p.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.json"))
    opts = p.parse_args(argv)
    cfg = load_config(opts.config)

    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                  BitsAndBytesConfig, TrainingArguments, Trainer)
    except ImportError as exc:
        sys.stderr.write(
            "train_lora: missing dependency (%s).\n"
            "Install on a GPU machine: pip install torch transformers peft datasets "
            "accelerate bitsandbytes\n" % exc)
        return 2

    train_path = os.path.join(opts.data, "train.jsonl")
    if not os.path.isfile(train_path):
        sys.stderr.write("train_lora: %s not found; run prepare_data.py first\n" % train_path)
        return 2

    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    ds = load_dataset("json", data_files={"train": train_path,
                                          "val": os.path.join(opts.data, "val.jsonl")})

    def fmt(example):
        text = tok.apply_chat_template(example["messages"], tokenize=False)
        out = tok(text, truncation=True, max_length=cfg["train"]["max_len"])
        out["labels"] = out["input_ids"].copy()
        return out

    ds = ds.map(fmt, remove_columns=ds["train"].column_names)

    quant = None
    if cfg.get("quantize_4bit") and torch.cuda.is_available():
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                   bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], quantization_config=quant,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    lc = cfg["lora"]
    model = get_peft_model(model, LoraConfig(
        r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
        target_modules=lc["target_modules"], task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    t = cfg["train"]
    args = TrainingArguments(
        output_dir=cfg["output_dir"], num_train_epochs=t["epochs"],
        per_device_train_batch_size=t["batch_size"],
        gradient_accumulation_steps=t["grad_accum"], learning_rate=t["lr"],
        warmup_ratio=t["warmup_ratio"], logging_steps=10, save_strategy="epoch",
        evaluation_strategy="epoch" if ds.get("val") else "no",
        bf16=torch.cuda.is_available(), report_to=[])
    Trainer(model=model, args=args, train_dataset=ds["train"],
            eval_dataset=ds.get("val"), tokenizer=tok).train()
    model.save_pretrained(cfg["output_dir"])
    tok.save_pretrained(cfg["output_dir"])
    print("adapter saved to", cfg["output_dir"])
    return 0


if __name__ == "__main__":
    sys.exit(main())