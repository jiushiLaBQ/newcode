# -*- coding: utf-8 -*-
"""P1: QLoRA 成员训练(冠军配方逐项照抄,seed=123,不做 dev 选 epoch)。

成员配方(--member 一键):
  baseline4b : Qwen/Qwen3-4B              r16 lr1e-4  bs1 ep3
  upgraded4b : Qwen/Qwen3-4B-Instruct-2507 r32 lr5e-5  bs4 ep4
  qwen8b     : Qwen/Qwen3-8B              r16 lr5e-5  bs2 ep3  adamw_bnb_8bit
统一: LoRA targets=all linear(q,k,v,o,gate,up,down), alpha=2r, dropout=0.05,
     NF4 双量化 + bf16, 有效 batch=8, warmup 0.05 + cosine, max_grad_norm=1,
     仅 assistant 段计算 loss, seed=123。

用法(云 GPU):
  python src/llm/train_qlora.py --member baseline4b --out checkpoints/llm/baseline4b
"""
import argparse
import inspect
import json
import math
import random
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          Trainer, TrainingArguments)

SEED = 123
MEMBERS = {
    "baseline4b": dict(model="Qwen/Qwen3-4B", r=16, lr=1e-4, bs=1, ep=3, optim="adamw_torch"),
    "upgraded4b": dict(model="Qwen/Qwen3-4B-Instruct-2507", r=32, lr=5e-5, bs=4, ep=4, optim="adamw_torch"),
    "qwen8b": dict(model="Qwen/Qwen3-8B", r=16, lr=5e-5, bs=2, ep=3, optim="adamw_bnb_8bit"),
}
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def set_seed(seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_sft(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def build_dataset(records, tokenizer, max_len=1024):
    """chat -> token;仅 assistant 段计 loss(labels 其余 -100)。"""
    input_ids_list, labels_list = [], []
    n_fallback = 0
    for rec in records:
        msgs = rec["messages"]
        ids = labels = None
        try:
            out = tokenizer.apply_chat_template(
                msgs, tokenize=True, return_dict=True, return_assistant_tokens_mask=True,
                truncation=True, max_length=max_len)
            ids = out["input_ids"]
            amask = out.get("assistant_masks") or out.get("assistant_tokens_mask")
            # 模板无 {% generation %} 时 mask 可能全 0:那就当不可用,走兜底
            if amask is not None and any(amask):
                labels = [t if m == 1 else -100 for t, m in zip(ids, amask)]
            else:
                ids = labels = None
        except Exception:
            ids = labels = None
        if ids is None or labels is None:
            n_fallback += 1
            full = tokenizer.apply_chat_template(msgs, tokenize=False,
                                                 add_generation_prompt=False)
            prompt = tokenizer.apply_chat_template(msgs[:-1], tokenize=False,
                                                   add_generation_prompt=True)
            ids = tokenizer(full, truncation=True, max_length=max_len)["input_ids"]
            p_len = len(tokenizer(prompt, truncation=True, max_length=max_len)["input_ids"])
            p_len = min(p_len, len(ids))
            labels = [-100] * p_len + ids[p_len:]
        input_ids_list.append(ids)
        labels_list.append(labels)
    if n_fallback:
        print(f"[build_dataset] assistant mask 不可用,走前缀切分兜底: {n_fallback}/{len(records)}")
    return Dataset.from_dict({"input_ids": input_ids_list, "labels": labels_list})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", required=True, choices=list(MEMBERS))
    ap.add_argument("--data", default="data/llm/sft_train.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--grad-ckpt", action="store_true", default=True)
    args = ap.parse_args()
    cfg = MEMBERS[args.member]
    set_seed()

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], quantization_config=quant, dtype=torch.bfloat16,
        device_map="auto", attn_implementation="sdpa")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.grad_ckpt)
    lora = LoraConfig(
        r=cfg["r"], lora_alpha=2 * cfg["r"], lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules=LORA_TARGETS)
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = build_dataset(load_sft(args.data), tokenizer, max_len=args.max_len)

    eff_bs = 8  # 有效 batch
    accum = max(1, eff_bs // (cfg["bs"] * max(1, torch.cuda.device_count())))
    kwargs = dict(
        output_dir=args.out, num_train_epochs=cfg["ep"],
        per_device_train_batch_size=cfg["bs"], gradient_accumulation_steps=accum,
        learning_rate=cfg["lr"], lr_scheduler_type="cosine",
        max_grad_norm=1.0, optim=cfg["optim"],
        bf16=True, logging_steps=10, save_strategy="epoch",
        report_to="none", seed=SEED, remove_unused_columns=False)
    # transformers 5.x 删了 warmup_ratio,改 warmup_steps;4.x 两者都在
    sig = inspect.signature(TrainingArguments.__init__).parameters
    if "warmup_ratio" in sig:
        kwargs["warmup_ratio"] = 0.05
    else:
        # 预告步数:trainer 之后会用 num_update_steps_per_epoch,这里先给个保守值
        n_cards = max(1, torch.cuda.device_count())
        steps_per_epoch = max(1, math.ceil(len(ds) / (cfg["bs"] * n_cards * accum)))
        kwargs["warmup_steps"] = max(1, int(0.05 * steps_per_epoch * cfg["ep"]))
    targs = TrainingArguments(**kwargs)
    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=lambda b: {
                          "input_ids": pad_to_max([x["input_ids"] for x in b], tokenizer.pad_token_id),
                          "labels": pad_to_max([x["labels"] for x in b], -100),
                      })
    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"saved -> {args.out}")


def pad_to_max(seqs, pad):
    n = max(len(s) for s in seqs)
    t = torch.full((len(seqs), n), pad, dtype=torch.long)
    for i, s in enumerate(seqs):
        t[i, :len(s)] = torch.tensor(s)
    return t


if __name__ == "__main__":
    main()
