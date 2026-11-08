# -*- coding: utf-8 -*-
"""P1: QLoRA 成员推理(冠军推理配置:贪心、左 padding、max_new_tokens=384、enable_thinking=False)。

输出 quads.jsonl 结构(带偏移,词面强制原文子串;幻觉词面编辑距离<=1 回贴,失败置 "_")。

用法:
  python src/llm/infer_qlora.py --base Qwen/Qwen3-4B --adapter checkpoints/llm/baseline4b \
      --input data/llm/test_prompts.jsonl --out data/llm/pred_baseline4b.jsonl [--limit N]
"""
import argparse
import json
import re

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

CATEGORIES = {"包装", "成分", "尺寸", "服务", "功效", "价格", "气味", "使用体验", "物流", "新鲜度", "真伪", "整体", "其他"}
POLARITIES = {"正面", "中性", "负面"}
PLACEHOLDER = "_"


def snap(term, text):
    """词面必须是原文子串:精确 -> 编辑距离<=1 滑窗 -> None。"""
    if term == PLACEHOLDER:
        return PLACEHOLDER
    if term in text:
        return term
    for L in (len(term) - 1, len(term), len(term) + 1):
        if L <= 0:
            continue
        for i in range(len(text) - L + 1):
            cand = text[i:i + L]
            if _lev(term, cand) <= 1:
                return cand
    return None


def _lev(a, b):
    if abs(len(a) - len(b)) > 1:
        return 2
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def parse_output(text, review):
    """生成文本 -> quads list(dict)。解析失败返回 None(上层决定重试)。"""
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
    except Exception:
        return None
    quads = []
    for item in arr:
        if not (isinstance(item, (list, tuple)) and len(item) == 4):
            continue
        a, o, c, p = (str(x).strip() for x in item)
        a = snap(a, review)
        o = snap(o, review)
        if a is None or o is None:
            continue  # 幻觉且不可回贴:丢该条
        if c not in CATEGORIES or p not in POLARITIES:
            continue  # 闭集外:丢该条
        if a == PLACEHOLDER and o == PLACEHOLDER:
            continue  # 双隐式禁止
        quads.append({
            "a": a, "a_start": review.find(a) if a != PLACEHOLDER else -1,
            "a_end": review.find(a) + len(a) if a != PLACEHOLDER else -1,
            "o": o, "o_start": review.find(o) if o != PLACEHOLDER else -1,
            "o_end": review.find(o) + len(o) if o != PLACEHOLDER else -1,
            "c": c, "p": p})
    return quads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--input", default="data/llm/test_prompts.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # decoder-only 生成必须左 padding
    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa")
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    records = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    if args.limit:
        records = records[:args.limit]
    # dev/confirm 文件没有 id:用行号补;_gold 存参考答案供离线评分
    for i, r in enumerate(records):
        r.setdefault("id", str(i))
        if len(r.get("messages", [])) >= 3:
            r["_gold"] = r["messages"][2]["content"]

    out = open(args.out, "w", encoding="utf-8", newline="")
    for i in range(0, len(records), args.batch_size):
        batch = records[i:i + args.batch_size]
        prompts = []
        for r in batch:
            msgs = r["messages"]
            if msgs and msgs[-1].get("role") == "assistant":
                msgs = msgs[:-1]  # dev/confirm:带答案的样本只取 prompt 段
            try:
                p = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False)
            except TypeError:
                p = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True)
            prompts.append(p)
        tok = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                        max_length=1024).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **tok, max_new_tokens=args.max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id)
        for j, r in enumerate(batch):
            new_tokens = gen[j][tok["input_ids"].shape[1]:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            review = next(m["content"] for m in r["messages"] if m.get("role") == "user")
            quads = parse_output(text, review)
            if quads is None:  # 解析失败重试一次:强制 JSON 提示
                retry_prompt = prompts[j] + "\n只输出JSON数组:"
                tok2 = tokenizer(retry_prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    g2 = model.generate(**tok2, max_new_tokens=args.max_new_tokens,
                                        do_sample=False, pad_token_id=tokenizer.pad_token_id)
                text = tokenizer.decode(g2[0][tok2["input_ids"].shape[1]:], skip_special_tokens=True)
                quads = parse_output(text, review) or []
            out.write(json.dumps({"id": r["id"], "text": review, "quads": quads,
                                  "_gold": r.get("_gold")},
                                 ensure_ascii=False) + "\n")
        out.flush()
        print(f"{min(i + args.batch_size, len(records))}/{len(records)}")
    out.close()
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
