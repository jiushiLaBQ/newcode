# -*- coding: utf-8 -*-
"""P1: 构建 QLoRA SFT 数据(chat 格式)与测试推理 prompt。

冠军格式(照抄其 system prompt 原文):
  system: 从化妆品评论抽取所有观点四元组。仅输出JSON数组,每项按[属性词,观点词,类别,极性]排列。...
  user:   <评论原文>
  assistant: [["属性词","观点词","类别","极性"], ...]   (无四元组时 [])
四元组排序:按 (属性显式则属性起点,否则观点起点) 升序;同起点按观点起点。
切分:冠军规模 train 2583 / dev 323 / confirm 323;近重复(NFKC 归一化同文)组不拆,确定性分配(冠军纪律:不用 dev 选 epoch)。

用法: python -I src/llm/build_sft_data.py
产物: data/llm/{sft_train,sft_dev,sft_confirm,test_prompts}.jsonl
"""
import json
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QUADS = ROOT / "data" / "processed" / "quads.jsonl"
TEST = ROOT / "data" / "processed" / "test_reviews.jsonl"
FOLDS = ROOT / "data" / "processed" / "folds.json"
OUT = ROOT / "data" / "llm"

SYSTEM_PROMPT = (
    "从化妆品评论抽取所有观点四元组。仅输出JSON数组，每项按[属性词,观点词,类别,极性]排列。"
    "属性词和观点词必须是原文连续片段，缺失用\"_\"。类别仅限：包装、成分、尺寸、服务、功效、价格、气味、"
    "使用体验、物流、新鲜度、真伪、整体、其他。极性仅限：正面、中性、负面。不要解释，不要漏掉不同观点。"
)

PLACEHOLDER = "_"


def nfkc(s):
    return unicodedata.normalize("NFKC", s)


def quad_sort_key(q):
    a_pos = q["a_start"] if q["a_start"] >= 0 else q["o_start"]
    return (a_pos, q["o_start"])


def assistant_text(quads):
    arr = [[q["a"], q["o"], q["c"], q["p"]] for q in sorted(quads, key=quad_sort_key)]
    return json.dumps(arr, ensure_ascii=False)


def chat_record(text, quads=None):
    rec = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
    }
    if quads is not None:
        rec["messages"].append({"role": "assistant", "content": assistant_text(quads)})
    return rec


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(QUADS, encoding="utf-8") if l.strip()]
    # 近重复组(归一化同文)不拆;确定性排序后按目标规模切 train 2583 / dev 323 / confirm 323
    groups = {}
    for r in rows:
        groups.setdefault(nfkc(r["text"]), []).append(r)
    order = sorted(groups)
    assign = {}
    n_dev = n_confirm = 0
    for key in order:
        g = groups[key]
        if n_dev + len(g) <= 323:
            split = "dev"
            n_dev += len(g)
        elif n_confirm + len(g) <= 323:
            split = "confirm"
            n_confirm += len(g)
        else:
            split = "train"
        for r in g:
            assign[r["id"]] = split

    counts = {"train": 0, "dev": 0, "confirm": 0}
    handles = {k: open(OUT / f"sft_{k}.jsonl", "w", encoding="utf-8", newline="") for k in counts}
    for r in rows:
        split = assign[r["id"]]
        handles[split].write(json.dumps(chat_record(r["text"], r["quads"]), ensure_ascii=False) + "\n")
        counts[split] += 1
    for h in handles.values():
        h.close()
    Path(OUT / "split.json").write_text(
        json.dumps({k: sorted(i for i, s in assign.items() if s == k) for k in counts},
                   ensure_ascii=False), encoding="utf-8")

    n_test = 0
    with open(OUT / "test_prompts.jsonl", "w", encoding="utf-8", newline="") as f:
        for l in open(TEST, encoding="utf-8"):
            if not l.strip():
                continue
            r = json.loads(l)
            rec = chat_record(r["text"])
            rec["id"] = r["id"]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_test += 1

    # 自检: assistant 输出可解析、词面闭集、显式词面为原文子串
    bad = 0
    for split in counts:
        for l in open(OUT / f"sft_{split}.jsonl", encoding="utf-8"):
            rec = json.loads(l)
            text = rec["messages"][1]["content"]
            arr = json.loads(rec["messages"][2]["content"])
            for a, o, c, p in arr:
                for term in (a, o):
                    if term != PLACEHOLDER and term not in text:
                        bad += 1
    print(f"splits: {counts}, test prompts: {n_test}, 子串自检失败: {bad}")


if __name__ == "__main__":
    main()
