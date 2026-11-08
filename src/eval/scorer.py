# -*- coding: utf-8 -*-
"""官方评测的本地镜像:同 ID 内四元组逐一匹配,四字段全对记 1。

匹配语义 = 多重集(Counter 交集):同 ID 内两条完全相同的四元组按出现次数配对。
  P = 预测四元组总数, G = 金标四元组总数, S = 命中数
  Precision = S/P, Recall = S/G, F1 = 2PR/(P+R)

用法:
  python -m src.eval.scorer --gold data/processed/quads.jsonl --pred submissions/Result.csv
  python -m src.eval.scorer --gold data/processed/quads.jsonl --pred pred.jsonl   (同 quads.jsonl 结构,便于 CV)
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

PLACEHOLDER = "_"


def load_gold_jsonl(path):
    """quads.jsonl -> {id: [(a,o,c,p), ...]}"""
    out = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for q in row["quads"]:
                out[row["id"]].append((q["a"], q["o"], q["c"], q["p"]))
    return dict(out)


def load_pred_csv(path):
    """提交格式(无表头): id,AspectTerm,OpinionTerm,Category,Polarity -> {id: [quad,...]}"""
    out = defaultdict(list)
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            if len(row) != 5:
                raise ValueError(f"提交行字段数 != 5: {row}")
            rid = int(row[0])
            out[rid].append((row[1], row[2], row[3], row[4]))
    return dict(out)


def load_pred_jsonl(path):
    out = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for q in row["quads"]:
                out[row["id"]].append((q["a"], q["o"], q["c"], q["p"]))
    return dict(out)


def score(gold, pred):
    """gold/pred: {id: [(a,o,c,p), ...]} -> {P,G,S,precision,recall,f1,per_id}"""
    S = P = G = 0
    per_id = {}
    for rid in set(gold) | set(pred):
        g = Counter(gold.get(rid, []))
        p = Counter(pred.get(rid, []))
        s = sum((g & p).values())
        ng, np_ = sum(g.values()), sum(p.values())
        S += s
        G += ng
        P += np_
        per_id[rid] = {"S": s, "G": ng, "P": np_}
    precision = S / P if P else 0.0
    recall = S / G if G else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"P": P, "G": G, "S": S, "precision": precision, "recall": recall, "f1": f1, "per_id": per_id}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--per-id", action="store_true", help="输出每个 id 的命中明细")
    args = ap.parse_args()

    gold = load_gold_jsonl(args.gold)
    if args.pred.endswith(".csv"):
        pred = load_pred_csv(args.pred)
    else:
        pred = load_pred_jsonl(args.pred)
    r = score(gold, pred)
    print(f"P={r['P']} G={r['G']} S={r['S']} "
          f"precision={r['precision']:.4f} recall={r['recall']:.4f} f1={r['f1']:.4f}")
    if args.per_id:
        for rid in sorted(r["per_id"]):
            print(rid, r["per_id"][rid])


if __name__ == "__main__":
    main()
