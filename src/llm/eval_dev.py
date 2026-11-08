# -*- coding: utf-8 -*-
"""dev/confirm 推理结果自评:用 infer_qlora.py 输出里的 _gold 直接算多重集 F1。

用法:
  python -I src/llm/eval_dev.py --pred data/llm/dev_pred_baseline4b.jsonl
  python -I src/llm/eval_dev.py --pred a.jsonl b.jsonl ...        # 多文件对比
"""
import argparse
import json
from collections import Counter


def gold_quads(raw):
    """assistant 文本 -> [(a,o,c,p), ...];解析失败返回 None。"""
    if raw is None:
        return None
    try:
        arr = json.loads(raw)
    except Exception:
        return None
    out = []
    for item in arr:
        if isinstance(item, (list, tuple)) and len(item) == 4:
            out.append(tuple(str(x).strip() for x in item))
    return out


def pred_quads(quads):
    return [(q["a"], q["o"], q["c"], q["p"]) for q in quads]


def score_file(path):
    S = P = G = 0
    n_rows = n_empty = n_bad_gold = 0
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        g = gold_quads(row.get("_gold"))
        if g is None:
            n_bad_gold += 1
            continue
        p = pred_quads(row.get("quads") or [])
        n_rows += 1
        if not p:
            n_empty += 1
        cg, cp = Counter(g), Counter(p)
        S += sum((cg & cp).values())
        G += sum(cg.values())
        P += sum(cp.values())
    prec = S / P if P else 0.0
    rec = S / G if G else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"file": path, "rows": n_rows, "empty_pred": n_empty,
            "bad_gold": n_bad_gold, "P": P, "G": G, "S": S,
            "precision": prec, "recall": rec, "f1": f1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", nargs="+", required=True)
    args = ap.parse_args()
    for path in args.pred:
        r = score_file(path)
        print(f"{r['file']}: rows={r['rows']} empty={r['empty_pred']} bad_gold={r['bad_gold']} "
              f"P={r['P']} G={r['G']} S={r['S']} "
              f"p={r['precision']:.4f} r={r['recall']:.4f} F1={r['f1']:.4f}")


if __name__ == "__main__":
    main()
