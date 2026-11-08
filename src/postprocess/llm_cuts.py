# -*- coding: utf-8 -*-
"""P2 删除族:LLM 融合结果的逐条删除规则(冠军同款),带 dev 判据自检。

每条规则按冠军判据评估:删 d 行含 t 命中,F1 提升充要 t/d < f/2。
dev 模式输出每条规则的 t/d 与独立 ΔF1;只有达标的才该在测试集上开。

规则:
  pol_conflict : 多成员在该 (a,o,c) 上极性不一致 -> 删
  cat_other    : Category == 其他 -> 删
  cat_size     : Category == 尺寸 -> 删
  cat_lex      : Category == 成分 且 观点词含尺寸/物流词 -> 删(可选,默认关)

用法:
  python -I src/postprocess/llm_cuts.py --in data/llm/dev_cross.jsonl \
      --rules pol_conflict cat_other cat_size --out data/llm/dev_cuts.jsonl
"""
import argparse
import json
from collections import Counter


def qkey(q):
    return (q["a"], q["o"], q["c"], q["p"])


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def gold_quads(raw):
    if raw is None:
        return None
    try:
        arr = json.loads(raw)
    except Exception:
        return None
    return [tuple(str(x).strip() for x in it)
            for it in arr if isinstance(it, (list, tuple)) and len(it) == 4]


def f1_of(rows):
    S = P = G = 0
    for r in rows:
        g = gold_quads(r.get("_gold"))
        if g is None:
            continue
        p = [qkey(q) for q in r["quads"]]
        cg, cp = Counter(g), Counter(p)
        S += sum((cg & cp).values())
        G += sum(cg.values())
        P += sum(cp.values())
    prec = S / P if P else 0.0
    rec = S / G if G else 0.0
    return (2 * prec * rec / (prec + rec)) if prec + rec else 0.0, prec, rec, S, G, P


RULES = {
    "pol_conflict": lambda q: bool(q.get("_pol_conflict")),
    "cat_other": lambda q: q["c"] == "其他",
    "cat_size": lambda q: q["c"] == "尺寸",
}


def apply_rule(rows, name):
    pred = RULES[name]
    out = []
    for r in rows:
        out.append(dict(r, quads=[q for q in r["quads"] if not pred(q)]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--rules", nargs="*", default=["pol_conflict", "cat_other", "cat_size"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = load(args.inp)
    has_gold = any(r.get("_gold") is not None for r in rows)
    base_f1 = None
    if has_gold:
        base_f1 = f1_of(rows)[0]
        print(f"base F1={base_f1:.4f} quads={sum(len(r['quads']) for r in rows)}")

    for name in args.rules:
        if name not in RULES:
            raise SystemExit(f"unknown rule: {name}")
        removed = total_hit = 0
        # 单条规则:统计候选删除的四元组有多少是真阳性
        for r in rows:
            g = gold_quads(r.get("_gold"))
            if g is None:
                continue
            gset = Counter(g)
            for q in r["quads"]:
                if RULES[name](q):
                    removed += 1
                    if gset[qkey(q)] > 0:
                        total_hit += 1
                        gset[qkey(q)] -= 1
        tdr = total_hit / removed if removed else 0.0
        line = f"{name}: 候选 {removed} 条, 其中真阳性 {total_hit}, t/d={tdr:.4f}"
        if has_gold:
            after = f1_of(apply_rule(rows, name))[0]
            line += f"  -> F1={after:.4f} delta={after - base_f1:+.4f}"
            line += f"  判据 t/d<{base_f1 / 2:.4f} " + ("PASS" if tdr < base_f1 / 2 else "FAIL")
        print(line)

    # 全部勾选规则叠加
    cur = rows
    for name in args.rules:
        cur = apply_rule(cur, name)
    if has_gold:
        f1, p, r_, S, G, P = f1_of(cur)
        print(f"combined F1={f1:.4f} (P={p:.4f} R={r_:.4f} S={S}/{G} pred={P}) "
              f"delta={f1 - base_f1:+.4f}")

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        for r in cur:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
