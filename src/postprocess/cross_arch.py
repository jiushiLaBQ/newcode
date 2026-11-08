# -*- coding: utf-8 -*-
"""P2: 跨架构后处理(LLM 融合 × MacBERT 抽取式),冠军两条规则 + 判据自检。

输入 fused_all.jsonl:fuse.py --min-votes 1 的输出,每个四元组带 _votes。

规则(顺序固定,各自可单独开关以便小步提交):
  1) 基线:保留 votes >= 2
  2) 跨架构否决:votes == 2 且 MacBERT 在该 id 上零产出 -> 删
  3) 跨架构确认:votes == 1 且 MacBERT 在该 id 上产出完全相同的四元组 -> 加
  4) 空集回退:该 id 过滤后为空 -> 用 fallback 成员的原始预测

判据(冠军方法论):删 d 行含 t 命中,F1 提升充要 t/d < f/2(f = 当前 F1);新增需 t/d > f/2。
dev 模式(--gold-from _gold)会打印每条规则的 t/d 与 ΔF1;不达标不该上线。

用法:
  # dev 自检(融合文件带 _gold)
  python -I src/postprocess/cross_arch.py --fused data/llm/dev_fused_all.jsonl \
      --macbert data/processed/oof_2seed_pf_sb.jsonl --out data/llm/dev_cross.jsonl
  # 测试集
  python -I src/postprocess/cross_arch.py --fused data/llm/fused_all.jsonl \
      --macbert data/processed/pred_p0_final.jsonl --out data/llm/test_cross.jsonl
"""
import argparse
import json
from collections import Counter, defaultdict


def qkey(q):
    return (q["a"], q["o"], q["c"], q["p"])


def qsort(q):
    a_pos = q.get("a_start", -1)
    if a_pos < 0:
        a_pos = q.get("o_start", 0)
    return (a_pos, q.get("o_start", 0))


def load_jsonl(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        if line.strip():
            rows.append(json.loads(line))
    return rows


def gold_quads(raw):
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


def f1_of(rows, key_quads):
    S = P = G = 0
    for r in rows:
        g = gold_quads(r.get("_gold"))
        if g is None:
            continue
        p = key_quads(r)
        cg, cp = Counter(g), Counter(p)
        S += sum((cg & cp).values())
        G += sum(cg.values())
        P += sum(cp.values())
    prec = S / P if P else 0.0
    rec = S / G if G else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"S": S, "P": P, "G": G, "precision": prec, "recall": rec, "f1": f1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fused", required=True, help="fuse.py --min-votes 1 的输出")
    ap.add_argument("--macbert", required=True, help="MacBERT 预测 jsonl(同 id 空间)")
    ap.add_argument("--fallback", default=None, help="空集回退的成员预测 jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-votes", type=int, default=2)
    ap.add_argument("--no-veto", action="store_true")
    ap.add_argument("--no-confirm", action="store_true")
    args = ap.parse_args()

    fused = load_jsonl(args.fused)
    mb, mb_by_text = {}, {}
    for r in load_jsonl(args.macbert):
        s = set(qkey(q) for q in (r.get("quads") or []))
        mb[str(r["id"])] = s
        if r.get("text"):
            mb_by_text[r["text"]] = s
    fb = {}
    if args.fallback:
        for r in load_jsonl(args.fallback):
            q = r.get("quads") or []
            fb[str(r["id"])] = q
            if r.get("text"):
                fb.setdefault("__text__" + r["text"], q)

    base_rows, out_rows = [], []
    veto = Counter()
    confirm = Counter()
    fallback_n = 0

    for r in fused:
        rid = str(r["id"])
        text = r.get("text")
        quads = list(r.get("quads") or [])
        # 测试集 id 对齐 MacBERT;dev 预测 id 是行号,退化为按原文匹配
        mb_set = mb.get(rid)
        if mb_set is None:
            mb_set = mb_by_text.get(text, set())
        mb_empty = len(mb_set) == 0

        kept = []
        for q in quads:
            v = int(q.get("_votes", 1))
            if v < args.min_votes:
                continue
            if not args.no_veto and v == args.min_votes and mb_empty:
                veto["rows"] += 1
                continue
            kept.append(q)
        base_rows.append(dict(r, quads=[q for q in quads if int(q.get("_votes", 1)) >= args.min_votes]))

        # 跨架构确认:单票且 MacBERT 完全一致
        if not args.no_confirm:
            have = set(qkey(q) for q in kept)
            for q in quads:
                if int(q.get("_votes", 1)) != 1:
                    continue
                if qkey(q) in mb_set and qkey(q) not in have:
                    kept.append(q)
                    have.add(qkey(q))
                    confirm["rows"] += 1

        fbq = fb.get(rid) or fb.get("__text__" + (text or ""))
        if not kept and fbq:
            kept = list(fbq)
            fallback_n += 1

        kept = sorted(kept, key=qsort)
        out_rows.append({"id": r["id"], "text": r.get("text"),
                         "quads": kept, "_gold": r.get("_gold")})

    print(f"ids={len(out_rows)} min_votes={args.min_votes} "
          f"vetoed={veto['rows']} confirmed={confirm['rows']} fallback={fallback_n}")

    if any(r.get("_gold") is not None for r in out_rows):
        b = f1_of(base_rows, lambda r: [qkey(q) for q in r["quads"]])
        a = f1_of(out_rows, lambda r: [qkey(q) for q in r["quads"]])
        print(f"base   F1={b['f1']:.4f} (P={b['precision']:.4f} R={b['recall']:.4f} "
              f"S={b['S']}/{b['G']} pred={b['P']})")
        print(f"after  F1={a['f1']:.4f} (P={a['precision']:.4f} R={a['recall']:.4f} "
              f"S={a['S']}/{a['G']} pred={a['P']})  delta={a['f1'] - b['f1']:+.4f}")
        print(f"判据: 删 {veto['rows']} 行需 t/d < {b['f1'] / 2:.4f};"
              f" 增 {confirm['rows']} 行需 t/d > {b['f1'] / 2:.4f}")

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
