# -*- coding: utf-8 -*-
"""P1: 多成员 QLoRA 预测融合(冠军策略)。

规则(按顺序):
  1. 全四元组投票:(a,o,c,p) 在 >= min_votes 个成员中出现 -> 保留
  2. 空集回退:该 id 投票后为空 -> 用 fallback 成员的原始预测(majority2_best_if_empty)
  3. 精确评论覆盖:测试评论文本与训练集原文逐字相同 -> 用训练金标替换(融合之后只做一次)

用法:
  # 测试集融合
  python -I src/postprocess/fuse.py --pred a.jsonl b.jsonl c.jsonl --fallback a.jsonl \
      --override data/processed/quads.jsonl --out data/llm/fused.jsonl
  # dev 融合自评(预测文件带 _gold,fuse 原样透传)
  python -I src/postprocess/fuse.py --pred dev_a.jsonl dev_b.jsonl dev_c.jsonl \
      --fallback dev_a.jsonl --out data/llm/dev_fused.jsonl
  python -I src/llm/eval_dev.py --pred data/llm/dev_fused.jsonl
"""
import argparse
import json
from collections import Counter, defaultdict

PLACEHOLDER = "_"


def load(path):
    out = {}
    order = []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        rid = row["id"]
        out[rid] = row
        order.append(rid)
    return out, order


def quad_key(q):
    return (q["a"], q["o"], q["c"], q["p"])


def quad_sort_key(q):
    a_pos = q["a_start"] if q.get("a_start", -1) >= 0 else q.get("o_start", 0)
    return (a_pos, q.get("o_start", 0))


def load_override(path):
    """训练集原文 -> 金标四元组(逐字相同才覆盖)。"""
    out = {}
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        out[row["text"]] = row["quads"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", nargs="+", required=True)
    ap.add_argument("--fallback", default=None, help="空集回退的成员预测文件")
    ap.add_argument("--min-votes", type=int, default=2)
    ap.add_argument("--override", default=None, help="训练集 quads.jsonl,做精确评论覆盖")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    simple = []
    first_order = None
    for p in args.pred:
        data, order = load(p)
        simple.append((p, data))
        if first_order is None:
            first_order = order

    fb = None
    if args.fallback:
        fb, _ = load(args.fallback)
    ov = load_override(args.override) if args.override else None

    fused_rows = []
    stats = Counter()
    votes_all = Counter()

    for rid in first_order:
        votes = Counter()
        pol_of = defaultdict(set)
        first_seen = {}
        text = gold = None
        for name, data in simple:
            row = data.get(rid)
            if row is None:
                continue
            if text is None:
                text = row.get("text")
            if gold is None and row.get("_gold") is not None:
                gold = row.get("_gold")
            for q in row.get("quads") or []:
                k = quad_key(q)
                votes[k] += 1
                pol_of[(q["a"], q["o"], q["c"])].add(q["p"])
                first_seen.setdefault(k, q)

        kept = [first_seen[k] for k, c in votes.items() if c >= args.min_votes]
        stats["kept_votes"] += len(kept)
        stats["dropped_low_vote"] += sum(1 for c in votes.values() if c < args.min_votes)
        for c in votes.values():
            votes_all[c] += 1

        if not kept and fb and rid in fb:
            kept = list(fb[rid].get("quads") or [])
            if kept:
                stats["empty_fallback"] += 1
        if not kept:
            stats["empty_final"] += 1

        overridden = ov is not None and text is not None and text in ov
        if overridden:
            # 覆盖来自训练金标:给足票数,避免被下游 min_votes 过滤
            kept = [dict(q, _votes=99, _pol_conflict=False) for q in ov[text]]
            stats["exact_override"] += 1
        else:
            kept = [dict(q, _votes=votes.get(quad_key(q), 1),
                         _pol_conflict=len(pol_of.get((q["a"], q["o"], q["c"]), ())) > 1)
                    for q in kept]
        kept.sort(key=quad_sort_key)
        fused_rows.append({"id": rid, "text": text, "quads": kept, "_gold": gold})

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        for r in fused_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"members={len(simple)} ids={len(fused_rows)} min_votes={args.min_votes}")
    print("vote hist (count -> #quads): " +
          ", ".join(f"{k}->{votes_all[k]}" for k in sorted(votes_all)))
    print("stats: " + ", ".join(f"{k}={stats[k]}" for k in sorted(stats)))
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
