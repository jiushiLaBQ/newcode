# -*- coding: utf-8 -*-
"""多预测融合 CLI:vote(跨文件一致才收)/ union(多重集并集)。

用法:
  python -I src/postprocess/ensemble.py --preds f1.jsonl f2.jsonl \
      --out merged.jsonl --mode vote --min-votes 2

语义(四元组同一性按 (a,o,c,p) 判定;输入/输出均与 quads.jsonl 同构):
  vote : 完全相同四元组在 >= min-votes 个不同文件中出现才收,收 1 次(不保留多重性)。
         同一文件内重复出现只算 1 票。
  union: 多重集并集——每个四元组保留次数 = 各文件出现次数的最大值。
         保留多重性:单文件内真重复(如 id=1375)不丢;两份相同预测合并幂等。
输出:{"id","text","quads":[...]},id 升序;任一输入出现过的 id 都输出(quads 可为空)。
quad 保留首次出现处的完整 dict,偏移按最终词面校正(输入偏移有效则保留,否则取最左)。
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.postprocess.rules import PLACEHOLDER, _resolve_offsets


def _quad_key(q):
    """四元组同一性键:(a,o,c,p)。"""
    return (q.get("a", PLACEHOLDER), q.get("o", PLACEHOLDER),
            q.get("c", PLACEHOLDER), q.get("p", PLACEHOLDER))


def load_pred_jsonl(path):
    """读一份预测 -> {id: {"text": str, "quads": [quad dict, ...]}}(保持文件内顺序)"""
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rid = int(row["id"])
            out[rid] = {"text": row.get("text", ""), "quads": list(row.get("quads", []))}
    return out


def merge(preds, mode="vote", min_votes=2):
    """融合多份预测。

    preds: [{id: {"text","quads"}}, ...],按给定顺序视为 文件1..文件N
    mode="vote" : 保留出现在 >=min-votes 个文件中的四元组,各收 1 次
    mode="union": 多重集并集,保留次数 = 各文件出现次数的最大值
    返回 {id: {"id","text","quads"}},id 升序。
    """
    if mode not in ("vote", "union"):
        raise ValueError(f"mode 必须是 vote/union,得到 {mode!r}")
    ids = sorted({rid for pr in preds for rid in pr})
    out = {}
    for rid in ids:
        lists = [pr[rid]["quads"] for pr in preds if rid in pr]
        text = ""
        for pr in preds:
            if rid in pr and pr[rid].get("text"):
                text = pr[rid]["text"]
                break

        # 首次出现顺序 + 首个完整 dict(偏移按最终词面校正)
        order, first = [], {}
        for lst in lists:
            for q in lst:
                key = _quad_key(q)
                if key not in first:
                    order.append(key)
                    first[key] = q

        if mode == "vote":
            votes = Counter()
            for lst in lists:          # 每文件内重复只算 1 票
                seen = set()
                for q in lst:
                    key = _quad_key(q)
                    if key not in seen:
                        seen.add(key)
                        votes[key] += 1
            chosen_keys = [k for k in order if votes[k] >= min_votes]
            counts = {k: 1 for k in chosen_keys}
        else:  # union: 多重集并集 = 逐键取各文件出现次数的最大值
            per_file = []
            for lst in lists:
                per_file.append(Counter(_quad_key(q) for q in lst))
            counts = {k: max(cnt.get(k, 0) for cnt in per_file) for k in order}
            chosen_keys = [k for k in order if counts[k] > 0]

        quads = []
        for k in chosen_keys:
            q = first[k]
            a, o = q.get("a", PLACEHOLDER), q.get("o", PLACEHOLDER)
            a_s, a_e = _resolve_offsets(a, text, q.get("a_start"), q.get("a_end"))
            o_s, o_e = _resolve_offsets(o, text, q.get("o_start"), q.get("o_end"))
            rec = {"a": a, "a_start": a_s, "a_end": a_e,
                   "o": o, "o_start": o_s, "o_end": o_e,
                   "c": q.get("c", PLACEHOLDER), "p": q.get("p", PLACEHOLDER)}
            quads.extend([rec] * counts[k])
        out[rid] = {"id": rid, "text": text, "quads": quads}
    return out


def write_jsonl(path, merged):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for rid in sorted(merged):
            f.write(json.dumps(merged[rid], ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description="多预测融合(vote/union)")
    ap.add_argument("--preds", nargs="+", required=True, help="若干预测 jsonl(同 quads.jsonl 格式)")
    ap.add_argument("--out", required=True, help="融合输出路径")
    ap.add_argument("--mode", choices=["vote", "union"], default="vote")
    ap.add_argument("--min-votes", type=int, default=2, help="vote 模式下最少出现文件数")
    args = ap.parse_args()

    preds = [load_pred_jsonl(p) for p in args.preds]
    merged = merge(preds, mode=args.mode, min_votes=args.min_votes)
    write_jsonl(args.out, merged)
    n_quads = sum(len(v["quads"]) for v in merged.values())
    print(f"[ensemble] mode={args.mode} min-votes={args.min_votes} "
          f"输入 {len(args.preds)} 份 -> {len(merged)} 条评论 / {n_quads} 个四元组 -> {args.out}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
