# -*- coding: utf-8 -*-
"""精确评论标签覆盖(exact override)——冠军方案的确定性组件,融合后单次应用。

机制:测试评论与某条训练评论**原文逐字相同**时,直接用训练金标整组替换该 id 的预测
(多重集语义,整组替换、不与原预测合并);未匹配的 id 不动。同一文本的金标必然相同,
这是确定性收益:不引入任何参数,不依赖模型输出。

多条训练匹配(同一文本在训练集出现多次)时:取各条训练金标(四元组多重集)的**多数版本**
——出现次数最多的那个版本,四元组(按评测同一性 (a,o,c,p),含多重性)完全一致才算同版本;
并列时取四元组更多者(期望命中更高:多出的四元组后验命中率 0.5 > F1/2≈0.365),
再并列取训练 id 最小者。多版本不一致即为冲突,连同各版本得票写入 <out 去后缀>_conflicts.json。

偏移语义:替换进来的训练金标偏移基于训练原文;原文逐字相同故偏移同样有效
(text[start:end]==词面,隐式 "_" 偏移恒为 -1)。--nfkc 下若两文本并非逐字相同,
替换前逐条校验 text[start:end]==词面,不符则回贴到测试文本中最左出现位置;
仍找不到则保留原偏移并记警告。

用法:
  PYTHONUTF8=1 python -I src/postprocess/exact_override.py \\
      --pred data/processed/pred_test_10v4_final.jsonl \\
      --gold data/processed/quads.jsonl \\
      --test data/processed/test_reviews.jsonl \\
      --out data/experiments/exact_override/pred_test_10v4_final_override.jsonl [--nfkc]
"""
import argparse
import copy
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

PLACEHOLDER = "_"


def load_jsonl(path):
    """读取 {"id","text","quads"} jsonl,保持行序;返回 (rows, {id: row})。"""
    rows, by_id = [], {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append(row)
            by_id[row["id"]] = row
    return rows, by_id


def norm_key(text, nfkc):
    return unicodedata.normalize("NFKC", text) if nfkc else text


def quad_eval_key(q):
    """评测同一性(a,o,c,p);偏移是渲染细节,不参与版本比较。"""
    return (q["a"], q["o"], q["c"], q["p"])


def version_key(quads):
    """四元组多重集的规范表示(排序后含重复),完全一致才算同版本。"""
    return tuple(sorted(quad_eval_key(q) for q in quads))


def pick_version(cand_rows):
    """多条训练匹配 -> (代表行, 版本得票 Counter, 是否冲突)。

    多数版本:票数最多;并列取四元组更多者;再并列取训练 id 最小者。
    """
    groups = defaultdict(list)  # version_key -> [row, ...]
    for row in cand_rows:
        groups[version_key(row["quads"])].append(row)
    votes = {vk: len(rs) for vk, rs in groups.items()}
    conflict = len(groups) > 1
    best = sorted(
        groups.items(),
        key=lambda kv: (-len(kv[1]), -len(kv[0]), min(r["id"] for r in kv[1])),
    )[0][1]
    representative = min(best, key=lambda r: r["id"])
    return representative, Counter(votes), conflict


def reanchor(quads, text, sid, warnings):
    """保证 text[start:end]==词面;不符回贴最左出现位置,找不到记警告。"""
    for q in quads:
        for term_k, s_k, e_k in (("a", "a_start", "a_end"), ("o", "o_start", "o_end")):
            term = q[term_k]
            if term == PLACEHOLDER:
                q[s_k], q[e_k] = -1, -1
                continue
            s, e = q[s_k], q[e_k]
            if 0 <= s and e <= len(text) and text[s:e] == term:
                continue
            idx = text.find(term)
            if idx >= 0:
                q[s_k], q[e_k] = idx, idx + len(term)
                warnings.append(
                    {"id": sid, "term": term, "old": [s, e], "new": [idx, idx + len(term)],
                     "reason": "offset_reanchor"}
                )
            else:
                warnings.append(
                    {"id": sid, "term": term, "old": [s, e], "reason": "term_not_in_test_text"}
                )
    return quads


def build_override_map(train_rows, test_rows, nfkc):
    """test id -> 替换用训练金标(整组);同时产出冲突记录与匹配清单。"""
    index = defaultdict(list)  # norm text -> [train row, ...]
    for row in train_rows:
        index[norm_key(row["text"], nfkc)].append(row)

    override_map, matched, conflicts = {}, [], []
    for trow in test_rows:
        cands = index.get(norm_key(trow["text"], nfkc))
        if not cands:
            continue
        representative, votes, conflict = pick_version(cands)
        entry = {
            "quads": copy.deepcopy(representative["quads"]),
            "train_ids": sorted(r["id"] for r in cands),
            "rep_train_id": representative["id"],
        }
        override_map[trow["id"]] = entry
        matched.append({"test_id": trow["id"], "train_ids": entry["train_ids"],
                        "rep_train_id": entry["rep_train_id"],
                        "n_quads": len(entry["quads"]), "conflict": conflict})
        if conflict:
            conflicts.append({
                "test_id": trow["id"],
                "train_ids": entry["train_ids"],
                "chosen_rep_train_id": entry["rep_train_id"],
                "chosen_version": [list(k) for k in sorted(version_key(representative["quads"]))],
                "votes": [
                    {"version": [list(k) for k in sorted(vk)], "count": int(n),
                     "train_ids": sorted(r["id"] for r in
                                         (x for x in cands if version_key(x["quads"]) == vk))}
                    for vk, n in sorted(votes.items(), key=lambda kv: (-kv[1], -len(kv[0])))
                ],
                "tie_break": "count -> larger_version -> smaller_train_id",
            })
    return override_map, matched, conflicts


def apply_override(pred_rows, override_map, test_by_id, warnings):
    """整组替换命中的 id;返回 (输出行, 统计)。未匹配的 id 原样保留。"""
    out_rows = []
    replaced_ids, quads_before, quads_after = [], 0, 0
    for row in pred_rows:
        sid = row["id"]
        new = {"id": sid, "text": row["text"], "quads": row["quads"]}
        if sid in override_map:
            gold_quads = copy.deepcopy(override_map[sid]["quads"])
            text = test_by_id[sid]["text"] if sid in test_by_id else row["text"]
            new["quads"] = reanchor(gold_quads, text, sid, warnings)
            replaced_ids.append(sid)
            quads_before += len(row["quads"])
            quads_after += len(new["quads"])
        out_rows.append(new)
    stats = {
        "n_pred": len(pred_rows),
        "n_replaced": len(replaced_ids),
        "replaced_ids": replaced_ids,
        "quads_before_on_replaced": quads_before,
        "quads_after_on_replaced": quads_after,
    }
    return out_rows, stats


def run_override(pred_path, gold_path, test_path, out_path, nfkc=False):
    """主流程;返回统计 dict(便于自测与批跑)。"""
    train_rows, _ = load_jsonl(gold_path)
    test_rows, test_by_id = load_jsonl(test_path)
    pred_rows, _ = load_jsonl(pred_path)

    override_map, matched, conflicts = build_override_map(train_rows, test_rows, nfkc)
    warnings = []
    out_rows, stats = apply_override(pred_rows, override_map, test_by_id, warnings)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    sidecar = out_path.with_name(out_path.stem + "_conflicts.json")
    with open(sidecar, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"nfkc": bool(nfkc), "matched": matched, "conflicts": conflicts,
                   "warnings": warnings}, f, ensure_ascii=False, indent=2)

    stats.update({
        "nfkc": bool(nfkc),
        "n_matched_test_ids": len(matched),
        "n_conflicts": len(conflicts),
        "n_warnings": len(warnings),
        "matched": matched,
        "conflicts_file": str(sidecar),
    })
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description="精确评论标签覆盖:同文本训练金标整组替换预测")
    ap.add_argument("--pred", required=True, help="待覆盖的预测 jsonl(与 quads.jsonl 同构)")
    ap.add_argument("--gold", required=True, help="训练金标 quads.jsonl")
    ap.add_argument("--test", required=True, help="测试评论 test_reviews.jsonl")
    ap.add_argument("--out", required=True, help="输出 jsonl(结构同输入)")
    ap.add_argument("--nfkc", action="store_true", help="按 NFKC 归一化后的文本比对")
    args = ap.parse_args(argv)

    stats = run_override(args.pred, args.gold, args.test, args.out, nfkc=args.nfkc)
    print(f"[exact_override] nfkc={stats['nfkc']}")
    print(f"  测试集匹配训练原文: {stats['n_matched_test_ids']} 条")
    print(f"  实际替换(在预测文件中): {stats['n_replaced']} 条 id")
    print(f"  替换处四元组: {stats['quads_before_on_replaced']} -> {stats['quads_after_on_replaced']}")
    print(f"  多版本冲突: {stats['n_conflicts']} 条;偏移回贴/警告: {stats['n_warnings']} 条")
    print(f"  输出: {args.out}")
    print(f"  冲突明细: {stats['conflicts_file']}")
    return stats


if __name__ == "__main__":
    main()
