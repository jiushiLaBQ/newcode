# -*- coding: utf-8 -*-
"""P0: 校验训练标注字符偏移，输出清洗后的四元组与数据画像。

输入:
  data/train_raw/TRAIN/Train_reviews.csv   (id,Reviews)
  data/train_raw/TRAIN/Train_labels.csv    (id,AspectTerms,A_start,A_end,OpinionTerms,O_start,O_end,Categories,Polarities)
  data/test_raw/TEST/Test_reviews.csv      (id,Reviews)
输出:
  data/processed/quads.jsonl   每条评论一行: {"id","text","quads":[{"a","a_start","a_end","o","o_start","o_end","c","p"}]}
  data/processed/stats.json    数据画像
  data/processed/conflicts.txt 偏移与原文不一致的修复记录

约定:
  - 字符偏移为 Python 切片语义: text[start:end] == 词面(start,end 为字符下标)。
  - 隐式字段词面为 "_"，偏移记 -1。
"""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = ROOT / "data" / "train_raw" / "TRAIN"
TEST_DIR = ROOT / "data" / "test_raw" / "TEST"
OUT_DIR = ROOT / "data" / "processed"

PLACEHOLDER = "_"
CATEGORIES = ["包装", "成分", "尺寸", "服务", "功效", "价格", "气味", "使用体验", "物流", "新鲜度", "真伪", "整体", "其他"]
POLARITIES = ["正面", "中性", "负面"]


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def parse_pos(v):
    v = (v or "").strip()
    return -1 if v == "" else int(v)


def verify_span(text, term, start, end, kind, rid, conflicts):
    """返回 (start,end)。词面必须能在原文中逐字找到；偏移不符则修复并记录。"""
    if term == PLACEHOLDER:
        return -1, -1
    if not term:  # 空词面直接丢弃(text.find('')==0 会伪造 (0,0) span)
        conflicts.append(f"id={rid} {kind} 词面为空,该四元组将被丢弃")
        return None
    if 0 <= start < end <= len(text) and text[start:end] == term:
        return start, end
    pos = text.find(term)
    if pos < 0:
        conflicts.append(f"id={rid} {kind}={term!r} 在原文中找不到,该四元组将被丢弃")
        return None
    conflicts.append(
        f"id={rid} {kind}={term!r} 偏移({start},{end}) 切片={text[start:end] if 0 <= start < end else '<越界>'!r} 修复为({pos},{pos + len(term)})"
    )
    return pos, pos + len(term)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    conflicts = []

    reviews_rows = read_csv(TRAIN_DIR / "Train_reviews.csv")
    labels_rows = read_csv(TRAIN_DIR / "Train_labels.csv")
    test_rows = read_csv(TEST_DIR / "Test_reviews.csv")

    reviews = {int(r["id"]): r["Reviews"] for r in reviews_rows}
    quads_by_id = defaultdict(list)

    for r in labels_rows:
        rid = int(r["id"])
        if rid not in reviews:
            conflicts.append(f"id={rid} 出现在标注中但不在评论文件中,丢弃")
            continue
        text = reviews[rid]
        a, o = r["AspectTerms"].strip(), r["OpinionTerms"].strip()
        c, p = r["Categories"].strip(), r["Polarities"].strip()
        a_span = verify_span(text, a, parse_pos(r["A_start"]), parse_pos(r["A_end"]), "Aspect", rid, conflicts)
        o_span = verify_span(text, o, parse_pos(r["O_start"]), parse_pos(r["O_end"]), "Opinion", rid, conflicts)
        if a_span is None or o_span is None:
            continue
        if a == PLACEHOLDER and o == PLACEHOLDER:
            conflicts.append(f"id={rid} 双隐式四元组(训练分布中不应存在),丢弃")
            continue
        quads_by_id[rid].append({
            "a": a, "a_start": a_span[0], "a_end": a_span[1],
            "o": o, "o_start": o_span[0], "o_end": o_span[1],
            "c": c, "p": p,
        })

    # ---- 画像统计 ----
    n_quads = sum(len(v) for v in quads_by_id.values())
    per_id = {rid: len(qs) for rid, qs in quads_by_id.items()}
    imp_a = sum(1 for v in quads_by_id.values() for q in v if q["a"] == PLACEHOLDER)
    imp_o = sum(1 for v in quads_by_id.values() for q in v if q["o"] == PLACEHOLDER)
    cross_order = sum(1 for v in quads_by_id.values() for q in v
                      if q["a_start"] >= 0 and q["o_start"] >= 0 and q["o_end"] <= q["a_start"])
    pol_dist = Counter(q["p"] for v in quads_by_id.values() for q in v)
    cat_dist = Counter(q["c"] for v in quads_by_id.values() for q in v)
    dup_ids = []
    for rid, v in quads_by_id.items():
        cnt = Counter((q["a"], q["o"], q["c"], q["p"]) for q in v)
        if any(n > 1 for n in cnt.values()):
            dup_ids.append(rid)
    # 同通道 span 重叠(给 build_tags 用):同句内属性-属性 或 观点-观点 span 交叉且不等同
    overlap_aa = overlap_oo = 0
    for v in quads_by_id.values():
        for key, counter_name in (("a", "aa"), ("o", "oo")):
            spans = sorted({(q[f"{key}_start"], q[f"{key}_end"]) for q in v if q[f"{key}_start"] >= 0})
            bad = sum(1 for i in range(len(spans)) for j in range(i + 1, len(spans))
                      if spans[i][0] < spans[j][1] and spans[j][0] < spans[i][1])
            if counter_name == "aa":
                overlap_aa += bad
            else:
                overlap_oo += bad
    lens = [len(t) for t in reviews.values()]
    test_lens = [len(r["Reviews"]) for r in test_rows]

    stats = {
        "n_train_reviews": len(reviews),
        "n_quads": n_quads,
        "quads_per_id_mean": round(n_quads / max(len(quads_by_id), 1), 2),
        "quads_per_id_max": max(per_id.values()) if per_id else 0,
        "implicit_aspect_rows": imp_a,
        "implicit_aspect_ratio": round(imp_a / max(n_quads, 1), 4),
        "implicit_opinion_rows": imp_o,
        "implicit_opinion_ratio": round(imp_o / max(n_quads, 1), 4),
        "cross_order_pairs": cross_order,
        "polarity_dist": dict(pol_dist),
        "category_dist": dict(cat_dist),
        "ids_with_duplicate_identical_quads": dup_ids,
        "overlap_aspect_spans": overlap_aa,
        "overlap_opinion_spans": overlap_oo,
        "text_len": {"min": min(lens), "mean": round(sum(lens) / len(lens), 1), "max": max(lens)},
        "n_test_reviews": len(test_rows),
        "test_text_len": {"min": min(test_lens), "mean": round(sum(test_lens) / len(test_lens), 1), "max": max(test_lens)},
        "offset_repairs": sum(1 for c in conflicts if "修复" in c),
        "dropped_quads": sum(1 for c in conflicts if "丢弃" in c),
    }

    with open(OUT_DIR / "quads.jsonl", "w", encoding="utf-8", newline="") as f:
        for rid in sorted(quads_by_id):
            f.write(json.dumps({"id": rid, "text": reviews[rid], "quads": quads_by_id[rid]}, ensure_ascii=False) + "\n")
    with open(OUT_DIR / "stats.json", "w", encoding="utf-8", newline="") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    with open(OUT_DIR / "conflicts.txt", "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(conflicts) + ("\n" if conflicts else ""))

    # 测试集也落盘一份(无标签),供推理与提交校验用
    with open(OUT_DIR / "test_reviews.jsonl", "w", encoding="utf-8", newline="") as f:
        for r in sorted(test_rows, key=lambda x: int(x["id"])):
            f.write(json.dumps({"id": int(r["id"]), "text": r["Reviews"]}, ensure_ascii=False) + "\n")

    print(json.dumps(stats, ensure_ascii=False, indent=1))
    if conflicts:
        print(f"[warn] {len(conflicts)} 条冲突记录 -> {OUT_DIR / 'conflicts.txt'}", file=sys.stderr)


if __name__ == "__main__":
    main()
