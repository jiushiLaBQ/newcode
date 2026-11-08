# -*- coding: utf-8 -*-
"""跨架构确认规则诊断:统计单票四元组与 MacBERT 的各种一致度。

用法:
  python -I src/postprocess/diag_cross.py --fused data/llm/dev_fused_all.jsonl \
      --macbert data/processed/oof_2seed_pf_sb.jsonl
"""
import argparse
import json


def qkey(q):
    return (q["a"], q["o"], q["c"], q["p"])


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fused", required=True)
    ap.add_argument("--macbert", required=True)
    args = ap.parse_args()

    mb_by_id, mb_by_text = {}, {}
    for r in load(args.macbert):
        s = set(qkey(q) for q in (r.get("quads") or []))
        mb_by_id[str(r["id"])] = s
        if r.get("text"):
            mb_by_text[r["text"]] = s

    n1 = 0
    full = ao_c = ao_p = ao = 0
    no_match_text = 0
    mb_empty_ids = 0
    examples = []

    for r in load(args.fused):
        rid, text = str(r["id"]), r.get("text")
        s = mb_by_id.get(rid)
        if s is None:
            s = mb_by_text.get(text)
            if s is None:
                no_match_text += 1
                s = set()
        if not s:
            mb_empty_ids += 1
            continue
        mb_ao = set((k[0], k[1]) for k in s)
        mb_aoc = set((k[0], k[1], k[2]) for k in s)
        for q in r.get("quads") or []:
            if int(q.get("_votes", 1)) != 1:
                continue
            n1 += 1
            k = qkey(q)
            if k in s:
                full += 1
            elif k in mb_aoc:
                ao_c += 1
                if len(examples) < 5:
                    examples.append(("c/p differs", k))
            elif (k[0], k[1]) in mb_ao:
                ao_p += 1
                if len(examples) < 5:
                    examples.append(("a,o same", k))
            else:
                ao += 1
                if len(examples) < 5:
                    examples.append(("no a,o", k))

    print(f"单票四元组={n1}  MacBERT全空的id={mb_empty_ids}  文本未匹配={no_match_text}")
    print(f"四字段全同={full}  (a,o,c)同p异={ao_c}  (a,o)同c异={ao_p}  全不同={ao}")
    for tag, k in examples:
        print(f"  ex {tag}: {k}")


if __name__ == "__main__":
    main()
