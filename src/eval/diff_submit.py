# -*- coding: utf-8 -*-
"""两版提交文件的逐行差异摘要(提交台账用)。

用法:
  python -I src/eval/diff_submit.py --old submissions/Result_p0.csv --new submissions/Result_llm.csv
输出:新增行数、删除行数、受影响 id 数、抽样若干条
"""
import argparse
import csv
from collections import Counter, defaultdict


def load(path):
    by_id = defaultdict(list)
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f):
            if len(row) != 5:
                continue
            by_id[int(row[0])].append(tuple(row[1:]))
    return by_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--show", type=int, default=5)
    args = ap.parse_args()

    old, new = load(args.old), load(args.new)
    ids = sorted(set(old) | set(new))
    added = removed = 0
    touched = []
    added_ids = set()
    removed_ids = set()
    for rid in ids:
        co, cn = Counter(old.get(rid, [])), Counter(new.get(rid, []))
        a = sum((cn - co).values())
        d = sum((co - cn).values())
        if a or d:
            touched.append((rid, a, d))
            added += a
            removed += d
            if a:
                added_ids.add(rid)
            if d:
                removed_ids.add(rid)

    print(f"old rows={sum(len(v) for v in old.values())} ids={len(old)}")
    print(f"new rows={sum(len(v) for v in new.values())} ids={len(new)}")
    print(f"added={added} removed={removed} net={added - removed:+d}")
    print(f"ids touched={len(touched)} (added in {len(added_ids)}, removed in {len(removed_ids)})")

    for rid, a, d in touched[:args.show]:
        print(f"  id={rid} +{a} -{d}")
        for q in list(Counter(new.get(rid, [])) - Counter(old.get(rid, [])))[:3]:
            print(f"    + {q}")
        for q in list(Counter(old.get(rid, [])) - Counter(new.get(rid, [])))[:3]:
            print(f"    - {q}")


if __name__ == "__main__":
    main()
