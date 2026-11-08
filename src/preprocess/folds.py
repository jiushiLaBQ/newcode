# -*- coding: utf-8 -*-
"""按 id 分层 5 折切分(防泄漏),输出 data/processed/folds.json。

防泄漏要点:
  - 绝不按样本行切(同句多四元组必须同折)。
  - 归一化后文本完全相同的评论(模板化短评近重复)整组进同一折。
  - 分层键 = (四元组数桶, 是否含负面) —— 让各折的难度与极性分布近似。
确定性: random.Random(502)(与 OpinioNet 同 seed,方便对照)。
"""
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QUADS = ROOT / "data" / "processed" / "quads.jsonl"
OUT = ROOT / "data" / "processed" / "folds.json"
SEED = 502
N_FOLDS = 5


def normalize(text):
    return re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE).lower()


def main():
    rows = []
    with open(QUADS, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    groups = defaultdict(list)  # 归一化文本 -> [id,...]
    for r in rows:
        groups[normalize(r["text"])].append(r["id"])
    strata = defaultdict(list)  # 分层键 -> [group,...]
    by_id = {r["id"]: r for r in rows}
    for key, ids in groups.items():
        nq = min(len(by_id[ids[0]]["quads"]), 4)
        has_neg = any(q["p"] == "负面" for i in ids for q in by_id[i]["quads"])
        strata[(nq, has_neg)].append(sorted(ids))

    rng = random.Random(SEED)
    folds = [[] for _ in range(N_FOLDS)]
    for key in sorted(strata):
        g = strata[key]
        rng.shuffle(g)
        for k, ids in enumerate(g):
            folds[k % N_FOLDS].extend(ids)
    for f in folds:
        f.sort()

    overlap = set()
    for f in folds:
        for i in f:
            if i in overlap:
                print(f"[fatal] id={i} 出现在多折!", file=sys.stderr)
                sys.exit(1)
            overlap.add(i)
    total = sum(len(f) for f in folds)
    if total != len(rows):
        print(f"[fatal] 折内 id 总数 {total} != 评论数 {len(rows)}", file=sys.stderr)
        sys.exit(1)

    with open(OUT, "w", encoding="utf-8", newline="") as f:
        json.dump({"seed": SEED, "n_folds": N_FOLDS,
                   "folds": folds,
                   "near_dup_groups": sum(1 for v in groups.values() if len(v) > 1)}, f, ensure_ascii=False)
    print(f"folds.json: {N_FOLDS} 折, 各折大小 {[len(f) for f in folds]}, "
          f"近重复组 {sum(1 for v in groups.values() if len(v) > 1)}")


if __name__ == "__main__":
    main()
