# -*- coding: utf-8 -*-
"""融合结果 -> 提交 CSV(id 升序、无表头、UTF-8 无 BOM、\\n 换行、空预测写一行全 "_")。

用法:
  python -I src/llm/to_submit.py --fused data/llm/fused.jsonl \
      --ids-from data/llm/test_prompts.jsonl --out submissions/Result_llm.csv
"""
import argparse
import csv
import json
from pathlib import Path

PLACEHOLDER = "_"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fused", required=True)
    ap.add_argument("--ids-from", default="data/llm/test_prompts.jsonl")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ids = []
    for line in open(args.ids_from, encoding="utf-8"):
        if line.strip():
            ids.append(int(json.loads(line)["id"]))
    ids = sorted(ids)

    preds = {}
    for line in open(args.fused, encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        preds[int(row["id"])] = row.get("quads") or []

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    missing = 0
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        for rid in ids:
            quads = preds.get(rid)
            if quads is None:
                missing += 1
            if not quads:
                w.writerow([rid, PLACEHOLDER, PLACEHOLDER, PLACEHOLDER, PLACEHOLDER])
                n_rows += 1
                continue
            for q in quads:
                w.writerow([rid, q["a"], q["o"], q["c"], q["p"]])
                n_rows += 1
    print(f"written {out}: ids={len(ids)} rows={n_rows} missing_ids={missing}")


if __name__ == "__main__":
    main()
