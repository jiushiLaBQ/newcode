# -*- coding: utf-8 -*-
"""预测记录 -> 提交文件 Result.csv。

规则(TEST README):
  列序 id,AspectTerm,OpinionTerm,Category,Polarity;无表头;UTF-8 无 BOM;\\n 换行;
  id 升序且与 Test_reviews.csv 全等——某 id 无预测也保留一行全 "_"。

用法: python -I src/eval/make_submission.py --pred data/processed/pred_rules.jsonl --out submissions/Result.csv
"""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLACEHOLDER = "_"


def load_records(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="quads.jsonl 结构的预测文件")
    ap.add_argument("--test", default=str(ROOT / "data" / "test_raw" / "TEST" / "Test_reviews.csv"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import csv as _csv
    test_ids = []
    with open(args.test, "r", encoding="utf-8-sig", newline="") as f:
        for row in _csv.DictReader(f):
            test_ids.append(int(row["id"]))
    test_ids = sorted(test_ids)

    preds = {r["id"]: r["quads"] for r in load_records(args.pred)}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        for rid in test_ids:
            quads = preds.get(rid) or []
            if not quads:
                w.writerow([rid, PLACEHOLDER, PLACEHOLDER, PLACEHOLDER, PLACEHOLDER])
                n_rows += 1
                continue
            for q in quads:
                w.writerow([rid, q["a"], q["o"], q["c"], q["p"]])
                n_rows += 1
    print(f"written {out}: {len(test_ids)} ids, {n_rows} rows")


if __name__ == "__main__":
    main()
