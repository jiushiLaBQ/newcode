# -*- coding: utf-8 -*-
"""用训练好的折权重对测试集推理,产出 quads.jsonl 结构预测。

用法: python -I src/train/predict_test.py --checkpoint checkpoints/a3/fold0.pt --out data/processed/pred_test_fold0.jsonl
"""
import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.preprocess.dataset import QuadDataset, load_records
from src.models.tagger import QuadTagger
from src.decode.pipeline import predict, write_records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hfl/chinese-macbert-base")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test", default=str(ROOT / "data" / "processed" / "test_reviews.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--pair-mode", default="nearest", choices=["nearest", "greedy"])
    ap.add_argument("--pair-threshold", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = QuadTagger(args.model).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    records = load_records(args.test)
    ds = QuadDataset(records, tokenizer, 128)
    preds = predict(model, ds, device, pair_mode=args.pair_mode,
                    threshold=args.pair_threshold, batch_size=args.batch_size,
                    pad_id=tokenizer.pad_token_id or 0)
    write_records(args.out, preds)
    nq = sum(len(r["quads"]) for r in preds)
    print(f"written {args.out}: {len(preds)} ids, {nq} quads")


if __name__ == "__main__":
    main()
