# -*- coding: utf-8 -*-
"""A3 训练入口:5 折 CV,早停指标=官方四元组 F1(不是 token F1),产出 OOF 预测。

用法:
  python -I src/train/train_a3.py --model hfl/chinese-macbert-base --fp16
  python -I src/train/train_a3.py --smoke          # 32 条 1 epoch,验证管线贯通
产物:
  data/processed/oof_a3/oof.jsonl        全折拼接的 OOF 预测(同 quads.jsonl 结构)
  data/processed/oof_a3/report.json      逐折 F1 + 均值±方差
  checkpoints/a3/fold{k}.pt              各折最优权重
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.preprocess.dataset import QuadDataset, load_records, collate
from src.models.tagger import QuadTagger
from src.decode.pipeline import predict, records_to_pred_dict, write_records
from src.eval.scorer import score


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def gold_dict(records):
    return {r["id"]: [(q["a"], q["o"], q["c"], q["p"]) for q in r.get("quads", [])] for r in records}


def train_one_fold(fold, train_recs, val_recs, args, tokenizer, device):
    set_seed(args.seed + fold)
    pad_id = tokenizer.pad_token_id or 0
    train_ds = QuadDataset(train_recs, tokenizer, args.max_len)
    val_ds = QuadDataset(val_recs, tokenizer, args.max_len)
    from torch.utils.data import DataLoader
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=lambda b: collate(b, pad_id))

    model = QuadTagger(args.model).to(device)
    model.train()
    steps = max(1, (len(loader) + args.accum_steps - 1) // args.accum_steps)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(
        optim, int(steps * args.epochs * args.warmup_ratio), steps * args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16 and device.type == "cuda")

    best_f1, best_state, best_epoch = -1.0, None, -1
    for ep in range(args.epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        optim.zero_grad(set_to_none=True)
        for bi, batch in enumerate(loader):
            with torch.amp.autocast("cuda", enabled=args.fp16 and device.type == "cuda"):
                out = model(input_ids=batch["input_ids"].to(device),
                            attention_mask=batch["attention_mask"].to(device),
                            aspect_tag_ids=batch["aspect_tag_ids"].to(device),
                            opinion_tag_ids=batch["opinion_tag_ids"].to(device),
                            pairs=batch["pairs"])
                loss = out["loss"] / args.accum_steps
            scaler.scale(loss).backward()
            if (bi + 1) % args.accum_steps == 0 or (bi + 1) == len(loader):  # 尾批也要 step
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
                sched.step()
                optim.zero_grad(set_to_none=True)
            tot += float(out["loss"])
            n += 1

        # 验证:解码四元组算官方 F1
        preds = predict(model, val_ds, device, pair_mode=args.pair_mode,
                        threshold=args.pair_threshold, batch_size=args.batch_size, pad_id=pad_id)
        r = score(gold_dict(val_recs), records_to_pred_dict(preds))
        print(f"[fold{fold}] ep{ep} loss={tot / max(n, 1):.4f} "
              f"valF1={r['f1']:.4f} P={r['precision']:.4f} R={r['recall']:.4f} "
              f"({time.time() - t0:.0f}s)")
        if r["f1"] > best_f1:
            best_f1, best_epoch = r["f1"], ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / f"fold{fold}.pt")
    oof = predict(model, val_ds, device, pair_mode=args.pair_mode,
                  threshold=args.pair_threshold, batch_size=args.batch_size, pad_id=pad_id)
    return oof, best_f1, best_epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hfl/chinese-macbert-base")
    ap.add_argument("--data", default=str(ROOT / "data" / "processed" / "quads.jsonl"))
    ap.add_argument("--folds", default=str(ROOT / "data" / "processed" / "folds.json"))
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "processed" / "oof_a3"))
    ap.add_argument("--model-dir", default=str(ROOT / "checkpoints" / "a3"))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--accum-steps", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=502)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--pair-mode", default="nearest", choices=["nearest", "greedy"])
    ap.add_argument("--pair-threshold", type=float, default=0.5)
    ap.add_argument("--fold", default="all")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--val-on-train", action="store_true", help="sanity: 在训练记录自身上评估(过拟合检查)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.epochs, args.limit, args.fold = 1, 32, "0"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} model={args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    records = load_records(args.data)
    if args.limit:
        records = records[:args.limit]
    folds = json.loads(Path(args.folds).read_text(encoding="utf-8"))
    fold_ids = [set(f) for f in folds["folds"]]
    by_id = {r["id"]: r for r in records}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    k_list = range(len(fold_ids)) if args.fold == "all" else [int(args.fold)]
    all_oof, fold_scores = [], []
    for k in k_list:
        val_ids = {i for i in fold_ids[k] if i in by_id}
        train_ids = set(by_id) - val_ids
        tr = [by_id[i] for i in sorted(train_ids)]
        va = [by_id[i] for i in sorted(val_ids)]
        if args.val_on_train:
            va = tr
        if args.limit and not args.val_on_train:  # smoke 时验证集也截断
            va = va[: max(8, args.limit // 4)]
        print(f"===== fold {k}: train {len(tr)} / val {len(va)} =====")
        oof, best_f1, best_ep = train_one_fold(k, tr, va, args, tokenizer, device)
        all_oof.extend(oof)
        fold_scores.append({"fold": k, "best_val_f1": best_f1, "best_epoch": best_ep})
        write_records(out_dir / f"oof_fold{k}.jsonl", oof)

    write_records(out_dir / "oof.jsonl", all_oof)
    f1s = [s["best_val_f1"] for s in fold_scores]
    pred_dict = records_to_pred_dict(all_oof)
    # 金标只取被 OOF 覆盖的 id(单折/smoke/limit 运行时口径才一致)
    covered = gold_dict([r for r in records if r["id"] in pred_dict])
    report = {"model": args.model, "folds": fold_scores,
              "f1_mean": sum(f1s) / len(f1s),
              "f1_std": (sum((x - sum(f1s) / len(f1s)) ** 2 for x in f1s) / len(f1s)) ** 0.5,
              "n_oof_ids": len(pred_dict),
              "overall_oof_f1": score(covered, pred_dict)["f1"] if pred_dict else None}
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
