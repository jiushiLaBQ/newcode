# -*- coding: utf-8 -*-
"""学习式四元组校验器(F1 项):给每个候选四元组打"正确概率",截断低置信项提精确率。

训练数据 = OOF 预测(每条都是模型没见过的样本)+ 金标匹配标签(多重集语义)。
特征全部来自输入文本与训练金标统计(不含模型内部概率),故可作用于任意模型的输出。

用法:
  # 交叉拟合评估(无偏 F1 增益)+ 训练全量模型并保存
  python -I src/postprocess/verifier.py train-eval \
      --oof data/processed/oof_a3/oof_all.jsonl --gold data/processed/quads.jsonl \
      --folds data/processed/folds.json --out data/processed/verifier.pt
  # 对测试集预测做过滤
  python -I src/postprocess/verifier.py apply \
      --pred data/processed/pred_test_vote2.jsonl --gold data/processed/quads.jsonl \
      --model data/processed/verifier.pt --threshold 0.5 --out data/processed/pred_test_vote2_vf.jsonl
"""
import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.eval.scorer import score

PLACEHOLDER = "_"


# ---------- 统计与特征 ----------
def build_stats(gold_records):
    st = {
        "a_freq": Counter(), "o_freq": Counter(), "pair_freq": Counter(),
        "o_cat": defaultdict(Counter), "o_pol": defaultdict(Counter),
        "a_cat": defaultdict(Counter), "cat_prior": Counter(), "pol_prior": Counter(),
        "total": 0,
    }
    for r in gold_records:
        for q in r["quads"]:
            st["total"] += 1
            if q["a"] != PLACEHOLDER:
                st["a_freq"][q["a"]] += 1
                st["a_cat"][q["a"]][q["c"]] += 1
            if q["o"] != PLACEHOLDER:
                st["o_freq"][q["o"]] += 1
                st["o_cat"][q["o"]][q["c"]] += 1
                st["o_pol"][q["o"]][q["p"]] += 1
            st["pair_freq"][(q["a"], q["o"])] += 1
            st["cat_prior"][q["c"]] += 1
            st["pol_prior"][q["p"]] += 1
    return st


FEAT_DIM = 16


def featurize(q, st):
    a, o, c, p = q["a"], q["o"], q["c"], q["p"]
    a_imp, o_imp = a == PLACEHOLDER, o == PLACEHOLDER
    dist = 0.0
    if not a_imp and not o_imp:
        dist = abs((q["a_start"] + q["a_end"]) / 2 - (q["o_start"] + q["o_end"]) / 2)
    o_cat = st["o_cat"].get(o, Counter())
    o_pol = st["o_pol"].get(o, Counter())
    a_cat = st["a_cat"].get(a, Counter())
    tot = max(st["total"], 1)

    def prior(counter, key):
        return math.log((counter.get(key, 0) + 1) / (sum(counter.values()) + len(counter) + 1))

    return [
        float(a_imp), float(o_imp),
        0.0 if a_imp else float(q["a_end"] - q["a_start"]),
        0.0 if o_imp else float(q["o_end"] - q["o_start"]),
        dist,
        math.log1p(st["a_freq"].get(a, 0)),
        math.log1p(st["o_freq"].get(o, 0)),
        math.log1p(st["pair_freq"].get((a, o), 0)),
        prior(st["cat_prior"], c),
        prior(st["pol_prior"], p),
        float(o_cat.get(c, 0)) / (sum(o_cat.values()) + 1),   # 观点词->类别一致性
        float(o_pol.get(p, 0)) / (sum(o_pol.values()) + 1),   # 观点词->极性一致性
        float(a_cat.get(c, 0)) / (sum(a_cat.values()) + 1),   # 属性词->类别一致性
        math.log1p(o_cat.get(c, 0)),
        math.log1p(o_pol.get(p, 0)),
        1.0 if (not o_imp and not a_imp and q["o_start"] < q["a_start"]) else 0.0,  # 跨序
    ]


# ---------- 多重集标注 ----------
def label_quads(pred_records, gold_by_id):
    """预测四元组 -> (特征行, 标签)。金标按多重集消耗。"""
    rows = []
    for r in pred_records:
        g = Counter(gold_by_id.get(r["id"], []))
        for q in r["quads"]:
            key = (q["a"], q["o"], q["c"], q["p"])
            if g.get(key, 0) > 0:
                g[key] -= 1
                rows.append((r["id"], q, 1))
            else:
                rows.append((r["id"], q, 0))
    return rows


# ---------- 模型 ----------
class Verifier(nn.Module):
    def __init__(self, d=FEAT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_verifier(rows, stats, seed=502, epochs=40, lr=1e-3):
    torch.manual_seed(seed)
    random.seed(seed)
    X = torch.tensor([featurize(q, stats) for _, q, _ in rows], dtype=torch.float32)
    y = torch.tensor([lab for _, _, lab in rows], dtype=torch.float32)
    n_pos = float(y.sum())
    pos_weight = torch.tensor([(len(y) - n_pos) / max(n_pos, 1.0)])
    model = Verifier()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        opt.step()
    model.eval()
    return model


@torch.no_grad()
def predict_scores(model, rows, stats):
    X = torch.tensor([featurize(q, stats) for _, q, _ in rows], dtype=torch.float32)
    return torch.sigmoid(model(X)).tolist()


def gold_dict_from(records):
    return {r["id"]: [(q["a"], q["o"], q["c"], q["p"]) for q in r["quads"]] for r in records}


def records_from_rows(records, rows, scores, threshold):
    keep = defaultdict(list)
    for (rid, q, _), s in zip(rows, scores):
        if s >= threshold:
            keep[rid].append(q)
    return [{"id": r["id"], "text": r["text"], "quads": keep.get(r["id"], [])} for r in records]


# ---------- 子命令 ----------
def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def cmd_train_eval(args):
    gold = load_jsonl(args.gold)
    oof = load_jsonl(args.oof)
    folds = json.loads(Path(args.folds).read_text(encoding="utf-8"))
    gold_by_id = {r["id"]: [(q["a"], q["o"], q["c"], q["p"]) for q in r["quads"]] for r in gold}
    gold_rec_by_id = {r["id"]: r for r in gold}

    # 交叉拟合:第 k 折的特征统计/verifier 都不得见第 k 折金标与预测
    fold_of = {}
    for k, ids in enumerate(folds["folds"]):
        for i in ids:
            fold_of[i] = k
    cross_rows, cross_scores = [], []
    for k in range(folds["n_folds"]):
        tr_recs = [r for r in gold if fold_of.get(r["id"]) != k]
        stats_k = build_stats(tr_recs)
        tr_rows = label_quads([r for r in oof if fold_of.get(r["id"]) != k], gold_by_id)
        te_rows = label_quads([r for r in oof if fold_of.get(r["id"]) == k], gold_by_id)
        if not tr_rows or not te_rows:
            continue
        model = train_verifier(tr_rows, stats_k)
        cross_rows.extend(te_rows)
        cross_scores.extend(predict_scores(model, te_rows, stats_k))

    base = score(gold_dict_from(gold), {r["id"]: [(q["a"], q["o"], q["c"], q["p"]) for q in r["quads"]] for r in oof})
    print(f"基线 OOF F1 = {base['f1']:.4f}")
    best = (base["f1"], 1.0)
    for th in [i / 20 for i in range(1, 20)]:
        filtered = records_from_rows(oof, cross_rows, cross_scores, th)
        r = score(gold_dict_from(gold), {x["id"]: [(q["a"], q["o"], q["c"], q["p"]) for q in x["quads"]] for x in filtered})
        print(f"  threshold={th:.2f} -> F1={r['f1']:.4f} (P={r['precision']:.4f} R={r['recall']:.4f} P数={r['P']})")
        if r["f1"] > best[0]:
            best = (r["f1"], th)
    print(f"交叉拟合最优: F1={best[0]:.4f} @ threshold={best[1]:.2f} (基线 {base['f1']:.4f}, Δ={best[0] - base['f1']:+.4f})")

    # 全量训练落盘
    all_rows = label_quads(oof, gold_by_id)
    stats_all = build_stats(gold)
    model = train_verifier(all_rows, stats_all)
    torch.save({"state_dict": model.state_dict(), "threshold": best[1]}, args.out)
    print(f"verifier 已保存 -> {args.out}")


def cmd_apply(args):
    gold = load_jsonl(args.gold)
    pred = load_jsonl(args.pred)
    ckpt = torch.load(args.model, map_location="cpu")
    stats = build_stats(gold)
    rows = label_quads(pred, {r["id"]: [] for r in gold})  # 全标 0,只为特征与对齐
    model = Verifier()
    model.load_state_dict(ckpt["state_dict"])
    th = args.threshold if args.threshold is not None else ckpt.get("threshold", 0.5)
    scores = predict_scores(model, rows, stats)
    out = records_from_rows(pred, rows, scores, th)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    nq = sum(len(r["quads"]) for r in out)
    print(f"threshold={th:.2f} 过滤后 {nq} 四元组 -> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("train-eval")
    p1.add_argument("--oof", required=True)
    p1.add_argument("--gold", required=True)
    p1.add_argument("--folds", required=True)
    p1.add_argument("--out", required=True)
    p1.set_defaults(func=cmd_train_eval)
    p2 = sub.add_parser("apply")
    p2.add_argument("--pred", required=True)
    p2.add_argument("--gold", required=True)
    p2.add_argument("--model", required=True)
    p2.add_argument("--threshold", type=float, default=None)
    p2.add_argument("--out", required=True)
    p2.set_defaults(func=cmd_apply)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
