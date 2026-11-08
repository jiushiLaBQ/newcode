# -*- coding: utf-8 -*-
"""交叉验证/错误分析报表工具:基于 scorer.score 的真实计算,不训练模型。

用法:
  python -I src/eval/run_cv.py --gold data/processed/quads.jsonl --pred data/processed/oof.jsonl \
      [--pred 更多预测文件做对比 ...] [--folds data/processed/folds.json] [--out report.json]

预测文件两种格式(自动识别):
  - .jsonl:与 quads.jsonl 同结构 {"id","quads":[{"a","o","c","p", 偏移可选}]}
  - .csv  :提交格式(无表头) id,AspectTerm,OpinionTerm,Category,Polarity

报表内容(全部真实计算):
  1. 总体 P/R/F1(多重集语义,scorer.score)+ 预测/金标条数比
  2. 逐类别 F1(13 类闭集)、逐极性 F1(3 类闭集)
  3. 错误分桶(相对金标):
     - 隐式属性四元组(a=="_")recall、显式四元组(a!="_")recall
     - 跨序对(o_end<=a_start)子集 F1:金标/预测两侧各自按偏移过滤后 scorer.score;
       预测文件无偏移(如提交 csv)时仅报 recall 并注明
     - 多四元组 id(金标 >=3 条)子集 F1:两侧取同一批 id 的全部四元组
     - 边界差一字命中数:精确匹配后剩余的金标/预测四元组中,同 id、同类别极性、
       属性+观点词面编辑距离合计恰为 1 的贪心一一配对数
     - 占位错误:剩余池中同 id、同类别极性、恰一字段占位错位("_" vs 显式)而
       另一字段全等的配对数,分「预测占位/金标显式」与「预测显式/金标占位」
     - 每 id 平均预测条数 vs 金标条数(并集 id 上)
     注:占位错误先配对,边界差一字在剩余池上配对,两桶互斥。
  4. 给定 folds.json 时:逐折 F1 及均值±方差(总体方差,ddof=0)

多重集语义:评测不去重(数据中 id=1375 存在重复四元组),预测侧同样不去重。
"""
import argparse
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.eval.scorer import score, load_pred_csv, PLACEHOLDER  # noqa: E402
from src.eval.check_submit import CATEGORIES, POLARITIES  # noqa: E402

CAT_ORDER = sorted(CATEGORIES)      # 13 类,稳定输出序
POL_ORDER = sorted(POLARITIES)      # 3 类


# ---------------------------------------------------------------- 数据加载

def load_quads(path):
    """quads.jsonl -> (gold: {id: [(a,o,c,p),...]}, rows: {id: [四元组 dict,...]})。

    dict 带字符偏移(a_start 等),供跨序对分桶用;元组供 scorer.score 用。
    """
    gold, rows = {}, {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rid = row["id"]
            rows[rid] = row["quads"]
            gold[rid] = [(q["a"], q["o"], q["c"], q["p"]) for q in row["quads"]]
    return gold, rows


def load_pred(path):
    """预测文件 -> (pred: {id: [(a,o,c,p),...]}, rows: {id: [dict,...]}, has_offsets)。"""
    if str(path).lower().endswith(".csv"):
        pred = load_pred_csv(path)
        rows = {rid: [{"a": a, "o": o, "c": c, "p": p} for (a, o, c, p) in qs]
                for rid, qs in pred.items()}
        return pred, rows, False
    pred, rows = load_quads(path)
    has_offsets = bool(rows) and all(
        "a_start" in q for qs in rows.values() for q in qs
    )
    return pred, rows, has_offsets


# ---------------------------------------------------------------- 分桶工具

def _filter_by_value(d, idx, value):
    """按元组第 idx 个字段过滤四元组(类别 idx=2 / 极性 idx=3)。"""
    return {rid: [q for q in qs if q[idx] == value] for rid, qs in d.items()}


def _filter_by_pred(d, pred_fn):
    """按谓词过滤四元组 dict 列表。"""
    return {rid: [q for q in qs if pred_fn(q)] for rid, qs in d.items()}


def _tup(rows):
    """{id: [dict,...]} -> {id: [(a,o,c,p),...]}"""
    return {rid: [(q["a"], q["o"], q["c"], q["p"]) for q in qs] for rid, qs in rows.items()}


def _cross_order(q):
    """跨序对:属性、观点都显式且观点在属性之前(含紧邻,o_end<=a_start)。"""
    return q.get("a_start", -1) >= 0 and q.get("o_start", -1) >= 0 and q["o_end"] <= q["a_start"]


def _lev(a, b):
    """Levenshtein 编辑距离(词面都很短,朴素 DP 足够)。"""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[lb]


def _placeholder_kind(g, p):
    """占位错位方向:0=不匹配该桶, 1=预测占位/金标显式, 2=预测显式/金标占位。

    条件:恰一字段占位错位,另一字段全等(类别极性由调用方保证)。
    g/p 为 (a,o,c,p) 元组。
    """
    ga, go, pa, po = g[0], g[1], p[0], p[1]
    if go == po:
        if pa == PLACEHOLDER and ga != PLACEHOLDER:
            return 1
        if pa != PLACEHOLDER and ga == PLACEHOLDER:
            return 2
    if ga == pa:
        if po == PLACEHOLDER and go != PLACEHOLDER:
            return 1
        if po != PLACEHOLDER and go == PLACEHOLDER:
            return 2
    return 0


def _pair_error_buckets(gold, pred):
    """精确匹配后的剩余池上做贪心一一配对(占位错误先,边界差一字后)。

    gold/pred: {id: [(a,o,c,p),...]} -> dict 计数。
    """
    boundary = ph_pred_ph = ph_gold_ph = 0
    for rid in set(gold) | set(pred):
        gc, pc = Counter(gold.get(rid, [])), Counter(pred.get(rid, []))
        rest_g = sorted((gc - (gc & pc)).elements())
        rest_p = sorted((pc - (gc & pc)).elements())
        used_p = [False] * len(rest_p)
        used_g = [False] * len(rest_g)

        # 1) 占位错误先配对(结构性错位优先)
        for gi, gq in enumerate(rest_g):
            for pi, pq in enumerate(rest_p):
                if used_p[pi]:
                    continue
                if gq[2] != pq[2] or gq[3] != pq[3]:   # 类别极性必须对
                    continue
                kind = _placeholder_kind(gq, pq)
                if kind:
                    used_g[gi] = used_p[pi] = True
                    if kind == 1:
                        ph_pred_ph += 1
                    else:
                        ph_gold_ph += 1
                    break

        # 2) 边界差一字:属性+观点词面编辑距离合计恰 1
        for gi, gq in enumerate(rest_g):
            if used_g[gi]:
                continue
            for pi, pq in enumerate(rest_p):
                if used_p[pi]:
                    continue
                if gq[2] != pq[2] or gq[3] != pq[3]:
                    continue
                if _lev(gq[0], pq[0]) + _lev(gq[1], pq[1]) == 1:
                    used_g[gi] = used_p[pi] = True
                    boundary += 1
                    break

    return {
        "boundary_off_by_one": boundary,
        "placeholder_pred_ph_gold_explicit": ph_pred_ph,
        "placeholder_pred_explicit_gold_ph": ph_gold_ph,
    }


# ---------------------------------------------------------------- 报表

def build_report(gold_path, pred_path, folds_path=None):
    """核心入口:返回完整报表 dict(全部指标经 scorer.score 或其多重集语义计算)。"""
    gold, gold_rows = load_quads(gold_path)
    pred, pred_rows, pred_has_offsets = load_pred(pred_path)

    overall_raw = score(gold, pred)
    overall = {
        "P": overall_raw["P"], "G": overall_raw["G"], "S": overall_raw["S"],
        "precision": overall_raw["precision"], "recall": overall_raw["recall"],
        "f1": overall_raw["f1"],
        "pred_gold_ratio": (overall_raw["P"] / overall_raw["G"]) if overall_raw["G"] else 0.0,
    }

    # 逐类别 / 逐极性 F1
    per_category = {}
    for c in CAT_ORDER:
        r = score(_filter_by_value(gold, 2, c), _filter_by_value(pred, 2, c))
        per_category[c] = {"G": r["G"], "P": r["P"], "S": r["S"],
                           "precision": r["precision"], "recall": r["recall"], "f1": r["f1"]}
    per_polarity = {}
    for p in POL_ORDER:
        r = score(_filter_by_value(gold, 3, p), _filter_by_value(pred, 3, p))
        per_polarity[p] = {"G": r["G"], "P": r["P"], "S": r["S"],
                           "precision": r["precision"], "recall": r["recall"], "f1": r["f1"]}

    buckets = {}

    # 隐式属性 / 显式四元组 recall(元组词面即可判定,无需偏移)
    for name, fn in (("implicit_aspect", lambda q: q[0] == PLACEHOLDER),
                     ("explicit_aspect", lambda q: q[0] != PLACEHOLDER)):
        g_sub = {rid: [q for q in qs if fn(q)] for rid, qs in gold.items()}
        r = score(g_sub, pred)
        buckets[name] = {"G": r["G"], "S": r["S"], "recall": r["recall"]}

    # 跨序对子集 F1(需要偏移;预测无偏移时退化为 recall + 注明)
    g_cross = _filter_by_pred(gold_rows, _cross_order)
    if pred_has_offsets:
        p_cross = _filter_by_pred(pred_rows, _cross_order)
        r = score(_tup(g_cross), _tup(p_cross))
        buckets["cross_order"] = {
            "G": r["G"], "P": r["P"], "S": r["S"],
            "precision": r["precision"], "recall": r["recall"], "f1": r["f1"],
            "note": "",
        }
    else:
        r = score(_tup(g_cross), pred)
        buckets["cross_order"] = {
            "G": r["G"], "P": None, "S": r["S"],
            "precision": None, "recall": r["recall"], "f1": None,
            "note": "预测文件无字符偏移,无法在预测侧过滤跨序对,仅报 recall",
        }

    # 多四元组 id(金标 >=3 条)子集 F1
    multi_ids = {rid for rid, qs in gold.items() if len(qs) >= 3}
    r = score({i: gold[i] for i in multi_ids if i in gold},
              {i: pred[i] for i in multi_ids if i in pred})
    buckets["multi_quad_ids"] = {
        "n_ids": len(multi_ids), "G": r["G"], "P": r["P"], "S": r["S"],
        "precision": r["precision"], "recall": r["recall"], "f1": r["f1"],
    }

    # 近邻错误配对(边界差一字 / 占位错误)
    buckets.update(_pair_error_buckets(gold, pred))

    # 每 id 平均条数
    all_ids = set(gold) | set(pred)
    buckets["per_id_avg"] = {
        "pred": overall_raw["P"] / len(all_ids) if all_ids else 0.0,
        "gold": overall_raw["G"] / len(all_ids) if all_ids else 0.0,
        "n_ids": len(all_ids),
    }

    # 逐折 F1
    fold_report = None
    if folds_path:
        folds_data = json.loads(Path(folds_path).read_text(encoding="utf-8"))
        per_fold = []
        for k, ids in enumerate(folds_data["folds"]):
            id_set = set(ids)
            r = score({i: gold[i] for i in id_set if i in gold},
                      {i: pred[i] for i in id_set if i in pred})
            per_fold.append({"fold": k, "n_ids": len(id_set), "f1": r["f1"],
                             "precision": r["precision"], "recall": r["recall"]})
        f1s = [x["f1"] for x in per_fold]
        n = len(f1s)
        mean = sum(f1s) / n if n else 0.0
        var = sum((x - mean) ** 2 for x in f1s) / n if n else 0.0
        fold_report = {
            "per_fold": per_fold,
            "mean": mean,
            "var": var,
            "std": var ** 0.5,
            "folds_file": str(folds_path),
        }

    return {
        "pred": str(pred_path),
        "n_gold_quads": overall_raw["G"],
        "n_gold_ids": len(gold),
        "n_pred_quads": overall_raw["P"],
        "n_pred_ids": len(pred),
        "pred_has_offsets": pred_has_offsets,
        "overall": overall,
        "per_category": per_category,
        "per_polarity": per_polarity,
        "buckets": buckets,
        "folds": fold_report,
    }


# ---------------------------------------------------------------- 打印

def _disp_width(s):
    """CJK 宽字符按 2 格计,保证表格对齐。"""
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in s)


def _pad(s, w):
    return s + " " * max(0, w - _disp_width(s))


def _fmt(v, spec=".4f"):
    return "-" if v is None else format(v, spec)


def print_report(rep):
    print(f"\n############ 预测文件: {rep['pred']} ############")
    o = rep["overall"]
    print("-- 总体 --")
    print(f"  金标四元组 G={o['G']} (id 数 {rep['n_gold_ids']}), "
          f"预测 P={o['P']} (id 数 {rep['n_pred_ids']}), 命中 S={o['S']}")
    print(f"  预测/金标条数比 = {o['pred_gold_ratio']:.4f}")
    print(f"  precision={o['precision']:.4f}  recall={o['recall']:.4f}  f1={o['f1']:.4f}")

    for title, table in (("逐类别 F1(13 类)", rep["per_category"]),
                         ("逐极性 F1(3 类)", rep["per_polarity"])):
        print(f"-- {title} --")
        print("  " + _pad("类别", 12) + _pad("G", 7) + _pad("P", 7) + _pad("S", 7)
              + _pad("recall", 9) + _pad("precision", 11) + "f1")
        for name, m in table.items():
            print("  " + _pad(name, 12) + _pad(str(m["G"]), 7) + _pad(str(m["P"]), 7)
                  + _pad(str(m["S"]), 7) + _pad(_fmt(m["recall"]), 9)
                  + _pad(_fmt(m["precision"]), 11) + _fmt(m["f1"]))

    b = rep["buckets"]
    print("-- 错误分桶(相对金标) --")
    ia, ea = b["implicit_aspect"], b["explicit_aspect"]
    print(f"  隐式属性四元组 recall : {ia['recall']:.4f}  (S={ia['S']} / G={ia['G']})")
    print(f"  显式四元组 recall     : {ea['recall']:.4f}  (S={ea['S']} / G={ea['G']})")
    co = b["cross_order"]
    note = f"  [{co['note']}]" if co["note"] else ""
    print(f"  跨序对(o_end<=a_start)子集 F1: {_fmt(co['f1'])}  "
          f"(G={co['G']} P={co['P']} S={co['S']} recall={_fmt(co['recall'])}){note}")
    mq = b["multi_quad_ids"]
    print(f"  多四元组 id(>=3)子集 F1    : {mq['f1']:.4f}  "
          f"(id 数={mq['n_ids']} G={mq['G']} P={mq['P']} S={mq['S']})")
    print(f"  边界差一字命中数           : {b['boundary_off_by_one']}")
    print(f"  占位错误(预测占位/金标显式): {b['placeholder_pred_ph_gold_explicit']}")
    print(f"  占位错误(预测显式/金标占位): {b['placeholder_pred_explicit_gold_ph']}")
    pa = b["per_id_avg"]
    print(f"  每 id 平均条数: 预测={pa['pred']:.4f} vs 金标={pa['gold']:.4f} (并集 id 数 {pa['n_ids']})")

    if rep["folds"]:
        fr = rep["folds"]
        print("-- 逐折 F1 --")
        for x in fr["per_fold"]:
            print(f"  fold {x['fold']}: f1={x['f1']:.4f}  "
                  f"(ids={x['n_ids']} precision={x['precision']:.4f} recall={x['recall']:.4f})")
        print(f"  均值±方差 = {fr['mean']:.4f} ± {fr['var']:.6f}  (标准差 {fr['std']:.6f})")
    else:
        print("-- 逐折 F1 --  (未提供 --folds,跳过)")


def print_comparison(reps):
    """多份预测横向对比。"""
    if len(reps) < 2:
        return
    print("\n==== 多预测对比 ====")
    print(_pad("预测文件", 40) + _pad("条数比", 9) + _pad("recall", 9)
          + _pad("precision", 11) + _pad("f1", 9) + "折F1均值±方差")
    for rep in reps:
        o = rep["overall"]
        fold_str = (f"{rep['folds']['mean']:.4f} ± {rep['folds']['var']:.6f}"
                    if rep["folds"] else "-")
        print(_pad(Path(rep["pred"]).name, 40) + _pad(f"{o['pred_gold_ratio']:.4f}", 9)
              + _pad(f"{o['recall']:.4f}", 9) + _pad(f"{o['precision']:.4f}", 11)
              + _pad(f"{o['f1']:.4f}", 9) + fold_str)


# ---------------------------------------------------------------- CLI

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="CV 评测/错误分析报表(基于 scorer.score)")
    ap.add_argument("--gold", required=True, help="金标 quads.jsonl")
    ap.add_argument("--pred", action="append", required=True,
                    help="预测文件(.jsonl 同 quads 结构 或 提交 .csv),可多次指定做对比")
    ap.add_argument("--folds", default=None, help="folds.json,给定则报逐折 F1 均值±方差")
    ap.add_argument("--out", default=None, help="报表 JSON 输出路径")
    args = ap.parse_args()

    reps = [build_report(args.gold, p, args.folds) for p in args.pred]
    for rep in reps:
        print_report(rep)
    print_comparison(reps)

    if args.out:
        out = {"gold": args.gold, "folds": args.folds, "predictions": reps}
        Path(args.out).write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n报表已写入: {args.out}")


if __name__ == "__main__":
    main()
