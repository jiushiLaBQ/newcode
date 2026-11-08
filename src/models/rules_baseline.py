# -*- coding: utf-8 -*-
"""A1 规则基线(无学习):词表最长匹配 + 最近距离配对 + 查表类别/极性。

定位:性能下限兜底 + 集成成员。全部统计只来自训练侧,不接触验证/测试标签。

流程:
  1. 从 quads.jsonl 训练部分统计:观点词表/属性词表/词->类别/词->极性(多数票)
  2. 对文本分别用最长匹配扫描观点/属性候选 span(贪心从左到右,每个起点取词表最长命中)
  3. 属性-观点按最近距离贪心配对(span 间字符 gap 优先,再按中心距离,并列取左)
  4. 无属性配上的观点 -> (_,o,c,p);无观点配上的属性 -> (a,_,c,p)
  5. 类别/极性查表(多数票,并列取计数大者再取字典序小者):
     类别优先属性词、再观点词;极性优先观点词、再属性词;
     查不到类别用"整体"、极性用"正面"
  6. 统一走 src/postprocess/rules.apply_rules 做闭集 mask/回贴/双隐式收尾(不去重)

CLI:
  # 全量训练 + 测试集预测
  python -I src/models/rules_baseline.py --train data/processed/quads.jsonl \
      --test data/processed/test_reviews.jsonl --out data/processed/pred_rules.jsonl
  # 按 folds.json 的 fold K 验证自评(打印真实 F1)
  python -I src/models/rules_baseline.py --train data/processed/quads.jsonl \
      --folds data/processed/folds.json --fold 0
  # 任意切分文件 CV(每行一个 id 或 JSON 列表;打印验证 F1,
  # 无 --test 时 --out 写验证集预测)
  python -I src/models/rules_baseline.py --train data/processed/quads.jsonl \
      --train-ids train_ids.txt --val-ids val_ids.txt --out cv_pred.jsonl
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.postprocess.rules import PLACEHOLDER, apply_rules
from src.eval.scorer import score as multiset_score


# ---------------------------------------------------------------- 数据加载

def load_rows(path):
    """读 quads.jsonl -> [{"id","text","quads"}, ...](保持文件顺序)"""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_ids(path):
    """切分文件:JSON 列表或每行一个 id -> set[int]"""
    raw = Path(path).read_text(encoding="utf-8").strip()
    if not raw:
        return set()
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return {int(x) for x in data}
    except json.JSONDecodeError:
        pass
    return {int(line.strip()) for line in raw.splitlines() if line.strip()}


# ---------------------------------------------------------------- 统计与词表

def _majority(counter):
    """多数票键;并列取计数大者,再并列取键字典序小者。"""
    return min(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0]


def build_stats(rows):
    """从训练行构建 观点/属性词表 + 词->类别 + 词->极性(多数票计数)。"""
    stats = {
        "opinion_terms": set(),
        "aspect_terms": set(),
        "cat": {"aspect": {}, "opinion": {}},   # {term: Counter({cat: n})}
        "pol": {"aspect": {}, "opinion": {}},   # {term: Counter({pol: n})}
    }
    for r in rows:
        for q in r.get("quads", []):
            a = q.get("a", PLACEHOLDER)
            o = q.get("o", PLACEHOLDER)
            c = q.get("c", PLACEHOLDER)
            p = q.get("p", PLACEHOLDER)
            if a != PLACEHOLDER:
                stats["aspect_terms"].add(a)
                stats["cat"]["aspect"].setdefault(a, Counter())[c] += 1
                stats["pol"]["aspect"].setdefault(a, Counter())[p] += 1
            if o != PLACEHOLDER:
                stats["opinion_terms"].add(o)
                stats["cat"]["opinion"].setdefault(o, Counter())[c] += 1
                stats["pol"]["opinion"].setdefault(o, Counter())[p] += 1
    # apply_rules 用的词典(类别共现 + 可选极性统计)
    stats["dicts"] = {
        "aspect": stats["cat"]["aspect"],
        "opinion": stats["cat"]["opinion"],
        "aspect_polarity": stats["pol"]["aspect"],
        "opinion_polarity": stats["pol"]["opinion"],
    }
    return stats


# ---------------------------------------------------------------- 扫描与配对

def longest_match_spans(text, term_set):
    """最长匹配扫描:贪心从左到右,每个起点取词表最长命中,命中即跳过该跨度。

    返回 [(start, end, term), ...],互不重叠、按起点升序。
    """
    if not text or not term_set:
        return []
    lengths = sorted({len(t) for t in term_set}, reverse=True)
    spans = []
    i, n = 0, len(text)
    while i < n:
        hit = None
        for l in lengths:
            if i + l <= n and text[i:i + l] in term_set:
                hit = l
                break
        if hit:
            spans.append((i, i + hit, text[i:i + hit]))
            i += hit
        else:
            i += 1
    return spans


def pair_nearest(a_spans, o_spans):
    """最近距离贪心配对:按 (字符 gap, 中心距离, 属性起点, 观点起点) 升序撮合。

    返回 [(ai, oi), ...];每个 span 至多配一次。
    """
    pairs = []
    for ai, (as_, ae, _) in enumerate(a_spans):
        for oi, (os_, oe, _) in enumerate(o_spans):
            gap = max(0, max(as_, os_) - min(ae, oe))
            center = abs((as_ + ae) - (os_ + oe))
            pairs.append((gap, center, as_, os_, ai, oi))
    pairs.sort()
    used_a, used_o, matched = set(), set(), []
    for _, _, _, _, ai, oi in pairs:
        if ai in used_a or oi in used_o:
            continue
        used_a.add(ai)
        used_o.add(oi)
        matched.append((ai, oi))
    return matched


# ---------------------------------------------------------------- 预测

def _lookup_cat(stats, term, role):
    if term == PLACEHOLDER:
        return None
    cnt = stats["cat"][role].get(term)
    return _majority(cnt) if cnt else None


def _lookup_pol(stats, term, role):
    if term == PLACEHOLDER:
        return None
    cnt = stats["pol"][role].get(term)
    return _majority(cnt) if cnt else None


def predict_text(text, stats):
    """对单条评论产出四元组 list[quad dict](经 apply_rules 收尾,保持多重性)。"""
    a_spans = longest_match_spans(text, stats["aspect_terms"])
    o_spans = longest_match_spans(text, stats["opinion_terms"])
    matched = pair_nearest(a_spans, o_spans)
    used_a = {ai for ai, _ in matched}
    used_o = {oi for _, oi in matched}

    raw = []  # (锚点起点, quad dict 未收尾)
    for ai, oi in matched:
        as_, ae, at = a_spans[ai]
        os_, oe, ot = o_spans[oi]
        c = _lookup_cat(stats, at, "aspect") or _lookup_cat(stats, ot, "opinion") or "整体"
        p = _lookup_pol(stats, ot, "opinion") or _lookup_pol(stats, at, "aspect") or "正面"
        raw.append((min(as_, os_), {
            "a": at, "a_start": as_, "a_end": ae,
            "o": ot, "o_start": os_, "o_end": oe,
            "c": c, "p": p}))
    for oi, (os_, oe, ot) in enumerate(o_spans):
        if oi in used_o:
            continue
        c = _lookup_cat(stats, ot, "opinion") or "整体"
        p = _lookup_pol(stats, ot, "opinion") or "正面"
        raw.append((os_, {
            "a": PLACEHOLDER, "a_start": -1, "a_end": -1,
            "o": ot, "o_start": os_, "o_end": oe,
            "c": c, "p": p}))
    for ai, (as_, ae, at) in enumerate(a_spans):
        if ai in used_a:
            continue
        c = _lookup_cat(stats, at, "aspect") or "整体"
        p = _lookup_pol(stats, at, "aspect") or "正面"
        raw.append((as_, {
            "a": at, "a_start": as_, "a_end": ae,
            "o": PLACEHOLDER, "o_start": -1, "o_end": -1,
            "c": c, "p": p}))

    raw.sort(key=lambda t: t[0])
    return apply_rules([q for _, q in raw], text, stats["dicts"])


def predict_rows(rows, stats):
    """对若干行预测 -> {id: [quad dict, ...]}"""
    out = {}
    for r in rows:
        out[r["id"]] = predict_text(r.get("text", ""), stats)
    return out


def to_tuples(pred_by_id):
    """{id: [quad dict]} -> {id: [(a,o,c,p), ...]}(scorer 入参形态)"""
    return {rid: [(q["a"], q["o"], q["c"], q["p"]) for q in qs]
            for rid, qs in pred_by_id.items()}


def gold_tuples(rows):
    """gold 行 -> {id: [(a,o,c,p), ...]}"""
    out = {}
    for r in rows:
        out[r["id"]] = [(q["a"], q["o"], q["c"], q["p"]) for q in r.get("quads", [])]
    return out


# ---------------------------------------------------------------- CLI

def write_jsonl(path, rows_pred, texts):
    """写 quads.jsonl 同构预测文件,id 升序。"""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for rid in sorted(rows_pred):
            rec = {"id": rid, "text": texts.get(rid, ""), "quads": rows_pred[rid]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description="A1 规则基线(无学习)")
    ap.add_argument("--train", required=True, help="训练 quads.jsonl")
    ap.add_argument("--test", default=None, help="test_reviews.jsonl")
    ap.add_argument("--out", default=None, help="预测输出(同 quads.jsonl 格式)")
    ap.add_argument("--train-ids", dest="train_ids", default=None, help="训练 id 切分文件")
    ap.add_argument("--val-ids", dest="val_ids", default=None, help="验证 id 切分文件(自评)")
    ap.add_argument("--folds", default=None, help="folds.json,配合 --fold 做验证自评")
    ap.add_argument("--fold", type=int, default=0, help="folds.json 中作为验证的折号")
    args = ap.parse_args()

    if args.out and not (args.test or args.val_ids):
        ap.error("--out 需要 --test(写测试集预测)或 --val-ids(写验证集 CV 预测)")
    if not (args.out or args.val_ids or args.folds):
        ap.error("无事可做:请给出 --out/--test 或验证切分(--val-ids / --folds)")

    rows = load_rows(args.train)
    all_ids = {r["id"] for r in rows}

    train_ids = load_ids(args.train_ids) if args.train_ids else None
    val_ids = load_ids(args.val_ids) if args.val_ids else None
    if args.folds and val_ids is None:
        folds = json.loads(Path(args.folds).read_text(encoding="utf-8"))
        if not (0 <= args.fold < len(folds["folds"])):
            ap.error(f"--fold {args.fold} 超出 folds 范围")
        val_ids = set(folds["folds"][args.fold])
    if val_ids is not None:
        val_ids &= all_ids
    # 训练侧:显式 --train-ids 优先;否则有验证折时用其余 id;否则全量
    if train_ids is None:
        train_ids = all_ids - val_ids if val_ids is not None else all_ids

    train_rows = [r for r in rows if r["id"] in train_ids]
    stats = build_stats(train_rows)
    print(f"[rules_baseline] 训练侧 {len(train_rows)} 条评论 / "
          f"属性词 {len(stats['aspect_terms'])} / 观点词 {len(stats['opinion_terms'])}")

    # ---- 自评:验证集真实 F1 ----
    if val_ids is not None:
        val_rows = [r for r in rows if r["id"] in val_ids]
        pred = predict_rows(val_rows, stats)
        res = multiset_score(gold_tuples(val_rows), to_tuples(pred))
        print(f"[self-eval] 训练 {len(train_rows)} 条 / 验证 {len(val_rows)} 条")
        print(f"[self-eval] P={res['P']} G={res['G']} S={res['S']} "
              f"precision={res['precision']:.4f} recall={res['recall']:.4f} f1={res['f1']:.4f}")

    # ---- 输出预测 ----
    if args.out:
        if args.test:
            out_rows = []
            with open(args.test, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        out_rows.append(json.loads(line))
            src = "测试集"
        else:
            out_rows = [r for r in rows if r["id"] in val_ids]
            src = "验证集"
        pred = predict_rows(out_rows, stats)
        texts = {r["id"]: r.get("text", "") for r in out_rows}
        write_jsonl(args.out, pred, texts)
        n_quads = sum(len(v) for v in pred.values())
        print(f"[rules_baseline] {src}预测 {len(pred)} 条评论 / {n_quads} 个四元组 -> {args.out}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
