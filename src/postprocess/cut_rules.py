# -*- coding: utf-8 -*-
"""删除族规则库 + CLI:对四元组预测(jsonl,与 quads.jsonl 同构)应用删除规则。

设计约定:
  - 所有规则都是"逐四元组谓词 + 删除",不做改判/修补(全字段精确匹配下改判风险远大于收益)。
  - 规则之间可交换、可串联:串联 = 删除集合并集,按 --rules 给定顺序依次作用于剩余集。
  - 词典类参数只从训练折金标构建(--dict-folds,默认 fold0-2),评测协议见
    data/experiments/cutrules/(冠军判据 t/d < f/2,双折门控,样本量>=15)。

规则清单(名称 -> 语义):
  cat_other   删除 c=其他 的四元组
  cat_size    删除 c=尺寸 的四元组
  pol_s503    双 seed 极性冲突 -> 删 seed503 侧
              冲突 = 同 (评论文本,a,o,c) 下 seed502/seed503 OOF 预测的极性集合不一致且并集>=2;
              seed503 侧 = pols503 - pols502(仅 seed503 独有的极性)。
              键含文本而非 id:训练/测试 id 数值撞号,文本才是评论身份。
  pol_both    双 seed 极性冲突 -> 两侧都删(对称差 pols502 Δ pols503)。
  pol_lex     双 seed 极性冲突 -> 只删与观点词词典极性冲突侧
              词典 lex(o) = 训练折金标中 o 的多数极性(并列取计数高者,再取字典序);
              o 为隐式 '_' 或词典未见 -> 不删;删除侧 = {p in 对称差 : p != lex(o)}。
  r1_gap      双显式(a,o 均有偏移)且属性-观点跨度间隙 gap>3 -> 删(pairfix R1 拆对/drop)
  r2_cross    跨序对(o_end<=a_start)且 (a,o) 在训练折金标零共现 -> 删(pairfix R2)
  r3_impo     预测 (a,'_') 且 a 在训练折中总是带显式观点(a_imp==0 且 a_exp>=1) -> 删(pairfix R3)
  r4_impa     预测 ('_',o) 且 o 在训练折中总是带显式属性(o_imp==0 且 o_exp>=1) -> 删(pairfix R4)

用法:
  PYTHONUTF8=1 .venv/Scripts/python -I src/postprocess/cut_rules.py \
      --pred data/processed/oof_2seed_inter2.jsonl --out /tmp/cut.jsonl \
      --rules pol_both,cat_other,cat_size \
      [--gold data/processed/quads.jsonl --folds data/processed/folds.json --dict-folds 0,1,2] \
      [--ref502 data/processed/oof_a3_ep12/oof_all.jsonl \
       --ref503 data/processed/oof_a3_ep12_s503/oof_all.jsonl]
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PLACEHOLDER = "_"
GOLD_DEFAULT = ROOT / "data/processed/quads.jsonl"
FOLDS_DEFAULT = ROOT / "data/processed/folds.json"
REF502_DEFAULT = ROOT / "data/processed/oof_a3_ep12/oof_all.jsonl"
REF503_DEFAULT = ROOT / "data/processed/oof_a3_ep12_s503/oof_all.jsonl"

# 串联固定顺序(冠军方案顺序:极性冲突 -> 类别 -> pairfix 四条)
RULE_ORDER = ["pol_s503", "pol_both", "pol_lex",
              "cat_other", "cat_size",
              "r1_gap", "r2_cross", "r3_impo", "r4_impa"]
POLICY_OF = {"pol_s503": "s503", "pol_both": "both", "pol_lex": "lex"}


# ---------------------------------------------------------------- 数据加载

def load_rows(path):
    """jsonl -> {id: {"id","text","quads":[...]}}(与 quads.jsonl 同构)。"""
    rows = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                rows[r["id"]] = r
    return rows


def rows_to_tuples(rows):
    return {rid: [(q["a"], q["o"], q["c"], q["p"]) for q in r["quads"]]
            for rid, r in rows.items()}


# ---------------------------------------------------------------- 词典(仅训练折金标)

def build_dicts(gold_rows, train_ids):
    """训练折金标 -> 规则所需统计词典。

    a_exp_op[a]: a 与显式观点同现次数; a_imp_op[a]: a 与隐式观点('_')同现次数
    o_exp_a[o] / o_imp_a[o]: 对偶; pair_cnt[(a,o)]: 双显式共现;
    op_pol[o]: 观点词 o 的极性计数(含隐式属性的四元组)。
    """
    a_exp_op, a_imp_op = Counter(), Counter()
    o_exp_a, o_imp_a = Counter(), Counter()
    pair_cnt = Counter()
    op_pol = defaultdict(Counter)
    for rid in train_ids:
        row = gold_rows.get(rid)
        if row is None:
            continue
        for q in row["quads"]:
            a, o = q["a"], q["o"]
            if o != PLACEHOLDER:
                op_pol[o][q["p"]] += 1
            if a != PLACEHOLDER and o != PLACEHOLDER:
                pair_cnt[(a, o)] += 1
                a_exp_op[a] += 1
                o_exp_a[o] += 1
            elif a != PLACEHOLDER and o == PLACEHOLDER:
                a_imp_op[a] += 1
            elif a == PLACEHOLDER and o != PLACEHOLDER:
                o_imp_a[o] += 1
    return {"a_exp_op": a_exp_op, "a_imp_op": a_imp_op,
            "o_exp_a": o_exp_a, "o_imp_a": o_imp_a,
            "pair_cnt": pair_cnt, "op_pol": op_pol}


def build_lex_pol(op_pol):
    """op_pol 计数 -> {o: 多数极性}(并列取计数高者,再取字典序,保证确定性)。"""
    return {o: max(cnt.items(), key=lambda kv: (kv[1], kv[0]))[0]
            for o, cnt in op_pol.items() if cnt}


# ---------------------------------------------------------------- 双 seed 极性冲突图

def build_pol_map(ref502_rows, ref503_rows):
    """两个 seed 的 OOF -> {(text, a, o, c): (pols502, pols503)}(仅冲突键)。

    冲突定义:同一 (评论文本, a, o, c) 下两 seed 的极性集合不一致,且并集大小 >= 2
    (两 seed 都给出了该键、但极性不同;单边缺席不算冲突)。
    注意键含文本而非 id:id 在训练/测试间数值撞号(train 1..3229 / test 1..2237),
    文本才是评论的身份;同文本多 id(近重复组)按同一键合并,语义一致。
    """
    def key_sets(rows):
        m = defaultdict(set)
        for rid, r in rows.items():
            for q in r["quads"]:
                m[(r["text"], q["a"], q["o"], q["c"])].add(q["p"])
        return m

    m502, m503 = key_sets(ref502_rows), key_sets(ref503_rows)
    out = {}
    for key in set(m502) & set(m503):
        s2, s3 = m502[key], m503[key]
        if s2 != s3 and len(s2 | s3) >= 2:
            out[key] = (s2, s3)
    return out


def build_pol_delete_sets(pol_map, policy, lex_pol):
    """冲突图 + 策略 -> {key: 待删极性集合}。"""
    out = {}
    for key, (s2, s3) in pol_map.items():
        sym = (s2 | s3) - (s2 & s3)          # 对称差 = 两侧冲突极性
        if policy == "s503":
            dele = s3 - s2                    # 只删 seed503 独有侧
        elif policy == "both":
            dele = sym                        # 两侧都删
        elif policy == "lex":
            _text, _a, o, _c = key
            pl = lex_pol.get(o) if o != PLACEHOLDER else None
            dele = set() if pl is None else {p for p in sym if p != pl}
        else:
            raise ValueError(policy)
        if dele:
            out[key] = dele
    return out


# ---------------------------------------------------------------- 跨序/间隙工具

def _both_explicit(q):
    return q.get("a_start", -1) >= 0 and q.get("o_start", -1) >= 0


def _span_gap(q):
    a0, a1, o0, o1 = q["a_start"], q["a_end"], q["o_start"], q["o_end"]
    if a1 <= o0:
        return o0 - a1
    if o1 <= a0:
        return a0 - o1
    return 0


def _cross_order(q):
    return _both_explicit(q) and q["o_end"] <= q["a_start"]


# ---------------------------------------------------------------- 规则(逐四元组谓词)

def make_predicates(rule_names, ctx):
    """规则名列表 -> [(name, fn(q, rid, text) -> bool)]。ctx 缺依赖时抛错。"""
    dicts = ctx.get("dicts")
    pol_del = ctx.get("pol_del")   # {policy: {key: set(pols)}}

    def need(cond, msg):
        if not cond:
            raise ValueError(f"规则依赖缺失: {msg}")

    out = []
    for name in rule_names:
        if name == "cat_other":
            out.append((name, lambda q, rid, text: q["c"] == "其他"))
        elif name == "cat_size":
            out.append((name, lambda q, rid, text: q["c"] == "尺寸"))
        elif name in POLICY_OF:
            need(pol_del is not None and POLICY_OF[name] in pol_del,
                 f"{name} 需要双 seed 参考预测(ref502/ref503)")
            dmap = pol_del[POLICY_OF[name]]
            out.append((name,
                        lambda q, rid, text, dmap=dmap:
                        q["p"] in dmap.get((text, q["a"], q["o"], q["c"]), ())))
        elif name == "r1_gap":
            out.append((name, lambda q, rid, text:
                        _both_explicit(q) and _span_gap(q) > 3))
        elif name == "r2_cross":
            need(dicts is not None, "r2_cross 需要训练折金标词典")
            out.append((name, lambda q, rid, text:
                        _cross_order(q) and dicts["pair_cnt"][(q["a"], q["o"])] < 1))
        elif name == "r3_impo":
            need(dicts is not None, "r3_impo 需要训练折金标词典")
            out.append((name, lambda q, rid, text:
                        q["o"] == PLACEHOLDER and q["a"] != PLACEHOLDER
                        and dicts["a_imp_op"][q["a"]] == 0
                        and dicts["a_exp_op"][q["a"]] >= 1))
        elif name == "r4_impa":
            need(dicts is not None, "r4_impa 需要训练折金标词典")
            out.append((name, lambda q, rid, text:
                        q["a"] == PLACEHOLDER and q["o"] != PLACEHOLDER
                        and dicts["o_imp_a"][q["o"]] == 0
                        and dicts["o_exp_a"][q["o"]] >= 1))
        else:
            raise ValueError(f"未知规则: {name}")
    return out


def build_ctx(gold_rows=None, train_ids=None, ref502_rows=None, ref503_rows=None,
              pol_policies=None):
    """组装规则上下文(词典只吃 train_ids 的金标;极性规则吃双 seed 参考)。"""
    ctx = {"dicts": None, "pol_del": None}
    if gold_rows is not None and train_ids is not None:
        dicts = build_dicts(gold_rows, train_ids)
        ctx["dicts"] = dicts
        ctx["lex_pol"] = build_lex_pol(dicts["op_pol"])
    if ref502_rows is not None and ref503_rows is not None:
        pol_map = build_pol_map(ref502_rows, ref503_rows)
        lex_pol = ctx.get("lex_pol", {})
        pols = pol_policies or sorted(POLICY_OF.values())
        ctx["pol_map"] = pol_map
        ctx["pol_del"] = {p: build_pol_delete_sets(pol_map, p, lex_pol) for p in pols}
    return ctx


def apply_rules(rows, rule_names, ctx):
    """按顺序串联规则(纯删除,可交换)。返回 (新 rows, 每条规则删除明细)。

    detail[name] = {rid: [删除的 quad dict, ...]}(被前序规则已删的不再计)。
    """
    preds = make_predicates(rule_names, ctx)
    remaining = {rid: list(r["quads"]) for rid, r in rows.items()}
    detail = {}
    for name, fn in preds:
        hits = {}
        for rid, quads in remaining.items():
            text = rows[rid]["text"]
            keep, drop = [], []
            for q in quads:
                (drop if fn(q, rid, text) else keep).append(q)
            if drop:
                hits[rid] = drop
            remaining[rid] = keep
        detail[name] = hits
    out = {rid: {"id": rid, "text": rows[rid]["text"], "quads": qs}
           for rid, qs in remaining.items()}
    return out, detail


# ---------------------------------------------------------------- CLI

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="删除族规则应用器(只删不改)")
    ap.add_argument("--pred", required=True, help="输入预测 jsonl(同 quads.jsonl 结构)")
    ap.add_argument("--out", required=True, help="输出 jsonl")
    ap.add_argument("--rules", required=True,
                    help=f"逗号分隔规则列表,可选: {','.join(RULE_ORDER)}")
    ap.add_argument("--gold", default=str(GOLD_DEFAULT), help="金标(构建词典用)")
    ap.add_argument("--folds", default=str(FOLDS_DEFAULT))
    ap.add_argument("--dict-folds", default="0,1,2",
                    help="用哪些折的金标建词典(默认 0,1,2=训练折,禁止用评测折)")
    ap.add_argument("--ref502", default=str(REF502_DEFAULT), help="seed502 OOF(极性规则参考)")
    ap.add_argument("--ref503", default=str(REF503_DEFAULT), help="seed503 OOF(极性规则参考)")
    args = ap.parse_args()

    rule_names = [r.strip() for r in args.rules.split(",") if r.strip()]
    unknown = [r for r in rule_names if r not in RULE_ORDER]
    if unknown:
        raise SystemExit(f"未知规则: {unknown};可选: {RULE_ORDER}")
    # 固定顺序串联(与 --rules 书写顺序无关)
    rule_names = [r for r in RULE_ORDER if r in set(rule_names)]

    gold_rows = load_rows(args.gold)
    folds = json.loads(Path(args.folds).read_text(encoding="utf-8"))["folds"]
    dict_folds = {int(x) for x in args.dict_folds.split(",") if x.strip() != ""}
    train_ids = set()
    for k in sorted(dict_folds):
        train_ids |= set(folds[k])

    need_pol = any(r in POLICY_OF for r in rule_names)
    ref502 = load_rows(args.ref502) if need_pol else None
    ref503 = load_rows(args.ref503) if need_pol else None
    ctx = build_ctx(gold_rows, train_ids, ref502, ref503,
                    pol_policies=[POLICY_OF[r] for r in rule_names if r in POLICY_OF])

    rows = load_rows(args.pred)
    out, detail = apply_rules(rows, rule_names, ctx)
    with open(args.out, "w", encoding="utf-8") as f:
        for rid in sorted(out):
            f.write(json.dumps(out[rid], ensure_ascii=False) + "\n")

    n0 = sum(len(r["quads"]) for r in rows.values())
    n1 = sum(len(r["quads"]) for r in out.values())
    print(f"输入 {n0} 条四元组 -> 输出 {n1} 条(共删 {n0 - n1})")
    for name in rule_names:
        d = detail[name]
        print(f"  {name}: 删 {sum(len(v) for v in d.values())} 条 (id 数 {len(d)})")
    print(f"已写入: {args.out}")


if __name__ == "__main__":
    main()
