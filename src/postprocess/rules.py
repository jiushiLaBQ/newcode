# -*- coding: utf-8 -*-
"""确定性后处理库(纯函数):词面回贴、闭集 mask、词典兜底、双隐式清理、多重性控制。

约定:
  - 四元组 dict 与 data/processed/quads.jsonl 的 quads 元素同构:
    {"a","a_start","a_end","o","o_start","o_end","c","p"}
  - 占位符 "_" 表示隐式字段,偏移恒为 -1;非占位词面必须是评论原文子串(text[start:end]==词面)
  - 多重集语义:默认不去重(评测不去重,数据里真有重复四元组)
"""
import json
from collections import Counter

PLACEHOLDER = "_"
# 赛题闭集,与 src/eval/check_submit.py 保持一致
CATEGORIES = {"包装", "成分", "尺寸", "服务", "功效", "价格", "气味", "使用体验", "物流", "新鲜度", "真伪", "整体", "其他"}
POLARITIES = {"正面", "中性", "负面"}


def _levenshtein(a, b):
    """编辑距离(插入/删除/替换各计 1),小串上直接 DP。"""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb]


def snap_back(term, text):
    """词面回贴:text 中含 term 则原样返回;否则找编辑距离 <=1 的 text 子串回贴。

    滑窗长度取 len(term)±1(含等长),取编辑距离最小者;并列取最左,
    同起点并列时优先等长窗口。找不到(含空词面/空文本)返回 None。
    """
    if not term or text is None:
        return None
    if term in text:
        return term
    L = len(term)
    n = len(text)
    best = None  # (距离, 起点, 词面)
    for s in range(n):
        for l in (L, L - 1, L + 1):
            if l <= 0 or s + l > n:
                continue
            cand = text[s:s + l]
            d = _levenshtein(term, cand)
            if d <= 1 and (best is None or d < best[0]):
                best = (d, s, cand)
    return best[2] if best else None


def build_term_category_dict(quads_path):
    """从 quads.jsonl 统计 词面->类别 共现计数。

    返回 {"aspect": {term: Counter({cat: n})}, "opinion": {...}};隐式词面 "_" 跳过。
    """
    dicts = {"aspect": {}, "opinion": {}}
    with open(quads_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for q in row.get("quads", []):
                a = q.get("a", PLACEHOLDER)
                o = q.get("o", PLACEHOLDER)
                c = q.get("c", PLACEHOLDER)
                if a != PLACEHOLDER:
                    dicts["aspect"].setdefault(a, Counter())[c] += 1
                if o != PLACEHOLDER:
                    dicts["opinion"].setdefault(o, Counter())[c] += 1
    return dicts


def category_fallback(term, dicts):
    """查词典多数类;词面未见(或为占位/空)返回 None。

    多数类并列时取计数大者,再并列取类别字典序小者,保证确定性。
    """
    if not term or term == PLACEHOLDER or not dicts:
        return None
    cnt = Counter()
    for role in ("aspect", "opinion"):
        sub = (dicts.get(role) or {}).get(term)
        if sub:
            cnt.update(sub)
    if not cnt:
        return None
    return min(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[0]


def _polarity_fallback(term, dicts):
    """从可选的极性统计里查多数极性;查不到返回 None。

    dicts 可携带 "polarity"/"aspect_polarity"/"opinion_polarity":
    {term: Counter({pol: n})};build_term_category_dict 的产物不含极性,此时恒返回 None。
    """
    if not term or term == PLACEHOLDER or not dicts:
        return None
    cnt = Counter()
    for key in ("polarity", "aspect_polarity", "opinion_polarity"):
        sub = (dicts.get(key) or {}).get(term)
        if sub:
            cnt.update(sub)
    if not cnt:
        return None
    return min(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[0]


def _resolve_offsets(term, text, start, end):
    """占位符 -> (-1,-1);输入偏移与最终词面一致则保留,否则取最左出现位置。"""
    if term == PLACEHOLDER:
        return -1, -1
    if (isinstance(start, int) and isinstance(end, int)
            and 0 <= start < end <= len(text) and text[start:end] == term):
        return start, end
    idx = text.find(term)
    if idx < 0:
        return -1, -1
    return idx, idx + len(term)


def apply_rules(quads, text, dicts, opts=None):
    """确定性后处理:闭集 mask -> 词面回贴 -> 双隐式丢弃 -> (可选)去重。

    (a) Category/Polarity 非法值(不在闭集且非占位)改由词典/多数类兜底:
        类别按 a、o 顺序查 category_fallback;极性查 dicts 中可选的极性统计
        (_polarity_fallback);兜不住置 "_"。
    (b) 非占位词面必须回贴为原文子串(用 snap_back),失败置 "_";偏移同步修正。
    (c) 双隐式:(a,o) 均为 "_" 的整条丢弃。
    (d) 相同 (a,o,c,p) 默认保留多重性不去重;opts={"dedup": True} 才按首见去重。

    返回 list[quad dict],格式同 quads.jsonl 的 quads 元素。
    """
    opts = opts or {}
    out = []
    for q in quads or []:
        a = q.get("a", PLACEHOLDER)
        o = q.get("o", PLACEHOLDER)
        c = q.get("c", PLACEHOLDER)
        p = q.get("p", PLACEHOLDER)

        # (a) 字段闭集 mask
        if c != PLACEHOLDER and c not in CATEGORIES:
            fixed = category_fallback(a, dicts) or category_fallback(o, dicts)
            c = fixed if fixed else PLACEHOLDER
        if p != PLACEHOLDER and p not in POLARITIES:
            fixed = _polarity_fallback(a, dicts) or _polarity_fallback(o, dicts)
            p = fixed if fixed else PLACEHOLDER

        # (b) 非占位词面回贴
        if a != PLACEHOLDER:
            snapped = snap_back(a, text)
            a = snapped if snapped is not None else PLACEHOLDER
        if o != PLACEHOLDER:
            snapped = snap_back(o, text)
            o = snapped if snapped is not None else PLACEHOLDER

        # (c) 双隐式丢弃
        if a == PLACEHOLDER and o == PLACEHOLDER:
            continue

        a_s, a_e = _resolve_offsets(a, text, q.get("a_start"), q.get("a_end"))
        o_s, o_e = _resolve_offsets(o, text, q.get("o_start"), q.get("o_end"))
        out.append({"a": a, "a_start": a_s, "a_end": a_e,
                    "o": o, "o_start": o_s, "o_end": o_e,
                    "c": c, "p": p})

    # (d) 多重性控制
    if opts.get("dedup"):
        seen = set()
        deduped = []
        for q in out:
            key = (q["a"], q["o"], q["c"], q["p"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(q)
        out = deduped
    return out
