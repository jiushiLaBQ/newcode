# -*- coding: utf-8 -*-
"""四元组 -> 字符级 BMES 双通道标签(属性通道 / 观点通道独立,互不冲突)。

标签集(每通道 9 个):
  B-A M-A E-A S-A  (属性 span 起/中/尾/单字)
  B-O M-O E-O S-O  (观点 span)
  O                (通道外)

规则:
  - 同通道内完全相同的 span 去重(同 span 属多个四元组只标一次)。
  - 同通道内 span 交叠且不等同(嵌套/交叉)时保留最长者,丢弃数记入 report。
  - 隐式字段(词面 "_")不产生标签,由专门的隐式头处理。
"""
from collections import Counter

TAGS = ["O", "B-A", "M-A", "E-A", "S-A", "B-O", "M-O", "E-O", "S-O"]
TAG2ID = {t: i for i, t in enumerate(TAGS)}


def spans_to_bmes(n, spans, kind):
    """spans: [(start,end),...] 半开区间;kind: 'A' 或 'O' -> (tags, dropped)。
    非法 span(e<=s 或越界)与交叠中较短者计入 dropped;交叠消解保留最长者。"""
    tags = ["O"] * n
    uniq = set(spans)
    valid = [(s, e) for s, e in uniq if 0 <= s < e <= n]
    dropped = len(uniq) - len(valid)
    kept = []
    for s, e in sorted(valid, key=lambda x: (-(x[1] - x[0]), x[0])):  # 长者优先
        if any(s < ke and ks < e for ks, ke in kept):  # 与已保留者交叠
            dropped += 1
            continue
        kept.append((s, e))
    for s, e in kept:
        if e - s == 1:
            tags[s] = f"S-{kind}"
        else:
            tags[s] = f"B-{kind}"
            for i in range(s + 1, e - 1):
                tags[i] = f"M-{kind}"
            tags[e - 1] = f"E-{kind}"
    return tags, dropped


def build_channels(text, quads):
    """quads: offset_check.py 的四元组结构 -> dict(含两通道标签 id 序列与丢弃统计)"""
    n = len(text)
    a_spans = [(q["a_start"], q["a_end"]) for q in quads if q["a_start"] >= 0]
    o_spans = [(q["o_start"], q["o_end"]) for q in quads if q["o_start"] >= 0]
    a_tags, drop_a = spans_to_bmes(n, a_spans, "A")
    o_tags, drop_o = spans_to_bmes(n, o_spans, "O")
    return {
        "aspect_tags": [TAG2ID[t] for t in a_tags],
        "opinion_tags": [TAG2ID[t] for t in o_tags],
        "dropped_aspect_spans": drop_a,
        "dropped_opinion_spans": drop_o,
    }


def bmes_decode(tag_ids):
    """标签 id 序列 -> [(start,end,kind)]。非法转移按单字 span 容错。"""
    tags = [TAGS[t] for t in tag_ids]
    spans = []
    i = 0
    while i < len(tags):
        t = tags[i]
        if t in ("B-A", "B-O"):
            kind = t[-1]
            j = i + 1
            while j < len(tags) and tags[j] == f"M-{kind}":
                j += 1
            if j < len(tags) and tags[j] == f"E-{kind}":
                spans.append((i, j + 1, kind))
                i = j + 1
            else:  # 非法收尾,退化为单字
                spans.append((i, i + 1, kind))
                i += 1
        elif t in ("S-A", "S-O"):
            spans.append((i, i + 1, t[-1]))
            i += 1
        else:
            i += 1
    return spans
