# -*- coding: utf-8 -*-
"""A3 解码管线:token 标签 -> 字符 span -> 属性/观点配对 -> 四元组。

配对策略:
  nearest —— 每个观点在 (最近属性,o) 与 (None,o) 隐式候选间按置信度选优(可重复服务多个观点);
             未被选中的属性 -> (a,_,c,p)。无双隐式(训练分布为 0,硬禁止)。
  greedy  —— 全候选(含隐式)按置信度(极性最大 softmax × 类别最大 softmax)贪心、阈值截断;
             收尾强制补剩余观点 -> (None,o)、未用属性 -> (a,None)。
词面恒为原文子串(从 text 切片),不可能幻觉。
"""
import torch
from torch.utils.data import DataLoader

from src.preprocess.build_tags import bmes_decode
from src.preprocess.dataset import collate, char_to_token_span, CATEGORIES, POLARITIES, PLACEHOLDER

try:
    from src.postprocess.rules import snap_back  # noqa: F401  (回贴修复,抽取式用不上但保留接口)
except ImportError:
    snap_back = None


def token_tags_to_char_tags(tok2char, tag_seq, n_chars):
    """token 级标签 -> 字符级标签(每字符取第一个 token 的标签)。"""
    out = [0] * n_chars
    seen = set()
    for t, c in enumerate(tok2char):
        if c is None or c in seen or t >= len(tag_seq):
            continue
        seen.add(c)
        out[c] = tag_seq[t]
    return out


def pair_spans(a_spans, o_spans, mode="nearest", scores=None, threshold=0.5):
    """返回 [(a_span|None, o_span|None), ...]。隐式候选 (None,o)/(a,None) 全程参与。

    scores: dict[(ai, oi)] -> 置信度;ai=-1 表示属性隐式,oi=-1 表示观点隐式。
      nearest —— 每个观点在 (最近属性,o) 与 (None,o) 之间取置信度高者(无 scores 时偏好显式);
                  未被任何观点选中的属性 -> (a,None)。
      greedy  —— 全候选(含隐式)按分贪心、阈值截断,观点至多用一次、属性隐式候选至多用一次;
                  收尾强制补:剩余观点 -> (None,o),未用属性 -> (a,None)。
    """
    S = scores or {}

    def conf(ai, oi):
        if (ai, oi) in S:
            return S[(ai, oi)]
        return 1.0 if ai >= 0 else 0.5  # 无分数时显式优先

    pairs = []
    used_a = set()
    if mode == "greedy" and scores:
        cand = sorted(S.items(), key=lambda kv: -kv[1])
        used_o = set()
        used_a_imp = set()
        for (ai, oi), s in cand:
            if s < threshold:
                break
            if oi >= 0:  # (ai|隐式, o)
                if oi in used_o:
                    continue
                used_o.add(oi)
                if ai >= 0:
                    used_a.add(ai)
                pairs.append((a_spans[ai] if ai >= 0 else None, o_spans[oi]))
            else:  # (a, None) 隐式观点候选
                if ai in used_a_imp or ai in used_a:
                    continue
                used_a_imp.add(ai)
                used_a.add(ai)
                pairs.append((a_spans[ai], None))
        for oi, o in enumerate(o_spans):
            if oi not in used_o:
                pairs.append((None, o))
        for ai, a in enumerate(a_spans):
            if ai not in used_a:
                pairs.append((a, None))
    else:  # nearest
        for oi, o in enumerate(o_spans):
            if not a_spans:
                pairs.append((None, o))
                continue
            best = min(range(len(a_spans)),
                       key=lambda i: abs((a_spans[i][0] + a_spans[i][1]) / 2 - (o[0] + o[1]) / 2))
            if conf(-1, oi) > conf(best, oi):  # 隐式属性候选胜出
                pairs.append((None, o))
            else:
                pairs.append((a_spans[best], o))
                used_a.add(best)
        for ai, a in enumerate(a_spans):
            if ai not in used_a:
                pairs.append((a, None))
    # 去掉双 None(不可能出现)
    return [(a, o) for a, o in pairs if a is not None or o is not None]


def build_quad(text, a_span, o_span, pol_id, cat_id):
    if a_span is None and o_span is None:
        return None  # 双隐式硬禁止(按 span 判定,不看词面)
    a = text[a_span[0]:a_span[1]] if a_span else PLACEHOLDER
    o = text[o_span[0]:o_span[1]] if o_span else PLACEHOLDER
    q = {"a": a, "a_start": a_span[0] if a_span else -1, "a_end": a_span[1] if a_span else -1,
         "o": o, "o_start": o_span[0] if o_span else -1, "o_end": o_span[1] if o_span else -1,
         "c": CATEGORIES[cat_id], "p": POLARITIES[pol_id]}
    return q


@torch.no_grad()
def predict(model, ds, device, pair_mode="nearest", threshold=0.5, batch_size=8, pad_id=0):
    """ds: QuadDataset -> [{id, text, quads}, ...](与 quads.jsonl 同构,可直接喂 scorer)"""
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=lambda b: collate(b, pad_id), shuffle=False)
    records = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        h = out["h"]
        for i, meta in enumerate(batch["meta"]):
            text = meta["text"]
            tok2char = meta["tok2char"]
            n_chars = len(text)
            ca = token_tags_to_char_tags(tok2char, out["aspect_tag_ids"][i], n_chars)
            co = token_tags_to_char_tags(tok2char, out["opinion_tag_ids"][i], n_chars)
            a_spans = [(s, e) for s, e, k in bmes_decode(ca) if k == "A"]
            o_spans = [(s, e) for s, e, k in bmes_decode(co) if k == "O"]
            a_toks = [char_to_token_span(tok2char, s, e) for s, e in a_spans]
            o_toks = [char_to_token_span(tok2char, s, e) for s, e in o_spans]

            # 全候选打分:显式组合 + 隐式观点候选 (None,o) + 隐式观点候选 (a,None)
            scores = {}
            for ai in range(len(a_spans)):
                for oi in range(len(o_spans)):
                    pol, cat = model.classify_pair(h[i], a_toks[ai], o_toks[oi])
                    scores[(ai, oi)] = float(pol.softmax(-1).max() * cat.softmax(-1).max())
            for oi in range(len(o_spans)):
                pol, cat = model.classify_pair(h[i], None, o_toks[oi])
                scores[(-1, oi)] = float(pol.softmax(-1).max() * cat.softmax(-1).max())
            for ai in range(len(a_spans)):
                pol, cat = model.classify_pair(h[i], a_toks[ai], None)
                scores[(ai, -1)] = float(pol.softmax(-1).max() * cat.softmax(-1).max())
            combos = pair_spans(a_spans, o_spans, pair_mode, scores, threshold)

            quads = []
            for a_span, o_span in combos:
                ai = a_spans.index(a_span) if a_span in a_spans else None
                oi = o_spans.index(o_span) if o_span in o_spans else None
                a_tok = a_toks[ai] if ai is not None else None
                o_tok = o_toks[oi] if oi is not None else None
                pol, cat = model.classify_pair(h[i], a_tok, o_tok)
                q = build_quad(text, a_span, o_span, int(pol.argmax(-1)), int(cat.argmax(-1)))
                if q is not None:
                    quads.append(q)
            records.append({"id": meta["id"], "text": text, "quads": quads})
    return records


def records_to_pred_dict(records):
    out = {}
    for r in records:
        out[r["id"]] = [(q["a"], q["o"], q["c"], q["p"]) for q in r["quads"]]
    return out


def write_records(path, records):
    import json
    with open(path, "w", encoding="utf-8", newline="") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
