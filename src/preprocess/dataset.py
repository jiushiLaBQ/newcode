# -*- coding: utf-8 -*-
"""A3 数据层:字符级文本 -> token 序列 + 双通道 BMES 标签 + 分类监督对。

对齐策略(不依赖 fast tokenizer):逐字符 encode,1 字符 -> 1..n token,
tok2char 记录每个 token 属于哪个字符,special token([CLS]/[SEP])记 None。
标签投影:token 的标签 = 其所属字符的标签;special 记 "O"(id 0)。
分类监督对:每条金标四元组一条,span 在 token 空间取 (首 token, 末 token) 闭区间;
隐式字段 span=None,表示时用 [CLS] 向量。
"""
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.preprocess.build_tags import build_channels, TAG2ID

ROOT = Path(__file__).resolve().parents[2]
PLACEHOLDER = "_"
CATEGORIES = ["包装", "成分", "尺寸", "服务", "功效", "价格", "气味", "使用体验", "物流", "新鲜度", "真伪", "整体", "其他"]
POLARITIES = ["正面", "中性", "负面"]
CAT2ID = {c: i for i, c in enumerate(CATEGORIES)}
POL2ID = {p: i for i, p in enumerate(POLARITIES)}


def load_records(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def encode_text(tokenizer, text, max_len):
    """逐字符编码。返回 input_ids, tok2char(截断后)。"""
    char_ids = []
    tok2char = []
    for ci, ch in enumerate(text):
        ids = tokenizer.encode(ch, add_special_tokens=False)
        if not ids:  # 空字符串防御
            ids = [tokenizer.unk_token_id]
        for t in ids:
            char_ids.append(t)
            tok2char.append(ci)
    # [CLS] + chars + [SEP],留 2 个 special 位
    budget = max_len - 2
    char_ids = char_ids[:budget]
    tok2char = tok2char[:budget]
    cls_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
    sep_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
    input_ids = [cls_id] + char_ids + [sep_id]
    tok2char = [None] + tok2char + [None]
    return input_ids, tok2char


def char_to_token_span(tok2char, start, end):
    """字符 span [start,end) -> token span (ts,te) 闭区间;被截断(含部分截断)返回 None。"""
    ts = te = None
    for t, c in enumerate(tok2char):
        if c is None:
            continue
        if c >= start and ts is None:
            ts = t
        if c < end:
            te = t
    if ts is None or te is None or te < ts:
        return None
    if tok2char[ts] != start or tok2char[te] != end - 1:
        return None  # 部分被截断,整体丢弃,不做缩短监督
    return ts, te


class QuadDataset(Dataset):
    """records: [{"id","text","quads"?}]。quads 结构同 data/processed/quads.jsonl。"""

    def __init__(self, records, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.items = []
        for r in records:
            self.items.append(self._build(r))

    def _build(self, rec):
        text = rec["text"]
        quads = rec.get("quads", [])
        input_ids, tok2char = encode_text(self.tokenizer, text, self.max_len)
        n_tok = len(input_ids)

        tags = build_channels(text, quads)  # 字符级
        aspect_tag_ids = [0] * n_tok
        opinion_tag_ids = [0] * n_tok
        for t, c in enumerate(tok2char):
            if c is None:
                continue
            aspect_tag_ids[t] = tags["aspect_tags"][c]
            opinion_tag_ids[t] = tags["opinion_tags"][c]

        pairs = []
        for q in quads:
            a_span = char_to_token_span(tok2char, q["a_start"], q["a_end"]) if q["a_start"] >= 0 else None
            o_span = char_to_token_span(tok2char, q["o_start"], q["o_end"]) if q["o_start"] >= 0 else None
            if q["a_start"] >= 0 and a_span is None:
                continue  # 被截断,丢弃
            if q["o_start"] >= 0 and o_span is None:
                continue
            pairs.append({
                "a": a_span, "o": o_span,
                "p": POL2ID[q["p"]], "c": CAT2ID[q["c"]],
            })

        return {
            "id": rec["id"],
            "text": text,
            "input_ids": input_ids,
            "attention_mask": [1] * n_tok,
            "aspect_tag_ids": aspect_tag_ids,
            "opinion_tag_ids": opinion_tag_ids,
            "pairs": pairs,
            "tok2char": tok2char,
        }

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch, pad_id=0):
    n = max(len(b["input_ids"]) for b in batch)
    input_ids, attention_mask = [], []
    aspect_tag_ids, opinion_tag_ids = [], []
    for b in batch:
        k = len(b["input_ids"])
        pad = n - k
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        attention_mask.append(b["attention_mask"] + [0] * pad)
        aspect_tag_ids.append(b["aspect_tag_ids"] + [0] * pad)
        opinion_tag_ids.append(b["opinion_tag_ids"] + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "aspect_tag_ids": torch.tensor(aspect_tag_ids, dtype=torch.long),
        "opinion_tag_ids": torch.tensor(opinion_tag_ids, dtype=torch.long),
        "pairs": [b["pairs"] for b in batch],
        "meta": [{"id": b["id"], "text": b["text"], "tok2char": b["tok2char"]} for b in batch],
    }
