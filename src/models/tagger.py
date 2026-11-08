# -*- coding: utf-8 -*-
"""A3 模型:共享 encoder + 双通道 BiLSTM-CRF(属性/观点)+ 极性/类别分类头。

隐式建模:
  - 属性/观点 span 缺失(词面 "_")时,分类头用 [CLS] 向量充当该侧表示。
  - 类别头输入 = 属性表示(显式)或观点表示(属性隐式);双隐式不建模(训练分布为 0,解码层禁止)。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
from transformers import AutoModel

N_TAGS = 9
N_POL = 3
N_CAT = 13


def span_rep(h, span):
    """h: [T, D];span: (s,e) 闭区间 token 索引,None -> [CLS] 向量。"""
    if span is None:
        return h[0]
    return h[span[0]:span[1] + 1].mean(dim=0)


class QuadTagger(nn.Module):
    def __init__(self, model_name, lstm_hidden=256, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.a_lstm = nn.LSTM(hidden, lstm_hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.o_lstm = nn.LSTM(hidden, lstm_hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.a_emit = nn.Linear(2 * lstm_hidden, N_TAGS)
        self.o_emit = nn.Linear(2 * lstm_hidden, N_TAGS)
        self.a_crf = CRF(N_TAGS, batch_first=True)
        self.o_crf = CRF(N_TAGS, batch_first=True)
        d = hidden  # 分类头用 encoder 表示
        self.pol_head = nn.Sequential(
            nn.Linear(2 * d, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, N_POL))
        self.cat_head = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, N_CAT))

    # ---------- 训练 ----------
    def forward(self, input_ids, attention_mask, aspect_tag_ids=None, opinion_tag_ids=None,
                pairs=None, loss_weights=None):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.bool()
        # BiLSTM 用 pack 屏蔽 pad 位置,保证输出与批内补齐无关(可复现性)
        lengths = attention_mask.sum(dim=1).cpu()
        packed = torch.nn.utils.rnn.pack_padded_sequence(self.dropout(h), lengths,
                                                         batch_first=True, enforce_sorted=False)
        lstm_a, _ = self.a_lstm(packed)
        lstm_o, _ = self.o_lstm(packed)
        h_a, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_a, batch_first=True, total_length=h.size(1))
        h_o, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_o, batch_first=True, total_length=h.size(1))
        ea = self.a_emit(h_a)
        eo = self.o_emit(h_o)

        out = {}
        if aspect_tag_ids is not None and self.training:
            loss_a = -self.a_crf(ea, aspect_tag_ids, mask, reduction="mean")
            loss_o = -self.o_crf(eo, opinion_tag_ids, mask, reduction="mean")
            pol_loss = h.new_zeros(())
            cat_loss = h.new_zeros(())
            n_pairs = 0
            for i, plist in enumerate(pairs):
                for pr in plist:
                    a_rep = span_rep(h[i], pr["a"])
                    o_rep = span_rep(h[i], pr["o"])
                    pol_logit = self.pol_head(torch.cat([a_rep, o_rep], dim=-1)).unsqueeze(0)
                    pol_loss = pol_loss + F.cross_entropy(pol_logit, torch.tensor([pr["p"]], device=h.device))
                    cat_in = a_rep if pr["a"] is not None else o_rep
                    cat_logit = self.cat_head(cat_in).unsqueeze(0)
                    cat_loss = cat_loss + F.cross_entropy(cat_logit, torch.tensor([pr["c"]], device=h.device))
                    n_pairs += 1
            if n_pairs:
                pol_loss = pol_loss / n_pairs
                cat_loss = cat_loss / n_pairs
            w = loss_weights or {}
            loss = (w.get("tag_a", 1.0) * loss_a + w.get("tag_o", 1.0) * loss_o
                    + w.get("pol", 1.0) * pol_loss + w.get("cat", 1.0) * cat_loss)
            out = {"loss": loss, "loss_a": loss_a.detach(), "loss_o": loss_o.detach(),
                   "loss_pol": pol_loss.detach(), "loss_cat": cat_loss.detach()}
            return out

        # 推理:返回解码标签 + 隐状态(供 decode/pipeline 用)
        with torch.no_grad():
            dec_a = self.a_crf.decode(ea, mask)
            dec_o = self.o_crf.decode(eo, mask)
        return {"aspect_tag_ids": dec_a, "opinion_tag_ids": dec_o, "h": h.detach()}

    # ---------- 推理辅助 ----------
    @torch.no_grad()
    def classify_pair(self, h, a_tok, o_tok):
        """h: [T,D];a_tok/o_tok: token span 或 None。返回 (pol_logits[N_POL], cat_logits[N_CAT])。"""
        a_rep = span_rep(h, a_tok)
        o_rep = span_rep(h, o_tok)
        pol = self.pol_head(torch.cat([a_rep, o_rep], dim=-1))
        cat_in = a_rep if a_tok is not None else o_rep
        if cat_in is None:  # 双隐式(不应出现),用 CLS
            cat_in = h[0]
        cat = self.cat_head(cat_in)
        return pol, cat
