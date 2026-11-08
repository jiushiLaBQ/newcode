# -*- coding: utf-8 -*-
"""规则/后处理/集成库单测:snap_back、词典构建、apply_rules、ensemble、规则基线冒烟。
运行: python -I tests/test_rules.py
"""
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.postprocess.rules import (
    snap_back, build_term_category_dict, category_fallback, apply_rules,
)
from src.postprocess.ensemble import load_pred_jsonl, merge
from src.models.rules_baseline import build_stats, predict_text

ROOT = Path(__file__).resolve().parents[1]


# ---------------- snap_back ----------------

def test_snap_back_hit():
    # term 是 text 子串 -> 原样返回
    assert snap_back("很好用", "这个很好用的") == "很好用"
    assert snap_back("好", "好好好") == "好"


def test_snap_back_snap():
    # 编辑距离 1 的回贴:替换
    assert snap_back("包装合", "这个包装盒不错") == "包装盒"
    # 词面多一字(回贴删除一字)
    assert snap_back("很好用用", "真的很好用") == "很好用"
    # 词面少一字(回贴补一字)
    assert snap_back("包装盒错", "包装盒不错") == "包装盒不"
    # 并列取最左:两个距离 1 的候选取左边那个
    assert snap_back("bc", "ac dc") == "ac"


def test_snap_back_fail():
    # 编辑距离 >1 -> None
    assert snap_back("完全不同的词", "面膜很好用") is None
    assert snap_back("", "任意文本") is None
    assert snap_back("词面", "") is None


# ---------------- 词典构建 ----------------

def _write_quads_jsonl(tmpdir, rows):
    p = Path(tmpdir) / "quads.jsonl"
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return p


def test_build_term_category_dict():
    rows = [
        {"id": 1, "text": "面膜很好用", "quads": [
            {"a": "面膜", "o": "好用", "c": "功效", "p": "正面"},
            {"a": "_", "o": "好用", "c": "整体", "p": "正面"}]},
        {"id": 2, "text": "超值", "quads": [
            {"a": "_", "o": "好用", "c": "功效", "p": "正面"}]},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        p = _write_quads_jsonl(tmp, rows)
        d = build_term_category_dict(p)
    assert d["aspect"]["面膜"] == Counter({"功效": 1})
    assert d["opinion"]["好用"] == Counter({"功效": 2, "整体": 1})
    assert "_" not in d["aspect"] and "_" not in d["opinion"]
    # 词典查多数类 / 未见词返回 None
    assert category_fallback("好用", d) == "功效"
    assert category_fallback("面膜", d) == "功效"
    assert category_fallback("没见过的词", d) is None
    assert category_fallback("_", d) is None


# ---------------- apply_rules ----------------

def test_apply_rules_closed_set_mask():
    text = "面膜很好用"
    dicts = {"aspect": {"面膜": Counter({"功效": 2})},
             "opinion": {"好用": Counter({"功效": 3, "整体": 1})}}
    quads = [
        # c 非法 -> 词典兜底(按 a 查到"功效")
        {"a": "面膜", "o": "好用", "c": "不存在的类", "p": "正面"},
        # c 非法且 a 未见于词典 -> 按 o 查多数类
        {"a": "很好", "o": "好用", "c": "坏类", "p": "正面"},
        # c/p 均非法且词典兜不住 -> "_"(p 无极性统计,恒兜不住)
        {"a": "面膜", "o": "好用", "c": "坏类x", "p": "坏极性"},
    ]
    out = apply_rules(quads, text, dicts)
    assert out[0]["c"] == "功效" and out[0]["p"] == "正面"
    assert out[1]["c"] == "功效"
    assert out[2]["c"] == "功效" and out[2]["p"] == "_"
    # 合法占位/合法值不被改动
    out2 = apply_rules([{"a": "_", "o": "好用", "c": "_", "p": "中性"}], text, dicts)
    assert out2[0]["c"] == "_" and out2[0]["p"] == "中性"
    # dicts 带极性统计时,非法 p 走多数票
    dicts2 = dict(dicts)
    dicts2["opinion_polarity"] = {"好用": Counter({"负面": 2, "正面": 1})}
    out3 = apply_rules([{"a": "_", "o": "好用", "c": "整体", "p": "坏极性"}], text, dicts2)
    assert out3[0]["p"] == "负面"


def test_apply_rules_double_implicit_dropped():
    text = "面膜很好用"
    dicts = {}
    quads = [
        {"a": "_", "o": "_", "c": "整体", "p": "正面"},          # 双隐式 -> 丢
        {"a": "包装盒质量差劲啊", "o": "_", "c": "包装", "p": "负面"},  # 回贴失败 -> a 置 "_"
        {"a": "_", "o": "好用", "c": "整体", "p": "正面"},          # 保留
    ]
    out = apply_rules(quads, text, dicts)
    assert len(out) == 1 and out[0]["o"] == "好用" and out[0]["a"] == "_"


def test_apply_rules_snap_and_offsets():
    text = "这个包装盒不错"
    q = {"a": "包装合", "a_start": 2, "a_end": 5, "o": "_", "o_start": -1, "o_end": -1,
         "c": "包装", "p": "正面"}
    out = apply_rules([q], text, {})
    assert out[0]["a"] == "包装盒" and out[0]["a_start"] == 2 and out[0]["a_end"] == 5
    # 输入偏移无效 -> 按最左出现位置重算
    q2 = {"a": "不错", "a_start": 0, "a_end": 2, "o": "_", "c": "整体", "p": "正面"}
    out2 = apply_rules([q2], text, {})
    assert out2[0]["a_start"] == 5 and out2[0]["a_end"] == 7


def test_apply_rules_multiset_default_keep():
    text = "面膜很好用"
    q = {"a": "面膜", "o": "好用", "c": "功效", "p": "正面"}
    out = apply_rules([q, dict(q)], text, {})
    assert len(out) == 2  # 默认保留多重性
    out2 = apply_rules([q, dict(q)], text, {}, {"dedup": True})
    assert len(out2) == 1


# ---------------- ensemble ----------------

def _q(a, o, c, p):
    return {"a": a, "a_start": -1, "a_end": -1, "o": o, "o_start": -1, "o_end": -1,
            "c": c, "p": p}


def test_ensemble_vote():
    qA, qB, qC = _q("面膜", "好用", "功效", "正面"), _q("_", "超值", "价格", "正面"), \
        _q("_", "太差", "整体", "负面")
    p1 = {1: {"text": "t1", "quads": [qA, qA, qB]}}   # qA 同文件重复只算 1 票
    p2 = {1: {"text": "t1", "quads": [qA, qC]}}
    m = merge([p1, p2], mode="vote", min_votes=2)
    assert [(q["a"], q["o"]) for q in m[1]["quads"]] == [("面膜", "好用")]  # 只收 qA,1 次
    m1 = merge([p1, p2], mode="vote", min_votes=1)
    assert len(m1[1]["quads"]) == 3  # 三个不同四元组各收 1 次
    # min-votes 超过文件数 -> 全部滤掉,但 id 仍在
    m0 = merge([p1, p2], mode="vote", min_votes=3)
    assert m0[1]["quads"] == [] and m0[1]["text"] == "t1"


def test_ensemble_union():
    qA, qB, qC = _q("面膜", "好用", "功效", "正面"), _q("_", "超值", "价格", "正面"), \
        _q("_", "太差", "整体", "负面")
    p1 = {1: {"text": "t1", "quads": [qA, qA, qB]}}   # qA 真重复 2 次
    p2 = {1: {"text": "t1", "quads": [qA, qC]}}
    m = merge([p1, p2], mode="union")
    got = [(q["a"], q["o"]) for q in m[1]["quads"]]
    assert got.count(("面膜", "好用")) == 2      # 多重集并集 = max(2,1)
    assert got.count(("_", "超值")) == 1 and got.count(("_", "太差")) == 1
    assert len(m[1]["quads"]) == 4
    # 幂等:两份相同预测 union 仍等于原预测
    m2 = merge([p1, p1], mode="union")
    assert [(q["a"], q["o"]) for q in m2[1]["quads"]] == [("面膜", "好用")] * 2 + [("_", "超值")]


def test_ensemble_cli_smoke():
    qA = _q("面膜", "好用", "功效", "正面")
    rows = [{"id": 7, "text": "面膜很好用", "quads": [qA]}]
    with tempfile.TemporaryDirectory() as tmp:
        f1 = Path(tmp) / "f1.jsonl"
        f2 = Path(tmp) / "f2.jsonl"
        out = Path(tmp) / "merged.jsonl"
        for p in (f1, f2):
            p.write_text(json.dumps(rows[0], ensure_ascii=False) + "\n", encoding="utf-8")
        r = subprocess.run(
            [sys.executable, "-I", str(ROOT / "src" / "postprocess" / "ensemble.py"),
             "--preds", str(f1), str(f2), "--out", str(out), "--mode", "vote", "--min-votes", "2"],
            capture_output=True, text=True, encoding="utf-8", timeout=120)
        assert r.returncode == 0, r.stderr
        got = load_pred_jsonl(out)
        assert got[7]["quads"][0]["a"] == "面膜" and got[7]["quads"][0]["a_start"] == 0


# ---------------- 规则基线冒烟 ----------------

def test_rules_baseline_smoke():
    rows = [
        {"id": 1, "text": "面膜很好用", "quads": [
            {"a": "面膜", "a_start": 0, "a_end": 2, "o": "很好用", "o_start": 2, "o_end": 5,
             "c": "功效", "p": "正面"}]},
        {"id": 2, "text": "超值", "quads": [
            {"a": "_", "a_start": -1, "a_end": -1, "o": "超值", "o_start": 0, "o_end": 2,
             "c": "价格", "p": "正面"}]},
    ]
    stats = build_stats(rows)
    quads = predict_text("这个面膜很好用", stats)
    assert len(quads) == 1
    got = quads[0]
    assert (got["a"], got["o"], got["c"], got["p"]) == ("面膜", "很好用", "功效", "正面")
    assert got["a_start"] == 2 and got["o_end"] == 7
    # 无属性配上的观点 -> 隐式属性
    q2 = predict_text("真超值", stats)
    assert len(q2) == 1 and q2[0]["a"] == "_" and q2[0]["o"] == "超值"
    assert (q2[0]["c"], q2[0]["p"]) == ("价格", "正面")
    # 词表外文本 -> 空
    assert predict_text("完全无关的句子", stats) == []


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
