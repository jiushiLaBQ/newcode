# -*- coding: utf-8 -*-
"""P0 核心单测: scorer 多重集语义、BMES 转换/解码、提交校验器。
运行: python -m tests.test_p0
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.scorer import score
from src.preprocess.build_tags import spans_to_bmes, bmes_decode, TAG2ID
from src.eval.check_submit import check
from src.decode.pipeline import pair_spans, build_quad
from src.preprocess.dataset import char_to_token_span

Q = lambda a, o, c, p: (a, o, c, p)


def test_scorer_exact_match():
    r = score({1: [Q("价格", "贵", "价格", "负面")]}, {1: [Q("价格", "贵", "价格", "负面")]})
    assert r["S"] == 1 and r["P"] == 1 and r["G"] == 1 and r["f1"] == 1.0


def test_scorer_multiset_duplicate():
    # 同 id 两条完全相同的四元组: 预测 2 条命中 2;只预测 1 条命中 1
    gold = {1: [Q("很好", "很好", "整体", "正面")] * 2}
    assert score(gold, {1: [Q("很好", "很好", "整体", "正面")] * 2})["S"] == 2
    r = score(gold, {1: [Q("很好", "很好", "整体", "正面")]})
    assert r["S"] == 1 and r["P"] == 1 and r["G"] == 2 and abs(r["recall"] - 0.5) < 1e-9
    # 预测 3 条(多报)命中 2
    r = score(gold, {1: [Q("很好", "很好", "整体", "正面")] * 3})
    assert r["S"] == 2 and r["P"] == 3 and abs(r["precision"] - 2 / 3) < 1e-9


def test_scorer_one_field_wrong():
    r = score({1: [Q("价格", "贵", "价格", "负面")]}, {1: [Q("价格", "贵", "价格", "正面")]})
    assert r["S"] == 0 and r["f1"] == 0.0


def test_scorer_placeholder_fields():
    gold = {2: [Q("_", "好贵", "价格", "负面")]}
    assert score(gold, {2: [Q("_", "好贵", "价格", "负面")]})["f1"] == 1.0
    assert score(gold, {2: [Q("价格", "好贵", "价格", "负面")]})["S"] == 0  # "_" != "价格"


def test_scorer_missing_and_extra_id():
    gold = {1: [Q("a", "b", "整体", "正面")], 3: [Q("c", "d", "整体", "正面")]}
    r = score(gold, {1: [Q("a", "b", "整体", "正面")], 2: [Q("x", "y", "整体", "正面")]})
    assert r["S"] == 1 and r["G"] == 2 and r["P"] == 2


def test_scorer_empty_pred():
    r = score({1: [Q("a", "b", "整体", "正面")]}, {})
    assert r["S"] == 0 and r["P"] == 0 and r["f1"] == 0.0


def test_bmes_roundtrip():
    text = "价格有点贵了"
    tags, dropped = spans_to_bmes(len(text), [(0, 2)], "A")
    assert dropped == 0 and tags[0] == "B-A" and tags[1] == "E-A" and tags[2] == "O"
    tags, _ = spans_to_bmes(len(text), [(0, 2)], "O")
    assert bmes_decode([TAG2ID[t] for t in tags]) == [(0, 2, "O")]
    # 单字 span
    tags, _ = spans_to_bmes(3, [(1, 2)], "A")
    assert tags == ["O", "S-A", "O"]
    # 同 span 去重
    tags, dropped = spans_to_bmes(3, [(0, 2), (0, 2)], "A")
    assert dropped == 0 and tags[0] == "B-A"
    # 交叠保长者
    tags, dropped = spans_to_bmes(4, [(0, 3), (1, 4)], "A")
    assert dropped == 1 and tags == ["B-A", "M-A", "E-A", "O"]


def test_bmes_decode_illegal():
    # B- 后面直接 O: 退化单字
    ids = [TAG2ID["B-A"], TAG2ID["O"]]
    assert bmes_decode(ids) == [(0, 1, "A")]


def test_bmes_zero_span_guard():
    # 零长/越界 span 不得污染标签(旧 bug: e-1 负索引写到全文末尾)
    tags, dropped = spans_to_bmes(5, [(2, 2), (0, 0)], "A")
    assert tags == ["O"] * 5 and dropped == 2
    tags, dropped = spans_to_bmes(3, [(1, 3), (2, 4)], "A")  # 越界者丢弃,保留 2 字符 span
    assert dropped == 1 and tags == ["O", "B-A", "E-A"]


def test_pair_spans_implicit_competition():
    # nearest: (None,o) 置信度更高时观点走隐式属性;未用属性补 (a,None)
    a, o1, o2 = (2, 4), (0, 2), (6, 8)
    scores = {(-1, 0): 0.9, (0, 0): 0.3, (0, 1): 0.8, (-1, 1): 0.2, (0, -1): 0.1}
    out = pair_spans([a], [o1, o2], "nearest", scores)
    assert (None, o1) in out and (a, o2) in out
    assert (a, None) not in out  # 属性已被 o2 使用,无剩余
    # 上一行修正:属性被用过则不补
    out2 = pair_spans([a], [o2], "nearest", {(0, 1): 0.8, (-1, 1): 0.2})
    assert out2 == [(a, o2)]
    # 无属性时全隐式
    assert pair_spans([], [o1, o2], "nearest") == [(None, o1), (None, o2)]


def test_pair_spans_greedy_leftover_aspect():
    # greedy: 落选属性必须补 (a,None)(旧 bug: 静默丢弃)
    a1, a2, o = (0, 2), (5, 7), (3, 5)
    scores = {(0, 0): 0.9, (1, 0): 0.2, (-1, 0): 0.1, (0, -1): 0.1, (1, -1): 0.1}
    out = pair_spans([a1, a2], [o], "greedy", scores, threshold=0.5)
    assert (a1, o) in out and (a2, None) in out
    # 隐式观点候选高分被接受;落选观点按契约强制补 (None,o)
    scores2 = {(0, 0): 0.2, (-1, 0): 0.1, (0, -1): 0.95, (1, -1): 0.1}
    out2 = pair_spans([a1], [o], "greedy", scores2, threshold=0.5)
    assert (a1, None) in out2 and (None, o) in out2 and len(out2) == 2


def test_build_quad_underscore_span():
    # 真实抽取到词面 "_" 的 span 不是隐式占位符,不应被双隐式规则误删
    q = build_quad("a_b", (1, 2), None, 0, 11)
    assert q is not None and q["a"] == "_" and q["o"] == "_"
    assert q["a_start"] == 1  # 词面占位符与偏移 -1 的隐式可以共存于词面 "_" 但偏移不同
    assert build_quad("abc", None, None, 0, 11) is None  # 真双隐式仍禁止


def test_char_to_token_span_truncation():
    # 部分截断必须整体丢弃(None),而非返回缩短 span
    tok2char = [None, 0, 1, 2, None]  # char 3 被截断
    assert char_to_token_span(tok2char, 0, 2) == (1, 2)
    assert char_to_token_span(tok2char, 1, 4) is None  # char 3 不在
    assert char_to_token_span(tok2char, 3, 4) is None


def _write(tmp, name, content_bytes):
    p = Path(tmp) / name
    p.write_bytes(content_bytes)
    return p


def test_check_submit():
    test_csv = "id,Reviews\n1,价格有点贵了\n2,很好用\n"
    with tempfile.TemporaryDirectory() as tmp:
        tp = _write(tmp, "t.csv", test_csv.encode("utf-8"))
        ok = _write(tmp, "ok.csv", "1,价格,有点贵,价格,负面\n2,_,很好用,整体,正面\n".encode("utf-8"))
        e, w = check(ok, tp)
        assert e == [], e

        bom = _write(tmp, "bom.csv", b"\xef\xbb\xbf1,_,_,_,_")
        e, _ = check(bom, tp)
        assert any("BOM" in x for x in e)

        crlf = _write(tmp, "crlf.csv", b"1,_,_,_,_\r\n2,_,_,_,_")
        e, _ = check(crlf, tp)
        assert any("0x0D" in x for x in e)

        header = _write(tmp, "h.csv", b"id,AspectTerm,OpinionTerm,Category,Polarity\n1,_,_,_,_")
        e, _ = check(header, tp)
        assert any("表头" in x for x in e)

        missing = _write(tmp, "m.csv", b"1,_,_,_,_")
        e, _ = check(missing, tp)
        assert any("缺失" in x for x in e)

        badcat = _write(tmp, "bc.csv", "1,_,_,物流,负面\n2,_,_,_,_\n".encode("utf-8"))
        e, _ = check(badcat, tp)
        # 物流是合法类别;改用真非法值
        badcat2 = _write(tmp, "bc2.csv", "1,_,_,不存在类,负面\n2,_,_,_,_\n".encode("utf-8"))
        e, _ = check(badcat2, tp)
        assert any("Category 非法" in x for x in e)

        halluc = _write(tmp, "hl.csv", "1,香水,有点贵,价格,负面\n2,_,_,_,_\n".encode("utf-8"))
        e, _ = check(halluc, tp)
        assert any("子串" in x for x in e)

        fullwidth = _write(tmp, "fw.csv", "1,＿,_,_,_\n2,_,_,_,_\n".encode("utf-8"))
        e, _ = check(fullwidth, tp)
        assert any("全角" in x for x in e)

        floatid = _write(tmp, "fi.csv", "1.0,_,_,_,_\n2,_,_,_,_\n".encode("utf-8"))
        e, _ = check(floatid, tp)
        assert any("纯数字" in x for x in e)


if __name__ == "__main__":
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
