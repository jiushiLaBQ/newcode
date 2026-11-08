# -*- coding: utf-8 -*-
"""P0/P1 集成测试(不起模型):真实数据上的 build_tags / scorer / check_submit / folds / run_cv。

运行: python -I tests/test_pipeline.py
说明: python -I 下 sys.path 无脚本目录与当前目录,故显式插入仓库根目录;
      输出统一 UTF-8(stdout 在 Windows 默认 GBK,必须 reconfigure,-I 下 PYTHONUTF8 无效)。
"""
import csv
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocess.build_tags import build_channels, bmes_decode
from src.eval.scorer import load_gold_jsonl, score
from src.eval.check_submit import check

ROOT = Path(__file__).resolve().parents[1]
QUADS = ROOT / "data" / "processed" / "quads.jsonl"
TEST_JSONL = ROOT / "data" / "processed" / "test_reviews.jsonl"
FOLDS = ROOT / "data" / "processed" / "folds.json"
STATS = ROOT / "data" / "processed" / "stats.json"
RAW_TEST_CSV = ROOT / "data" / "test_raw" / "TEST" / "Test_reviews.csv"
RUN_CV = ROOT / "src" / "eval" / "run_cv.py"
PLACEHOLDER = "_"
PH = PLACEHOLDER


def _load_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------- build_tags

def test_build_tags_real_data():
    """真实 quads.jsonl 抽 50 条:标签长度==字符数,BMES 解码 span 与原 span 一致(显式字段)。"""
    rows = _load_rows(QUADS)
    rng = random.Random(502)
    sample = rng.sample(rows, 50)
    n_a = n_o = 0
    for row in sample:
        text, quads, rid = row["text"], row["quads"], row["id"]
        ch = build_channels(text, quads)
        assert len(ch["aspect_tags"]) == len(text), f"id={rid} 属性标签长度 != 字符数"
        assert len(ch["opinion_tags"]) == len(text), f"id={rid} 观点标签长度 != 字符数"

        for tags, skey, ekey, wkey, kind in (
            (ch["aspect_tags"], "a_start", "a_end", "a", "A"),
            (ch["opinion_tags"], "o_start", "o_end", "o", "O"),
        ):
            expect = set()
            for q in quads:
                if q[skey] >= 0:
                    # 偏移切片语义: text[start:end] == 词面
                    assert text[q[skey]:q[ekey]] == q[wkey], f"id={rid} {kind} 偏移切片不一致"
                    expect.add((q[skey], q[ekey]))
            got = {(s, e) for s, e, k in bmes_decode(tags) if k == kind}
            assert got == expect, f"id={rid} {kind} 解码 {sorted(got)} != 原 span {sorted(expect)}"
            if kind == "A":
                n_a += len(expect)
            else:
                n_o += len(expect)

    # 防止抽样恰好全隐式、断言被空跑
    assert n_a > 0, "样本中无显式属性 span,测试未真正覆盖"
    assert n_o > 0, "样本中无显式观点 span,测试未真正覆盖"


# ---------------------------------------------------------------- scorer

def test_scorer_gold_self_and_drop10():
    """真实全量数据 gold vs gold == 1.0;随机删 10% 行(按 id)后 recall 约 0.9。"""
    gold = load_gold_jsonl(QUADS)
    r = score(gold, gold)
    assert r["precision"] == 1.0 and r["recall"] == 1.0 and r["f1"] == 1.0
    assert r["P"] == r["G"] == r["S"]
    stats = json.loads(STATS.read_text(encoding="utf-8"))
    assert r["G"] == stats["n_quads"] == 6633, f"金标条数 {r['G']} != stats.n_quads"

    ids = sorted(gold)
    rng = random.Random(502)
    drop = set(rng.sample(ids, k=int(round(len(ids) * 0.1))))
    pred = {i: q for i, q in gold.items() if i not in drop}
    r2 = score(gold, pred)
    assert r2["precision"] == 1.0, f"删行后 precision={r2['precision']} != 1.0"
    assert abs(r2["recall"] - 0.9) < 0.05, f"删 10% 行后 recall={r2['recall']:.4f} 偏离 0.9 过大"
    assert r2["S"] == r2["P"] and r2["G"] == 6633


# ---------------------------------------------------------------- check_submit

def test_check_submit_end_to_end():
    """真实端到端:全占位提交(每 id 一行 "id,_,_,_,_",无 BOM/无 CR/\\n 结尾)必须全绿;加 BOM 必须报错。"""
    rows = _load_rows(TEST_JSONL)
    ids = sorted(r["id"] for r in rows)
    stats = json.loads(STATS.read_text(encoding="utf-8"))
    assert len(ids) == stats["n_test_reviews"] == 2237, f"测试 id 数 {len(ids)} 异常"

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # 由真实 test_reviews.jsonl 生成 check_submit 用的测试 CSV(id,Reviews)
        test_csv = tmp / "Test_reviews.csv"
        with open(test_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "Reviews"])
            for r in sorted(rows, key=lambda x: x["id"]):
                w.writerow([r["id"], r["text"]])

        # 全占位提交:严格无 BOM、无 0x0D、以 \n 结尾
        content = "\n".join(f"{i},{PH},{PH},{PH},{PH}" for i in ids) + "\n"
        raw = content.encode("utf-8")
        assert not raw.startswith(b"\xef\xbb\xbf"), "提交不应带 BOM"
        assert b"\x0d" not in raw, "提交不应含回车 0x0D"
        assert raw.endswith(b"\n"), "提交应以 \\n 结尾"
        pred_path = tmp / "Result.csv"
        pred_path.write_bytes(raw)

        errors, warns = check(pred_path, test_csv)
        assert errors == [], f"全占位提交应全绿,实际报错: {errors[:5]}"
        assert warns == [], f"全占位提交不应告警: {warns[:5]}"

        # 故意加 BOM -> 必须报错
        bom_path = tmp / "Result_bom.csv"
        bom_path.write_bytes(b"\xef\xbb\xbf" + raw)
        errors2, _ = check(bom_path, test_csv)
        assert any("BOM" in e for e in errors2), f"加 BOM 应报错,实际: {errors2[:5]}"

        # 真实 Test_reviews.csv 若在,同一提交也必须全绿(id 集合一致性由 check 把关)
        if RAW_TEST_CSV.exists():
            errors3, warns3 = check(pred_path, RAW_TEST_CSV)
            assert errors3 == [], f"对真实 Test_reviews.csv 报错: {errors3[:5]}"
            assert warns3 == [], f"对真实 Test_reviews.csv 告警: {warns3[:5]}"


# ---------------------------------------------------------------- folds

def test_folds_partition():
    """folds.json:各折无交集、无重复,并集 == quads.jsonl 全部 id。"""
    gold = load_gold_jsonl(QUADS)
    all_ids = set(gold)
    folds = json.loads(FOLDS.read_text(encoding="utf-8"))
    assert folds["n_folds"] == len(folds["folds"]), "n_folds 与折数不一致"
    seen = set()
    for k, f in enumerate(folds["folds"]):
        s = set(f)
        assert len(s) == len(f), f"fold {k} 内有重复 id"
        inter = s & seen
        assert not inter, f"fold {k} 与之前折有交集: {sorted(inter)[:5]}"
        seen |= s
    assert seen == all_ids, (
        f"并集 != 全部 id: 缺 {sorted(all_ids - seen)[:5]} 多 {sorted(seen - all_ids)[:5]}"
    )


# ---------------------------------------------------------------- run_cv

def test_run_cv_report_self_and_buckets():
    """run_cv 报表:gold vs gold 全 1.0;合成错例上错误分桶数字精确。"""
    from src.eval.run_cv import build_report

    # 1) 真实数据 gold vs gold 自测
    rep = build_report(QUADS, QUADS, FOLDS)
    assert abs(rep["overall"]["f1"] - 1.0) < 1e-12
    assert abs(rep["overall"]["pred_gold_ratio"] - 1.0) < 1e-12
    for c, m in rep["per_category"].items():
        if m["G"] > 0:
            assert abs(m["f1"] - 1.0) < 1e-12, f"类别 {c} 自测 f1={m['f1']}"
    for p, m in rep["per_polarity"].items():
        if m["G"] > 0:
            assert abs(m["f1"] - 1.0) < 1e-12, f"极性 {p} 自测 f1={m['f1']}"
    b = rep["buckets"]
    assert b["implicit_aspect"]["recall"] == 1.0
    assert b["explicit_aspect"]["recall"] == 1.0
    assert b["cross_order"]["f1"] == 1.0 and b["cross_order"]["G"] == 109
    assert b["multi_quad_ids"]["f1"] == 1.0 and b["multi_quad_ids"]["n_ids"] == 908
    assert b["boundary_off_by_one"] == 0
    assert b["placeholder_pred_ph_gold_explicit"] == 0
    assert b["placeholder_pred_explicit_gold_ph"] == 0
    assert rep["folds"]["mean"] == 1.0 and rep["folds"]["var"] == 0.0

    # 2) 合成错例:边界差一字 / 占位错误 / 跨序对 / 隐式 recall 精确断言
    #    text: 有0 点1 贵2 的3 手4 机5 价6 格7 不8 错9 啊10
    text = "有点贵的手机价格不错啊"
    gold_rows = [{
        "id": 1, "text": text,
        "quads": [
            # q1 跨序对: 观点(0,3) 在属性(6,8) 之前(o_end<=a_start)
            {"a": "价格", "a_start": 6, "a_end": 8,
             "o": "有点贵", "o_start": 0, "o_end": 3, "c": "价格", "p": "负面"},
            # q2 隐式属性(a=="_"),观点词面(8,10)
            {"a": "_", "a_start": -1, "a_end": -1,
             "o": "不错", "o_start": 8, "o_end": 10, "c": "整体", "p": "正面"},
            # q3 显式属性(非跨序: o_end=10>a_start=6),预测侧将给成占位
            {"a": "手机", "a_start": 4, "a_end": 6,
             "o": "不错", "o_start": 8, "o_end": 10, "c": "使用体验", "p": "负面"},
        ],
    }]
    pred_rows = [{
        "id": 1, "text": text,
        "quads": [
            # p1 与 q1 完全一致 -> 唯一命中
            {"a": "价格", "a_start": 6, "a_end": 8,
             "o": "有点贵", "o_start": 0, "o_end": 3, "c": "价格", "p": "负面"},
            # p2 观点词面多一字(不错->不错啊) -> 边界差一字
            {"a": "_", "a_start": -1, "a_end": -1,
             "o": "不错啊", "o_start": 8, "o_end": 11, "c": "整体", "p": "正面"},
            # p3 属性给成占位 -> 占位错误(预测占位/金标显式)
            {"a": "_", "a_start": -1, "a_end": -1,
             "o": "不错", "o_start": 8, "o_end": 10, "c": "使用体验", "p": "负面"},
        ],
    }]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        gp = tmp / "gold.jsonl"
        pp = tmp / "pred.jsonl"
        gp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in gold_rows) + "\n",
                      encoding="utf-8")
        pp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in pred_rows) + "\n",
                      encoding="utf-8")

        rep2 = build_report(gp, pp)
        o = rep2["overall"]
        assert o["G"] == 3 and o["P"] == 3 and o["S"] == 1
        assert abs(o["f1"] - 1 / 3) < 1e-9
        b2 = rep2["buckets"]
        assert b2["implicit_aspect"] == {"G": 1, "S": 0, "recall": 0.0}
        assert b2["explicit_aspect"]["G"] == 2 and b2["explicit_aspect"]["S"] == 1
        assert abs(b2["explicit_aspect"]["recall"] - 0.5) < 1e-9
        assert b2["cross_order"]["G"] == 1 and b2["cross_order"]["S"] == 1
        assert b2["cross_order"]["f1"] == 1.0 and b2["cross_order"]["note"] == ""
        assert b2["multi_quad_ids"]["n_ids"] == 1
        assert abs(b2["multi_quad_ids"]["f1"] - 1 / 3) < 1e-9
        assert b2["boundary_off_by_one"] == 1
        assert b2["placeholder_pred_ph_gold_explicit"] == 1
        assert b2["placeholder_pred_explicit_gold_ph"] == 0
        assert b2["per_id_avg"]["pred"] == 3.0 and b2["per_id_avg"]["gold"] == 3.0

        # 3) CLI 冒烟: 文件路径方式跑 run_cv.py,断言退出码 0 且 --out JSON 可解析
        out_path = tmp / "report.json"
        proc = subprocess.run(
            [sys.executable, "-I", str(RUN_CV),
             "--gold", str(gp), "--pred", str(pp),
             "--pred", str(gp),  # 第二份做对比
             "--out", str(out_path)],
            capture_output=True,
        )
        assert proc.returncode == 0, f"run_cv CLI 退出码 {proc.returncode}: {proc.stdout.decode('utf-8', 'replace')}{proc.stderr.decode('utf-8', 'replace')}"
        assert out_path.exists(), "run_cv 未写出 --out JSON"
        rep_json = json.loads(out_path.read_text(encoding="utf-8"))
        assert len(rep_json["predictions"]) == 2
        assert abs(rep_json["predictions"][0]["overall"]["f1"] - 1 / 3) < 1e-9
        assert rep_json["predictions"][1]["overall"]["f1"] == 1.0


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
