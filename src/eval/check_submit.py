# -*- coding: utf-8 -*-
"""提交文件 Result.csv 的字节级校验器。格式错=整份作废,提交前必须全绿。

校验项:
  1. UTF-8 无 BOM(头 3 字节 != EF BB BF),不含回车 0x0D(Windows CRLF 翻车点)
  2. 无表头:首行首字段必须是纯数字 id
  3. 每行恰好 5 字段: id,AspectTerm,OpinionTerm,Category,Polarity
  4. id 正则 ^[0-9]+$(防 pandas 浮点 "1.0"),升序,与 Test_reviews.csv 的 id 集合全等(不缺不多)
  5. 占位符 "_" 必须恰为 0x5F 单字符(防全角"＿"),且只能整字段出现
  6. Category ∈ 13 类闭集(或占位), Polarity ∈ 3 类闭集(或占位)
  7. AspectTerm/OpinionTerm 非占位时必须是对应评论原文的逐字子串(防幻觉/归一化)
  8. 行数断言: 每个 id 至少 1 行(全空行合法)

用法: python -m src.eval.check_submit --pred submissions/Result.csv [--test data/test_raw/TEST/Test_reviews.csv]
退出码: 0 全绿, 1 有错。
"""
import argparse
import csv
import io
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLACEHOLDER = "_"
PLACEHOLDER_BYTE = 0x5F
CATEGORIES = {"包装", "成分", "尺寸", "服务", "功效", "价格", "气味", "使用体验", "物流", "新鲜度", "真伪", "整体", "其他"}
POLARITIES = {"正面", "中性", "负面"}
ID_RE = re.compile(r"^[0-9]+$")


def load_test_texts(path):
    texts = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            texts[int(row["id"])] = row["Reviews"]
    return texts


def check(pred_path, test_path):
    errors, warns = [], []
    raw = Path(pred_path).read_bytes()

    if raw[:3] == b"\xef\xbb\xbf":
        errors.append("存在 UTF-8 BOM(EF BB BF)")
    if b"\x0d" in raw:
        errors.append("存在回车 0x0D(应为纯 \\n 换行)")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        errors.append(f"UTF-8 解码失败: {e}")
        return errors, warns

    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()  # 文件末尾换行
    if not lines:
        errors.append("文件为空")
        return errors, warns

    first = lines[0].split(",")
    if not (first and ID_RE.match(first[0].strip())):
        errors.append(f"疑似表头或非法首行: {lines[0][:50]!r}")

    test_texts = load_test_texts(test_path)
    seen_ids = []
    rows_by_id = {}
    for i, line in enumerate(lines, 1):
        parts = line.split(",")
        if len(parts) != 5:
            errors.append(f"第 {i} 行字段数 {len(parts)} != 5: {line[:60]!r}")
            continue
        rid_s, a, o, c, p = parts
        if not ID_RE.match(rid_s):
            errors.append(f"第 {i} 行 id 非纯数字: {rid_s!r}")
            continue
        rid = int(rid_s)
        seen_ids.append(rid)
        rows_by_id.setdefault(rid, []).append((a, o, c, p))

        for name, val in (("AspectTerm", a), ("OpinionTerm", o), ("Category", c), ("Polarity", p)):
            if val == "":
                errors.append(f"id={rid} 第 {i} 行 {name} 为空字符串(应填 '_')")
            elif val != PLACEHOLDER and PLACEHOLDER in val:
                warns.append(f"id={rid} 第 {i} 行 {name} 内嵌下划线: {val!r}")
            if val == "＿":
                errors.append(f"id={rid} 第 {i} 行 {name} 是全角下划线")
        if c not in CATEGORIES and c != PLACEHOLDER:
            errors.append(f"id={rid} Category 非法: {c!r}")
        if p not in POLARITIES and p != PLACEHOLDER:
            errors.append(f"id={rid} Polarity 非法: {p!r}")

        txt = test_texts.get(rid)
        if txt is None:
            errors.append(f"id={rid} 不在 Test_reviews.csv 中")
        else:
            for name, val in (("AspectTerm", a), ("OpinionTerm", o)):
                if val != PLACEHOLDER and val not in txt:
                    errors.append(f"id={rid} {name}={val!r} 不是评论原文子串")

    if seen_ids != sorted(seen_ids):
        errors.append("id 未升序")
    dup = len(seen_ids) - len(set(seen_ids))
    if dup:
        warns.append(f"存在 {dup} 个重复 id 行(同 id 多行=多四元组,合法;此处仅提示确认非误重复)")
    missing = set(test_texts) - set(seen_ids)
    extra = set(seen_ids) - set(test_texts)
    if missing:
        errors.append(f"缺失 {len(missing)} 个测试 id(必须保留并用 '_' 占位): {sorted(missing)[:10]}...")
    if extra:
        errors.append(f"多出 {len(extra)} 个非测试 id: {sorted(extra)[:10]}")

    # 占位符字节级确认
    for i, line in enumerate(lines, 1):
        for fld in line.split(","):
            if fld == PLACEHOLDER and len(fld.encode("utf-8")) != 1:
                errors.append(f"第 {i} 行占位符不是单字节 0x5F")

    return errors, warns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--test", default=str(ROOT / "data" / "test_raw" / "TEST" / "Test_reviews.csv"))
    args = ap.parse_args()
    errors, warns = check(args.pred, args.test)
    for w in warns:
        print(f"[warn] {w}")
    if errors:
        for e in errors:
            print(f"[error] {e}")
        print(f"\nFAIL: {len(errors)} 个错误")
        sys.exit(1)
    print("PASS: 提交格式校验全绿")


if __name__ == "__main__":
    main()
