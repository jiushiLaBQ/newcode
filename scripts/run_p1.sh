#!/usr/bin/env bash
# P1: A3 主力基线全流程。Windows Git Bash / Linux 通用。
# 依赖: 项目 venv(.venv),由 python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt 准备。
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -x ".venv/Scripts/python" ]]; then PY=".venv/Scripts/python"; else PY="python"; fi
export PYTHONUTF8=1

echo "== P0 复跑(幂等) =="
"$PY" -I src/preprocess/offset_check.py >/dev/null
"$PY" -I src/preprocess/folds.py
"$PY" -I tests/test_p0.py

echo "== A3 冒烟 =="
"$PY" -I src/train/train_a3.py --smoke

echo "== A3 5 折训练(fp16,base 模型) =="
"$PY" -I src/train/train_a3.py --model hfl/chinese-macbert-base --fp16 \
  --epochs 8 --batch-size 8 --accum-steps 2 --lr 3e-5

echo "== OOF 评测报表 =="
"$PY" -I src/eval/run_cv.py --gold data/processed/quads.jsonl \
  --pred data/processed/oof_a3/oof.jsonl --folds data/processed/folds.json \
  --out data/processed/oof_a3/cv_report.json
