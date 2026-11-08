#!/usr/bin/env bash
# P0/P1 一键回归:偏移校验 -> 折切分 -> P0 单测 -> 集成测试。
# 用法: bash scripts/run_p0.sh(在任意位置执行,脚本内自动 cd 到仓库根目录)
# 注意: python -I 下 -m 不可用(隔离模式不把仓库根加进 sys.path),一律用文件路径方式。
#       -I 连带 -E,PYTHONUTF8/PYTHONIOENCODING 均被忽略;各 py 脚本内部自行
#       sys.stdout.reconfigure(encoding="utf-8") 保证输出 UTF-8。
set -euo pipefail
cd "$(dirname "$0")/.."

python -I src/preprocess/offset_check.py && \
python -I src/preprocess/folds.py && \
python -I tests/test_p0.py && \
python -I tests/test_pipeline.py

echo "run_p0.sh: ALL GREEN"
