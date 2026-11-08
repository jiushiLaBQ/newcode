# ABSA 属性级情感分析

面向中文 ABSA（Aspect-Based Sentiment Analysis）【天池经典打榜赛】赛道六-评论观点挖掘赛的建模与提交管线。

## 结构

```
src/
  preprocess/   数据清洗、标签构建、交叉验证划分
  models/       BiLSTM/CRF 规则基线 + 序列标注模型
  train/        训练与测试集推理
  decode/       解码（四元组抽取）
  postprocess/  规则修正、多数投票融合、LLM 切分、校验
  eval/         本地评分、CV、提交生成与检查
  llm/          QLoRA 微调数据构建、训练、推理、后处理
scripts/        P0/P1 运行脚本与云端作业手册
tests/          单元测试
submissions/    历史提交 CSV
data/
  train_raw/    原始训练数据
  test_raw/     原始测试数据
  llm/          SFT 数据与切分（小体量，入库）
  processed/    预处理产物（本地生成，不入库）
  experiments/  实验记录（本地生成，不入库）
```

## 环境

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows
pip install -r requirements.txt   # 见 scripts/ 与各模块依赖
```

主要依赖：`torch`、`transformers`、`peft`、`bitsandbytes`、`datasets`、`pandas`、`scikit-learn`。

## 快速开始

```bash
# P0：规则 + 序列标注基线
bash scripts/run_p0.sh

# P1：QLoRA 成员训练（需 GPU，见 scripts/cloud_setup.md）
bash scripts/run_p1.sh
```

## 评测

多重集（multiset）四元组匹配，详见 [src/eval/scorer.py](src/eval/scorer.py)。

## 复现说明

- 模型权重（`checkpoints/`，约 6 GB）与预处理中间产物不入库，按 `scripts/` 脚本重新生成。
- 所有实验使用固定随机种子；训练配置与日志保留在 `data/experiments/`。
