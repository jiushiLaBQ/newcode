# P1 云端 GPU 作业手册（AutoDL / 恒源云，4090 24GB 即可）

目标：训练 3 个 QLoRA 成员（总 6-12h，约 ¥20-60），推理测试集，回传预测。

## 1. 租卡

- AutoDL（autodl.com）或恒源云（gpushare.com）→ 租 **RTX 4090 24GB**（8B 成员 bs2+梯度检查点可跑；更稳可选 A100 40GB）。
- 镜像选 **PyTorch 2.x + Python 3.11 + CUDA 12.x**。
- 数据盘挂载后上传项目（或只上传 `data/llm/` + `src/llm/`）：

```bash
# 本地打包
tar czf llm_job.tar.gz src/llm data/llm
# 云实例上解压到 /root/newcode
```

## 2. 环境

```bash
pip install torch transformers datasets peft bitsandbytes accelerate
# 国内镜像可加速权重下载:
export HF_ENDPOINT=https://hf-mirror.com
```

## 3. 训练（三成员，配方固定 seed=123，不做 dev 选 epoch）

```bash
cd /root/newcode
python src/llm/train_qlora.py --member baseline4b --out checkpoints/llm/baseline4b
python src/llm/train_qlora.py --member upgraded4b --out checkpoints/llm/upgraded4b
python src/llm/train_qlora.py --member qwen8b     --out checkpoints/llm/qwen8b
```

## 4. 推理（贪心、enable_thinking=False）

```bash
python src/llm/infer_qlora.py --base Qwen/Qwen3-4B              --adapter checkpoints/llm/baseline4b --out data/llm/pred_baseline4b.jsonl
python src/llm/infer_qlora.py --base Qwen/Qwen3-4B-Instruct-2507 --adapter checkpoints/llm/upgraded4b --out data/llm/pred_upgraded4b.jsonl
python src/llm/infer_qlora.py --base Qwen/Qwen3-8B              --adapter checkpoints/llm/qwen8b     --out data/llm/pred_qwen8b.jsonl
# dev/confirm 自评同理,--input 换 sft_dev/sft_confirm 的 prompt 版(用 build_sft_data 产物转)
```

## 5. 回传

```bash
tar czf preds.tar.gz data/llm/pred_*.jsonl checkpoints/llm  # adapter 小,可一并回传
```

## 6. 本地融合

```bash
python -I src/postprocess/fuse.py --preds data/llm/pred_baseline4b.jsonl data/llm/pred_upgraded4b.jsonl data/llm/pred_qwen8b.jsonl \
  --fallback baseline4b --override data/processed/quads.jsonl --out data/llm/fused.jsonl
```

## 备忘

- Qwen3 系列 Apache-2.0，审查合规。
- 训练日志/配置全部保留（审查包）。
- 8B 若 OOM：`--bs` 降 1 + 梯度检查点默认已开；或换 4090→A100。
