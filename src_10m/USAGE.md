# 10M Dataset Quick Usage Guide

## 최소 실행 명령어

### 1. Dense MoE
```bash
uv run python3 src_10m/training/train_dense_moe.py --epochs 10
```

### 2. PPO-MoE (Batch)
```bash
uv run python3 src_10m/training/train_ppo_moe.py --epochs 10 --batch_size 512
```

### 3. GRPO-MoE (Batch)
```bash
uv run python3 src_10m/training/train_grpo_moe.py --epochs 10 --batch_size 512
```

### 4. PPO-MoE (Standard) - 논문용
```bash
uv run python3 src_10m/training/train_ppo_moe_standard.py --epochs 10 --batch_size 1024
```

### 5. GRPO-MoE (Standard) - 논문용
```bash
uv run python3 src_10m/training/train_grpo_moe_standard.py --epochs 10 --batch_size 1024
```

## 평가
```bash
uv run python3 src_10m/training/evaluate.py \
    --dense_checkpoint checkpoints_10m/dense_moe/dense_moe_best.pt \
    --output_file results_10m/eval.json
```

## 시각화
```bash
uv run python3 src_10m/visualization/publication_plots.py \
    --results_file results_10m/eval.json \
    --output_dir results_10m/plots
```

## 주요 차이점 (vs 100k)
- 데이터: 10M ratings (100배)
- 임베딩: 128차원 (2배)
- Expert: 16개 (2배)
- Batch size: 2048 for Dense, 512 for RL
- 학습 시간: ~30-60분/epoch (vs 2-4분)

## GPU 메모리
- Dense: 2048 batch → ~12GB
- PPO/GRPO Batch: 512 batch → ~8GB
- Standard: 1024 batch → ~18GB
