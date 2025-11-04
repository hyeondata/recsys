# 사용법 (간단 참조용)

> ⚠️ **상세 사용법은 [`00_using.md`](./00_using.md) 참조**
>
> 이 파일은 간단한 참조용입니다. 전체 사용법, 파라미터 설명, 버전 히스토리는 00_using.md에서 확인하세요.

## Quick Commands

### 학습 (uv 필수)

**Dense MoE** (베이스라인)
```bash
uv run python3 src/training/train_dense_moe.py --epochs 10 --batch_size 1024
```

**PPO/GRPO (Batch)** (메모리 효율적)
```bash
uv run python3 src/training/train_ppo_moe.py --epochs 10 --batch_size 256 --ppo_epochs 4
uv run python3 src/training/train_grpo_moe.py --epochs 10 --batch_size 256 --grpo_epochs 4
```

**PPO/GRPO (Standard)** (논문 재현용) ⭐
```bash
uv run python3 src/training/train_ppo_moe_standard.py --epochs 10 --batch_size 512 --mini_batch_size 256 --ppo_epochs 4
uv run python3 src/training/train_grpo_moe_standard.py --epochs 10 --batch_size 512 --mini_batch_size 256 --grpo_epochs 4
```

### 평가
```bash
uv run python3 src/training/evaluate.py \
    --dense_checkpoint checkpoints/dense_moe/dense_moe_best.pt \
    --ppo_checkpoint checkpoints/ppo_moe/ppo_moe_best.pt \
    --grpo_checkpoint checkpoints/grpo_moe/grpo_moe_best.pt \
    --output_file results/evaluation.json
```

### 시각화
```bash
# 기본 성능 비교
uv run python3 src/visualization/publication_plots.py \
    --results_file results/all_models_evaluation.json \
    --output_dir results/publication

# 종합 지표 비교 (논문용) ⭐
uv run python3 src/visualization/paper_metrics_comparison.py \
    --results_file results/all_models_evaluation.json \
    --output_dir results/publication
```

---

📖 **전체 문서**: [`00_using.md`](./00_using.md)
