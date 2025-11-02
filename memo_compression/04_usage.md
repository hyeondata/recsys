# 사용법

## 학습 (uv 사용 필수!)

### Dense MoE
```bash
uv run python3 src/training/train_dense_moe.py \
    --epochs 10 --batch_size 1024 --lr 0.001 \
    --checkpoint_dir checkpoints/dense_moe
```

### GRPO-MoE
```bash
uv run python3 src/training/train_grpo_moe.py \
    --epochs 10 --batch_size 256 --grpo_epochs 4 --lr 0.0003 \
    --checkpoint_dir checkpoints/grpo_moe
```

### PPO-MoE
```bash
uv run python3 src/training/train_ppo_moe.py \
    --epochs 10 --batch_size 256 --ppo_epochs 4 --lr 0.0003 \
    --checkpoint_dir checkpoints/ppo_moe
```

## 평가
```bash
uv run python3 src/training/evaluate.py \
    --dense_checkpoint checkpoints/dense_moe/dense_moe_best.pt \
    --ppo_checkpoint checkpoints/ppo_moe/ppo_moe_best.pt \
    --grpo_checkpoint checkpoints/grpo_moe/grpo_moe_best.pt \
    --output_file results/evaluation.json
```

## Git
```bash
git init
git remote add origin https://github.com/hyeondata/recsys.git
git checkout -b moe
git add .
git commit -m "feat: MoE recommendation system"
git push -u origin moe
```

## 환경 설정
- uv 패키지 관리자 사용
- Python 3.8+
- PyTorch 2.0+
- CUDA 필수
