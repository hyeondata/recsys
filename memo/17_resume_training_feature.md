# Resume Training 기능 구현

**작성일**: 2025-11-08
**버전**: v2.1

---

## 📌 개요

학습 중 예기치 않은 중단(GPU 드라이버 크래시, 시스템 종료 등)이 발생해도 이어서 학습할 수 있는 **Resume 기능** 추가

---

## 🎯 구현 목표

1. ✅ 매 epoch마다 최신 체크포인트 자동 저장
2. ✅ 학습 중단 시 원래 epoch부터 이어서 학습
3. ✅ Optimizer, Scheduler, Early Stopping 상태 완전 복원
4. ✅ 모든 학습 스크립트에 일관된 방식으로 적용

---

## 🔧 수정된 파일

### 1. src/utils/trainer_utils.py

#### CheckpointManager 개선

**`save_checkpoint()` 메서드 확장**:
```python
def save_checkpoint(self, model, optimizer, epoch, metrics, is_best=False, scheduler=None, **kwargs):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }

    # Scheduler state 저장
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    # 추가 정보 저장 (early_stopping state 등)
    checkpoint.update(kwargs)

    # 기존: epoch_N.pt, best.pt
    # 새로 추가: latest.pt (resume용)
    latest_filename = f"{self.model_name}_latest.pt"
    latest_filepath = self.checkpoint_dir / latest_filename
    torch.save(checkpoint, latest_filepath)
```

**`load_checkpoint()` 메서드 확장**:
```python
def load_checkpoint(self, model, optimizer=None, scheduler=None, filename=None, resume=False):
    if filename is None:
        if resume:
            filename = f"{self.model_name}_latest.pt"  # Resume용
        else:
            filename = f"{self.model_name}_best.pt"     # 평가용

    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # 모든 추가 정보 반환
    return info
```

#### EarlyStopping 상태 저장/복원

```python
class EarlyStopping:
    def state_dict(self):
        """Early stopping 상태 반환"""
        return {
            'counter': self.counter,
            'best_score': self.best_score,
            'early_stop': self.early_stop
        }

    def load_state_dict(self, state_dict):
        """Early stopping 상태 로드"""
        self.counter = state_dict['counter']
        self.best_score = state_dict['best_score']
        self.early_stop = state_dict['early_stop']
```

---

### 2. 모든 학습 스크립트 (5개)

- `src/training/train_dense_moe.py`
- `src/training/train_ppo_moe.py`
- `src/training/train_grpo_moe.py`
- `src/training/train_ppo_moe_standard.py`
- `src/training/train_grpo_moe_standard.py`

#### Resume 로직 추가 (공통 패턴)

```python
# Resume 기능
start_epoch = 1
best_val_rmse = float('inf')

if args.resume:
    try:
        checkpoint_info = checkpoint_manager.load_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,  # Dense만 해당
            resume=True
        )
        start_epoch = checkpoint_info['epoch'] + 1
        best_val_rmse = checkpoint_info.get('best_val_rmse',
                                           checkpoint_info['metrics'].get('rmse', float('inf')))

        # Early stopping state 복원
        if 'early_stopping_state' in checkpoint_info:
            early_stopping.load_state_dict(checkpoint_info['early_stopping_state'])

        print(f"\nResumed from epoch {checkpoint_info['epoch']}")
        print(f"Best validation RMSE: {best_val_rmse:.4f}\n")
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        print("Starting training from scratch...")

# 학습 루프
for epoch in range(start_epoch, args.epochs + 1):  # start_epoch부터 시작
    ...
```

#### Checkpoint 저장 시 추가 정보 포함

```python
checkpoint_manager.save_checkpoint(
    model=model,
    optimizer=optimizer,
    epoch=epoch,
    metrics=val_metrics,
    is_best=is_best,
    scheduler=scheduler,  # Dense만 해당
    early_stopping_state=early_stopping.state_dict(),
    best_val_rmse=best_val_rmse
)
```

#### 명령행 인자 추가

```python
parser.add_argument("--resume", action="store_true",
                   help="Resume training from latest checkpoint")
```

---

## 📊 체크포인트 저장 방식

### 저장되는 파일 (3종류)

| 파일명 | 용도 | 저장 시점 | 개수 |
|--------|------|----------|------|
| `{model}_epoch_{N}.pt` | 일반 체크포인트 | 매 epoch | 최근 3개 |
| `{model}_best.pt` | 최고 성능 모델 | Best metric 갱신 시 | 1개 |
| `{model}_latest.pt` | Resume용 | **매 epoch** | 1개 |

### 저장되는 정보

```python
checkpoint = {
    # 기본 정보
    'epoch': int,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': dict,
    'metrics': dict,

    # 추가 정보 (resume용)
    'scheduler_state_dict': dict,  # Dense MoE만
    'early_stopping_state': {
        'counter': int,
        'best_score': float,
        'early_stop': bool
    },
    'best_val_rmse': float  # 또는 best_val_loss
}
```

---

## 🚀 사용 방법

### 1. 일반 학습 (처음부터)

```bash
uv run python3 src/training/train_ppo_moe.py \
    --epochs 100 \
    --batch_size 256 \
    --lr 0.0003 \
    --checkpoint_dir checkpoints/ppo_moe
```

### 2. 중단된 학습 이어서 하기

```bash
# --resume 플래그만 추가
uv run python3 src/training/train_ppo_moe.py \
    --epochs 100 \
    --batch_size 256 \
    --lr 0.0003 \
    --checkpoint_dir checkpoints/ppo_moe \
    --resume
```

**출력 예시**:
```
==================================================
Resumed from epoch 45
Best validation RMSE: 1.0452
==================================================

==================================================
Epoch 46/100
==================================================
```

### 3. 모든 모델에 동일하게 적용

```bash
# Dense MoE
uv run python3 src/training/train_dense_moe.py --epochs 50 --resume

# PPO-MoE (Batch)
uv run python3 src/training/train_ppo_moe.py --epochs 50 --resume

# GRPO-MoE (Batch)
uv run python3 src/training/train_grpo_moe.py --epochs 50 --resume

# PPO-MoE (Standard)
uv run python3 src/training/train_ppo_moe_standard.py --epochs 50 --resume

# GRPO-MoE (Standard)
uv run python3 src/training/train_grpo_moe_standard.py --epochs 50 --resume
```

---

## 📈 실전 사용 시나리오

### 시나리오 1: GPU 드라이버 크래시

```bash
# 1. 학습 시작
uv run python3 src/training/train_ppo_moe.py --epochs 100 --batch_size 256
# ... Epoch 45/100 학습 중 GPU 드라이버 크래시

# 2. 시스템 재부팅 후
uv run python3 src/training/train_ppo_moe.py --epochs 100 --batch_size 256 --resume
# ✅ Epoch 46부터 자동으로 이어서 학습
```

### 시나리오 2: 하이퍼파라미터 변경 후 재학습

```bash
# 1. 초기 학습 (lr=0.001로 시작)
uv run python3 src/training/train_ppo_moe.py --epochs 50 --lr 0.001
# ... Epoch 50/50 완료

# 2. 더 작은 lr로 fine-tuning
uv run python3 src/training/train_ppo_moe.py --epochs 100 --lr 0.0001 --resume
# ✅ Epoch 51부터 lr=0.0001로 학습
```

### 시나리오 3: Early Stopping 후 추가 학습

```bash
# 1. Early stopping으로 epoch 60에서 종료
uv run python3 src/training/train_ppo_moe.py --epochs 100 --patience 10
# ... Early stopping triggered at epoch 60

# 2. Patience 늘려서 추가 학습
uv run python3 src/training/train_ppo_moe.py --epochs 100 --patience 20 --resume
# ✅ Epoch 61부터 이어서 학습, early stopping counter 복원됨
```

---

## ⚠️ 주의사항

### 1. 하이퍼파라미터 일치

Resume 시 **원래 학습과 동일한 하이퍼파라미터** 사용 권장:
- `--batch_size`
- `--lr`
- `--num_experts`
- `--embedding_dim`
- 기타 모델 구조 관련 파라미터

**잘못된 예**:
```bash
# 원래: batch_size=256
uv run python3 src/training/train_ppo_moe.py --epochs 100 --batch_size 256

# Resume: batch_size=512로 변경 (권장하지 않음)
uv run python3 src/training/train_ppo_moe.py --epochs 100 --batch_size 512 --resume
# ⚠️ Optimizer 상태와 batch_size가 맞지 않을 수 있음
```

### 2. Checkpoint 디렉토리

사용자 지정 경로를 사용했다면 resume 시에도 동일한 경로 지정:

```bash
# 원래
uv run python3 src/training/train_ppo_moe.py \
    --checkpoint_dir checkpoints/experiment_1

# Resume (경로 일치 필요)
uv run python3 src/training/train_ppo_moe.py \
    --checkpoint_dir checkpoints/experiment_1 \
    --resume
```

### 3. 체크포인트 파일 존재 확인

Resume 시 `{model_name}_latest.pt`가 없으면 자동으로 처음부터 학습:

```bash
$ uv run python3 src/training/train_ppo_moe.py --resume
Warning: Checkpoint not found: checkpoints/ppo_moe_latest.pt
Starting training from scratch...
```

### 4. Best Metric 복원

Resume 시 이전 best metric이 유지되므로 개선된 모델만 `_best.pt`로 저장됨:

```bash
# Epoch 45까지 best RMSE = 1.0452
# Resume 후 Epoch 50에서 RMSE = 1.0300
# → _best.pt 업데이트됨 ✅

# Epoch 55에서 RMSE = 1.0500
# → _best.pt 업데이트 안됨 (1.0300이 더 좋음)
```

---

## 🔍 복원되는 상태 검증

### 체크포인트 내용 확인

```python
import torch

checkpoint = torch.load('checkpoints/ppo_moe/ppo_moe_latest.pt')
print(f"Epoch: {checkpoint['epoch']}")
print(f"Best RMSE: {checkpoint['best_val_rmse']:.4f}")
print(f"Early stopping counter: {checkpoint['early_stopping_state']['counter']}")
print(f"Metrics: {checkpoint['metrics']}")
```

**출력 예시**:
```
Epoch: 45
Best RMSE: 1.0452
Early stopping counter: 3
Metrics: {'mse': 1.0925, 'rmse': 1.0452, 'mae': 0.8234, ...}
```

---

## 📊 메모리 및 디스크 사용량

### 체크포인트 파일 크기

| 모델 | 파일 크기 (approximate) |
|------|------------------------|
| Dense MoE | ~15 MB |
| PPO-MoE | ~20 MB |
| GRPO-MoE | ~20 MB |
| PPO-MoE Standard | ~20 MB |
| GRPO-MoE Standard | ~20 MB |

### 디스크 사용량 (100 epochs)

- `epoch_N.pt`: 15-20 MB × 3개 = 45-60 MB
- `best.pt`: 15-20 MB × 1개 = 15-20 MB
- `latest.pt`: 15-20 MB × 1개 = 15-20 MB
- **총합**: ~75-100 MB

---

## 🎓 Best Practices

### 1. 긴 학습 시 주기적으로 중간 저장 확인

```bash
# 10 epochs마다 checkpoint 확인
for epoch in {10,20,30,40,50}; do
    ls -lh checkpoints/ppo_moe/ppo_moe_epoch_${epoch}.pt
done
```

### 2. 중요한 실험은 별도 디렉토리에 저장

```bash
# 실험 1
uv run python3 src/training/train_ppo_moe.py \
    --checkpoint_dir checkpoints/ppo_lr_0.001

# 실험 2
uv run python3 src/training/train_ppo_moe.py \
    --checkpoint_dir checkpoints/ppo_lr_0.0001
```

### 3. Resume 전 latest.pt 백업

```bash
cp checkpoints/ppo_moe/ppo_moe_latest.pt \
   checkpoints/ppo_moe/ppo_moe_latest_backup.pt

uv run python3 src/training/train_ppo_moe.py --resume
```

---

## 🐛 트러블슈팅

### 문제 1: Resume 시 epoch이 1부터 시작됨

**원인**: `--resume` 플래그를 빼먹음

**해결**:
```bash
# ❌ 잘못된 명령어
uv run python3 src/training/train_ppo_moe.py --epochs 100

# ✅ 올바른 명령어
uv run python3 src/training/train_ppo_moe.py --epochs 100 --resume
```

### 문제 2: "Checkpoint not found" 에러

**원인**: 잘못된 checkpoint 디렉토리 또는 파일이 없음

**해결**:
```bash
# 파일 확인
ls checkpoints/ppo_moe/

# 올바른 경로 지정
uv run python3 src/training/train_ppo_moe.py \
    --checkpoint_dir checkpoints/ppo_moe \
    --resume
```

### 문제 3: Resume 후 성능이 떨어짐

**원인**: Learning rate 또는 batch size 변경

**해결**: 원래 하이퍼파라미터와 동일하게 유지

---

## 📚 관련 문서

- `12_code_refactoring_tqdm.md` - tqdm 진행률 표시
- `14_batch_loading_fix.md` - 배치 로딩 수정
- `16_standard_rl_implementation.md` - 표준 RL 구현

---

## 🔮 향후 개선 방향

### 1. Auto-resume 기능
- 학습 시작 시 latest.pt가 있으면 자동으로 resume

### 2. Multi-checkpoint 관리
- 여러 시점의 체크포인트 유지 (epoch 10, 20, 30, ...)

### 3. Checkpoint 압축
- 파일 크기 줄이기 (현재 15-20 MB → 5-10 MB)

### 4. Cloud 백업
- S3, GCS 등에 자동 업로드

---

**작성자**: Claude + User
**버전**: v2.1
**마지막 업데이트**: 2025-11-08
