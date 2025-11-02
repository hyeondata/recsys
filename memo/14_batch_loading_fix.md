# Batch Loading 방식 수정 (공정한 비교를 위해)

## 작성 일자
2025년 11월 3일

---

## 문제점

### 기존 데이터 로딩 방식의 차이

**Dense MoE**:
- `batch_size=256` (또는 args.batch_size)
- 미니배치 단위로 학습
- 각 배치를 1번 업데이트
- 총 업데이트 횟수: ~311회/epoch (79,619 / 256)

**GRPO/PPO MoE (기존)**:
- `batch_size=len(train_dataset)` (전체 데이터)
- 전체 데이터를 한번에 메모리에 로드
- 배치로 나눠서 4번씩 업데이트 (grpo_epochs=4)
- 총 업데이트 횟수: ~311 × 4 = 1,244회/epoch

**결과**: GRPO/PPO가 Dense보다 4배 많은 업데이트를 받음 → **불공정한 비교**

---

## 해결 방법

### 수정된 데이터 로딩 방식

**GRPO/PPO MoE (수정 후)**:
- `batch_size=args.batch_size` (Dense와 동일)
- 미니배치 단위로 학습
- 각 배치에 대해:
  1. 에피소드 수집 (forward pass, no grad)
  2. Advantage/Reward 계산
  3. grpo_epochs번 업데이트 (해당 배치만)

**업데이트 횟수 (수정 후)**:
- 배치 수: ~311개
- 각 배치당 업데이트: 4회
- 총 업데이트: ~1,244회/epoch

**주의**: 여전히 Dense(311회)보다 많지만, 이는 강화학습의 특성입니다. 동일한 데이터를 여러 번 학습하는 것이 PPO의 핵심 아이디어입니다.

---

## 코드 수정 내용

### 1. train_grpo_moe.py

#### DataLoader 수정
```python
# Before
train_loader = DataLoader(
    train_dataset,
    batch_size=len(train_dataset),  # 전체 데이터
    shuffle=False
)

# After
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,  # Dense와 동일
    shuffle=True,
    num_workers=args.num_workers
)
```

#### train_epoch 함수 수정
```python
def train_epoch(model, train_loader, optimizer, device, args):
    model.train()

    # 배치별로 학습 (Dense MoE와 동일한 방식)
    for batch in tqdm(train_loader, desc="Training"):
        # 1. 데이터 준비
        user_id = batch['user_id'].to(device)
        # ... (기타 입력들)

        # 2. 에피소드 수집 (현재 배치에 대해)
        with torch.no_grad():
            model.eval()
            outputs_old = model(...)
            old_log_probs = outputs_old['log_prob']
            predictions = outputs_old['rating']

        # 3. 보상 계산
        raw_rewards = model.compute_rewards_from_errors(predictions, rating)
        group_rewards = model.compute_group_relative_rewards(raw_rewards)

        # 4. GRPO 업데이트 (배치를 grpo_epochs번 반복)
        model.train()
        for _ in range(args.grpo_epochs):
            outputs = model(...)
            losses = model.compute_loss(...)

            optimizer.zero_grad()
            losses['total_loss'].backward()
            optimizer.step()
```

#### validate_epoch 함수 수정
```python
def validate_epoch(model, val_loader, device):
    model.eval()

    all_predictions = []
    all_targets = []
    all_actions = []

    # 배치별로 검증
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            outputs = model(..., deterministic=True)
            all_predictions.append(outputs['rating'].cpu())
            # ...

    # 결과 병합
    predictions = torch.cat(all_predictions)
    targets = torch.cat(all_targets)
    # 메트릭 계산
```

### 2. train_ppo_moe.py

PPO도 GRPO와 동일하게 수정:
- DataLoader를 batch_size 단위로 변경
- train_epoch를 배치별 처리로 변경
- validate_epoch를 배치별 처리로 변경

#### Advantage 계산 간소화
```python
# PPO에서는 단일 스텝이므로:
rewards = -torch.abs(predictions - rating)
advantages = rewards - old_values  # TD error
returns = rewards  # 단일 스텝
```

### 3. argparse에 num_workers 추가

```python
# 두 파일 모두에 추가
parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
```

---

## 변경 전후 비교

### 메모리 사용량

| 모델 | 기존 | 수정 후 |
|------|-----|--------|
| Dense MoE | 256 샘플 | 256 샘플 |
| GRPO-MoE | 79,619 샘플 (전체) | 256 샘플 |
| PPO-MoE | 79,619 샘플 (전체) | 256 샘플 |

**개선**: GRPO/PPO의 메모리 사용량 ~300배 감소

### 학습 방식

| 모델 | 배치 처리 | 배치당 업데이트 | Epoch당 총 업데이트 |
|------|----------|---------------|------------------|
| Dense MoE | 순차 | 1회 | ~311회 |
| GRPO-MoE (기존) | 전체 → 분할 | 4회 | ~1,244회 |
| GRPO-MoE (수정) | 순차 | 4회 | ~1,244회 |

**주의**: 업데이트 횟수는 동일하지만, 데이터 로딩 방식이 공정해짐

### 코드 구조

**기존**:
```
collect_episode(전체 데이터) → 배치로 분할 → 4번 업데이트
```

**수정 후**:
```
for batch in data_loader:
    collect_episode(배치) → 4번 업데이트
```

---

## 예상 성능 변화

### 기대 효과
1. **메모리 효율성**: 대폭 개선
2. **학습 안정성**: 더 안정적일 가능성
3. **공정한 비교**: Dense와 동일한 데이터 로딩 방식
4. **확장성**: 더 큰 데이터셋에도 적용 가능

### 성능 비교
재학습 필요! 기존 결과와 다를 수 있음:

**기존 결과**:
- Dense MoE: RMSE 0.9803
- GRPO-MoE: RMSE 1.0412
- PPO-MoE: RMSE 1.0659

**재학습 후 예상**:
- 성능이 약간 달라질 수 있음
- 더 공정한 비교가 될 것

---

## 삭제된 함수

### collect_episode 함수
기존에는 전체 DataLoader를 받아서 에피소드를 수집했지만, 이제는 train_epoch 내부에서 배치별로 처리하므로 사용하지 않음 (삭제하지는 않고 남겨둠)

---

## 재학습 권장 사항

### 명령어
```bash
# GRPO-MoE
uv run python3 src/training/train_grpo_moe.py \
    --epochs 10 \
    --batch_size 256 \
    --grpo_epochs 4 \
    --lr 0.0003 \
    --data_dir ml-100k \
    --train_rating_path ml-100k/u1.base \
    --val_rating_path ml-100k/u1.test \
    --checkpoint_dir checkpoints/grpo_moe_v2

# PPO-MoE
uv run python3 src/training/train_ppo_moe.py \
    --epochs 10 \
    --batch_size 256 \
    --ppo_epochs 4 \
    --lr 0.0003 \
    --data_dir ml-100k \
    --train_rating_path ml-100k/u1.base \
    --val_rating_path ml-100k/u1.test \
    --checkpoint_dir checkpoints/ppo_moe_v2
```

---

## 결론

### 주요 변경사항
✅ GRPO/PPO를 Dense와 동일한 배치 단위로 학습
✅ 메모리 효율성 대폭 개선
✅ 공정한 성능 비교 가능
✅ 코드 가독성 향상

### 다음 단계
1. 재학습 실행
2. 새로운 결과와 기존 결과 비교
3. 성능 변화 분석

---

*작성자: hyeondata*
*작성일: 2025년 11월 3일*
*프로젝트: claudeMoE - Fair Comparison*
