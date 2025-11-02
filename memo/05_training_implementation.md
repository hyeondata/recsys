# 학습 및 평가 구현 완료

## 구현 일자
2025년 11월 2일

## 구현 개요
MoE 모델들의 학습 및 평가 파이프라인을 완성했습니다.

---

## 구현된 파일 목록

```
src/
├── utils/                          # 유틸리티 모듈
│   ├── __init__.py
│   ├── metrics.py                  # 평가 지표 계산
│   └── trainer_utils.py            # 학습 헬퍼 함수
│
├── training/                       # 학습 스크립트
│   ├── __init__.py
│   ├── train_dense_moe.py         # Dense MoE 학습
│   ├── train_ppo_moe.py           # PPO-MoE 학습
│   ├── train_grpo_moe.py          # GRPO-MoE 학습
│   └── evaluate.py                 # 통합 평가

configs/                            # 실험 설정
├── dense_moe.yaml
├── ppo_moe.yaml
└── grpo_moe.yaml

README.md                           # 프로젝트 문서
```

---

## 1. 유틸리티 모듈 (`src/utils/`)

### metrics.py

#### 주요 함수

**1. `denormalize_rating(normalized_rating)`**
- 정규화된 평점 (0-1)을 원본 범위 (1-5)로 변환
- 공식: `rating = normalized * 4.0 + 1.0`

**2. `compute_mse(predictions, targets, denormalize=False)`**
- MSE (Mean Squared Error) 계산
- denormalize=True면 원본 스케일로 변환 후 계산

**3. `compute_rmse(predictions, targets, denormalize=False)`**
- RMSE (Root Mean Squared Error) 계산
- RMSE = sqrt(MSE)

**4. `compute_mae(predictions, targets, denormalize=False)`**
- MAE (Mean Absolute Error) 계산
- 평균 절대 오차

**5. `compute_all_metrics(predictions, targets, denormalize=True)`**
- 모든 지표를 한번에 계산
- 반환: `{'mse': float, 'rmse': float, 'mae': float}`

**6. `compute_expert_distribution(actions)`**
- Expert 선택 분포 계산 (강화학습 모델용)
- 반환: `{expert_idx: count}`

**7. `compute_expert_performance(actions, errors, num_experts=8)`**
- 각 Expert의 평균 성능 계산
- 반환: `{expert_idx: avg_error}`

### trainer_utils.py

#### EarlyStopping 클래스

**목적**: 검증 성능이 개선되지 않으면 학습 조기 종료

**파라미터**:
- `patience`: 개선 없이 기다리는 epoch 수
- `min_delta`: 개선으로 간주할 최소 변화량
- `mode`: 'min' (낮을수록 좋음) 또는 'max' (높을수록 좋음)

**사용 예**:
```python
early_stopping = EarlyStopping(patience=10, mode='min')

for epoch in range(epochs):
    val_loss = validate()
    improved = early_stopping(val_loss)

    if early_stopping.early_stop:
        print("Early stopping!")
        break
```

#### CheckpointManager 클래스

**목적**: 모델 체크포인트 저장 및 관리

**파라미터**:
- `checkpoint_dir`: 저장 디렉토리
- `model_name`: 모델 이름
- `max_keep`: 최대 유지할 체크포인트 개수

**주요 메서드**:

1. `save_checkpoint(model, optimizer, epoch, metrics, is_best=False)`
   - 체크포인트 저장
   - is_best=True면 별도로 best 모델 저장
   - 오래된 체크포인트 자동 삭제

2. `load_checkpoint(model, optimizer=None, filename=None)`
   - 체크포인트 로드
   - filename=None이면 best 모델 로드

**사용 예**:
```python
checkpoint_mgr = CheckpointManager("checkpoints", "dense_moe", max_keep=3)

# 저장
checkpoint_mgr.save_checkpoint(model, optimizer, epoch, metrics, is_best=True)

# 로드
info = checkpoint_mgr.load_checkpoint(model, optimizer)
```

#### 기타 유틸리티 함수

- `set_seed(seed)`: 재현성을 위한 시드 설정
- `count_parameters(model)`: 학습 가능한 파라미터 수 계산
- `get_device()`: CUDA/CPU 디바이스 반환
- `AverageMeter`: 평균 계산 헬퍼 클래스
- `print_metrics(metrics, prefix)`: 지표 출력 포맷팅

---

## 2. Dense MoE 학습 (`train_dense_moe.py`)

### 학습 방식
**Supervised Learning**: 일반적인 지도 학습

### 학습 루프

```python
for epoch in range(epochs):
    # 1. 학습 단계
    for batch in train_loader:
        outputs = model(user_id, movie_id, age_group, gender, occupation)
        loss = model.compute_loss(outputs, rating)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 2. 검증 단계
    val_metrics = validate(model, val_loader)

    # 3. Learning rate scheduling
    scheduler.step(val_metrics['loss'])

    # 4. Checkpoint 저장
    checkpoint_manager.save_checkpoint(...)

    # 5. Early stopping 확인
    if early_stopping.early_stop:
        break
```

### 주요 특징

- **손실 함수**: MSE Loss
- **Optimizer**: Adam (lr=0.001)
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=5)
- **Batch Size**: 256
- **Early Stopping**: patience=15

### 실행 명령

```bash
python src/training/train_dense_moe.py \
    --data_dir ml-100k \
    --train_rating_path ml-100k/u1.base \
    --val_rating_path ml-100k/u1.test \
    --batch_size 256 \
    --epochs 100 \
    --lr 0.001
```

---

## 3. PPO-MoE 학습 (`train_ppo_moe.py`)

### 학습 방식
**On-policy Reinforcement Learning**: PPO 알고리즘

### 학습 루프

```python
for epoch in range(epochs):
    # 1. 에피소드 수집 (전체 데이터)
    episode_data = collect_episode(model, train_loader)
    # -> actions, log_probs, values, rewards, predictions, targets

    # 2. Advantages와 Returns 계산 (GAE)
    advantages, returns = model.compute_advantages(
        rewards, values, next_values, dones,
        gamma=0.99, lam=0.95
    )

    # 3. Advantages 정규화
    advantages = (advantages - mean) / (std + eps)

    # 4. PPO 업데이트 (여러 번 반복)
    for _ in range(ppo_epochs):  # 기본 4회
        for batch in mini_batches:
            outputs = model(...)
            losses = model.compute_loss(
                outputs, targets,
                old_log_probs, advantages, returns
            )

            optimizer.zero_grad()
            losses['total_loss'].backward()
            clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

    # 5. 검증
    val_metrics = validate(model, val_loader)
```

### 강화학습 요소

#### State
- User 임베딩 + Movie 임베딩 (128차원)

#### Action
- 8개 Expert 중 하나 선택 (0-7)

#### Reward
- `reward = -|predicted_rating - actual_rating|`
- 오차가 작을수록 높은 보상

#### PPO 손실 함수

```python
total_loss = policy_loss + value_coef * value_loss
             + entropy_coef * entropy_loss + rating_loss
```

**구성 요소**:

1. **Policy Loss** (Clipped Objective):
   ```python
   ratio = exp(log_prob - old_log_prob)
   surr1 = ratio * advantages
   surr2 = clamp(ratio, 1-ε, 1+ε) * advantages
   policy_loss = -min(surr1, surr2).mean()
   ```
   - ε = 0.2 (clip_epsilon)
   - 정책 변화를 제한하여 안정적 학습

2. **Value Loss**:
   ```python
   value_loss = MSE(value, returns)
   ```
   - Value Network의 예측 정확도 향상

3. **Entropy Loss**:
   ```python
   entropy_loss = -entropy.mean()
   ```
   - 탐색(exploration) 촉진

4. **Rating Loss**:
   ```python
   rating_loss = MSE(predicted_rating, actual_rating)
   ```
   - 실제 평점 예측 정확도

### 주요 하이퍼파라미터

```python
batch_size = 256          # PPO 업데이트 배치 크기
ppo_epochs = 4            # 에피소드당 업데이트 횟수
gamma = 0.99              # 할인율
lam = 0.95                # GAE lambda
entropy_coef = 0.01       # 엔트로피 계수
value_coef = 0.5          # Value 손실 계수
max_grad_norm = 0.5       # Gradient clipping
lr = 0.0003               # Learning rate (Adam)
```

### 실행 명령

```bash
python src/training/train_ppo_moe.py \
    --data_dir ml-100k \
    --train_rating_path ml-100k/u1.base \
    --val_rating_path ml-100k/u1.test \
    --batch_size 256 \
    --ppo_epochs 4 \
    --lr 0.0003
```

---

## 4. GRPO-MoE 학습 (`train_grpo_moe.py`)

### 학습 방식
**On-policy Reinforcement Learning**: GRPO 알고리즘

### PPO와의 차이점

| 요소 | PPO | GRPO |
|------|-----|------|
| **보상** | 절대 오차 | 상대 오차 (정규화) |
| **Advantage** | GAE 사용 | Baseline 대비 상대 차이 |
| **추가 네트워크** | Value Network | Baseline Network |

### 학습 루프

```python
for epoch in range(epochs):
    # 1. 에피소드 수집
    episode_data = collect_episode(model, train_loader)
    # -> actions, log_probs, baselines, predictions, targets

    # 2. 원본 보상 계산
    raw_rewards = -|predictions - targets|

    # 3. 그룹 상대 보상 계산 (Z-score 정규화)
    group_rewards = compute_group_relative_rewards(raw_rewards)
    # group_rewards = (rewards - mean) / (std + eps)

    # 4. GRPO 업데이트 (여러 번 반복)
    for _ in range(grpo_epochs):  # 기본 4회
        for batch in mini_batches:
            outputs = model(...)
            losses = model.compute_loss(
                outputs, targets,
                old_log_probs, group_rewards
            )

            optimizer.zero_grad()
            losses['total_loss'].backward()
            clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

    # 5. 검증
    val_metrics = validate(model, val_loader)
```

### GRPO 핵심: 그룹 상대 보상

#### 보상 정규화

```python
def compute_group_relative_rewards(rewards):
    """
    배치(그룹) 내에서 상대적 성능 계산
    """
    mean_reward = rewards.mean()
    std_reward = rewards.std() + 1e-8
    normalized_rewards = (rewards - mean_reward) / std_reward
    return normalized_rewards
```

**장점**:
- 절대적 보상 크기에 덜 민감
- 배치 내 상대적 성능에 집중
- 더 안정적인 학습

### GRPO 손실 함수

```python
total_loss = policy_loss + baseline_coef * baseline_loss
             + entropy_coef * entropy_loss + rating_loss
```

**구성 요소**:

1. **Policy Loss** (Clipped Objective):
   ```python
   advantages = group_rewards - baseline.detach()
   ratio = exp(log_prob - old_log_prob)
   surr1 = ratio * advantages
   surr2 = clamp(ratio, 1-ε, 1+ε) * advantages
   policy_loss = -min(surr1, surr2).mean()
   ```

2. **Baseline Loss**:
   ```python
   baseline_loss = MSE(baseline, group_rewards)
   ```

3. **Entropy Loss**: (PPO와 동일)

4. **Rating Loss**: (PPO와 동일)

### 주요 하이퍼파라미터

```python
batch_size = 256          # GRPO 업데이트 배치 크기
grpo_epochs = 4           # 에피소드당 업데이트 횟수
entropy_coef = 0.01       # 엔트로피 계수
baseline_coef = 0.5       # Baseline 손실 계수
max_grad_norm = 0.5       # Gradient clipping
temperature = 1.0         # Softmax 온도
lr = 0.0003               # Learning rate (Adam)
```

### 실행 명령

```bash
python src/training/train_grpo_moe.py \
    --data_dir ml-100k \
    --train_rating_path ml-100k/u1.base \
    --val_rating_path ml-100k/u1.test \
    --batch_size 256 \
    --grpo_epochs 4 \
    --temperature 1.0
```

---

## 5. 통합 평가 스크립트 (`evaluate.py`)

### 기능

1. **세 가지 모델 통합 평가**: Dense, PPO, GRPO
2. **체크포인트 로드**: Best 모델 자동 로드
3. **성능 비교**: MSE, RMSE, MAE 비교 표 출력
4. **Expert 분석**:
   - Expert 선택 분포 (강화학습 모델)
   - Expert별 성능 (평균 오차)
5. **결과 저장**: JSON 형식으로 저장

### 평가 프로세스

```python
# 1. 모델 로드
model = load_model(model_type, checkpoint_path, model_args, device)

# 2. 평가
if model_type == 'dense':
    metrics = evaluate_dense_moe(model, test_loader, device)
    # -> mse, rmse, mae, avg_gate_probs, gate_entropy
else:  # ppo or grpo
    metrics = evaluate_rl_moe(model, test_loader, device)
    # -> mse, rmse, mae, expert_distribution, expert_performance

# 3. 비교 및 출력
print_comparison_table(results)

# 4. 결과 저장
save_results_to_json(results, output_file)
```

### Dense MoE 평가

**추가 지표**:
- `avg_gate_probs`: 각 Expert의 평균 가중치
- `gate_entropy`: Gating 분포의 엔트로피 (다양성)

**예시 출력**:
```
Dense MoE Results:
  mse: 0.8234
  rmse: 0.9074
  mae: 0.7123
  gate_entropy: 1.8456
```

### 강화학습 MoE 평가

**추가 지표**:
- `expert_distribution`: Expert 선택 빈도
- `expert_performance`: Expert별 평균 오차

**예시 출력**:
```
PPO-MoE Results:
  mse: 0.8156
  rmse: 0.9031
  mae: 0.7089
  Expert distribution: {0: 12450, 1: 9876, 2: 11234, ...}
  Expert performance: {0: 0.6543, 1: 0.7234, 2: 0.6912, ...}
```

### 모델 비교

**출력 형식**:
```
Model Comparison
--------------------------------------------------
Model           MSE        RMSE       MAE
--------------------------------------------------
dense_moe       0.8234     0.9074     0.7123
ppo_moe         0.8156     0.9031     0.7089
grpo_moe        0.8098     0.8999     0.7045
```

### 실행 명령

```bash
# 세 모델 모두 평가
python src/training/evaluate.py \
    --data_dir ml-100k \
    --test_rating_path ml-100k/u1.test \
    --dense_checkpoint checkpoints/dense_moe/dense_moe_best.pt \
    --ppo_checkpoint checkpoints/ppo_moe/ppo_moe_best.pt \
    --grpo_checkpoint checkpoints/grpo_moe/grpo_moe_best.pt \
    --output_file evaluation_results.json

# 단일 모델 평가
python src/training/evaluate.py \
    --test_rating_path ml-100k/u1.test \
    --ppo_checkpoint checkpoints/ppo_moe/ppo_moe_best.pt
```

---

## 6. 실험 설정 파일 (`configs/`)

### 파일 구조

```yaml
# Data
data:
  data_dir: "ml-100k"
  train_rating_path: "ml-100k/u1.base"
  val_rating_path: "ml-100k/u1.test"

# Model
model:
  embedding_dim: 64
  num_experts: 8
  ...

# Training
training:
  batch_size: 256
  epochs: 100
  lr: 0.001
  ...
```

### 설정 파일 비교

| 파라미터 | Dense | PPO | GRPO |
|----------|-------|-----|------|
| **Learning Rate** | 0.001 | 0.0003 | 0.0003 |
| **Gating Hidden** | 128 | - | - |
| **Policy Hidden** | - | 128 | 128 |
| **Value Hidden** | - | 128 | - |
| **PPO Epochs** | - | 4 | - |
| **GRPO Epochs** | - | - | 4 |
| **Temperature** | - | - | 1.0 |

---

## 학습 플로우 비교

### Dense MoE
```
Data → Model → Loss → Backprop → Update
(단순 지도 학습)
```

### PPO-MoE
```
Data → Collect Episode → Compute GAE → PPO Update (4회)
                ↓
        State, Action, Reward
                ↓
        Policy + Value Loss
```

### GRPO-MoE
```
Data → Collect Episode → Group Relative Rewards → GRPO Update (4회)
                ↓
        State, Action, Raw Reward
                ↓
        Z-score Normalization
                ↓
        Policy + Baseline Loss
```

---

## 예상 실험 절차

### 1단계: 베이스라인 학습 (Dense MoE)
```bash
python src/training/train_dense_moe.py --epochs 100
```

### 2단계: PPO 학습
```bash
python src/training/train_ppo_moe.py --epochs 100
```

### 3단계: GRPO 학습
```bash
python src/training/train_grpo_moe.py --epochs 100
```

### 4단계: 통합 평가
```bash
python src/training/evaluate.py \
    --dense_checkpoint checkpoints/dense_moe/dense_moe_best.pt \
    --ppo_checkpoint checkpoints/ppo_moe/ppo_moe_best.pt \
    --grpo_checkpoint checkpoints/grpo_moe/grpo_moe_best.pt
```

---

## 주요 특징 및 장점

### 모듈화
✅ 각 컴포넌트가 독립적으로 동작
✅ 쉬운 확장 및 수정

### 재현성
✅ 시드 고정 (`set_seed`)
✅ 설정 파일 기반 실험
✅ 체크포인트 자동 저장

### 안정성
✅ Early stopping
✅ Gradient clipping
✅ Learning rate scheduling
✅ PPO clipping

### 분석 도구
✅ 다양한 평가 지표
✅ Expert 선택 분석
✅ 모델 비교 자동화

---

## 다음 단계

### 필요한 작업
1. ⏳ **실제 학습 실행**
   - 세 모델 학습 및 비교
   - 하이퍼파라미터 튜닝

2. ⏳ **Ablation Study**
   - Expert 개수 변화 (4, 8, 16)
   - 임베딩 차원 변화
   - 학습률 실험

3. ⏳ **결과 시각화**
   - 학습 곡선 그래프
   - Expert 선택 히트맵
   - 성능 비교 차트

4. ⏳ **추가 실험**
   - Cross-validation (u1-u5)
   - 다른 데이터 분할 (ua, ub)

---

## 기술 스택 요약

- **프레임워크**: PyTorch
- **데이터**: MovieLens 100k
- **모델**: MoE (8 experts)
- **강화학습**: PPO, GRPO
- **평가**: MSE, RMSE, MAE
- **도구**: Early stopping, Checkpointing, Metrics

---

## 파일 크기

```
src/utils/metrics.py           4.2 KB
src/utils/trainer_utils.py     5.8 KB
src/training/train_dense_moe.py   8.1 KB
src/training/train_ppo_moe.py    11.3 KB
src/training/train_grpo_moe.py   10.8 KB
src/training/evaluate.py        10.5 KB
configs/                         ~1 KB (총)
README.md                        5.2 KB
```

**총 구현 코드: ~57 KB**
