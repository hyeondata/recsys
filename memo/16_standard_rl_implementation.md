# 표준 RL 구현 방식 추가

날짜: 2025-11-04

## 개요

기존의 배치 단위 즉시 업데이트 방식 외에, 논문에서 일반적으로 사용되는 **표준 RL 방식**을 구현했습니다.

## 새로 추가된 파일

1. **src/training/train_ppo_moe_standard.py**
   - 표준 PPO 구현
   - 전체 epoch 데이터 수집 → 여러 번 업데이트

2. **src/training/train_grpo_moe_standard.py**
   - 표준 GRPO 구현
   - 전체 epoch 데이터 수집 → 여러 번 업데이트

## 구현 방식 비교

### 기존 방식 (Batch-wise Immediate Update)

```python
# train_ppo_moe.py, train_grpo_moe.py
for batch in dataloader:  # 각 배치마다 (256 샘플)
    # 1. 현재 배치에서 에피소드 수집
    old_outputs = model(batch)
    rewards = compute_rewards(batch)

    # 2. 같은 배치를 4번 업데이트
    for _ in range(4):
        outputs = model(batch)
        loss = compute_loss(outputs, rewards)
        optimizer.step()
```

**특징:**
- 메모리: 256 샘플 (배치 크기)
- 업데이트: 각 배치당 4번
- Epoch당 총 업데이트: `(79,619 / 256) × 4 = 약 1,245번`
- 장점: 메모리 효율적 (300배 절약)
- 단점: 작은 데이터로 정책 업데이트

### 표준 RL 방식 (Epoch-wise Collection + Multiple Updates)

```python
# train_ppo_moe_standard.py, train_grpo_moe_standard.py
for epoch in range(num_epochs):
    # Phase 1: 전체 epoch 데이터 수집
    all_episodes = []
    for batch in dataloader:
        episodes = collect_episode(batch)
        all_episodes.append(episodes)

    # Phase 2: 전체 데이터에 대한 reward/advantage 계산
    rewards = compute_rewards(all_episodes)  # 79,619 샘플

    # Phase 3: 수집된 데이터로 여러 epoch 학습
    for update_epoch in range(4):
        for mini_batch in create_mini_batches(all_episodes, size=256):
            outputs = model(mini_batch)
            loss = compute_loss(outputs, rewards)
            optimizer.step()
```

**특징:**
- 메모리: 79,619 샘플 (전체 데이터)
- 업데이트: `(79,619 / 256) × 4 = 약 1,245번`
- Epoch당 총 업데이트: 약 1,245번 (동일)
- 장점:
  - 충분한 exploration
  - 전체 데이터로 정확한 advantage 계산
  - 논문 재현에 적합
- 단점: 메모리 사용량 높음

## 주요 차이점

| 항목 | 기존 방식 | 표준 RL 방식 |
|------|----------|-------------|
| 데이터 수집 범위 | 배치 단위 (256) | Epoch 단위 (79,619) |
| 메모리 사용 | 낮음 (256 샘플) | 높음 (79,619 샘플) |
| Reward 계산 범위 | 배치 내 (256) | 전체 epoch (79,619) |
| GRPO Group 크기 | 배치 크기 (256) | 설정 가능 (기본 256) |
| 업데이트 횟수 | 동일 (약 1,245번) | 동일 (약 1,245번) |
| 논문 재현성 | 낮음 | 높음 |
| 구현 복잡도 | 낮음 | 중간 |

## 알고리즘 비교

### PPO Standard

```python
# Phase 1: Episode Collection
episodes = collect_full_epoch_episodes(model, train_loader, device)
# → 79,619 샘플의 (state, action, log_prob, value, prediction) 수집

# Phase 2: Advantage Computation
rewards = -|predictions - targets|  # 79,619개
advantages = rewards - values       # 전체 데이터에 대해 계산
advantages = normalize(advantages)

# Phase 3: Multiple Update Epochs
for ppo_epoch in range(4):
    for mini_batch in create_mini_batches(episodes, batch_size=256):
        # 새로운 log_prob으로 ratio 계산
        ratio = exp(new_log_prob - old_log_prob)
        policy_loss = -min(ratio * adv, clip(ratio) * adv)

        optimizer.step()
```

### GRPO Standard

```python
# Phase 1: Episode Collection
episodes = collect_full_epoch_episodes(model, train_loader, device)
# → 79,619 샘플의 (state, action, log_prob, baseline, prediction) 수집

# Phase 2: Group Relative Reward Computation
raw_rewards = compute_rewards_from_errors(predictions, targets)  # 79,619개

# 전체 데이터를 그룹으로 나눠서 상대적 보상 계산
group_rewards = []
for group in split_into_groups(raw_rewards, group_size=256):
    group_mean = group.mean()
    group_std = group.std()
    relative = (group - group_mean) / group_std
    group_rewards.append(relative)

# Phase 3: Multiple Update Epochs
for grpo_epoch in range(4):
    for mini_batch in create_mini_batches(episodes, batch_size=256):
        ratio = exp(new_log_prob - old_log_prob)
        policy_loss = -ratio * group_rewards

        optimizer.step()
```

## 사용법

### 표준 PPO-MoE

```bash
uv run python3 src/training/train_ppo_moe_standard.py \
    --epochs 10 \
    --batch_size 512 \
    --mini_batch_size 256 \
    --ppo_epochs 4 \
    --lr 0.0003 \
    --checkpoint_dir checkpoints/ppo_moe_standard
```

### 표준 GRPO-MoE

```bash
uv run python3 src/training/train_grpo_moe_standard.py \
    --epochs 10 \
    --batch_size 512 \
    --mini_batch_size 256 \
    --grpo_epochs 4 \
    --group_size 256 \
    --lr 0.0003 \
    --checkpoint_dir checkpoints/grpo_moe_standard
```

### 주요 파라미터

- `--batch_size`: 에피소드 수집 시 배치 크기 (512 권장, 메모리 허용 시 더 크게 가능)
- `--mini_batch_size`: 업데이트 시 mini-batch 크기 (256)
- `--ppo_epochs` / `--grpo_epochs`: 수집된 데이터로 몇 번 학습할지 (4)
- `--group_size`: (GRPO) Group relative reward 계산 시 그룹 크기 (256)

## 예상 성능 차이

### 기존 방식의 장점
1. **메모리 효율성**: GPU 메모리 사용량 최소화
2. **안정성**: 작은 배치로 학습하여 GPU 드라이버 크래시 방지
3. **빠른 업데이트**: 배치마다 즉시 업데이트

### 표준 RL 방식의 장점
1. **더 나은 Exploration**: 전체 데이터에서 다양한 상황 학습
2. **정확한 Advantage**: 전체 epoch 데이터로 더 정확한 추정
3. **안정적인 학습**: 충분한 데이터로 정책 학습
4. **논문 재현**: 일반적인 RL 논문과 동일한 방식

## 예상 결과

표준 RL 방식이 **더 나은 성능**을 보일 가능성이 높습니다:

1. **PPO-MoE Standard**: RMSE < 1.0659 (기존)
2. **GRPO-MoE Standard**: RMSE < 1.0412 (기존)

특히 GRPO의 경우 전체 데이터에서 그룹을 나눠 상대적 보상을 계산하므로, 더 정확한 보상 신호를 제공할 수 있습니다.

## 권장사항

### 언제 기존 방식을 사용할까?
- GPU 메모리가 제한적일 때
- 빠른 프로토타이핑이 필요할 때
- Dense MoE와 공정한 비교가 필요할 때

### 언제 표준 RL 방식을 사용할까?
- 최고 성능을 원할 때
- 논문 출판을 위한 벤치마크
- 충분한 GPU 메모리가 있을 때 (24GB+)
- 일반적인 RL 알고리즘과 비교할 때

## 구현 세부사항

### 메모리 관리

표준 RL 방식은 전체 epoch 데이터를 메모리에 저장하므로:

```python
# 메모리 사용량 추정
episodes = {
    'user_ids': (79619,),           # 4 bytes × 79619 ≈ 318KB
    'movie_ids': (79619,),          # 4 bytes × 79619 ≈ 318KB
    'ratings': (79619,),            # 4 bytes × 79619 ≈ 318KB
    'old_log_probs': (79619,),      # 4 bytes × 79619 ≈ 318KB
    'predictions': (79619,),        # 4 bytes × 79619 ≈ 318KB
    # ... 기타 필드들
}
# 총 약 10개 필드 × 318KB ≈ 3.2MB (CPU 메모리)
```

GPU 메모리는 mini-batch만 사용하므로 문제없습니다.

### 성능 최적화

1. **Batch size for collection**: 512 이상 (메모리 허용 시)
2. **Mini-batch size**: 256 (업데이트 안정성)
3. **PPO/GRPO epochs**: 4-10 (표준 범위)
4. **Workers**: 4-8 (데이터 로딩 속도)

## 다음 단계

1. 표준 RL 방식으로 학습 실행
2. 기존 방식과 성능 비교
3. Learning curves 비교 시각화
4. 논문용 결과 업데이트

## 참고 문헌

- **PPO**: Schulman et al. (2017) - Proximal Policy Optimization Algorithms
- **OpenAI Baselines**: 표준 PPO 구현 (timesteps=2048, epochs=10)
- **GRPO**: Group Relative Policy Optimization (그룹 상대 보상)
