# MoE 모델 구현 완료

## 구현 일자
2025년 11월 2일

## 구현 개요
MovieLens 100k 데이터셋을 활용한 MoE(Mixture of Experts) 기반 영화 추천 시스템의 모델 아키텍처를 완성했습니다.

---

## 구현된 파일 목록

```
src/models/
├── __init__.py              # 모델 패키지 초기화 및 export
├── expert_network.py        # Expert Network 모듈
├── base_moe.py             # MoE 기본 클래스
├── dense_moe.py            # Dense Gating MoE 모델
├── ppo_moe.py              # PPO 강화학습 MoE 모델
└── grpo_moe.py             # GRPO 강화학습 MoE 모델
```

---

## 1. Expert Network (`expert_network.py`)

### ExpertNetwork 클래스
**목적**: 단일 Expert Network (FFN 구조)

**아키텍처**:
```
Input (combined_dim=128)
  ↓
Linear(input_dim → hidden_dim=256)
  ↓
ReLU + Dropout(0.1)
  ↓
Linear(hidden_dim → hidden_dim//2=128)
  ↓
ReLU + Dropout(0.1)
  ↓
Linear(hidden_dim//2 → output_dim=1)
  ↓
Output (rating prediction)
```

### ExpertEnsemble 클래스
**목적**: 8개의 Expert를 관리하는 앙상블

**기능**:
- `forward(x, expert_indices=None)`:
  - `expert_indices=None`: 모든 Expert 출력 반환 `[batch, 8, 1]`
  - `expert_indices` 지정: 선택된 Expert만 실행 `[batch, 1]`

---

## 2. Base MoE Model (`base_moe.py`)

### BaseMoEModel 클래스
**목적**: 모든 MoE 모델의 기본 클래스

**주요 컴포넌트**:

#### 임베딩 레이어
| 임베딩 | 입력 | 차원 | 설명 |
|--------|------|------|------|
| `user_embedding` | user_id | 64 | 사용자 기본 임베딩 |
| `age_embedding` | age_group (0-7) | 16 | 나이 그룹 임베딩 |
| `gender_embedding` | gender (0:M, 1:F) | 16 | 성별 임베딩 |
| `occupation_embedding` | occupation (0-20) | 16 | 직업 임베딩 |
| `movie_embedding` | movie_id | 64 | 영화 임베딩 |

#### User Projection
```
User Features (64 + 16 + 16 + 16 = 112)
  ↓
Linear(112 → 64) + ReLU + Dropout
  ↓
User Embedding (64)
```

#### State 생성
```
State = [User Embedding (64) | Movie Embedding (64)]
State Shape: [batch_size, 128]
```

**주요 메서드**:
- `get_user_embedding()`: User 특성 결합 및 임베딩
- `get_movie_embedding()`: Movie 임베딩
- `get_state()`: 강화학습의 State 벡터 생성
- `forward()`: 하위 클래스에서 구현 필요

---

## 3. Dense MoE (`dense_moe.py`)

### 모델 구조
**Gating 방식**: Fully Connected Layer (비강화학습)

```
State (128)
  ↓
Gating Network:
  Linear(128 → 128) + ReLU + Dropout
  ↓
  Linear(128 → 64) + ReLU + Dropout
  ↓
  Linear(64 → 8)  # Expert 개수
  ↓
Softmax → Gate Weights [batch_size, 8]

모든 Expert 실행 → Expert Outputs [batch_size, 8, 1]
  ↓
Weighted Sum (Gate Weights × Expert Outputs)
  ↓
Sigmoid → Rating [0, 1]
```

### Forward 출력
```python
{
    'rating': [batch_size],              # 예측 평점 (0-1)
    'gate_logits': [batch_size, 8],      # Gating logits
    'gate_probs': [batch_size, 8],       # Softmax 확률
    'expert_outputs': [batch_size, 8, 1] # 각 Expert 출력
}
```

### 손실 함수
- **Rating Loss**: MSE(predicted_rating, target_rating)
- **특징**: 단순하고 안정적인 베이스라인 모델

---

## 4. PPO-MoE (`ppo_moe.py`)

### 모델 구조
**Gating 방식**: PPO(Proximal Policy Optimization) 강화학습

#### Policy Network (Actor)
```
State (128)
  ↓
Linear(128 → 128) + ReLU + Dropout
  ↓
Linear(128 → 64) + ReLU + Dropout
  ↓
Linear(64 → 8)  # Expert 개수
  ↓
Softmax → Action Probs [batch_size, 8]
  ↓
Categorical Sampling → Action (Expert Index)
```

#### Value Network (Critic)
```
State (128)
  ↓
Linear(128 → 128) + ReLU + Dropout
  ↓
Linear(128 → 64) + ReLU + Dropout
  ↓
Linear(64 → 1)
  ↓
State Value [batch_size]
```

### Forward 출력
```python
{
    'rating': [batch_size],          # 예측 평점
    'action': [batch_size],          # 선택된 Expert 인덱스
    'log_prob': [batch_size],        # Action의 로그 확률
    'entropy': [batch_size],         # 정책 엔트로피
    'value': [batch_size],           # State 가치
    'action_probs': [batch_size, 8]  # Expert 선택 확률
}
```

### 강화학습 요소

#### State
- User 임베딩 + Movie 임베딩 (128차원)

#### Action
- 8개 Expert 중 하나 선택 (0-7)

#### Reward
- `reward = -|predicted_rating - actual_rating|`
- 예측 오차가 작을수록 높은 보상

### 손실 함수
```python
total_loss = policy_loss + value_coef * value_loss
             + entropy_coef * entropy_loss + rating_loss
```

**구성 요소**:
1. **Policy Loss** (PPO Clipped Objective):
   ```
   ratio = exp(log_prob - old_log_prob)
   surr1 = ratio * advantages
   surr2 = clamp(ratio, 1-ε, 1+ε) * advantages
   loss = -min(surr1, surr2).mean()
   ```
   - `ε = 0.2` (clip_epsilon)

2. **Value Loss**:
   ```
   loss = MSE(value, returns)
   ```

3. **Entropy Loss**:
   ```
   loss = -entropy.mean()
   ```
   - 탐색(exploration) 촉진

4. **Rating Loss**:
   ```
   loss = MSE(predicted_rating, actual_rating)
   ```

### GAE (Generalized Advantage Estimation)
```python
advantages, returns = compute_advantages(
    rewards, values, next_values, dones,
    gamma=0.99,  # 할인율
    lam=0.95     # GAE lambda
)
```

---

## 5. GRPO-MoE (`grpo_moe.py`)

### 모델 구조
**Gating 방식**: GRPO(Group Relative Policy Optimization)

#### Policy Network
```
State (128)
  ↓
Linear(128 → 128) + ReLU + Dropout
  ↓
Linear(128 → 64) + ReLU + Dropout
  ↓
Linear(64 → 8) / temperature
  ↓
Softmax → Action Probs [batch_size, 8]
```

#### Baseline Network
```
State (128)
  ↓
Linear(128 → 128) + ReLU + Dropout
  ↓
Linear(128 → 64) + ReLU + Dropout
  ↓
Linear(64 → 1)
  ↓
Group Baseline [batch_size]
```

### Forward 출력
```python
{
    'rating': [batch_size],          # 예측 평점
    'action': [batch_size],          # 선택된 Expert 인덱스
    'log_prob': [batch_size],        # Action의 로그 확률
    'entropy': [batch_size],         # 정책 엔트로피
    'baseline': [batch_size],        # Baseline 예측
    'action_probs': [batch_size, 8]  # Expert 선택 확률
}
```

### GRPO 핵심: Group Relative Rewards

#### 보상 정규화 (Z-score)
```python
mean_reward = rewards.mean()
std_reward = rewards.std() + 1e-8
normalized_rewards = (rewards - mean_reward) / std_reward
```

**특징**:
- 배치(그룹) 내에서 상대적 성능 비교
- 절대적 보상 크기에 덜 민감
- 더 안정적인 학습

### 손실 함수
```python
total_loss = policy_loss + baseline_coef * baseline_loss
             + entropy_coef * entropy_loss + rating_loss
```

**구성 요소**:
1. **Policy Loss** (GRPO Clipped Objective):
   ```
   advantages = group_rewards - baseline.detach()
   ratio = exp(log_prob - old_log_prob)
   surr1 = ratio * advantages
   surr2 = clamp(ratio, 1-ε, 1+ε) * advantages
   loss = -min(surr1, surr2).mean()
   ```

2. **Baseline Loss**:
   ```
   loss = MSE(baseline, group_rewards)
   ```

3. **Entropy Loss**: (PPO와 동일)

4. **Rating Loss**: (PPO와 동일)

---

## 모델 비교

| 특징 | Dense MoE | PPO-MoE | GRPO-MoE |
|------|-----------|---------|----------|
| **Gating 방식** | FC Layer | PPO RL | GRPO RL |
| **Expert 선택** | 가중합 (모두 사용) | 단일 선택 | 단일 선택 |
| **학습 방식** | Supervised | On-policy RL | On-policy RL |
| **보상 기준** | - | 절대 오차 | 상대 오차 (정규화) |
| **안정성** | 높음 | 중간 | 높음 |
| **복잡도** | 낮음 | 높음 | 높음 |
| **탐색** | 없음 | Entropy 기반 | Entropy 기반 |
| **추가 네트워크** | Gating | Policy + Value | Policy + Baseline |

---

## 하이퍼파라미터 설정

### 공통 파라미터
```python
embedding_dim = 64          # User/Movie 임베딩 차원
num_experts = 8             # Expert 개수
expert_hidden_dim = 256     # Expert 은닉층 차원
dropout = 0.1               # 드롭아웃 비율
```

### Dense MoE
```python
gating_hidden_dim = 128     # Gating Network 은닉층
```

### PPO-MoE
```python
policy_hidden_dim = 128     # Policy Network 은닉층
value_hidden_dim = 128      # Value Network 은닉층
clip_epsilon = 0.2          # PPO clipping
gamma = 0.99                # 할인율
lam = 0.95                  # GAE lambda
entropy_coef = 0.01         # 엔트로피 계수
value_coef = 0.5            # Value 손실 계수
```

### GRPO-MoE
```python
policy_hidden_dim = 128     # Policy Network 은닉층
clip_epsilon = 0.2          # Clipping
temperature = 1.0           # Softmax 온도
entropy_coef = 0.01         # 엔트로피 계수
baseline_coef = 0.5         # Baseline 손실 계수
```

---

## 데이터 입력 형식

모든 모델은 다음 입력을 받습니다:

```python
inputs = {
    'user_id': torch.Tensor,      # [batch_size] - 0부터 시작
    'movie_id': torch.Tensor,     # [batch_size] - 0부터 시작
    'age_group': torch.Tensor,    # [batch_size] - 0~7
    'gender': torch.Tensor,       # [batch_size] - 0:M, 1:F
    'occupation': torch.Tensor,   # [batch_size] - 0~20
}

# EnhancedMovieRatingDataset에서 제공
```

---

## 예상 사용 방법

### 1. Dense MoE (베이스라인)
```python
from src.models import DenseMoE

model = DenseMoE(
    num_users=943,
    num_movies=1682,
    num_age_groups=8,
    num_occupations=21
)

# Forward
outputs = model(user_id, movie_id, age_group, gender, occupation)
loss = model.compute_loss(outputs, target_ratings)
```

### 2. PPO-MoE
```python
from src.models import PPOMoE

model = PPOMoE(
    num_users=943,
    num_movies=1682,
    num_age_groups=8,
    num_occupations=21
)

# Forward
outputs = model(user_id, movie_id, age_group, gender, occupation)

# Compute advantages
advantages, returns = model.compute_advantages(
    rewards, values, next_values, dones
)

# Loss
losses = model.compute_loss(
    outputs, target_ratings,
    old_log_probs, advantages, returns
)
```

### 3. GRPO-MoE
```python
from src.models import GRPOMoE

model = GRPOMoE(
    num_users=943,
    num_movies=1682,
    num_age_groups=8,
    num_occupations=21
)

# Forward
outputs = model(user_id, movie_id, age_group, gender, occupation)

# Compute group relative rewards
raw_rewards = model.compute_rewards_from_errors(
    outputs['rating'], target_ratings
)
group_rewards = model.compute_group_relative_rewards(raw_rewards)

# Loss
losses = model.compute_loss(
    outputs, target_ratings,
    old_log_probs, group_rewards
)
```

---

## 다음 단계

### 필요한 구현
1. ✅ 모델 아키텍처 (완료)
2. ⏳ 학습 스크립트 (`train.py`)
   - DataLoader 설정
   - 학습 루프
   - 강화학습 에피소드 관리
3. ⏳ 평가 스크립트 (`evaluate.py`)
   - MSE, RMSE, MAE 계산
   - Expert 선택 분포 분석
4. ⏳ 실험 설정 (`config.yaml`)
5. ⏳ 결과 시각화 및 분석

### 실험 계획
1. **베이스라인**: Dense MoE 학습 및 평가
2. **PPO 비교**: PPO-MoE vs Dense MoE
3. **GRPO 비교**: GRPO-MoE vs PPO-MoE vs Dense MoE
4. **Ablation Study**:
   - Expert 개수 변화 (4, 8, 16)
   - 임베딩 차원 변화
   - 하이퍼파라미터 튜닝

---

## 구현 특징

### 강점
✅ **모듈화**: 각 컴포넌트가 명확히 분리됨
✅ **확장성**: 새로운 Gating 방식 추가 용이
✅ **표준 인터페이스**: 모든 모델이 동일한 입출력 형식
✅ **강화학습 지원**: PPO와 GRPO 구현
✅ **사용자 특성 활용**: 나이, 성별, 직업 임베딩

### 기술 스택
- **프레임워크**: PyTorch
- **강화학습**: On-policy (PPO, GRPO)
- **최적화**: Adam (예정)
- **평가 지표**: MSE, RMSE, MAE

---

## 참고 사항

### 강화학습 학습 시 고려사항
1. **Exploration-Exploitation Trade-off**
   - Entropy 계수로 탐색 조절
   - Temperature로 확률 분포 조절

2. **Reward Shaping**
   - 오차 기반 보상: `-|pred - actual|`
   - 정규화 중요 (GRPO)

3. **안정성**
   - PPO clipping으로 정책 변화 제한
   - Value/Baseline 네트워크로 분산 감소

4. **배치 크기**
   - 강화학습은 큰 배치 권장 (256+)
   - GRPO는 그룹 통계 계산을 위해 특히 중요
