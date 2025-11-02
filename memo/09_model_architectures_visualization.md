# MoE 모델 아키텍처 시각화

## 작성 일자
2025년 11월 2일

---

## 목차
1. [Dense MoE 아키텍처](#1-dense-moe-아키텍처)
2. [PPO-MoE 아키텍처](#2-ppo-moe-아키텍처)
3. [GRPO-MoE 아키텍처](#3-grpo-moe-아키텍처)
4. [아키텍처 비교](#4-아키텍처-비교)

---

## 1. Dense MoE 아키텍처

### 개요
- **Gating 방식**: Softmax (모든 Expert 가중치 조합)
- **Expert 활용**: 8개 Expert 모두 사용
- **출력**: Weighted sum of all experts
- **특징**: 안정적이고 예측 가능한 학습

### 구조 다이어그램

```
┌─────────────────────────────────────────────────────────────┐
│                        Input Layer                          │
│  User ID │ Movie ID │ Age Group │ Gender │ Occupation      │
└────┬─────────────┬──────────┬──────────┬─────────┬─────────┘
     │             │          │          │         │
     ▼             ▼          ▼          ▼         ▼
┌─────────┐  ┌─────────┐  ┌────┐  ┌────┐  ┌────────┐
│User Emb │  │Movie Emb│  │Age │  │Gen │  │Occup   │
│ (64)    │  │ (64)    │  │(8) │  │(2) │  │Emb(21) │
└────┬────┘  └────┬────┘  └─┬──┘  └─┬──┘  └───┬────┘
     └────────────┴─────────┴───────┴─────────┘
                    │
                    ▼
          ┌─────────────────┐
          │   State Vector  │
          │     (concat)    │
          └────────┬────────┘
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
┌──────────────────┐  ┌──────────────────┐
│ Gating Network   │  │  Expert Networks │
│                  │  │   (8 Experts)    │
│  FC(state, 128)  │  │                  │
│     ReLU         │  │  Expert 0        │
│  Dropout(0.1)    │  │  Expert 1        │
│  FC(128, 8)      │  │  Expert 2        │
│   Softmax        │  │  Expert 3        │
│                  │  │  Expert 4        │
│  ┌─────────┐    │  │  Expert 5        │
│  │ Gate    │    │  │  Expert 6        │
│  │ Probs   │    │  │  Expert 7        │
│  │  (8)    │    │  │                  │
│  └────┬────┘    │  │  각 Expert:      │
│       │         │  │  FC(state, 256)  │
└───────┼─────────┘  │  ReLU            │
        │            │  Dropout(0.1)    │
        │            │  FC(256, 1)      │
        │            └────────┬─────────┘
        │                     │
        │         ┌───────────┴──────────┐
        │         │   Expert Outputs     │
        │         │   [o0, o1, ..., o7]  │
        │         └───────────┬──────────┘
        │                     │
        └──────────┬──────────┘
                   │
                   ▼
          ┌────────────────┐
          │  Weighted Sum  │
          │                │
          │  output =      │
          │  Σ gate[i] *   │
          │    expert[i]   │
          └────────┬───────┘
                   │
                   ▼
          ┌────────────────┐
          │  Final Rating  │
          │   Prediction   │
          │   (1-5 scale)  │
          └────────────────┘
```

### 수식

**State 생성:**
```
state = concat([user_emb, movie_emb, age_emb, gender_emb, occup_emb])
```

**Gating 확률:**
```
gate_logits = FC_gating2(ReLU(Dropout(FC_gating1(state))))
gate_probs = Softmax(gate_logits)  # shape: [batch_size, 8]
```

**Expert 출력:**
```
expert_i = FC_expert2(ReLU(Dropout(FC_expert1(state))))  # shape: [batch_size, 1]
expert_outputs = stack([expert_0, ..., expert_7])  # shape: [batch_size, 8]
```

**최종 예측:**
```
rating = Σ(gate_probs[i] * expert_outputs[i])  # shape: [batch_size, 1]
```

### 파라미터 수
- User Embedding: 943 × 64 = 60,352
- Movie Embedding: 1,682 × 64 = 107,648
- Age Embedding: 8 × 8 = 64
- Occupation Embedding: 21 × 16 = 336
- Gating Network: (146 × 128) + 128 + (128 × 8) + 8 = 19,816
- Expert Networks: 8 × [(146 × 256) + 256 + (256 × 1) + 1] = 300,288
- **총 파라미터: 729,408**

### 학습 방식
```python
# Forward
outputs = model(user_id, movie_id, age_group, gender, occupation)
rating_pred = outputs['rating']
gate_probs = outputs['gate_probs']

# Loss
loss = MSE(rating_pred, rating_true)

# Backward
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

---

## 2. PPO-MoE 아키텍처

### 개요
- **Gating 방식**: Categorical Sampling (하나의 Expert 선택)
- **Expert 활용**: 8개 중 1개 선택
- **강화학습**: PPO 알고리즘
- **특징**: 효율적이지만 학습이 불안정할 수 있음

### 구조 다이어그램

```
┌─────────────────────────────────────────────────────────────┐
│                        Input Layer                          │
│  User ID │ Movie ID │ Age Group │ Gender │ Occupation      │
└────┬─────────────┬──────────┬──────────┬─────────┬─────────┘
     │             │          │          │         │
     ▼             ▼          ▼          ▼         ▼
┌─────────┐  ┌─────────┐  ┌────┐  ┌────┐  ┌────────┐
│User Emb │  │Movie Emb│  │Age │  │Gen │  │Occup   │
│ (64)    │  │ (64)    │  │(8) │  │(2) │  │Emb(21) │
└────┬────┘  └────┬────┘  └─┬──┘  └─┬──┘  └───┬────┘
     └────────────┴─────────┴───────┴─────────┘
                    │
                    ▼
          ┌─────────────────┐
          │   State Vector  │
          │     (concat)    │
          └────┬────────────┘
               │
      ┌────────┴────────┐
      │                 │
      ▼                 ▼
┌──────────────┐  ┌──────────────┐
│Policy Network│  │Value Network │
│              │  │              │
│FC(state, 128)│  │FC(state, 128)│
│   ReLU       │  │   ReLU       │
│Dropout(0.1)  │  │Dropout(0.1)  │
│FC(128, 8)    │  │FC(128, 1)    │
│              │  │              │
│┌───────────┐ │  │┌───────────┐ │
││  Logits   │ │  ││  Value    │ │
││   (8)     │ │  ││   (1)     │ │
│└─────┬─────┘ │  │└───────────┘ │
│      │       │  │              │
│      ▼       │  │  Critic for  │
│  Categorical │  │  advantage   │
│   Sampling   │  │  estimation  │
│      │       │  │              │
│  ┌───┴────┐  │  └──────────────┘
│  │ Action │  │
│  │  (0-7) │  │
│  └───┬────┘  │
│      │       │
│  ┌───┴────┐  │
│  │Log Prob│  │
│  └────────┘  │
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│ Expert Networks  │
│  (8 Experts)     │
│                  │
│ SELECT ONE based │
│   on action      │
│                  │
│  Expert[action]  │
│  FC(state, 256)  │
│     ReLU         │
│  Dropout(0.1)    │
│  FC(256, 1)      │
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│  Final Rating   │
│   Prediction    │
│  (1-5 scale)    │
└─────────────────┘

┌─────────────────────────────────────┐
│       PPO Training Loop             │
│                                     │
│ 1. Collect Episodes:                │
│    - Run policy (stochastic)        │
│    - Store (s, a, log_p, r, v)      │
│                                     │
│ 2. Compute Advantages:              │
│    - GAE (λ=0.95, γ=0.99)           │
│    - Normalize advantages           │
│                                     │
│ 3. PPO Update (4 epochs):           │
│    - Resample batches               │
│    - Compute ratio = π_new / π_old  │
│    - Clip ratio to [1-ε, 1+ε]      │
│    - Policy loss (clipped)          │
│    - Value loss (MSE)               │
│    - Entropy bonus                  │
│                                     │
│ 4. Repeat for each epoch            │
└─────────────────────────────────────┘
```

### 수식

**Policy (Expert 선택):**
```
policy_logits = FC_policy2(ReLU(Dropout(FC_policy1(state))))
action_probs = Softmax(policy_logits)
action ~ Categorical(action_probs)  # 0-7
log_prob = log(action_probs[action])
```

**Value (상태 가치):**
```
value = FC_value2(ReLU(Dropout(FC_value1(state))))
```

**Expert 출력:**
```
selected_expert = experts[action]
rating = selected_expert(state)
```

**Reward:**
```
reward = -|rating_pred - rating_true|
```

**Advantage (GAE):**
```
δ_t = reward_t + γ * value_{t+1} - value_t
advantage_t = Σ (γλ)^k * δ_{t+k}
```

**PPO Loss:**
```
ratio = exp(log_prob_new - log_prob_old)
ratio_clipped = clip(ratio, 1-ε, 1+ε)

policy_loss = -min(
    ratio * advantage,
    ratio_clipped * advantage
)

value_loss = (value - returns)^2

entropy = -Σ action_probs * log(action_probs)

total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
```

### 파라미터 수
- Embeddings: 168,400 (same as Dense)
- Policy Network: (146 × 128) + 128 + (128 × 8) + 8 = 19,816
- Value Network: (146 × 128) + 128 + (128 × 1) + 1 = 18,817
- Expert Networks: 300,288 (same as Dense)
- **총 파라미터: 754,241**

### 학습 방식
```python
# 1. Episode Collection (eval mode)
with torch.no_grad():
    outputs = model(user_id, movie_id, age_group, gender, occupation)
    action = outputs['action']
    log_prob = outputs['log_prob']
    value = outputs['value']
    rating_pred = outputs['rating']
    reward = -abs(rating_pred - rating_true)

# 2. Compute Advantages
advantages, returns = compute_advantages(rewards, values, gamma, lam)

# 3. PPO Update (train mode, 4 epochs)
for ppo_epoch in range(4):
    for batch in minibatches:
        outputs = model(batch_inputs)

        # Ratio
        ratio = exp(outputs['log_prob'] - old_log_prob)
        ratio_clipped = clip(ratio, 1-epsilon, 1+epsilon)

        # Losses
        policy_loss = -min(ratio * adv, ratio_clipped * adv).mean()
        value_loss = (outputs['value'] - returns).pow(2).mean()
        entropy = -(outputs['action_probs'] * log(outputs['action_probs'])).sum(-1).mean()

        total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
```

---

## 3. GRPO-MoE 아키텍처

### 개요
- **Gating 방식**: Categorical Sampling (하나의 Expert 선택)
- **Expert 활용**: 8개 중 1개 선택
- **강화학습**: GRPO (Group Relative Policy Optimization)
- **특징**: 상대적 보상으로 더 안정적인 학습

### 구조 다이어그램

```
┌─────────────────────────────────────────────────────────────┐
│                        Input Layer                          │
│  User ID │ Movie ID │ Age Group │ Gender │ Occupation      │
└────┬─────────────┬──────────┬──────────┬─────────┬─────────┘
     │             │          │          │         │
     ▼             ▼          ▼          ▼         ▼
┌─────────┐  ┌─────────┐  ┌────┐  ┌────┐  ┌────────┐
│User Emb │  │Movie Emb│  │Age │  │Gen │  │Occup   │
│ (64)    │  │ (64)    │  │(8) │  │(2) │  │Emb(21) │
└────┬────┘  └────┬────┘  └─┬──┘  └─┬──┘  └───┬────┘
     └────────────┴─────────┴───────┴─────────┘
                    │
                    ▼
          ┌─────────────────┐
          │   State Vector  │
          │     (concat)    │
          └────┬────────────┘
               │
      ┌────────┴────────┐
      │                 │
      ▼                 ▼
┌──────────────┐  ┌──────────────┐
│Policy Network│  │Baseline Net  │
│              │  │              │
│FC(state, 128)│  │FC(state, 128)│
│   ReLU       │  │   ReLU       │
│Dropout(0.1)  │  │Dropout(0.1)  │
│FC(128, 8)    │  │FC(128, 1)    │
│   /T (temp)  │  │              │
│              │  │┌───────────┐ │
│┌───────────┐ │  ││ Baseline  │ │
││  Logits   │ │  ││   (1)     │ │
││   (8)     │ │  │└───────────┘ │
│└─────┬─────┘ │  │              │
│      │       │  │ For relative │
│      ▼       │  │   rewards    │
│  Categorical │  └──────────────┘
│   Sampling   │
│      │       │
│  ┌───┴────┐  │
│  │ Action │  │
│  │  (0-7) │  │
│  └───┬────┘  │
│      │       │
│  ┌───┴────┐  │
│  │Log Prob│  │
│  └────────┘  │
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│ Expert Networks  │
│  (8 Experts)     │
│                  │
│ SELECT ONE based │
│   on action      │
│                  │
│  Expert[action]  │
│  FC(state, 256)  │
│     ReLU         │
│  Dropout(0.1)    │
│  FC(256, 1)      │
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│  Final Rating   │
│   Prediction    │
│  (1-5 scale)    │
└─────────────────┘

┌──────────────────────────────────────┐
│       GRPO Training Loop             │
│                                      │
│ 1. Collect Episodes:                 │
│    - Run policy (stochastic)         │
│    - Store (s, a, log_p, r, b)       │
│    - r = -|pred - true|              │
│                                      │
│ 2. Compute Group Relative Rewards:   │
│    - Normalize raw rewards           │
│    - relative_r = (r - mean) / std   │
│    - More stable gradient            │
│                                      │
│ 3. GRPO Update (4 epochs):           │
│    - Resample batches                │
│    - Policy loss: -log_prob * rel_r  │
│    - Baseline loss: MSE(b, raw_r)    │
│    - Entropy bonus                   │
│                                      │
│ 4. Repeat for each epoch             │
└──────────────────────────────────────┘
```

### 수식

**Policy (Expert 선택):**
```
policy_logits = FC_policy2(ReLU(Dropout(FC_policy1(state)))) / temperature
action_probs = Softmax(policy_logits)
action ~ Categorical(action_probs)  # 0-7
log_prob = log(action_probs[action])
```

**Baseline (보상 예측):**
```
baseline = FC_baseline2(ReLU(Dropout(FC_baseline1(state))))
```

**Expert 출력:**
```
selected_expert = experts[action]
rating = selected_expert(state)
```

**Raw Reward:**
```
raw_reward = -|rating_pred - rating_true|
```

**Group Relative Reward:**
```
mean_reward = mean(raw_rewards)
std_reward = std(raw_rewards) + 1e-8
relative_reward = (raw_reward - mean_reward) / std_reward
```

**GRPO Loss:**
```
policy_loss = -log_prob * relative_reward

baseline_loss = (baseline - raw_reward)^2

entropy = -Σ action_probs * log(action_probs)

total_loss = policy_loss + 0.5 * baseline_loss - 0.01 * entropy
```

### 파라미터 수
- Embeddings: 168,400 (same as Dense)
- Policy Network: (146 × 128) + 128 + (128 × 8) + 8 = 19,816
- Baseline Network: (146 × 128) + 128 + (128 × 1) + 1 = 18,817
- Expert Networks: 300,288 (same as Dense)
- **총 파라미터: 754,241**

### 학습 방식
```python
# 1. Episode Collection (eval mode)
with torch.no_grad():
    outputs = model(user_id, movie_id, age_group, gender, occupation)
    action = outputs['action']
    log_prob = outputs['log_prob']
    baseline = outputs['baseline']
    rating_pred = outputs['rating']
    raw_reward = -abs(rating_pred - rating_true)

# 2. Compute Group Relative Rewards
mean_reward = raw_rewards.mean()
std_reward = raw_rewards.std() + 1e-8
relative_rewards = (raw_rewards - mean_reward) / std_reward

# 3. GRPO Update (train mode, 4 epochs)
for grpo_epoch in range(4):
    for batch in minibatches:
        outputs = model(batch_inputs)

        # Losses
        policy_loss = -(outputs['log_prob'] * batch_relative_rewards).mean()
        baseline_loss = (outputs['baseline'] - batch_raw_rewards).pow(2).mean()
        entropy = -(outputs['action_probs'] * log(outputs['action_probs'])).sum(-1).mean()

        total_loss = policy_loss + 0.5 * baseline_loss - 0.01 * entropy

        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
```

---

## 4. 아키텍처 비교

### 4.1 Gating 메커니즘 비교

| 특성 | Dense MoE | PPO-MoE | GRPO-MoE |
|------|-----------|---------|----------|
| **선택 방식** | Soft (Weighted) | Hard (Sampling) | Hard (Sampling) |
| **활용 Expert** | 모두 (8개) | 하나 (1/8) | 하나 (1/8) |
| **Gating Network** | FC → Softmax | FC → Categorical | FC / T → Categorical |
| **출력** | Weighted Sum | Single Expert | Single Expert |
| **추론 효율** | 낮음 | 높음 | 높음 |
| **학습 안정성** | 높음 | 중간 | 높음 |

### 4.2 학습 방식 비교

| 특성 | Dense MoE | PPO-MoE | GRPO-MoE |
|------|-----------|---------|----------|
| **학습 방법** | 지도 학습 | 강화학습 (PPO) | 강화학습 (GRPO) |
| **Loss 함수** | MSE | Clipped Policy + Value | Policy + Baseline |
| **보상 함수** | - | -\|pred - true\| | -\|pred - true\| |
| **Advantage** | - | GAE (λ=0.95) | Relative Reward |
| **업데이트** | 1회/epoch | 4회/epoch | 4회/epoch |
| **Critic** | 없음 | Value Network | Baseline Network |

### 4.3 네트워크 구성 비교

```
┌──────────────────────────────────────────────────────────┐
│                    Dense MoE                             │
├──────────────────────────────────────────────────────────┤
│  State → Gating (Softmax) → All Experts → Weighted Sum  │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│                    PPO-MoE                               │
├──────────────────────────────────────────────────────────┤
│  State → Policy (Categorical) → One Expert               │
│       ↘ Value → Advantage → PPO Update                  │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│                    GRPO-MoE                              │
├──────────────────────────────────────────────────────────┤
│  State → Policy (Categorical/T) → One Expert             │
│       ↘ Baseline → Relative Reward → GRPO Update        │
└──────────────────────────────────────────────────────────┘
```

### 4.4 Forward Pass 비교

**Dense MoE:**
```python
state = create_state(user_id, movie_id, age, gender, occup)
gate_probs = gating_network(state)  # [batch, 8]
expert_outputs = [expert_i(state) for i in range(8)]  # 8 × [batch, 1]
rating = sum(gate_probs[:, i] * expert_outputs[i])  # [batch, 1]
```

**PPO-MoE:**
```python
state = create_state(user_id, movie_id, age, gender, occup)
action_probs = policy_network(state)  # [batch, 8]
value = value_network(state)  # [batch, 1]
action = Categorical(action_probs).sample()  # [batch]
log_prob = log(action_probs[action])  # [batch]
rating = experts[action](state)  # [batch, 1]
```

**GRPO-MoE:**
```python
state = create_state(user_id, movie_id, age, gender, occup)
action_probs = policy_network(state) / temperature  # [batch, 8]
baseline = baseline_network(state)  # [batch, 1]
action = Categorical(action_probs).sample()  # [batch]
log_prob = log(action_probs[action])  # [batch]
rating = experts[action](state)  # [batch, 1]
```

### 4.5 성능 비교 (MovieLens 100k)

| 모델 | RMSE | MAE | 학습 시간 | Expert 활용 |
|------|------|-----|----------|------------|
| **Dense MoE** | **0.9803** | **0.7756** | 70초 | 균등 (다양성 87%) |
| **GRPO-MoE** | 1.0412 | 0.8276 | 150초 | 편향 (1개 68%) |
| **PPO-MoE** | 1.0659 | 0.8575 | 85초 | 편향 (3개 90%) |

### 4.6 장단점 비교

#### Dense MoE
**장점:**
- ✅ 가장 높은 성능
- ✅ 안정적인 학습
- ✅ 빠른 수렴
- ✅ 예측 가능성

**단점:**
- ❌ 추론 시 모든 Expert 계산
- ❌ 메모리 사용량 높음
- ❌ Expert 특화 불명확

#### PPO-MoE
**장점:**
- ✅ 추론 효율 (1개 Expert만)
- ✅ Expert 특화 학습
- ✅ 탐색-활용 균형

**단점:**
- ❌ 학습 불안정
- ❌ 성능 낮음 (3위)
- ❌ 하이퍼파라미터 민감

#### GRPO-MoE
**장점:**
- ✅ 추론 효율 (1개 Expert만)
- ✅ PPO보다 안정적
- ✅ 상대적 보상으로 분산 감소
- ✅ 2위 성능

**단점:**
- ❌ Dense보다 성능 낮음
- ❌ 학습 시간 가장 긺
- ❌ Expert 편향 심함

---

## 5. 코드 구현 예시

### 5.1 Dense MoE - Forward

```python
def forward(self, user_id, movie_id, age_group, gender, occupation):
    # State
    state = self.create_state(user_id, movie_id, age_group, gender, occupation)

    # Gating
    gate_logits = self.gating_layer2(
        self.dropout(torch.relu(self.gating_layer1(state)))
    )
    gate_probs = F.softmax(gate_logits, dim=-1)  # [batch, 8]

    # All Experts
    expert_outputs = []
    for expert in self.experts:
        output = expert(state)  # [batch, 1]
        expert_outputs.append(output)
    expert_outputs = torch.stack(expert_outputs, dim=-1)  # [batch, 1, 8]

    # Weighted Sum
    rating = torch.sum(gate_probs.unsqueeze(1) * expert_outputs, dim=-1)

    return {
        'rating': rating,
        'gate_probs': gate_probs
    }
```

### 5.2 PPO-MoE - Forward

```python
def forward(self, user_id, movie_id, age_group, gender, occupation, deterministic=False):
    # State
    state = self.create_state(user_id, movie_id, age_group, gender, occupation)

    # Policy
    policy_logits = self.policy_layer2(
        self.dropout(torch.relu(self.policy_layer1(state)))
    )
    action_probs = F.softmax(policy_logits, dim=-1)

    # Value
    value = self.value_layer2(
        self.dropout(torch.relu(self.value_layer1(state)))
    )

    # Action Selection
    if deterministic:
        action = torch.argmax(action_probs, dim=-1)
    else:
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()

    log_prob = torch.log(action_probs.gather(1, action.unsqueeze(1)) + 1e-10)

    # Selected Expert
    rating = self._apply_selected_expert(state, action)

    return {
        'rating': rating,
        'action': action,
        'log_prob': log_prob,
        'value': value,
        'action_probs': action_probs
    }
```

### 5.3 GRPO-MoE - Forward

```python
def forward(self, user_id, movie_id, age_group, gender, occupation, deterministic=False):
    # State
    state = self.create_state(user_id, movie_id, age_group, gender, occupation)

    # Policy (with temperature)
    policy_logits = self.policy_layer2(
        self.dropout(torch.relu(self.policy_layer1(state)))
    ) / self.temperature
    action_probs = F.softmax(policy_logits, dim=-1)

    # Baseline
    baseline = self.baseline_layer2(
        self.dropout(torch.relu(self.baseline_layer1(state)))
    )

    # Action Selection
    if deterministic:
        action = torch.argmax(action_probs, dim=-1)
    else:
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()

    log_prob = torch.log(action_probs.gather(1, action.unsqueeze(1)) + 1e-10)

    # Selected Expert
    rating = self._apply_selected_expert(state, action)

    return {
        'rating': rating,
        'action': action,
        'log_prob': log_prob,
        'baseline': baseline,
        'action_probs': action_probs
    }
```

---

## 6. 핵심 차이점 요약

### Gating 메커니즘
1. **Dense MoE**: `Softmax(FC(state))` → 모든 Expert 가중치 조합
2. **PPO-MoE**: `Categorical(Softmax(FC(state)))` → 하나 선택 + Value Network
3. **GRPO-MoE**: `Categorical(Softmax(FC(state)/T))` → 하나 선택 + Baseline Network

### 학습 신호
1. **Dense MoE**: 직접적인 MSE 손실
2. **PPO-MoE**: GAE 기반 Advantage × Clipped Ratio
3. **GRPO-MoE**: Normalized Relative Reward

### 탐색-활용
1. **Dense MoE**: 암묵적 탐색 (Softmax 엔트로피)
2. **PPO-MoE**: 명시적 탐색 (Entropy Bonus + Stochastic Sampling)
3. **GRPO-MoE**: Temperature 기반 탐색 + Entropy Bonus

---

## 결론

### 언제 어떤 모델을 사용할까?

**Dense MoE 추천:**
- ✅ 최고 성능이 필요할 때
- ✅ 학습 안정성이 중요할 때
- ✅ 추론 시간이 중요하지 않을 때
- ✅ 베이스라인 모델로 사용

**PPO-MoE 추천:**
- ✅ 추론 효율이 중요할 때
- ✅ Expert 특화가 필요할 때
- ✅ 대규모 데이터셋에서

**GRPO-MoE 추천:**
- ✅ 추론 효율 + 안정성 둘 다 필요할 때
- ✅ PPO보다 나은 성능 원할 때
- ✅ 상대적 비교가 의미있는 태스크

---

*작성자: Claude Code*
*날짜: 2025년 11월 2일*
*프로젝트: claudeMoE*
