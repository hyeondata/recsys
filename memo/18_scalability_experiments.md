# 18. 확장성 실험 시스템 (Scalability Experiments)

## 개요

Expert 개수를 늘렸을 때 Dense MoE와 RL-based MoE의 **확장성(Scalability)**과 **효율성(Efficiency)** 차이를 정량적으로 비교하는 실험 시스템을 구축했습니다.

**작성일**: 2025-11-09
**버전**: v2.2

---

## 배경 및 동기

### 문제 인식

기존 실험에서는 Expert 8개로 고정하여 비교했으나, MoE의 핵심 장점인 **확장성**을 충분히 보여주지 못했습니다:

- Dense MoE: 모든 Expert를 가중평균으로 사용
- RL MoE: Policy Network로 단일 Expert 선택

이론적으로:
- Dense MoE는 Expert 개수에 비례하여 계산량 증가 (O(n))
- RL MoE는 Expert 개수와 무관하게 일정한 계산량 (O(1))

### 목표

**"Expert가 많아질수록 RL MoE가 더 효율적"**이라는 가설을 실험적으로 검증합니다.

---

## 구현 내용

### 1. 실험 프레임워크 (`experiments/scalability_experiment.py`)

#### 핵심 기능

**다양한 Expert 개수 테스트**
```python
# 기본: 4, 8, 16, 32
# 확장: 2, 4, 8, 16, 32, 64, 128
```

**측정 지표**

1. **학습 시간**: 전체 학습 소요 시간
2. **추론 시간**: 테스트 데이터셋 전체에 대한 추론 시간
3. **메모리 사용량**: GPU 메모리 할당량
4. **성능 지표**: RMSE, MAE, MSE
5. **파라미터 수**: 모델 크기

**측정 함수**
```python
def measure_inference_time(model, test_loader, device, num_iterations=5):
    """워밍업 후 5회 반복 평균"""

def measure_memory_usage(model, batch, device):
    """CUDA 메모리 peak 측정"""
```

#### 실험 프로토콜

1. 각 (모델, Expert 개수) 조합에 대해:
   - 모델 생성 및 파라미터 카운트
   - N epochs 학습 (시간 측정)
   - 테스트 평가 (성능 측정)
   - 추론 시간 측정 (5회 평균)
   - 메모리 사용량 측정

2. 결과를 JSON으로 저장
   ```json
   {
     "models": {
       "dense": {
         "4": {...},
         "8": {...},
         ...
       }
     }
   }
   ```

### 2. 시각화 시스템 (`experiments/visualize_scalability.py`)

#### 생성되는 그래프

**1. 전체 비교 (`scalability_overview.png`)**
- 2×3 서브플롯
- 학습 시간, 추론 시간, 메모리, RMSE, 파라미터 수, MAE

**2. 학습 시간 확장성 (`train_time_scalability.png`)**
- Dense MoE의 선형 증가 추세 강조
- 선형 근사 라인 표시 (y = ax + b)

**3. 추론 시간 확장성 (`inference_time_scalability.png`)**
- RL MoE의 일정한 추론 시간 강조

**4. 효율성 비교 (`efficiency_comparison.png`)**
- Efficiency = 1 / (RMSE × Time)
- 높을수록 좋음 (정확하면서 빠름)

**5. 정규화 비교 (`normalized_comparison.png`)**
- 모든 메트릭을 0-1로 정규화
- 종합 점수로 비교

#### 테이블 생성

**Markdown 테이블** (`scalability_comparison_table.md`)
- Expert 개수별 상세 비교
- 성장률 분석 (4→8, 8→16, 16→32)

### 3. 실행 스크립트

#### 전체 실험 (`run_scalability_experiment.sh`)
```bash
# Expert: 4, 8, 16, 32
# Models: Dense, PPO, GRPO
# Epochs: 5
```

#### 빠른 테스트 (`quick_test.sh`)
```bash
# Expert: 4, 8
# Models: Dense, PPO
# Epochs: 3
```

---

## 실험 설계

### 독립 변수
- **Expert 개수**: 4, 8, 16, 32 (기본) / 64, 128 (확장)
- **모델 타입**: Dense MoE, PPO-MoE, GRPO-MoE

### 종속 변수
- **학습 시간** (초)
- **추론 시간** (초)
- **메모리 사용량** (MB)
- **RMSE** (낮을수록 좋음)

### 통제 변수
- 데이터셋: MovieLens 100k (u1.base/u1.test)
- Embedding dimension: 64
- Expert hidden dimension: 256
- Batch size: 256
- Learning rate: 0.001
- Seed: 42

---

## 예상 결과

### 가설 1: 학습 시간 확장성

**Dense MoE**
```
Expert 4:  30s
Expert 8:  60s  (2배)
Expert 16: 120s (2배)
Expert 32: 240s (2배)
```
→ 선형 증가 (O(n))

**PPO/GRPO MoE**
```
Expert 4:  85s
Expert 8:  85s  (일정)
Expert 16: 85s  (일정)
Expert 32: 85s  (일정)
```
→ 일정 (O(1))

### 가설 2: Crossover Point

```
Expert ≤ 8:  Dense MoE가 더 빠름
Expert ≥ 16: RL MoE가 더 빠름
```

### 가설 3: 성능 (RMSE)

```
모든 Expert 개수에서 비슷한 RMSE 유지
(Expert 개수는 성능보다 확장성에 영향)
```

---

## 사용법

### 빠른 테스트 (5분)
```bash
chmod +x experiments/quick_test.sh
./experiments/quick_test.sh
```

### 전체 실험 (1-2시간)
```bash
chmod +x experiments/run_scalability_experiment.sh
./experiments/run_scalability_experiment.sh
```

### 커스터마이징
```bash
uv run python3 experiments/scalability_experiment.py \
    --models dense ppo grpo \
    --num_experts_list 4 8 16 32 64 \
    --num_epochs 10 \
    --experiment_id custom_exp
```

---

## 파일 구조

```
experiments/
├── scalability_experiment.py       # 실험 실행 스크립트
├── visualize_scalability.py        # 시각화 스크립트
├── run_scalability_experiment.sh   # 전체 실험 셸 스크립트
├── quick_test.sh                   # 빠른 테스트 셸 스크립트
├── README_SCALABILITY.md           # 상세 가이드
├── QUICKSTART.md                   # 빠른 시작 가이드
├── results/                        # 실험 결과 JSON
│   └── scalability_results_{id}.json
└── visualizations/                 # 시각화 결과
    └── {experiment_id}/
        ├── scalability_overview.png
        ├── train_time_scalability.png
        ├── inference_time_scalability.png
        ├── efficiency_comparison.png
        ├── normalized_comparison.png
        ├── *.pdf (논문용)
        └── scalability_comparison_table.md
```

---

## 기술적 세부사항

### 1. 추론 시간 측정

```python
# 워밍업 (첫 배치는 제외)
with torch.no_grad():
    for batch in test_loader:
        _ = model(...)
        break

# 실제 측정 (5회 반복 평균)
for _ in range(5):
    start = time.time()
    for batch in test_loader:
        _ = model(...)
    times.append(time.time() - start)

avg_time = mean(times)
```

### 2. 메모리 측정

```python
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# Forward pass
_ = model(...)

# Peak memory
peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
```

### 3. 모델 파라미터 계산

```python
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```

---

## 주의사항

### 1. GPU 메모리
- Expert 32개 이상: GPU 메모리 부족 가능
- 해결: `--batch_size 128` 또는 더 작게

### 2. 실험 시간
- 전체 실험 (3 모델 × 4 expert 구성 × 5 epochs): 약 1.5~2시간
- 빠른 테스트: 약 5분

### 3. 재현성
- `--seed 42` 고정
- 동일한 데이터 split 사용

---

## 확장 가능성

### 1. 더 많은 Expert
```bash
--num_experts_list 2 4 8 16 32 64 128 256
```

### 2. 다양한 데이터셋
```bash
--data_dir ml-1m   # 더 큰 데이터셋
--data_dir ml-10m
```

### 3. 다양한 모델 크기
```bash
--expert_hidden_dim 128   # 작은 모델
--expert_hidden_dim 512   # 큰 모델
```

### 4. Standard RL 비교
```python
# Standard PPO/GRPO도 추가 가능
from src.models.ppo_moe_standard import PPOMoEStandard
```

---

## 기대 효과

### 1. 확장성 입증
- Dense MoE의 선형 증가
- RL MoE의 일정한 성능

### 2. Crossover Point 발견
- 어느 시점부터 RL MoE가 유리한지

### 3. 실용적 가이드라인
- Expert 개수 선택 기준
- 모델 선택 가이드

---

## 다음 단계

1. **실험 실행**: 다양한 Expert 개수로 실험
2. **결과 분석**: Crossover point 확인
3. **논문 작성**: 시각화 자료 활용
4. **추가 실험**:
   - 더 큰 데이터셋 (MovieLens 1M, 10M)
   - 더 많은 Expert (64, 128)
   - Top-k expert selection

---

## 관련 파일

- `experiments/README_SCALABILITY.md`: 상세 가이드
- `experiments/QUICKSTART.md`: 빠른 시작
- `memo/04_model_implementation.md`: 모델 아키텍처
- `memo_compression/01_summary.md`: 프로젝트 요약

---

## 요약

확장성 실험 시스템을 통해:
- ✅ Expert 개수 증가에 따른 성능 변화 측정
- ✅ Dense MoE vs RL MoE 확장성 비교
- ✅ 시각화 및 정량적 분석
- ✅ 실용적인 모델 선택 가이드라인 제공

**핵심 메시지**: "Expert가 많을수록 RL-based MoE가 효율적"
