# MoE 모델 비교 시각화 가이드

## 작성 일자
2025년 11월 2일

---

## 개요

이 문서는 Dense MoE, PPO-MoE, GRPO-MoE 세 모델을 비교하는 시각화에 대한 가이드입니다.

---

## 시각화 실행 방법

### 기본 사용법

```bash
# uv를 사용하여 실행 (필수!)
uv run python3 src/visualization/compare_models.py \
    --results_file results/all_models_evaluation.json \
    --output_dir results/visualizations
```

### 매개변수

| 매개변수 | 설명 | 기본값 |
|---------|------|--------|
| `--results_file` | 평가 결과 JSON 파일 경로 | `results/all_models_evaluation.json` |
| `--output_dir` | 시각화 출력 디렉토리 | `results/visualizations` |

---

## 생성되는 시각화

### 1. performance_comparison.png (167KB)

**내용**: MSE, RMSE, MAE 성능 비교 바 차트

**특징**:
- 3개의 서브플롯 (MSE, RMSE, MAE)
- 각 막대에 정확한 값 표시
- 색상 구분: Dense MoE (파란색), PPO-MoE (보라색), GRPO-MoE (주황색)

**해석**:
- 낮을수록 좋은 메트릭
- Dense MoE가 모든 메트릭에서 가장 우수
- GRPO-MoE가 PPO-MoE보다 약간 더 나은 성능

**실제 결과**:
```
MSE:  Dense 0.9611 < GRPO 1.0841 < PPO 1.1361
RMSE: Dense 0.9803 < GRPO 1.0412 < PPO 1.0659
MAE:  Dense 0.7756 < GRPO 0.8276 < PPO 0.8575
```

---

### 2. expert_distribution.png (214KB)

**내용**: Expert 선택/가중치 분포 비교

**세 개의 서브플롯**:

#### 2.1 Dense MoE - Gate Weights
- **표시**: 각 Expert에 할당된 평균 가중치
- **특징**: 모든 Expert 활용 (Softmax 가중치)
- **빨간 점선**: 균등 분포 기준선 (12.5%)
- **해석**:
  - Expert 4가 26.1%로 가장 높음
  - Expert 5가 4.4%로 가장 낮음
  - Gate Entropy: 1.81 (적절한 다양성)

#### 2.2 PPO-MoE - Expert Selection
- **표시**: 각 Expert가 선택된 비율 (%)
- **특징**: 한 번에 하나의 Expert만 선택
- **해석**:
  - Expert 0, 5, 6에 집중 (90.3%)
  - Expert 3은 전혀 선택되지 않음
  - 강한 편향 존재

#### 2.3 GRPO-MoE - Expert Selection
- **표시**: 각 Expert가 선택된 비율 (%)
- **특징**: 한 번에 하나의 Expert만 선택
- **해석**:
  - Expert 5에 매우 집중 (68.5%)
  - Expert 0은 전혀 선택되지 않음
  - PPO보다 더 강한 편향

**핵심 인사이트**:
- Dense MoE: 균형잡힌 Expert 활용
- PPO-MoE: 3개 Expert에 집중
- GRPO-MoE: 1개 Expert에 매우 집중

---

### 3. expert_performance.png (108KB)

**내용**: Expert별 평균 에러 비교 (PPO/GRPO만)

**두 개의 서브플롯**:

#### 3.1 PPO-MoE Expert Performance
- 각 Expert의 평균 예측 오차
- N/A: 선택되지 않은 Expert
- **최고 성능**: Expert 4 (0.1839 에러)
- **최저 성능**: Expert 2 (0.2516 에러)

#### 3.2 GRPO-MoE Expert Performance
- 각 Expert의 평균 예측 오차
- N/A: 선택되지 않은 Expert
- **최고 성능**: Expert 2 (0.1858 에러)
- **최저 성능**: Expert 3 (0.2036 에러)

**해석**:
- 선택된 Expert들이 대체로 낮은 에러를 가짐
- 강화학습이 좋은 Expert를 찾아냄
- 하지만 전체 성능은 Dense보다 낮음 (다양성 부족)

---

### 4. model_comparison_radar.png (547KB)

**내용**: 5가지 차원에서의 모델 비교 레이더 차트

**평가 차원**:

1. **Accuracy (1/RMSE)**: 정확도 (높을수록 좋음)
   - Dense: 1.0 (최고)
   - GRPO: 0.94
   - PPO: 0.92

2. **Precision (1/MAE)**: 정밀도 (높을수록 좋음)
   - Dense: 1.0 (최고)
   - GRPO: 0.94
   - PPO: 0.91

3. **Expert Diversity**: Expert 활용 다양성 (높을수록 좋음)
   - Dense: 0.87 (Gate Entropy 기반)
   - PPO: 0.45 (선택 분포 엔트로피)
   - GRPO: 0.32 (가장 편향됨)

4. **Inference Efficiency**: 추론 효율성 (높을수록 좋음)
   - PPO: 1.0 (1개 Expert)
   - GRPO: 1.0 (1개 Expert)
   - Dense: 0.125 (8개 Expert)

5. **Training Stability**: 학습 안정성 (추정)
   - Dense: 1.0 (지도 학습)
   - GRPO: 0.8 (상대적 보상)
   - PPO: 0.6 (GAE 기반)

**핵심 인사이트**:
- Dense MoE: 정확도와 다양성에서 우수
- PPO/GRPO-MoE: 추론 효율성에서 우수
- Trade-off: 성능 vs 효율성

---

### 5. architecture_comparison.png (261KB)

**내용**: 세 모델의 아키텍처 다이어그램 비교

**세 개의 다이어그램**:

#### 5.1 Dense MoE Architecture
```
State → Gating Network (Softmax) → 8 Experts (All Active) → Weighted Sum → Prediction
```

**특징**:
- ✓ All 8 Experts Used
- ✓ Weighted Combination
- ✗ High Inference Cost

#### 5.2 PPO-MoE Architecture
```
State → Policy Network (Categorical) → Select 1 Expert → Prediction
     ↘ Value Network (Critic)
```

**특징**:
- ✓ Only 1 Expert Used
- ✓ Low Inference Cost
- ✓ PPO Training

#### 5.3 GRPO-MoE Architecture
```
State → Policy Network (Categorical/T) → Select 1 Expert → Prediction
     ↘ Baseline Network (Reward Est.)
```

**특징**:
- ✓ Only 1 Expert Used
- ✓ Low Inference Cost
- ✓ GRPO Training

**핵심 차이점**:
1. **Expert 활용**: Dense는 모두, PPO/GRPO는 하나
2. **Gating**: Dense는 Softmax, PPO/GRPO는 Categorical Sampling
3. **보조 네트워크**: Dense는 없음, PPO는 Value, GRPO는 Baseline

---

### 6. summary_table.png (165KB)

**내용**: 모델 비교 요약 테이블

**포함된 정보**:

| Metric | Dense MoE | PPO-MoE | GRPO-MoE |
|--------|-----------|---------|----------|
| **MSE** | **0.9611** ✓ | 1.1361 | 1.0841 |
| **RMSE** | **0.9803** ✓ | 1.0659 | 1.0412 |
| **MAE** | **0.7756** ✓ | 0.8575 | 0.8276 |
| **Experts Used** | All (8) | One (1) | One (1) |
| **Inference Efficiency** | Low (8x) | High (1x) | High (1x) |
| **Training Method** | Supervised | PPO (RL) | GRPO (RL) |

**색상 코딩**:
- 파란색 헤더: 메트릭 이름
- 연한 파란색: 메트릭 레이블
- 녹색: 최고 성능 하이라이트
- 흰색: 일반 셀

**사용 용도**:
- 발표 자료에 직접 삽입 가능
- 한눈에 모든 정보 파악
- 명확한 비교 가능

---

## 시각화 해석 가이드

### 성능 순위
1. 🥇 **Dense MoE**: 최고 성능 (RMSE 0.9803)
2. 🥈 **GRPO-MoE**: 2위 (RMSE 1.0412)
3. 🥉 **PPO-MoE**: 3위 (RMSE 1.0659)

### Expert 활용 패턴
- **Dense MoE**: 균형잡힌 활용 (다양성 87%)
- **PPO-MoE**: 3개 Expert 집중 (편향 중간)
- **GRPO-MoE**: 1개 Expert 집중 (편향 심함)

### 추론 효율성
- **Dense MoE**: 8개 Expert 모두 실행 (느림)
- **PPO-MoE**: 1개 Expert만 실행 (8배 빠름)
- **GRPO-MoE**: 1개 Expert만 실행 (8배 빠름)

### Trade-off 분석

#### Dense MoE의 선택 이유
- ✅ 최고 성능이 필요할 때
- ✅ 추론 속도가 중요하지 않을 때
- ✅ 안정적인 베이스라인이 필요할 때
- ❌ 대규모 서비스 배포 시

#### PPO-MoE의 선택 이유
- ✅ 추론 효율이 중요할 때
- ✅ 대규모 서비스 배포 시
- ✅ Expert 특화가 필요할 때
- ❌ 최고 성능이 필요할 때

#### GRPO-MoE의 선택 이유
- ✅ 추론 효율 + 안정성 둘 다 필요할 때
- ✅ PPO보다 나은 성능 원할 때
- ✅ 대규모 서비스 배포 시
- ❌ 최고 성능이 필요할 때

---

## 활용 방법

### 1. 논문/보고서 작성
- summary_table.png: 결과 섹션
- performance_comparison.png: 성능 비교
- expert_distribution.png: Expert 분석
- model_comparison_radar.png: 종합 비교

### 2. 발표 자료
- architecture_comparison.png: 방법론 설명
- performance_comparison.png: 결과 발표
- model_comparison_radar.png: 종합 평가

### 3. 기술 문서
- 모든 시각화 포함
- 상세한 해석 추가

---

## 커스터마이징

### 색상 변경
```python
# src/visualization/compare_models.py 수정
colors = ['#2E86AB', '#A23B72', '#F18F01']  # Dense, PPO, GRPO
```

### 메트릭 추가
```python
# performance_comparison 함수에 서브플롯 추가
fig, axes = plt.subplots(1, 4, figsize=(20, 5))  # 4개로 변경
```

### 스타일 변경
```python
# setup_plot_style 함수 수정
plt.style.use('ggplot')  # 다른 스타일
plt.rcParams['figure.figsize'] = (20, 12)  # 더 큰 크기
```

---

## 문제 해결

### 1. matplotlib 에러
```bash
# matplotlib 설치
uv pip install matplotlib
```

### 2. 폰트 경고
```
UserWarning: Glyph 10003 (\N{CHECK MARK}) missing from font(s)
```
- 무시해도 됨 (시각화는 정상 생성)
- 또는 다른 폰트 사용

### 3. 결과 파일 없음
```bash
# 평가 먼저 실행
uv run python3 src/training/evaluate.py \
    --data_dir ml-100k \
    --test_rating_path ml-100k/u1.test \
    --dense_checkpoint checkpoints/dense_moe/dense_moe_best.pt \
    --ppo_checkpoint checkpoints/ppo_moe/ppo_moe_best.pt \
    --grpo_checkpoint checkpoints/grpo_moe/grpo_moe_best.pt \
    --output_file results/all_models_evaluation.json
```

---

## 시각화 품질

### 해상도
- **DPI**: 300 (출판 품질)
- **형식**: PNG (투명 배경 지원)
- **크기**: 자동 조정 (tight_layout)

### 최적화
- 벡터 형식 필요 시: `plt.savefig(..., format='pdf')`
- 더 높은 해상도: `dpi=600`
- 압축: PNG는 자동으로 최적 압축

---

## 확장 가능성

### 추가 가능한 시각화

1. **학습 곡선 비교**
   - Epoch별 Loss 변화
   - 학습 안정성 분석

2. **추론 시간 비교**
   - 실제 추론 시간 측정
   - 배치 크기별 비교

3. **Expert 특화 분석**
   - 각 Expert가 담당하는 사용자/영화 분석
   - 특성별 Expert 선택 패턴

4. **에러 분포 분석**
   - 예측 에러 히스토그램
   - 평점별 성능 비교

---

## 파일 구조

```
results/
├── all_models_evaluation.json       # 평가 결과 (입력)
└── visualizations/                   # 시각화 출력
    ├── performance_comparison.png   # 성능 비교
    ├── expert_distribution.png      # Expert 분포
    ├── expert_performance.png       # Expert 성능
    ├── model_comparison_radar.png   # 레이더 차트
    ├── architecture_comparison.png  # 아키텍처 비교
    └── summary_table.png            # 요약 테이블
```

---

## 참고 코드

### 완전한 실행 예시
```bash
# 1. 모델 학습 (이미 완료)
# Dense MoE, PPO-MoE, GRPO-MoE 학습 완료

# 2. 통합 평가
uv run python3 src/training/evaluate.py \
    --data_dir ml-100k \
    --test_rating_path ml-100k/u1.test \
    --dense_checkpoint checkpoints/dense_moe/dense_moe_best.pt \
    --ppo_checkpoint checkpoints/ppo_moe/ppo_moe_best.pt \
    --grpo_checkpoint checkpoints/grpo_moe/grpo_moe_best.pt \
    --output_file results/all_models_evaluation.json

# 3. 시각화 생성
uv run python3 src/visualization/compare_models.py \
    --results_file results/all_models_evaluation.json \
    --output_dir results/visualizations

# 4. 결과 확인
ls -lh results/visualizations/
```

---

## 결론

이 시각화 도구는 Dense MoE, PPO-MoE, GRPO-MoE 세 모델을 다각도로 비교할 수 있게 해줍니다.

**핵심 발견**:
- Dense MoE: 최고 성능, 낮은 효율성
- GRPO-MoE: 중간 성능, 높은 효율성
- PPO-MoE: 낮은 성능, 높은 효율성

**실무 적용**:
- 연구/프로토타입: Dense MoE
- 프로덕션 서비스: GRPO-MoE
- 대규모 배포: PPO/GRPO-MoE (효율성)

---

*작성자: Claude Code*
*작성일: 2025년 11월 2일*
*프로젝트: claudeMoE*
