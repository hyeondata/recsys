# Scalability Experiments for MoE Models

Expert 개수를 변화시켜 **확장성(Scalability)**과 **효율성(Efficiency)**을 비교하는 실험 도구입니다.

---

## 📋 개요

MoE 모델에서 Expert 개수를 늘리면:
- ✅ **장점**: 모델 표현력 증가 → 성능 향상 가능
- ❌ **단점**: 학습 시간, 메모리 사용량 증가

이 실험은 다양한 Expert 개수(8, 16, 32, 64, 128개)로 학습하여:
1. **성능** (RMSE)
2. **학습 시간** (Time per epoch)
3. **메모리 사용량** (GPU memory)
4. **Expert 활용도** (Entropy, Gini coefficient)

를 측정하고 비교합니다.

---

## 🚀 빠른 시작

### 1. 단일 모델 실험 (테스트용)

```bash
# Dense MoE로 빠르게 테스트 (8, 16, 32 experts, 5 epochs)
uv run python3 src_10m/experiments/quick_experiment.py \
    --model_type dense \
    --expert_counts 8,16,32 \
    --epochs 5
```

**예상 시간**: ~30-45분 (RTX 3090 기준)

### 2. 전체 실험 (모든 모델)

```bash
# Dense, PPO, GRPO 모두 실행 (8, 16, 32, 64, 128 experts)
bash src_10m/experiments/run_all_scalability_experiments.sh
```

**예상 시간**: ~3-4시간 (모델 3개 × Expert 설정 5개 × Epoch 10)

### 3. 시각화

```bash
# 그래프 생성
uv run python3 src_10m/experiments/visualize_scalability.py \
    --results_dir results_10m/scalability \
    --output_dir results_10m/scalability/plots
```

---

## 📁 파일 구조

```
experiments/
├── README.md                          # 이 파일
├── efficiency_metrics.py              # 효율성 측정 도구
├── run_scalability_experiment.py      # 실험 실행 메인 스크립트
├── visualize_scalability.py           # 시각화 도구
├── quick_experiment.py                # 빠른 실험 (테스트용)
└── run_all_scalability_experiments.sh # 전체 실험 실행 스크립트
```

---

## 🔬 측정 지표

### 1. 성능 지표
- **RMSE**: Root Mean Squared Error (낮을수록 좋음)
- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error

### 2. 효율성 지표

#### 시간
- **avg_epoch_time**: 평균 Epoch 학습 시간
- **avg_throughput**: 평균 처리량 (samples/second)
- **time_efficiency**: 시간 효율성 (>1이면 효율적)

#### 메모리
- **peak_gpu_memory**: 최대 GPU 메모리 사용량 (GB)
- **memory_efficiency**: 메모리 효율성

#### Expert 활용도
- **normalized_entropy**: 정규화된 엔트로피 (0~1, 높을수록 균등)
  - 1.0: 모든 Expert가 동일하게 사용됨
  - 0.0: 한 Expert만 사용됨
- **gini**: Gini 계수 (0~1, 낮을수록 균등)
  - 0.0: 완전히 균등한 분포
  - 1.0: 한 Expert에 집중
- **usage_ratio**: 최대/최소 사용 비율

---

## 📊 생성되는 시각화

모든 그래프는 **PDF (600 DPI)**와 **PNG** 형식으로 저장됩니다.

### 1. `performance_scaling.pdf`
- Expert 개수 vs RMSE
- 모델별 성능 비교

### 2. `time_scaling.pdf`
- Expert 개수 vs 학습 시간
- 시간 효율성 (Time Efficiency)

### 3. `memory_scaling.pdf`
- Expert 개수 vs GPU 메모리
- Expert 개수 vs 파라미터 수

### 4. `expert_utilization.pdf`
- Expert 개수 vs Entropy (균등도)
- Expert 개수 vs Gini (집중도)

### 5. `efficiency_comparison.pdf`
- 종합 비교 (6개 서브플롯)
  - Performance, Time, Memory
  - Throughput, RMSE vs Time, RMSE vs Memory

### 6. `summary_table.tex`
- LaTeX 형식 요약 테이블
- 논문 작성용

---

## 🎯 사용 예시

### 예시 1: Dense MoE만 테스트

```bash
uv run python3 src_10m/experiments/run_scalability_experiment.py \
    --model_type dense \
    --expert_counts 8 16 32 64 \
    --epochs 10 \
    --batch_size 1024 \
    --output_dir results_10m/scalability
```

### 예시 2: PPO-MoE로 빠른 테스트

```bash
uv run python3 src_10m/experiments/run_scalability_experiment.py \
    --model_type ppo \
    --expert_counts 16 32 64 \
    --epochs 5 \
    --batch_size 512 \
    --output_dir results_10m/scalability
```

### 예시 3: 커스텀 설정

```bash
# 더 많은 Expert로 실험 (128, 256)
uv run python3 src_10m/experiments/run_scalability_experiment.py \
    --model_type dense \
    --expert_counts 64 128 256 \
    --epochs 15 \
    --batch_size 512 \
    --embedding_dim 128 \
    --lr 0.0005
```

---

## 📈 예상 결과

### 성능 (RMSE)
- **8 experts**: 0.80-0.85
- **16 experts**: 0.75-0.80 (baseline)
- **32 experts**: 0.73-0.78 (약간 개선)
- **64 experts**: 0.72-0.77 (소폭 개선)
- **128 experts**: 0.72-0.76 (미미한 개선)

→ **수확 체감 법칙**: Expert가 많을수록 성능 개선 폭이 줄어듦

### 학습 시간
- **8 experts**: ~20분/epoch (baseline)
- **16 experts**: ~30분/epoch (1.5x)
- **32 experts**: ~50분/epoch (2.5x)
- **64 experts**: ~90분/epoch (4.5x)
- **128 experts**: ~150분/epoch (7.5x)

→ **준선형 증가**: Expert 2배 → 시간 1.5-2배

### GPU 메모리
- **8 experts**: ~8GB
- **16 experts**: ~12GB
- **32 experts**: ~18GB
- **64 experts**: ~26GB (RTX 3090 한계)
- **128 experts**: ~40GB (A100 필요)

### Expert 활용도
- **Entropy**: Expert가 많을수록 감소 (일부만 집중 사용)
- **Gini**: Expert가 많을수록 증가 (불균등 심화)

→ **비효율 증가**: Expert 수만 늘린다고 모두 활용되지 않음

---

## 💡 인사이트

### 확장성 (Scalability)
- ✅ **선형 확장**: 16 → 32 experts는 비교적 효율적
- ⚠️ **비선형 증가**: 64 이상에서는 메모리/시간 급증
- ❌ **한계**: 128 이상은 일반 GPU에서 불가능

### 효율성 (Efficiency)
- ✅ **최적점**: 16-32 experts가 성능/효율 밸런스 좋음
- ⚠️ **과도한 Expert**: 64+ experts는 활용도 떨어짐
- 💡 **권장**: 데이터 크기와 GPU 메모리에 맞춰 선택

### 모델별 차이
- **Dense MoE**: 가장 빠르고 효율적
- **GRPO-MoE**: 중간 수준
- **PPO-MoE**: 가장 느리지만 Expert 활용도 높음

---

## 🔧 고급 옵션

### 실험 파라미터

```bash
run_scalability_experiment.py [OPTIONS]

Options:
  --model_type {dense,ppo,grpo}   모델 타입
  --expert_counts [N ...]         Expert 개수 리스트 (예: 8 16 32)
  --baseline_experts N            비교 기준 Expert 수 (default: 16)
  --epochs N                      Epoch 수 (default: 10)
  --batch_size N                  배치 크기 (default: 1024)
  --embedding_dim N               임베딩 차원 (default: 128)
  --lr FLOAT                      학습률 (default: 0.001)
  --patience N                    Early stopping patience (default: 5)
  --output_dir PATH               결과 저장 경로
```

### 시각화 옵션

```bash
visualize_scalability.py [OPTIONS]

Options:
  --results_dir PATH    결과 JSON 파일 경로
  --output_dir PATH     그래프 저장 경로
```

---

## 📊 결과 파일 구조

```
results_10m/scalability/
├── scalability_results_dense.json       # Dense MoE 중간 결과
├── scalability_results_ppo.json         # PPO MoE 중간 결과
├── scalability_results_grpo.json        # GRPO MoE 중간 결과
├── scalability_analysis_dense.json      # Dense MoE 최종 분석
├── scalability_analysis_ppo.json        # PPO MoE 최종 분석
├── scalability_analysis_grpo.json       # GRPO MoE 최종 분석
└── plots/
    ├── performance_scaling.pdf
    ├── time_scaling.pdf
    ├── memory_scaling.pdf
    ├── expert_utilization.pdf
    ├── efficiency_comparison.pdf
    └── summary_table.tex
```

---

## ⚠️ 주의사항

### GPU 메모리 요구사항

| Expert 개수 | Dense MoE | PPO/GRPO (Batch 512) | 권장 GPU |
|------------|-----------|---------------------|----------|
| 8          | ~6GB      | ~5GB                | RTX 3060 (12GB) |
| 16         | ~12GB     | ~8GB                | RTX 3090 (24GB) |
| 32         | ~18GB     | ~14GB               | RTX 3090 (24GB) |
| 64         | ~26GB     | ~22GB               | A100 (40GB) |
| 128        | ~40GB     | ~35GB               | A100 (80GB) |

### 학습 시간 예상

- **빠른 테스트** (3개 Expert 설정, 5 epochs): ~1시간
- **기본 실험** (5개 Expert 설정, 10 epochs): ~2-3시간
- **전체 실험** (3개 모델, 5개 설정, 10 epochs): ~6-9시간

### 메모리 부족 시

```bash
# Batch 크기 줄이기
--batch_size 512  # 또는 256

# Expert 수 줄이기
--expert_counts 8 16 32  # 64, 128 제외

# Epoch 수 줄이기 (빠른 테스트)
--epochs 5
```

---

## 📚 관련 문서

- **README_10M.md**: 전체 프로젝트 설명
- **memo_compression/00_using.md**: 사용법 통합 문서
- **src_10m/USAGE.md**: 빠른 실행 가이드

---

## 🎓 연구 활용

### 논문 작성 시
1. `efficiency_comparison.pdf`: Figure로 삽입
2. `summary_table.tex`: Table로 삽입
3. `scalability_analysis_*.json`: 수치 인용

### 발표 자료
- PNG 파일 사용 (고해상도)
- 모델별 비교 강조

### 추가 분석
- JSON 파일에서 원시 데이터 추출
- 커스텀 분석/시각화 가능

---

## 🔬 실험 디자인 팁

### 1. 빠른 프로토타이핑
```bash
# 적은 Expert, 짧은 Epoch
--expert_counts 8 16 --epochs 3
```

### 2. 정밀 실험
```bash
# 더 많은 Expert, 충분한 Epoch
--expert_counts 8 16 32 64 --epochs 20
```

### 3. 극한 테스트
```bash
# 매우 많은 Expert (A100 필요)
--expert_counts 64 128 256 --epochs 15
```

---

## 📞 문의

실험 관련 이슈는 GitHub Issues에 등록해주세요.

---

**버전**: v1.0
**마지막 업데이트**: 2025-11-09
**작성자**: Claude + User
