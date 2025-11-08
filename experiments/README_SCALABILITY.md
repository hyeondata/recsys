# MoE 확장성 실험 (Scalability Experiments)

## 목적

Expert 개수를 늘렸을 때 **Dense MoE**와 **RL-based MoE (PPO/GRPO)**의 **확장성(Scalability)**과 **효율성(Efficiency)** 차이를 명확하게 비교합니다.

## 핵심 가설

### Dense MoE
- **모든 Expert를 사용**: 가중평균 방식
- **Expert 개수 증가 시**: 계산량이 **선형적으로 증가** (O(n))
- **장점**: 모든 expert의 지식 활용
- **단점**: 확장성 제한 (expert 많아질수록 느림)

### RL-based MoE (PPO/GRPO)
- **단일 Expert 선택**: 정책 네트워크로 하나만 선택
- **Expert 개수 증가 시**: 계산량이 **일정** (O(1))
- **장점**: 뛰어난 확장성 (expert 많아져도 일정한 속도)
- **단점**: 선택되지 않은 expert는 미활용

---

## 실험 구조

### 1. 테스트할 Expert 개수
```
4, 8, 16, 32
```

### 2. 측정 지표

#### 확장성(Scalability) 지표
- **학습 시간**: Expert 개수 증가에 따른 학습 시간 변화
- **추론 시간**: Expert 개수 증가에 따른 추론 시간 변화
- **메모리 사용량**: Expert 개수 증가에 따른 GPU 메모리 사용

#### 효율성(Efficiency) 지표
- **파라미터 수**: 모델 크기
- **시간 대비 성능**: Efficiency = 1 / (RMSE × Time)

#### 성능(Performance) 지표
- **RMSE, MAE, MSE**: 추천 정확도

---

## 사용법

### 방법 1: 스크립트 실행 (권장)

```bash
chmod +x experiments/run_scalability_experiment.sh
./experiments/run_scalability_experiment.sh
```

### 방법 2: 직접 실행

#### Step 1: 실험 실행
```bash
uv run python3 experiments/scalability_experiment.py \
    --data_dir ml-100k \
    --train_rating_path ml-100k/u1.base \
    --test_rating_path ml-100k/u1.test \
    --models dense ppo grpo \
    --num_experts_list 4 8 16 32 \
    --num_epochs 5 \
    --experiment_id exp1 \
    --output_dir experiments/results \
    --batch_size 256
```

#### Step 2: 시각화 생성
```bash
uv run python3 experiments/visualize_scalability.py \
    --result_path experiments/results/scalability_results_exp1.json \
    --output_dir experiments/visualizations/exp1 \
    --dpi 600
```

---

## 실험 파라미터 설정

### 빠른 테스트 (개발/디버깅)
```bash
--num_experts_list 4 8 \
--num_epochs 3 \
--batch_size 256
```

### 표준 실험
```bash
--num_experts_list 4 8 16 32 \
--num_epochs 5 \
--batch_size 256
```

### 완전한 실험 (논문용)
```bash
--num_experts_list 4 8 16 32 64 \
--num_epochs 10 \
--batch_size 256
```

---

## 출력 파일

### 1. 실험 결과 JSON
```
experiments/results/scalability_results_{experiment_id}.json
```

구조:
```json
{
  "config": { ... },
  "models": {
    "dense": {
      "4": {
        "num_parameters": 1234567,
        "train_time": 45.2,
        "inference_time": 0.234,
        "memory": {"allocated_mb": 123.4},
        "test_metrics": {"rmse": 0.98, "mae": 0.76}
      },
      "8": { ... },
      ...
    },
    "ppo": { ... },
    "grpo": { ... }
  }
}
```

### 2. 시각화 파일

**PNG 파일 (고해상도, 600 DPI):**
- `scalability_overview.png`: 전체 비교 (2×3 서브플롯)
- `train_time_scalability.png`: 학습 시간 확장성
- `inference_time_scalability.png`: 추론 시간 확장성
- `efficiency_comparison.png`: 효율성 비교
- `normalized_comparison.png`: 정규화된 종합 점수

**PDF 파일:** 위 파일들의 PDF 버전 (LaTeX 논문용)

**Markdown 테이블:**
- `scalability_comparison_table.md`: 수치 비교 테이블

---

## 예상 결과

### Dense MoE
```
Experts: 4  → Train: 30s,  Inference: 0.15s
Experts: 8  → Train: 60s,  Inference: 0.30s  (2배 증가)
Experts: 16 → Train: 120s, Inference: 0.60s  (2배 증가)
Experts: 32 → Train: 240s, Inference: 1.20s  (2배 증가)
```
→ **선형 증가 (O(n))**

### PPO/GRPO MoE
```
Experts: 4  → Train: 85s,  Inference: 0.25s
Experts: 8  → Train: 85s,  Inference: 0.25s  (변화 없음)
Experts: 16 → Train: 85s,  Inference: 0.25s  (변화 없음)
Experts: 32 → Train: 85s,  Inference: 0.25s  (변화 없음)
```
→ **일정 (O(1))**

### 해석
- **Expert 4~8개**: Dense MoE가 더 빠름 (단순한 가중평균)
- **Expert 16개 이상**: RL MoE가 더 빠름 (확장성 우수)
- **Expert 32개 이상**: RL MoE가 압도적으로 빠름

---

## 실험 커스터마이징

### 특정 모델만 테스트
```bash
--models dense ppo  # GRPO 제외
```

### Expert 범위 변경
```bash
--num_experts_list 2 4 8 16 32 64 128  # 더 넓은 범위
```

### 더 긴 학습 (정확도 향상)
```bash
--num_epochs 20  # 더 많은 epoch
```

### GPU 메모리 제한 시
```bash
--batch_size 128  # 배치 크기 줄이기
```

---

## 주의사항

### 1. 메모리 관리
- Expert 32개 이상일 때 GPU 메모리 부족 가능
- `--batch_size`를 줄여서 조정

### 2. 실험 시간
- 전체 실험은 수 시간 소요 가능
- 모델 3개 × Expert 4개 = 12개 구성
- 각 구성당 5 epochs

예상 시간:
```
Dense (4개) ≈ 3분
Dense (8개) ≈ 6분
Dense (16개) ≈ 12분
Dense (32개) ≈ 24분
PPO/GRPO (각각 모든 구성) ≈ 15분

총 예상: 약 1.5~2시간
```

### 3. 재현성
- `--seed 42` 고정으로 재현 가능
- 동일한 데이터셋 사용 (u1.base/u1.test)

---

## 결과 분석 예시

### 1. 확장성 비교
```
Dense MoE: 학습 시간이 Expert 개수에 비례
PPO/GRPO MoE: 학습 시간이 일정
```

### 2. Crossover Point
```
Expert <= 8: Dense MoE가 더 빠름
Expert >= 16: RL MoE가 더 빠름
```

### 3. 효율성
```
Dense MoE: Expert 4~8개일 때 최고 효율
RL MoE: Expert 개수와 무관하게 일정한 효율
```

---

## 추가 실험 아이디어

### 1. 더 큰 Expert 개수
```bash
--num_experts_list 4 8 16 32 64 128
```

### 2. 다양한 모델 크기
```bash
--expert_hidden_dim 128  # 작은 모델
--expert_hidden_dim 512  # 큰 모델
```

### 3. 다른 데이터셋
```bash
--data_dir ml-1m  # MovieLens 1M
--data_dir ml-10m  # MovieLens 10M
```

---

## 참고

### 관련 파일
- `experiments/scalability_experiment.py`: 실험 실행 스크립트
- `experiments/visualize_scalability.py`: 시각화 스크립트
- `experiments/run_scalability_experiment.sh`: 실행 셸 스크립트

### 메모 참고
- `memo/04_model_implementation.md`: 모델 아키텍처 상세
- `memo_compression/01_summary.md`: 프로젝트 요약

---

## 문의

실험 관련 문제가 있을 경우:
1. `memo_compression/03_known_issues.md` 확인
2. GPU 메모리 부족 시 `--batch_size` 줄이기
3. 학습 실패 시 `--num_epochs` 줄이기

## 라이선스

이 프로젝트는 claudeMoE의 일부입니다.
