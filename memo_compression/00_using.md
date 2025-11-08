# MoE 영화 추천 시스템 - 사용법 통합 가이드

**최종 업데이트**: 2025-11-09
**버전**: v2.2 (확장성 실험 추가)

---

## 📋 목차
1. [환경 설정](#환경-설정)
2. [데이터 준비](#데이터-준비)
3. [모델 학습](#모델-학습)
4. [Resume 기능 (중단된 학습 이어하기)](#resume-기능)
5. [모델 평가](#모델-평가)
6. [시각화](#시각화)
7. [확장성 실험](#확장성-실험-scalability-experiments--새로운-기능-v22)
8. [Git 관리](#git-관리)
9. [버전 히스토리](#버전-히스토리)

---

## 🔧 환경 설정

### 필수 요구사항
- **Python**: 3.8+
- **PyTorch**: 2.0+
- **CUDA**: 필수
- **GPU**: RTX 3090 (24GB) 권장
- **패키지 관리자**: uv (필수)

### 설치
```bash
# uv 설치 (없는 경우)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 의존성 설치
uv sync
```

---

## 📊 데이터 준비

### MovieLens 100k 다운로드
```bash
# 데이터가 없는 경우 자동으로 다운로드됨
# 또는 수동 다운로드:
wget https://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip
```

### 데이터 구조
```
ml-100k/
├── u.data          # 전체 평점 데이터
├── u1.base         # 학습 데이터 (79,619)
├── u1.test         # 검증 데이터 (20,381)
├── u.item          # 영화 정보
└── u.user          # 사용자 정보
```

---

## 🚀 모델 학습

### 1. Dense MoE (베이스라인)

**특징**: FC Layer Gating, 모든 Expert 사용

```bash
uv run python3 src/training/train_dense_moe.py \
    --epochs 10 \
    --batch_size 1024 \
    --lr 0.001 \
    --checkpoint_dir checkpoints/dense_moe
```

**주요 파라미터**:
- `--batch_size`: 1024 (Dense는 큰 배치 사용 가능)
- `--lr`: 0.001
- `--num_experts`: 8 (기본값)

**예상 결과**: RMSE ~0.98

---

### 2. PPO-MoE (배치 단위 방식)

**특징**: 메모리 효율적, 각 배치 즉시 업데이트

```bash
uv run python3 src/training/train_ppo_moe.py \
    --epochs 10 \
    --batch_size 256 \
    --ppo_epochs 4 \
    --lr 0.0003 \
    --checkpoint_dir checkpoints/ppo_moe
```

**주요 파라미터**:
- `--batch_size`: 256 (권장)
- `--ppo_epochs`: 4 (각 배치 업데이트 횟수)
- `--lr`: 0.0003
- `--clip_epsilon`: 0.2 (PPO clipping)
- `--entropy_coef`: 0.01
- `--value_coef`: 0.5

**장점**: 메모리 사용량 300배 감소, GPU 안정성
**예상 결과**: RMSE ~1.07

---

### 3. GRPO-MoE (배치 단위 방식)

**특징**: 그룹 상대 보상, 메모리 효율적

```bash
uv run python3 src/training/train_grpo_moe.py \
    --epochs 10 \
    --batch_size 256 \
    --grpo_epochs 4 \
    --lr 0.0003 \
    --checkpoint_dir checkpoints/grpo_moe
```

**주요 파라미터**:
- `--batch_size`: 256 (권장)
- `--grpo_epochs`: 4 (각 배치 업데이트 횟수)
- `--lr`: 0.0003
- `--clip_epsilon`: 0.2
- `--entropy_coef`: 0.01
- `--baseline_coef`: 0.5

**장점**: 메모리 효율적, 상대적 보상
**예상 결과**: RMSE ~1.04

---

### 4. PPO-MoE Standard (표준 RL 방식) ⭐ 새로운 버전

**특징**: 논문 재현용, 전체 epoch 데이터 수집 후 업데이트

```bash
uv run python3 src/training/train_ppo_moe_standard.py \
    --epochs 10 \
    --batch_size 512 \
    --mini_batch_size 256 \
    --ppo_epochs 4 \
    --lr 0.0003 \
    --checkpoint_dir checkpoints/ppo_moe_standard
```

**주요 파라미터**:
- `--batch_size`: 512 (에피소드 수집용, 더 크게 가능)
- `--mini_batch_size`: 256 (업데이트용)
- `--ppo_epochs`: 4 (전체 데이터로 4번 학습)
- `--gamma`: 0.99 (discount factor)
- `--lam`: 0.95 (GAE lambda)

**장점**: 더 정확한 Advantage, 논문 재현
**단점**: 메모리 사용량 높음 (79,619 샘플)
**권장 용도**: 최고 성능, 논문 출판, 벤치마크

---

### 5. GRPO-MoE Standard (표준 RL 방식) ⭐ 새로운 버전

**특징**: 논문 재현용, 전체 epoch 그룹 상대 보상

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

**주요 파라미터**:
- `--batch_size`: 512 (에피소드 수집용)
- `--mini_batch_size`: 256 (업데이트용)
- `--grpo_epochs`: 4 (전체 데이터로 4번 학습)
- `--group_size`: 256 (그룹 크기)

**장점**: 더 정확한 상대 보상, 논문 재현
**단점**: 메모리 사용량 높음
**권장 용도**: 최고 성능, 논문 출판

---

### 학습 방식 비교표

| 방식 | 메모리 | 업데이트 횟수 | 성능 | 용도 |
|------|--------|-------------|------|------|
| **Dense MoE** | 낮음 | 표준 | 🥇 최고 | 베이스라인 |
| **PPO-MoE (Batch)** | 매우 낮음 | 표준 | 🥉 | 빠른 실험 |
| **GRPO-MoE (Batch)** | 매우 낮음 | 표준 | 🥈 | 빠른 실험 |
| **PPO-MoE (Standard)** | 높음 | 표준 | ⭐ 우수 | 논문 재현 |
| **GRPO-MoE (Standard)** | 높음 | 표준 | ⭐ 우수 | 논문 재현 |

---

## 🔄 Resume 기능

### 중단된 학습 이어서 하기 ⭐ 새로운 기능 (v2.1)

학습 중 예기치 않은 중단(GPU 크래시, 시스템 종료 등)이 발생해도 이어서 학습 가능!

### 기본 사용법

```bash
# 일반 학습 (처음부터)
uv run python3 src/training/train_ppo_moe.py --epochs 100 --batch_size 256

# 중단된 학습 이어하기 (--resume 플래그만 추가)
uv run python3 src/training/train_ppo_moe.py --epochs 100 --batch_size 256 --resume
```

### 자동 저장되는 체크포인트

매 epoch마다 3가지 체크포인트가 자동 저장됨:
- `{model}_epoch_{N}.pt` - 일반 체크포인트 (최근 3개 유지)
- `{model}_best.pt` - 최고 성능 모델
- `{model}_latest.pt` - **Resume용 최신 체크포인트**

### 복원되는 정보

- ✅ 모델 가중치 (완전 복원)
- ✅ Optimizer 상태 (학습률, momentum 등)
- ✅ Scheduler 상태 (Dense MoE만)
- ✅ Early Stopping 상태 (counter, best score)
- ✅ 현재 Epoch 번호
- ✅ Best validation metric

### 사용 예시

```bash
# 학습 시작
uv run python3 src/training/train_ppo_moe.py --epochs 100 --batch_size 256
# ... Epoch 45/100 학습 중 GPU 크래시 발생

# 시스템 재시작 후 이어서 학습
uv run python3 src/training/train_ppo_moe.py --epochs 100 --batch_size 256 --resume
# ✅ Epoch 46부터 자동으로 이어서 학습!

# 출력 예시:
# ==================================================
# Resumed from epoch 45
# Best validation RMSE: 1.0452
# ==================================================
# Epoch 46/100
# ...
```

### 모든 모델 지원

```bash
# Dense MoE
uv run python3 src/training/train_dense_moe.py --epochs 50 --resume

# PPO-MoE
uv run python3 src/training/train_ppo_moe.py --epochs 50 --resume

# GRPO-MoE
uv run python3 src/training/train_grpo_moe.py --epochs 50 --resume

# PPO-MoE Standard
uv run python3 src/training/train_ppo_moe_standard.py --epochs 50 --resume

# GRPO-MoE Standard
uv run python3 src/training/train_grpo_moe_standard.py --epochs 50 --resume
```

### ⚠️ 주의사항

1. **하이퍼파라미터 일치**: Resume 시 원래와 동일한 파라미터 사용 권장
   ```bash
   # ❌ 잘못된 사용
   # 원래: --batch_size 256
   # Resume: --batch_size 512  (권장하지 않음)

   # ✅ 올바른 사용
   # 원래: --batch_size 256
   # Resume: --batch_size 256  (동일하게 유지)
   ```

2. **Checkpoint 경로**: `--checkpoint_dir`을 사용했다면 resume 시에도 동일하게 지정
   ```bash
   uv run python3 src/training/train_ppo_moe.py \
       --checkpoint_dir checkpoints/experiment_1 \
       --resume
   ```

3. **자동 처음부터 시작**: latest.pt 파일이 없으면 자동으로 처음부터 학습

### 상세 문서

- `memo/17_resume_training_feature.md` - Resume 기능 상세 설명

---

## 📈 모델 평가

### 단일 모델 평가
```bash
uv run python3 src/training/evaluate.py \
    --dense_checkpoint checkpoints/dense_moe/dense_moe_best.pt \
    --output_file results/dense_evaluation.json
```

### 전체 모델 비교 평가
```bash
uv run python3 src/training/evaluate.py \
    --dense_checkpoint checkpoints/dense_moe/dense_moe_best.pt \
    --ppo_checkpoint checkpoints/ppo_moe/ppo_moe_best.pt \
    --grpo_checkpoint checkpoints/grpo_moe/grpo_moe_best.pt \
    --output_file results/all_models_evaluation.json
```

### Standard 모델 포함 평가 (5개 모델)
```bash
uv run python3 src/training/evaluate.py \
    --dense_checkpoint checkpoints/dense_moe/dense_moe_best.pt \
    --ppo_checkpoint checkpoints/ppo_moe/ppo_moe_best.pt \
    --grpo_checkpoint checkpoints/grpo_moe/grpo_moe_best.pt \
    --ppo_standard_checkpoint checkpoints/ppo_moe_standard/ppo_moe_standard_best.pt \
    --grpo_standard_checkpoint checkpoints/grpo_moe_standard/grpo_moe_standard_best.pt \
    --output_file results/all_models_with_standard_evaluation.json
```

**출력 지표**:
- MSE, RMSE, MAE
- Expert 분포 (PPO/GRPO)
- Expert별 성능

---

## 📊 시각화

### 1. 기본 성능 비교 (3개 모델)

```bash
uv run python3 src/visualization/publication_plots.py \
    --results_file results/all_models_evaluation.json \
    --output_dir results/publication
```

**생성 파일**:
- `mse_comparison.pdf/.png`
- `rmse_comparison.pdf/.png`
- `mae_comparison.pdf/.png`
- `performance_combined.pdf/.png`
- `expert_distribution.pdf/.png`
- `comparison_table.pdf/.png`
- `comparison_table.tex` (LaTeX)

**특징**:
- 600 DPI PDF (논문 제출용)
- Colorblind-friendly 색상
- LaTeX 통합 지원

---

### 2. 종합 지표 비교 (논문용) ⭐ 새로운 기능

```bash
uv run python3 src/visualization/paper_metrics_comparison.py \
    --results_file results/all_models_with_standard_evaluation.json \
    --output_dir results/publication
```

**생성 파일**:
- `comprehensive_comparison.pdf/.png` (4-panel)
  - (a) Performance: MAE, RMSE
  - (b) Scalability: 파라미터 수
  - (c) Training Efficiency: Epoch당 시간
  - (d) Inference Efficiency: 샘플당 시간
- `performance_vs_efficiency.pdf/.png` (산점도)
- `comprehensive_table.tex` (LaTeX)

**특징**:
- 확장성, 효율성, 성능 종합 비교
- 논문 Figure용 고품질 시각화
- 5개 모델 동시 비교 지원

---

### 3. 학습 곡선

```bash
uv run python3 src/visualization/training_curves.py \
    --checkpoint_dir checkpoints \
    --output_dir results/publication
```

**생성 파일**:
- `training_curves_combined.pdf/.png`
- 모델별 Loss/RMSE 변화

---

### 시각화 비교표

| 도구 | 용도 | 지표 | 모델 수 |
|------|------|------|---------|
| `publication_plots.py` | 기본 성능 비교 | MSE, RMSE, MAE | 3개 |
| `paper_metrics_comparison.py` ⭐ | 종합 비교 | 성능, 확장성, 효율성 | 5개 |
| `training_curves.py` | 학습 과정 | Loss, RMSE | 모든 모델 |

---

## 🔬 확장성 실험 (Scalability Experiments) ⭐ 새로운 기능 (v2.2)

### Expert 개수를 늘려서 확장성 비교

**목적**: Dense MoE vs RL MoE의 확장성 차이를 정량적으로 측정

**핵심 차이**:
- Dense MoE: 모든 Expert 사용 → Expert 늘어나면 느려짐 (O(n))
- RL MoE: 하나만 선택 → Expert 늘어나도 속도 일정 (O(1))

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
    --num_epochs 5 \
    --experiment_id my_exp
```

### 측정 지표
- **학습 시간**: Expert 개수에 따른 학습 시간
- **추론 시간**: Expert 개수에 따른 추론 시간
- **메모리 사용량**: GPU 메모리
- **성능**: RMSE, MAE
- **효율성**: 1 / (RMSE × Time)

### 결과 확인
```bash
# 시각화 그래프
ls experiments/visualizations/{실험ID}/*.png

# 수치 테이블
cat experiments/visualizations/{실험ID}/scalability_comparison_table.md
```

### 상세 가이드
- **빠른 시작**: `experiments/QUICKSTART.md`
- **상세 가이드**: `experiments/README_SCALABILITY.md`
- **메모**: `memo/18_scalability_experiments.md`

---

## 🔄 Git 관리

### 초기 설정
```bash
git init
git remote add origin https://github.com/hyeondata/recsys.git
git checkout -b moe
```

### 커밋 및 푸시
```bash
git add .
git commit -m "feat: Add standard RL implementation and comprehensive metrics"
git push -u origin moe
```

### 주요 커밋 메시지 규칙
- `feat:` - 새로운 기능 추가
- `fix:` - 버그 수정
- `refactor:` - 코드 리팩토링
- `docs:` - 문서 업데이트
- `test:` - 테스트 추가/수정

---

## 📝 버전 히스토리

### v2.2 (2025-11-09) - 확장성 실험
- ✅ Expert 개수 확장성 실험 시스템 (`experiments/scalability_experiment.py`)
- ✅ 학습/추론 시간, 메모리 자동 측정
- ✅ 시각화 자동 생성 (`experiments/visualize_scalability.py`)
- ✅ Dense MoE vs RL MoE 확장성 비교
- ✅ 빠른 테스트 스크립트 (`quick_test.sh`)
- 📄 메모: memo/18_scalability_experiments.md
- 📄 가이드: experiments/README_SCALABILITY.md

### v2.1 (2025-11-08) - Resume 기능
- ✅ Resume 기능 추가 (중단된 학습 이어하기)
- ✅ 매 epoch마다 `_latest.pt` 자동 저장
- ✅ Optimizer, Scheduler, Early Stopping 상태 복원
- ✅ 모든 학습 스크립트에 `--resume` 플래그 추가
- ✅ `CheckpointManager` 개선
- 📄 메모: memo/17_resume_training_feature.md

### v2.0 (2025-11-04) - 표준 RL 구현
- ✅ `train_ppo_moe_standard.py` 추가
- ✅ `train_grpo_moe_standard.py` 추가
- ✅ `paper_metrics_comparison.py` 추가
- ✅ 종합 지표 비교 시각화
- ✅ 5개 모델 동시 비교 지원
- 📄 메모: memo/16_standard_rl_implementation.md

### v1.2 (2025-11-03)
- ✅ Publication-quality 시각화
- ✅ 600 DPI PDF 출력
- ✅ LaTeX 테이블 자동 생성
- 📄 메모: memo/15_publication_visualization.md

### v1.1 (2025-11-03)
- ✅ Batch loading 방식 수정
- ✅ 메모리 효율성 300배 향상
- ✅ Dense MoE와 공정한 비교
- 📄 메모: memo/14_batch_loading_fix.md

### v1.0 (2025-11-02)
- ✅ Dense MoE 구현
- ✅ PPO-MoE 구현
- ✅ GRPO-MoE 구현
- ✅ 기본 시각화
- 📄 메모: memo/01-10 참조

---

## 🎯 Quick Start (추천 워크플로우)

### 1. 빠른 실험 (메모리 제한)
```bash
# 1. 학습
uv run python3 src/training/train_dense_moe.py --epochs 10 --batch_size 1024
uv run python3 src/training/train_ppo_moe.py --epochs 10 --batch_size 256
uv run python3 src/training/train_grpo_moe.py --epochs 10 --batch_size 256

# 2. 평가
uv run python3 src/training/evaluate.py \
    --dense_checkpoint checkpoints/dense_moe/dense_moe_best.pt \
    --ppo_checkpoint checkpoints/ppo_moe/ppo_moe_best.pt \
    --grpo_checkpoint checkpoints/grpo_moe/grpo_moe_best.pt \
    --output_file results/results.json

# 3. 시각화
uv run python3 src/visualization/publication_plots.py \
    --results_file results/results.json \
    --output_dir results/figures
```

### 2. 논문 출판용 (최고 성능)
```bash
# 1. 모든 모델 학습
uv run python3 src/training/train_dense_moe.py --epochs 20 --batch_size 1024
uv run python3 src/training/train_ppo_moe_standard.py --epochs 20 --batch_size 512
uv run python3 src/training/train_grpo_moe_standard.py --epochs 20 --batch_size 512

# 2. 종합 평가
uv run python3 src/training/evaluate.py \
    --dense_checkpoint checkpoints/dense_moe/dense_moe_best.pt \
    --ppo_standard_checkpoint checkpoints/ppo_moe_standard/ppo_moe_standard_best.pt \
    --grpo_standard_checkpoint checkpoints/grpo_moe_standard/grpo_moe_standard_best.pt \
    --output_file results/publication_results.json

# 3. 종합 시각화
uv run python3 src/visualization/paper_metrics_comparison.py \
    --results_file results/publication_results.json \
    --output_dir results/publication
```

---

## ⚠️ 주의사항

### GPU 메모리
- Dense MoE: batch_size 1024 (안전)
- PPO/GRPO (Batch): batch_size 256 (권장)
- PPO/GRPO (Standard): batch_size 512 (24GB GPU 필요)

### 안정성
- Standard 방식: 더 많은 메모리 필요하지만 성능 우수
- Batch 방식: 메모리 효율적, GPU 드라이버 크래시 방지

### 학습 시간 (RTX 3090 기준)
- Dense MoE: ~2분/epoch
- PPO-MoE (Batch): ~4분/epoch
- PPO-MoE (Standard): ~5분/epoch
- GRPO-MoE (Batch): ~4분/epoch
- GRPO-MoE (Standard): ~5분/epoch

---

## 📚 추가 문서

### 상세 문서 (memo/)
- `01_project_overview.md` - 프로젝트 개요
- `04_model_implementation.md` - 모델 구현 세부사항
- `05_training_implementation.md` - 학습 구현
- `16_standard_rl_implementation.md` - 표준 RL 구현 (최신)

### 압축 문서 (memo_compression/)
- `01_summary.md` - 전체 요약
- `02_technical_details.md` - 기술 세부사항
- `03_known_issues.md` - 알려진 이슈

---

## 🆘 트러블슈팅

### CUDA Out of Memory
```bash
# Batch 크기 줄이기
--batch_size 128  # 또는 64

# Standard 방식 대신 Batch 방식 사용
train_ppo_moe.py  # train_ppo_moe_standard.py 대신
```

### 학습이 너무 느림
```bash
# Workers 수 조정
--num_workers 8  # 기본값: 4

# Batch 크기 증가 (GPU 메모리 허용 시)
--batch_size 512
```

### 성능이 낮음
- Epoch 수 증가: `--epochs 20`
- Learning rate 조정: `--lr 0.0001` (낮추기)
- Early stopping patience 증가: `--patience 20`

---

**마지막 업데이트**: 2025-11-04
**작성자**: Claude + User
**버전**: v2.0
