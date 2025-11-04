# MoE 영화 추천 시스템 - 사용법 통합 가이드

**최종 업데이트**: 2025-11-04
**버전**: v2.0 (표준 RL 구현 추가)

---

## 📋 목차
1. [환경 설정](#환경-설정)
2. [데이터 준비](#데이터-준비)
3. [모델 학습](#모델-학습)
4. [모델 평가](#모델-평가)
5. [시각화](#시각화)
6. [Git 관리](#git-관리)
7. [버전 히스토리](#버전-히스토리)

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
