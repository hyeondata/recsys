# MoE-based Movie Recommendation System

MovieLens 100k 데이터셋을 활용한 **Mixture of Experts(MoE) 기반 영화 추천 시스템**

> 🔥 **전체 사용법**: [`memo_compression/00_using.md`](./memo_compression/00_using.md)

---

## 📋 프로젝트 개요

사용자 특성(나이, 성별, 직업)과 영화 정보를 활용하여 평점을 예측하는 추천 시스템입니다. **5가지** 다른 Gating 전략을 비교합니다.

### 🎯 모델 종류

| 모델 | 방식 | RMSE | 특징 |
|------|------|------|------|
| **Dense MoE** | FC Gating | 0.9803 🥇 | 베이스라인, 모든 Expert 사용 |
| **GRPO-MoE (Batch)** | 그룹 상대 보상 | 1.0412 🥈 | 메모리 효율적 |
| **PPO-MoE (Batch)** | PPO RL | 1.0659 🥉 | 메모리 효율적 |
| **PPO-MoE (Standard)** ⭐ | 표준 PPO | - | 논문 재현용 |
| **GRPO-MoE (Standard)** ⭐ | 표준 GRPO | - | 논문 재현용 |

---

## 📁 프로젝트 구조

```
claudeMoE/
├── ml-100k/              # MovieLens 100k 데이터셋
├── src/
│   ├── data/             # 데이터 전처리 및 로딩
│   │   ├── movie_preprocessor.py
│   │   ├── user_preprocessor.py
│   │   ├── dataset.py
│   │   └── enhanced_dataset.py
│   ├── models/           # MoE 모델들 (3종)
│   │   ├── base_moe.py
│   │   ├── dense_moe.py
│   │   ├── ppo_moe.py
│   │   └── grpo_moe.py
│   ├── training/         # 학습 스크립트 (5종)
│   │   ├── train_dense_moe.py
│   │   ├── train_ppo_moe.py              # Batch 방식
│   │   ├── train_grpo_moe.py             # Batch 방식
│   │   ├── train_ppo_moe_standard.py     ⭐ Standard RL
│   │   ├── train_grpo_moe_standard.py    ⭐ Standard RL
│   │   └── evaluate.py
│   ├── visualization/    # 시각화 도구 (3종)
│   │   ├── publication_plots.py          # 기본 비교
│   │   ├── paper_metrics_comparison.py   ⭐ 종합 비교
│   │   └── training_curves.py
│   └── utils/            # 유틸리티
│       ├── metrics.py
│       └── trainer_utils.py
├── memo/                 # 상세 문서 (16개 파일)
├── memo_compression/     # 압축 문서
│   ├── 00_using.md       🔥 전체 사용법 통합
│   ├── 01_summary.md
│   ├── 02_technical_details.md
│   ├── 03_known_issues.md
│   └── 04_usage.md
└── README.md
```

---

## 🚀 Quick Start

### 1. 환경 설정

```bash
# uv 설치 (패키지 관리자)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 의존성 설치
uv sync
```

**필수 요구사항**:
- Python 3.8+
- PyTorch 2.0+
- CUDA
- GPU: RTX 3090 (24GB) 권장

---

### 2. 데이터 준비

MovieLens 100k 데이터셋이 `ml-100k/` 디렉토리에 있어야 합니다.

```bash
# 자동으로 다운로드되거나, 수동 다운로드:
wget https://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip
```

---

### 3. 모델 학습

#### Dense MoE (베이스라인)
```bash
uv run python3 src/training/train_dense_moe.py \
    --epochs 10 \
    --batch_size 1024 \
    --lr 0.001 \
    --checkpoint_dir checkpoints/dense_moe
```

#### PPO/GRPO (Batch 방식 - 메모리 효율적)
```bash
# PPO-MoE
uv run python3 src/training/train_ppo_moe.py \
    --epochs 10 --batch_size 256 --ppo_epochs 4 --lr 0.0003

# GRPO-MoE
uv run python3 src/training/train_grpo_moe.py \
    --epochs 10 --batch_size 256 --grpo_epochs 4 --lr 0.0003
```

#### PPO/GRPO (Standard 방식 - 논문 재현용) ⭐
```bash
# PPO-MoE Standard
uv run python3 src/training/train_ppo_moe_standard.py \
    --epochs 10 --batch_size 512 --mini_batch_size 256 --ppo_epochs 4

# GRPO-MoE Standard
uv run python3 src/training/train_grpo_moe_standard.py \
    --epochs 10 --batch_size 512 --mini_batch_size 256 --grpo_epochs 4
```

---

### 4. 평가

```bash
# 전체 모델 비교
uv run python3 src/training/evaluate.py \
    --dense_checkpoint checkpoints/dense_moe/dense_moe_best.pt \
    --ppo_checkpoint checkpoints/ppo_moe/ppo_moe_best.pt \
    --grpo_checkpoint checkpoints/grpo_moe/grpo_moe_best.pt \
    --output_file results/evaluation.json
```

---

### 5. 시각화

```bash
# 기본 성능 비교
uv run python3 src/visualization/publication_plots.py \
    --results_file results/evaluation.json \
    --output_dir results/publication

# 종합 지표 비교 (논문용) ⭐
uv run python3 src/visualization/paper_metrics_comparison.py \
    --results_file results/evaluation.json \
    --output_dir results/publication
```

---

## 📊 주요 특징

### 1. 학습 방식 비교

| 방식 | 메모리 사용량 | 업데이트 방식 | 용도 |
|------|-------------|-------------|------|
| **Batch** | 매우 낮음 (256 샘플) | 각 배치 즉시 업데이트 | 빠른 실험, GPU 제한 |
| **Standard** ⭐ | 높음 (79,619 샘플) | 전체 epoch 수집 후 업데이트 | 논문 재현, 최고 성능 |

### 2. 시각화 도구

| 도구 | 기능 | 지표 |
|------|------|------|
| `publication_plots.py` | 기본 성능 비교 | MSE, RMSE, MAE |
| `paper_metrics_comparison.py` ⭐ | 종합 비교 | 성능 + 확장성 + 효율성 |
| `training_curves.py` | 학습 과정 | Loss, RMSE 곡선 |

### 3. 주요 개선사항

- ✅ Batch loading 방식: **메모리 300배 절약**
- ✅ Standard RL 구현: **논문 재현 가능**
- ✅ 논문 출판용 시각화: **600 DPI PDF**
- ✅ 종합 지표 비교: **확장성, 효율성, 성능**

---

## 📖 모델 아키텍처

### 공통 구조
- **User Embedding**: user_id, age, gender, occupation (64차원)
- **Movie Embedding**: movie_id (64차원)
- **State**: User + Movie 임베딩 concat (128차원)
- **Expert Network**: 8개의 독립적인 FFN (256-128-64-1)
- **총 파라미터**: ~750K

### Gating 전략 비교

| 특징 | Dense MoE | PPO-MoE | GRPO-MoE |
|------|-----------|---------|----------|
| **Gating** | FC Layer → Softmax | Policy Network | Policy Network |
| **학습 방식** | Supervised | On-policy RL (PPO) | On-policy RL (GRPO) |
| **Expert 선택** | 가중합 (모두 사용) | Categorical 샘플링 | Categorical 샘플링 |
| **보상** | - | -\|error\| | Group Relative |
| **추가 Network** | - | Value Network | Baseline Network |
| **특징** | 안정적 | GAE, PPO clipping | Group 상대 보상 |

---

## 📈 평가 지표

### 성능 지표
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error

### 확장성 지표
- **Parameters**: 모델 파라미터 수

### 효율성 지표
- **Training Time**: Epoch당 학습 시간
- **Inference Time**: 샘플당 추론 시간

---

## 📚 문서

### 상세 문서 (memo/)
- `01_project_overview.md` - 프로젝트 개요
- `04_model_implementation.md` - 모델 구현
- `16_standard_rl_implementation.md` - **표준 RL 구현 (최신)**

### 압축 문서 (memo_compression/)
- **`00_using.md`** - 🔥 **전체 사용법 통합 문서 (필수)**
- `01_summary.md` - 전체 요약
- `02_technical_details.md` - 기술 세부사항
- `03_known_issues.md` - 알려진 이슈

---

## ⚠️ 주의사항

### GPU 메모리
- Dense MoE: `batch_size 1024` (안전)
- PPO/GRPO (Batch): `batch_size 256` (권장)
- PPO/GRPO (Standard): `batch_size 512` (24GB GPU 필요)

### 학습 시간 (RTX 3090 기준)
- Dense MoE: ~2분/epoch
- PPO-MoE (Batch): ~4분/epoch
- GRPO-MoE (Batch): ~4분/epoch
- PPO-MoE (Standard): ~5분/epoch
- GRPO-MoE (Standard): ~5분/epoch

---

## 🔄 버전 히스토리

### v2.0 (2025-11-04) - 표준 RL 구현
- ✅ Standard RL 구현 (PPO, GRPO)
- ✅ 종합 지표 비교 시각화
- ✅ 5개 모델 동시 비교
- ✅ 전체 사용법 통합 문서

### v1.2 (2025-11-03) - 논문용 시각화
- ✅ 600 DPI PDF 출력
- ✅ LaTeX 테이블 자동 생성
- ✅ Colorblind-friendly 색상

### v1.1 (2025-11-03) - Batch Loading 개선
- ✅ 메모리 효율성 300배 향상
- ✅ Dense MoE와 공정한 비교

---

## 📜 참고 문헌

- **데이터셋**: MovieLens 100k (GroupLens Research)
- **논문**: F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context.
- **강화학습**:
  - PPO: Schulman et al., 2017. Proximal Policy Optimization Algorithms
  - GRPO: Group Relative Policy Optimization

---

## 📞 문의

프로젝트 관련 문의나 이슈는 [GitHub Issues](https://github.com/hyeondata/recsys/issues)에 등록해주세요.

---

**마지막 업데이트**: 2025-11-04
**버전**: v2.0
**작성자**: Claude + User
