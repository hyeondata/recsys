# MoE-based Movie Recommendation System

MovieLens 100k 데이터셋을 활용한 **Mixture of Experts(MoE) 기반 영화 추천 시스템**

## 프로젝트 개요

사용자 특성(나이, 성별, 직업)과 영화 정보를 활용하여 평점을 예측하는 추천 시스템입니다. 세 가지 다른 Gating 전략을 비교합니다:

1. **Dense MoE**: Fully Connected Layer 기반 (비강화학습 베이스라인)
2. **PPO-MoE**: PPO(Proximal Policy Optimization) 강화학습 기반
3. **GRPO-MoE**: GRPO(Group Relative Policy Optimization) 강화학습 기반

## 프로젝트 구조

```
claudeMoE/
├── ml-100k/              # MovieLens 100k 데이터셋
├── src/
│   ├── data/             # 데이터 전처리 및 로딩
│   │   ├── movie_preprocessor.py
│   │   ├── user_preprocessor.py
│   │   ├── dataset.py
│   │   └── enhanced_dataset.py
│   ├── models/           # MoE 모델들
│   │   ├── expert_network.py
│   │   ├── base_moe.py
│   │   ├── dense_moe.py
│   │   ├── ppo_moe.py
│   │   └── grpo_moe.py
│   ├── utils/            # 유틸리티 함수들
│   │   ├── metrics.py
│   │   └── trainer_utils.py
│   └── training/         # 학습 스크립트
│       ├── train_dense_moe.py
│       ├── train_ppo_moe.py
│       ├── train_grpo_moe.py
│       └── evaluate.py
├── configs/              # 실험 설정 파일들
│   ├── dense_moe.yaml
│   ├── ppo_moe.yaml
│   └── grpo_moe.yaml
├── memo/                 # 프로젝트 문서
└── README.md
```

## 설치

### 요구사항

- Python 3.8+
- PyTorch 1.10+
- pandas, numpy, scikit-learn, tqdm

### 설치 방법

```bash
pip install torch pandas numpy scikit-learn tqdm pyyaml
```

## 데이터 준비

MovieLens 100k 데이터셋이 `ml-100k/` 디렉토리에 있어야 합니다.

데이터셋 구성:
- `u.data`: 평점 데이터 (100,000개)
- `u.item`: 영화 정보 (1,682개)
- `u.user`: 사용자 정보 (943명)
- `u1.base`, `u1.test`: 학습/테스트 분할

## 학습

### 1. Dense MoE (베이스라인)

```bash
python src/training/train_dense_moe.py \
    --data_dir ml-100k \
    --train_rating_path ml-100k/u1.base \
    --val_rating_path ml-100k/u1.test \
    --batch_size 256 \
    --epochs 100 \
    --lr 0.001 \
    --checkpoint_dir checkpoints/dense_moe
```

### 2. PPO-MoE (강화학습)

```bash
python src/training/train_ppo_moe.py \
    --data_dir ml-100k \
    --train_rating_path ml-100k/u1.base \
    --val_rating_path ml-100k/u1.test \
    --batch_size 256 \
    --ppo_epochs 4 \
    --epochs 100 \
    --lr 0.0003 \
    --checkpoint_dir checkpoints/ppo_moe
```

### 3. GRPO-MoE (강화학습)

```bash
python src/training/train_grpo_moe.py \
    --data_dir ml-100k \
    --train_rating_path ml-100k/u1.base \
    --val_rating_path ml-100k/u1.test \
    --batch_size 256 \
    --grpo_epochs 4 \
    --epochs 100 \
    --lr 0.0003 \
    --checkpoint_dir checkpoints/grpo_moe
```

## 평가

학습된 모델을 평가하고 비교합니다:

```bash
python src/training/evaluate.py \
    --data_dir ml-100k \
    --test_rating_path ml-100k/u1.test \
    --dense_checkpoint checkpoints/dense_moe/dense_moe_best.pt \
    --ppo_checkpoint checkpoints/ppo_moe/ppo_moe_best.pt \
    --grpo_checkpoint checkpoints/grpo_moe/grpo_moe_best.pt \
    --output_file evaluation_results.json
```

## 모델 아키텍처

### 공통 구조

- **User Embedding**: user_id, age, gender, occupation 결합
- **Movie Embedding**: movie_id
- **State**: User + Movie 임베딩 concat (128차원)
- **Expert Network**: 8개의 독립적인 FFN

### Gating 전략 비교

| 특징 | Dense MoE | PPO-MoE | GRPO-MoE |
|------|-----------|---------|----------|
| Gating | FC Layer | Policy Network | Policy Network |
| 학습 방식 | Supervised | On-policy RL | On-policy RL |
| Expert 선택 | 가중합 (모두 사용) | 단일 선택 | 단일 선택 |
| 보상 | - | 절대 오차 | 상대 오차 (정규화) |
| 추가 네트워크 | - | Value Network | Baseline Network |

## 평가 지표

- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error

평점은 1-5 범위로 denormalize되어 평가됩니다.

## 하이퍼파라미터

자세한 설정은 `configs/` 폴더의 YAML 파일을 참조하세요.

### 주요 하이퍼파라미터

- **embedding_dim**: 64
- **num_experts**: 8
- **expert_hidden_dim**: 256
- **batch_size**: 256
- **learning_rate**: 0.001 (Dense), 0.0003 (RL)

## 결과 예시

```
Model Comparison
--------------------------------------------------
Model           MSE        RMSE       MAE
--------------------------------------------------
dense_moe       0.8234     0.9074     0.7123
ppo_moe         0.8156     0.9031     0.7089
grpo_moe        0.8098     0.8999     0.7045
```

## 문서

자세한 구현 내용은 `memo/` 폴더의 문서를 참조하세요:

- `01_project_overview.md`: 프로젝트 전체 개요
- `02_dataset_analysis.md`: 데이터셋 분석
- `03_code_structure.md`: 코드 구조 분석
- `04_model_implementation.md`: 모델 구현 상세

## 참고

- **데이터셋**: MovieLens 100k (GroupLens Research)
- **논문**: F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context.
- **강화학습**: PPO (Schulman et al., 2017), GRPO (Group Relative Policy Optimization)

## 라이선스

이 프로젝트는 교육 및 연구 목적으로 사용됩니다.
