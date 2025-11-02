# 프로젝트 전체 개요

## 프로젝트 구조
```
claudeMoE/
├── ml-100k/         # MovieLens 100k 데이터셋
├── src/             # 소스 코드
│   ├── data/        # 데이터 전처리 및 로딩
│   │   ├── movie_preprocessor.py
│   │   ├── user_preprocessor.py
│   │   ├── dataset.py
│   │   └── enhanced_dataset.py
│   ├── models/      # MoE 모델 정의 (완료)
│   │   ├── expert_network.py    # Expert Network
│   │   ├── base_moe.py          # Base MoE Model
│   │   ├── dense_moe.py         # Dense MoE (베이스라인)
│   │   ├── ppo_moe.py           # PPO-MoE (강화학습)
│   │   └── grpo_moe.py          # GRPO-MoE (강화학습)
│   ├── utils/       # 유틸리티 (완료)
│   │   ├── metrics.py           # 평가 지표
│   │   └── trainer_utils.py     # 학습 헬퍼
│   └── training/    # 학습 스크립트 (완료)
│       ├── train_dense_moe.py
│       ├── train_ppo_moe.py
│       ├── train_grpo_moe.py
│       └── evaluate.py
├── configs/         # 실험 설정 파일 (완료)
│   ├── dense_moe.yaml
│   ├── ppo_moe.yaml
│   └── grpo_moe.yaml
├── memo/            # 프로젝트 문서
│   ├── 01_project_overview.md
│   ├── 02_dataset_analysis.md
│   ├── 03_code_structure.md
│   ├── 04_model_implementation.md
│   └── 05_training_implementation.md
└── README.md        # 프로젝트 가이드
```

## 프로젝트 목적
- MovieLens 100k 데이터셋을 활용한 **MoE 기반 영화 추천 시스템** 구현
- 영화 제목, 사용자 특성(나이, 성별, 직업)을 활용한 평점 예측
- **강화학습(PPO, GRPO)과 비강화학습(Dense) 방식 비교**

## 현재 구현 상태

### ✅ 데이터 전처리 코드 완료
- 영화 데이터 전처리 (MoviePreprocessor)
- 사용자 데이터 전처리 (UserPreprocessor)
- 데이터셋 클래스 (MovieRatingDataset, EnhancedMovieRatingDataset)

### ✅ MoE 모델 구현 완료 (2025-11-02)
- **Expert Network**: 8개의 독립적인 FFN
- **Base MoE Model**: User/Movie 임베딩 및 State 생성
- **Dense MoE**: FC Layer 기반 Gating (베이스라인)
- **PPO-MoE**: PPO 강화학습 기반 Expert 선택
- **GRPO-MoE**: 그룹 상대 보상 기반 Expert 선택

### ✅ 유틸리티 및 학습 파이프라인 완료 (2025-11-02)
- **평가 지표**: MSE, RMSE, MAE, Expert 분석
- **학습 헬퍼**: EarlyStopping, CheckpointManager, AverageMeter
- **Dense MoE 학습 스크립트**: Supervised Learning
- **PPO-MoE 학습 스크립트**: On-policy RL with GAE
- **GRPO-MoE 학습 스크립트**: Group Relative Rewards
- **통합 평가 스크립트**: 모든 모델 비교 평가

### ✅ 실험 설정 및 문서 완료 (2025-11-02)
- **실험 설정 파일**: YAML 형식 (dense, ppo, grpo)
- **README.md**: 프로젝트 가이드 및 사용법
- **프로젝트 문서**: 5개 문서 (overview, dataset, code, model, training)

## 다음 단계
1. ⏳ **실제 학습 실행**
   - Dense MoE 학습 및 성능 확인
   - PPO-MoE 학습 및 비교
   - GRPO-MoE 학습 및 비교

2. ⏳ **하이퍼파라미터 튜닝**
   - Learning rate 조정
   - Expert 개수 실험 (4, 8, 16)
   - 임베딩 차원 실험

3. ⏳ **결과 분석 및 시각화**
   - 학습 곡선 그래프
   - Expert 선택 히트맵
   - 성능 비교 차트

4. ⏳ **추가 실험**
   - 5-fold Cross-validation (u1-u5)
   - Ablation Study
