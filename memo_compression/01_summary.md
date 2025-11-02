# 프로젝트 요약

## 목표
MovieLens 100k 기반 MoE 영화 추천 시스템 구현 및 비교

## 구현된 모델
1. **Dense MoE**: FC Layer Gating (베이스라인)
2. **PPO-MoE**: Proximal Policy Optimization
3. **GRPO-MoE**: Group Relative Policy Optimization

## 최종 결과 (batch loading 수정 전)
- Dense MoE: RMSE **0.9803** 🥇
- GRPO-MoE: RMSE **1.0412** 🥈
- PPO-MoE: RMSE **1.0659** 🥉

## 구조
```
src/
├── data/              # 전처리
├── models/            # MoE 모델 3종
├── training/          # 학습 스크립트
├── utils/             # 메트릭, Trainer 유틸
└── visualization/     # 논문용 시각화
    ├── publication_plots.py    # 성능 비교 (PDF 600 DPI)
    ├── training_curves.py      # 학습 곡선
    └── README.md               # 사용 가이드
```

## 주요 수정사항
1. tqdm 진행률 표시 추가 (12_code_refactoring_tqdm.md)
2. Batch loading 방식 수정 (14_batch_loading_fix.md)
   - GRPO/PPO를 Dense와 동일하게 batch 단위로 학습
   - 메모리 효율성 300배 향상
3. Publication-quality 시각화 (15_publication_visualization.md)
   - IEEE/ACM 기준 충족 (600 DPI PDF)
   - Colorblind-friendly 색상
   - LaTeX 통합 지원
