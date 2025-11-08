# 프로젝트 요약

**최종 업데이트**: 2025-11-09
**버전**: v2.2

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

experiments/
├── scalability_experiment.py      # 확장성 실험 ⭐ 신규
├── visualize_scalability.py       # 확장성 시각화 ⭐ 신규
├── run_scalability_experiment.sh  # 실행 스크립트
└── quick_test.sh                  # 빠른 테스트
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
4. **Resume 기능 추가 (17_resume_training_feature.md)**
   - 중단된 학습 이어서 하기 (`--resume` 플래그)
   - 매 epoch마다 `_latest.pt` 자동 저장
   - Optimizer, Scheduler, Early Stopping 상태 완전 복원
   - 모든 학습 스크립트 지원
5. **확장성 실험 시스템 (18_scalability_experiments.md)** ⭐ 최신
   - Expert 개수 증가에 따른 성능 비교 (4, 8, 16, 32, 64)
   - 학습/추론 시간, 메모리 자동 측정
   - Dense MoE vs RL MoE 확장성 차이 정량화
   - 시각화 자동 생성 (600 DPI)
