# 기술적 세부사항

## 데이터셋
- MovieLens 100k: 943 사용자, 1,682 영화, 100,000 평점
- Train/Test: u1.base (79,619) / u1.test (20,381)
- 사용자 특성: 나이, 성별, 직업

## 모델 아키텍처
- User/Movie 임베딩: 64차원
- Expert 수: 8개
- Expert 구조: FFN (256-128-64-1)
- 총 파라미터: ~750K

## 핵심 차이점

### Dense MoE
- Gating: FC Layer → Softmax
- 출력: 모든 Expert의 가중 평균

### PPO-MoE
- Gating: Policy Network (RL)
- Expert 선택: Categorical 샘플링
- Loss: Policy + Value + Rating
- 특징: GAE, PPO clipping

### GRPO-MoE
- Gating: Policy Network (RL)
- Expert 선택: Categorical 샘플링
- Loss: Policy + Baseline + Rating
- 특징: Group Relative Rewards

## Batch Loading 수정 (중요!)
**기존**: 전체 데이터 로드 → 배치로 분할 → 4번 업데이트
**수정후**: 배치 단위 로드 → 각 배치 4번 업데이트

메모리: 79,619 샘플 → 256 샘플 (300배 감소)
