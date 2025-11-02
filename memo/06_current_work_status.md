# 현재 작업 상태 및 진행 계획

## 작성 일자
2025년 11월 2일

---

## 프로젝트 현황 요약

### ✅ 완료된 작업

#### 1. 데이터 전처리 모듈 (100% 완료)
- `src/data/movie_preprocessor.py`: 영화 제목 인코딩 및 장르 추출
- `src/data/user_preprocessor.py`: 사용자 특성 전처리 (나이, 성별, 직업)
- `src/data/dataset.py`: 기본 데이터셋
- `src/data/enhanced_dataset.py`: 사용자 특성 포함 데이터셋

#### 2. MoE 모델 구현 (100% 완료)
- `src/models/expert_network.py`: 8개 Expert FFN
- `src/models/base_moe.py`: 공통 임베딩 및 State 생성
- `src/models/dense_moe.py`: FC Layer Gating (베이스라인)
- `src/models/ppo_moe.py`: PPO 강화학습 Gating
- `src/models/grpo_moe.py`: GRPO 강화학습 Gating

#### 3. 학습 파이프라인 (100% 완료)
- `src/utils/metrics.py`: MSE, RMSE, MAE, Expert 분석
- `src/utils/trainer_utils.py`: EarlyStopping, CheckpointManager, AverageMeter
- `src/training/train_dense_moe.py`: Dense MoE 학습 스크립트
- `src/training/train_ppo_moe.py`: PPO-MoE 학습 스크립트
- `src/training/train_grpo_moe.py`: GRPO-MoE 학습 스크립트
- `src/training/evaluate.py`: 통합 평가 스크립트

#### 4. 실험 설정 및 문서 (100% 완료)
- `configs/dense_moe.yaml`: Dense MoE 설정
- `configs/ppo_moe.yaml`: PPO-MoE 설정
- `configs/grpo_moe.yaml`: GRPO-MoE 설정
- `README.md`: 프로젝트 가이드
- `memo/`: 5개의 상세 문서

---

## 패키지 관리: uv 사용

### uv란?
- Rust로 작성된 초고속 Python 패키지 관리자
- pip, pip-tools, virtualenv를 대체
- 10-100배 빠른 패키지 설치

### 필요한 의존성

```toml
# pyproject.toml (예상)
[project]
name = "claude-moe"
version = "0.1.0"
requires-python = ">=3.8"
dependencies = [
    "torch>=2.0.0",
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "scikit-learn>=1.3.0",
    "pyyaml>=6.0",
    "tqdm>=4.65.0",
]

[project.optional-dependencies]
dev = [
    "jupyter>=1.0.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
]
```

### uv 설치 및 사용

```bash
# uv 설치 (이미 설치되어 있을 수 있음)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 프로젝트 초기화 (pyproject.toml 생성)
uv init

# 의존성 설치
uv pip install torch numpy pandas scikit-learn pyyaml tqdm

# 가상환경 생성 및 활성화
uv venv
source .venv/bin/activate

# 또는 uv run으로 직접 실행
uv run python src/training/train_dense_moe.py
```

---

## 다음 실행 단계

### Phase 1: 환경 설정 및 검증
- [ ] uv를 사용한 패키지 설치
- [ ] 데이터셋 경로 확인 (ml-100k/)
- [ ] GPU 가용성 확인
- [ ] 간단한 데이터 로딩 테스트

### Phase 2: Dense MoE 학습 (베이스라인)
```bash
uv run python src/training/train_dense_moe.py \
    --data_dir ml-100k \
    --train_rating_path ml-100k/u1.base \
    --val_rating_path ml-100k/u1.test \
    --batch_size 256 \
    --epochs 100 \
    --lr 0.001 \
    --output_dir checkpoints/dense_moe
```

**예상 소요 시간**: 10-20분 (GPU 기준)

### Phase 3: PPO-MoE 학습
```bash
uv run python src/training/train_ppo_moe.py \
    --data_dir ml-100k \
    --train_rating_path ml-100k/u1.base \
    --val_rating_path ml-100k/u1.test \
    --batch_size 256 \
    --epochs 100 \
    --lr 0.0003 \
    --ppo_epochs 4 \
    --output_dir checkpoints/ppo_moe
```

**예상 소요 시간**: 30-60분 (GPU 기준)

### Phase 4: GRPO-MoE 학습
```bash
uv run python src/training/train_grpo_moe.py \
    --data_dir ml-100k \
    --train_rating_path ml-100k/u1.base \
    --val_rating_path ml-100k/u1.test \
    --batch_size 256 \
    --epochs 100 \
    --lr 0.0003 \
    --grpo_epochs 4 \
    --temperature 1.0 \
    --output_dir checkpoints/grpo_moe
```

**예상 소요 시간**: 30-60분 (GPU 기준)

### Phase 5: 통합 평가
```bash
uv run python src/training/evaluate.py \
    --data_dir ml-100k \
    --test_rating_path ml-100k/u1.test \
    --dense_checkpoint checkpoints/dense_moe/dense_moe_best.pt \
    --ppo_checkpoint checkpoints/ppo_moe/ppo_moe_best.pt \
    --grpo_checkpoint checkpoints/grpo_moe/grpo_moe_best.pt \
    --output_file results/evaluation_results.json
```

### Phase 6: 결과 분석 및 시각화
- [ ] 학습 곡선 플롯
- [ ] Expert 선택 분포 분석
- [ ] 성능 비교 테이블
- [ ] Ablation Study 설계

---

## 예상 실험 결과

### 성능 비교 (예상)

| 모델 | MSE | RMSE | MAE | 특징 |
|------|-----|------|-----|------|
| Dense MoE | 0.82 | 0.91 | 0.71 | 안정적 베이스라인 |
| PPO-MoE | 0.80-0.83 | 0.89-0.91 | 0.70-0.72 | 단일 Expert 선택 |
| GRPO-MoE | 0.79-0.82 | 0.89-0.91 | 0.69-0.71 | 상대적 보상 |

### 가설
1. **Dense MoE**: 모든 Expert를 활용하므로 안정적이지만 최고 성능은 아닐 수 있음
2. **PPO-MoE**: 단일 Expert 선택으로 효율적, 엔트로피 기반 탐색
3. **GRPO-MoE**: 상대적 보상으로 더 안정적 학습, 약간 더 나은 성능 가능

---

## 하이퍼파라미터 튜닝 계획

### 실험 1: Expert 개수
- 4, 8, 16개 Expert 비교
- 가설: 8개가 최적 (too few vs too many trade-off)

### 실험 2: 임베딩 차원
- 32, 64, 128 차원 비교
- 현재: 64 (user/movie 각각)

### 실험 3: 학습률
- Dense: 0.0005, 0.001, 0.002
- RL: 0.0001, 0.0003, 0.001

### 실험 4: 강화학습 하이퍼파라미터
- PPO: clip_epsilon (0.1, 0.2, 0.3), entropy_coef (0.01, 0.05, 0.1)
- GRPO: temperature (0.5, 1.0, 2.0)

---

## Cross-Validation 계획

### 5-fold CV
MovieLens 100k는 5개의 분할을 제공:
- u1.base / u1.test (80%/20%)
- u2.base / u2.test
- u3.base / u3.test
- u4.base / u4.test
- u5.base / u5.test

**실행 계획**:
```bash
for i in {1..5}; do
    uv run python src/training/train_dense_moe.py \
        --train_rating_path ml-100k/u${i}.base \
        --val_rating_path ml-100k/u${i}.test \
        --output_dir checkpoints/dense_moe/fold${i}
done
```

---

## 체크포인트 관리

### 디렉토리 구조
```
checkpoints/
├── dense_moe/
│   ├── dense_moe_best.pt          # Best 모델
│   ├── dense_moe_epoch_010.pt     # 중간 체크포인트
│   ├── dense_moe_epoch_020.pt
│   └── ...
├── ppo_moe/
│   └── ...
└── grpo_moe/
    └── ...
```

### 체크포인트 내용
```python
{
    'epoch': int,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': OrderedDict,
    'metrics': {
        'train_loss': float,
        'val_loss': float,
        'val_mse': float,
        'val_rmse': float,
        'val_mae': float,
    },
    'config': dict,  # 학습 설정
}
```

---

## 로깅 및 모니터링

### TensorBoard 지원 (추가 구현 필요)
```bash
# 학습 중
tensorboard --logdir runs/

# 로그 예시
runs/
├── dense_moe_20251102_143022/
│   ├── train_loss
│   ├── val_loss
│   └── val_rmse
├── ppo_moe_20251102_150145/
└── grpo_moe_20251102_153302/
```

### 콘솔 출력 형식
```
Epoch [10/100]
  Train Loss: 0.8456
  Val Loss: 0.8234
  Val RMSE: 0.9074
  Val MAE: 0.7123
  Learning Rate: 0.001000
  Best Val Loss: 0.8234 (saved)
  Time: 45.2s
```

---

## 디버깅 체크리스트

### 학습 시작 전
- [ ] 데이터 로딩 확인 (shape, dtype)
- [ ] 모델 입력/출력 shape 검증
- [ ] 작은 배치로 forward/backward 테스트
- [ ] GPU 메모리 사용량 확인

### 학습 중 모니터링
- [ ] Loss가 감소하는지 확인
- [ ] Gradient norm 폭발/소실 확인
- [ ] Expert 선택이 다양한지 확인 (RL)
- [ ] 학습 속도 (초/에포크)

### 문제 발생 시
- [ ] NaN/Inf 체크
- [ ] Gradient clipping 강화
- [ ] Learning rate 감소
- [ ] 배치 크기 조정

---

## 결과 시각화 계획

### 1. 학습 곡선
```python
import matplotlib.pyplot as plt

plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Curve - Dense MoE')
```

### 2. 모델 비교 차트
- Bar chart: MSE, RMSE, MAE 비교
- 신뢰구간 포함 (5-fold CV)

### 3. Expert 분석
- **Dense MoE**: Gating 가중치 히트맵
- **RL MoE**: Expert 선택 빈도 막대 그래프
- Expert별 성능 (평균 오차)

### 4. Confusion Matrix (Optional)
- 평점을 범주로 간주 (1-5점)
- 실제 vs 예측 분포

---

## Git 관리 (권장)

### .gitignore
```
# 데이터
ml-100k/

# 체크포인트
checkpoints/
*.pt
*.pth

# 결과
results/
runs/

# 환경
.venv/
__pycache__/
*.pyc

# 기타
.DS_Store
.vscode/
```

### Commit 메시지 예시
```
feat: Add Dense MoE training script
fix: Fix Expert selection bug in PPO-MoE
docs: Update README with uv instructions
exp: Run baseline Dense MoE experiment
```

---

## 다음 작업 우선순위

### 높은 우선순위 (즉시 실행)
1. ⏳ uv로 의존성 설치 및 환경 설정
2. ⏳ 간단한 데이터 로딩 테스트
3. ⏳ Dense MoE 학습 실행 (베이스라인)

### 중간 우선순위 (Dense MoE 완료 후)
4. ⏳ PPO-MoE 학습 실행
5. ⏳ GRPO-MoE 학습 실행
6. ⏳ 세 모델 통합 평가

### 낮은 우선순위 (기본 실험 완료 후)
7. ⏳ 하이퍼파라미터 튜닝
8. ⏳ 5-fold Cross-validation
9. ⏳ 결과 시각화 및 보고서 작성

---

## 예상 이슈 및 대응

### 이슈 1: GPU 메모리 부족
**대응**: 배치 크기 감소 (256 → 128 → 64)

### 이슈 2: 강화학습 모델 불안정
**대응**:
- Learning rate 감소 (0.0003 → 0.0001)
- Gradient clipping 강화 (0.5 → 0.3)
- Entropy coefficient 증가 (탐색 촉진)

### 이슈 3: Expert 편향 (특정 Expert만 선택)
**대응**:
- Entropy coefficient 증가
- Temperature 조정 (GRPO)
- Expert 초기화 변경

### 이슈 4: 학습 속도 느림
**대응**:
- 데이터 로딩 최적화 (num_workers)
- Mixed precision 학습 (torch.amp)
- 배치 크기 증가 (메모리 허용 시)

---

## 성공 기준

### 최소 목표
- [ ] 세 모델 모두 성공적으로 학습 완료
- [ ] RMSE < 1.0 (모든 모델)
- [ ] 통합 평가 리포트 생성

### 목표
- [ ] GRPO-MoE가 Dense MoE보다 나은 성능
- [ ] Expert 선택이 다양함 (편향 < 50%)
- [ ] 5-fold CV 결과 재현 가능

### 도전 목표
- [ ] RMSE < 0.90
- [ ] Ablation Study 완료
- [ ] 논문 수준의 분석 및 시각화

---

## 참고 자료

### 논문
- Shazeer et al. (2017): "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"
- Schulman et al. (2017): "Proximal Policy Optimization Algorithms"

### 코드베이스
- Hugging Face Transformers (MoE 구현)
- OpenAI Spinning Up (PPO 구현)

### MovieLens 데이터셋
- F. Maxwell Harper and Joseph A. Konstan (2015)
- https://grouplens.org/datasets/movielens/

---

## 요약

### 코드 구현: ✅ 100% 완료
- 모든 모듈 구현 완료
- 학습 스크립트 준비 완료
- 평가 시스템 준비 완료

### 다음 단계: ⏳ 실행 대기
1. uv로 환경 설정
2. Dense MoE 학습
3. PPO-MoE 학습
4. GRPO-MoE 학습
5. 통합 평가 및 분석

### 최종 목표
**강화학습 기반 MoE가 Dense MoE보다 나은 성능을 보이는지 실험적으로 검증**
