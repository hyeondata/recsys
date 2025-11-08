# 확장성 실험 빠른 시작 (Quick Start)

## 1분 요약

**목적**: Expert 개수를 늘렸을 때 Dense MoE vs RL MoE의 확장성 차이 확인

**핵심 차이**:
- Dense MoE: 모든 expert 사용 → Expert 늘어나면 느려짐 ⚠️
- RL MoE: 하나만 선택 → Expert 늘어나도 속도 일정 ✅

---

## 즉시 실행

### 방법 1: 빠른 테스트 (5분)
```bash
chmod +x experiments/quick_test.sh
./experiments/quick_test.sh
```

### 방법 2: 전체 실험 (1~2시간)
```bash
chmod +x experiments/run_scalability_experiment.sh
./experiments/run_scalability_experiment.sh
```

---

## 결과 확인

### 1. 수치 결과
```bash
cat experiments/visualizations/{실험ID}/scalability_comparison_table.md
```

### 2. 시각화 그래프
```bash
ls experiments/visualizations/{실험ID}/*.png
```

주요 그래프:
- `scalability_overview.png`: 전체 비교
- `train_time_scalability.png`: 학습 시간 비교
- `efficiency_comparison.png`: 효율성 비교

---

## 예상 결과

```
Expert 4개:
  Dense MoE:  30초 (빠름) ✓
  PPO MoE:    85초

Expert 8개:
  Dense MoE:  60초
  PPO MoE:    85초 (차이 줄어듦)

Expert 16개:
  Dense MoE:  120초
  PPO MoE:    85초 (역전!) ✓

Expert 32개:
  Dense MoE:  240초 (매우 느림)
  PPO MoE:    85초 (여전히 일정) ✓✓
```

**결론**: Expert가 많아질수록 RL MoE가 유리!

---

## 커스터마이징

### 더 많은 Expert 테스트
```bash
uv run python3 experiments/scalability_experiment.py \
    --num_experts_list 4 8 16 32 64 \
    --num_epochs 5
```

### 특정 모델만
```bash
uv run python3 experiments/scalability_experiment.py \
    --models dense ppo \
    --num_experts_list 4 8 16
```

---

## 문제 해결

### GPU 메모리 부족
```bash
# 배치 크기 줄이기
--batch_size 128
```

### 너무 오래 걸림
```bash
# Epoch 줄이기
--num_epochs 3

# 또는 빠른 테스트 사용
./experiments/quick_test.sh
```

---

## 더 자세한 정보

- 전체 가이드: `experiments/README_SCALABILITY.md`
- 프로젝트 요약: `memo_compression/01_summary.md`
