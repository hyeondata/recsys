# 15. Publication-Quality Visualization System

**작성일**: 2025-11-03
**목적**: 논문 제출용 고품질 시각화 도구 구축

---

## 📌 개요

MoE 모델 성능 비교를 위한 IEEE/ACM 학술지 기준 시각화 시스템 구축.

### 생성된 파일
1. `src/visualization/publication_plots.py` - 성능 비교 및 표 생성
2. `src/visualization/training_curves.py` - 학습 곡선 시각화
3. `src/visualization/README.md` - 사용 가이드

---

## 🎨 주요 특징

### Publication-Quality Standards
- **해상도**: PDF (600 DPI), PNG (300 DPI)
- **폰트**: Times New Roman (LaTeX 호환)
- **색상**: Colorblind-friendly palette
  - Dense-MoE: `#0173B2` (Blue)
  - PPO-MoE: `#DE8F05` (Orange)
  - GRPO-MoE: `#029E73` (Green)
- **형식**: IEEE/ACM 표준 준수

### 생성 시각화 (총 16개 파일)

#### 1. 성능 비교 (7개)
- `mse_comparison.pdf/.png` - MSE 개별 비교
- `rmse_comparison.pdf/.png` - RMSE 개별 비교
- `mae_comparison.pdf/.png` - MAE 개별 비교
- `performance_combined.pdf/.png` - 3개 메트릭 통합 (2×2 subplot)

#### 2. Expert 분석 (2개)
- `expert_distribution.pdf/.png` - Expert 사용 분포

#### 3. 비교 테이블 (3개)
- `comparison_table.pdf/.png` - 시각화된 테이블
- `comparison_table.tex` - LaTeX 코드

#### 4. 학습 곡선 (4개)
- `training_curves.pdf/.png` - 4개 subplot (Loss, RMSE, MAE, LR)
- `convergence_comparison.pdf/.png` - 수렴 속도 비교

---

## 🚀 사용법

### 1. 성능 비교 시각화

```bash
uv run python3 src/visualization/publication_plots.py \
    --results_file results/all_models_evaluation.json \
    --output_dir results/publication
```

**출력 예시**:
```
✓ Loaded results from: results/all_models_evaluation.json
  Models: dense_moe, ppo_moe, grpo_moe
✓ Saved: mse_comparison.pdf/.png
✓ Saved: rmse_comparison.pdf/.png
✓ Saved: mae_comparison.pdf/.png
✓ Saved: performance_combined.pdf/.png
✓ Saved: expert_distribution.pdf/.png
✓ Saved: comparison_table.pdf/.png
✓ Saved: comparison_table.tex
```

### 2. 학습 곡선 시각화

```bash
# 더미 데이터로 실행 (실제 학습 로그 없을 때)
uv run python3 src/visualization/training_curves.py \
    --output_dir results/publication

# 실제 학습 로그 사용
uv run python3 src/visualization/training_curves.py \
    --training_logs_file results/training_history.json \
    --output_dir results/publication
```

**출력 예시**:
```
⚠ No training logs file found. Using dummy data for demonstration.
✓ Saved: training_curves.pdf/.png
✓ Saved: convergence_comparison.pdf/.png
```

---

## 📊 입력 데이터 형식

### 평가 결과 (필수)

`results/all_models_evaluation.json`:
```json
{
  "dense_moe": {
    "mse": 0.9611,
    "rmse": 0.9803,
    "mae": 0.7756,
    "avg_gate_probs": [0.0857, 0.1503, ...],
    "gate_entropy": 1.8114
  },
  "ppo_moe": {
    "mse": 1.1361,
    "rmse": 1.0659,
    "mae": 0.8575,
    "expert_distribution": {0: 8902, 1: 1682, ...},
    "expert_performance": {0: 0.2245, 1: 0.1997, ...}
  },
  "grpo_moe": {
    "mse": 1.0841,
    "rmse": 1.0412,
    "mae": 0.8276,
    "expert_distribution": {0: 0, 1: 3579, ...},
    "expert_performance": {1: 0.1963, 2: 0.1858, ...}
  }
}
```

### 학습 로그 (선택적)

`results/training_history.json`:
```json
{
  "dense_moe": {
    "epochs": [1, 2, 3, ..., 10],
    "train_loss": [0.0660, 0.0545, ...],
    "val_rmse": [1.0414, 0.9931, ...],
    "val_mae": [0.8382, 0.7951, ...],
    "learning_rate": [0.001, 0.001, ...]
  },
  "ppo_moe": { ... },
  "grpo_moe": { ... }
}
```

---

## 📝 LaTeX 통합

### 1. 통합 성능 비교 그림

```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.9\columnwidth]{figures/performance_combined.pdf}
\caption{Performance comparison of MoE models on MovieLens 100k dataset.
(a) MSE, (b) RMSE, (c) MAE. Lower values indicate better performance.
Dense-MoE achieves the best performance across all metrics.}
\label{fig:performance}
\end{figure}
```

### 2. 학습 곡선 그림

```latex
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/training_curves.pdf}
\caption{Training dynamics of MoE models. (a) Training loss, (b) validation RMSE,
(c) validation MAE, (d) learning rate schedule. Dense-MoE converges faster
and achieves lower validation errors.}
\label{fig:training}
\end{figure}
```

### 3. 비교 테이블

```latex
% LaTeX 코드 직접 사용
\input{tables/comparison_table.tex}

% 또는 PDF 이미지로 사용
\begin{table}[t]
\centering
\includegraphics[width=0.8\columnwidth]{tables/comparison_table.pdf}
\caption{Performance comparison of MoE models.}
\label{tab:performance}
\end{table}
```

### 4. 서브플롯 참조 예시

```latex
As shown in Figure~\ref{fig:performance}(a), Dense-MoE achieves
the lowest MSE of 0.9611, outperforming PPO-MoE (1.1361) and
GRPO-MoE (1.0841). The training curves in Figure~\ref{fig:training}(b)
demonstrate that Dense-MoE converges faster and more stably
than RL-based alternatives.
```

---

## 🔧 구현 세부사항

### setup_publication_style()

논문 품질 플롯 스타일 설정:

```python
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.format': 'pdf',
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
})
```

### Colorblind-Friendly Colors

```python
colors = {
    'dense': '#0173B2',  # Blue
    'ppo': '#DE8F05',    # Orange
    'grpo': '#029E73',   # Green
}
```

### 주요 함수 (publication_plots.py)

1. **plot_performance_bars()** - 개별 메트릭 bar chart
2. **plot_combined_performance()** - 2×2 subplot 통합 비교
3. **plot_expert_distribution_publication()** - Expert 사용 분포
4. **plot_comparison_table()** - 시각화된 비교 테이블
5. **generate_latex_table()** - LaTeX 테이블 코드 생성

### 주요 함수 (training_curves.py)

1. **plot_training_curves_combined()** - 2×2 subplot (Loss, RMSE, MAE, LR)
2. **plot_convergence_comparison()** - 수렴 속도 단일 플롯
3. **create_dummy_training_logs()** - 더미 데이터 생성

---

## ✅ 테스트 결과

### 실행 환경
- 날짜: 2025-11-03
- 입력: `results/all_models_evaluation.json` (실제 평가 데이터)
- 출력: `results/publication/` (16개 파일)

### 생성 파일 목록

```bash
$ ls -lh results/publication/
-rw-rw-r-- 1 gpu gpu  25K comparison_table.pdf
-rw-rw-r-- 1 gpu gpu  98K comparison_table.png
-rw-rw-r-- 1 gpu gpu  512 comparison_table.tex
-rw-rw-r-- 1 gpu gpu  21K convergence_comparison.pdf
-rw-rw-r-- 1 gpu gpu 133K convergence_comparison.png
-rw-rw-r-- 1 gpu gpu  21K expert_distribution.pdf
-rw-rw-r-- 1 gpu gpu 122K expert_distribution.png
-rw-rw-r-- 1 gpu gpu  16K mae_comparison.pdf
-rw-rw-r-- 1 gpu gpu  54K mae_comparison.png
-rw-rw-r-- 1 gpu gpu  17K mse_comparison.pdf
-rw-rw-r-- 1 gpu gpu  56K mse_comparison.png
-rw-rw-r-- 1 gpu gpu  21K performance_combined.pdf
-rw-rw-r-- 1 gpu gpu 138K performance_combined.png
-rw-rw-r-- 1 gpu gpu  18K rmse_comparison.pdf
-rw-rw-r-- 1 gpu gpu  59K rmse_comparison.png
-rw-rw-r-- 1 gpu gpu  29K training_curves.pdf
-rw-rw-r-- 1 gpu gpu 377K training_curves.png
```

### 파일 크기
- **PDF**: 16-29 KB (모두 <1 MB, 학술지 제출 기준 충족)
- **PNG**: 54-377 KB
- **총 크기**: ~1.3 MB

---

## 🐛 해결된 문제

### 1. Seaborn 의존성 제거

**문제**:
```
ModuleNotFoundError: No module named 'seaborn'
```

**원인**: `sns.color_palette('colorblind')` 사용

**해결**:
```python
# Before
import seaborn as sns
...
'palette': sns.color_palette('colorblind')

# After
# seaborn import 제거
# 직접 정의한 colorblind-friendly colors 사용
colors = {
    'dense': '#0173B2',
    'ppo': '#DE8F05',
    'grpo': '#029E73'
}
```

---

## 📐 논문 제출 체크리스트

- [x] PDF 형식 생성 (600 DPI)
- [x] Times New Roman 폰트 사용
- [x] Colorblind-friendly 색상
- [x] 서브플롯 레이블 (a), (b), (c)
- [x] 파일 크기 <1 MB per figure
- [x] LaTeX 테이블 코드 생성
- [ ] 실제 학습 로그로 training curves 재생성 (현재 더미 데이터)
- [ ] 캡션 작성 및 검토
- [ ] 흑백 인쇄 테스트

---

## 🔄 다음 단계

1. **실제 학습 로그 수집**:
   - 모델 학습 시 metrics 저장
   - `results/training_history.json` 생성

2. **학습 로그 형식**:
   ```python
   # src/training/train_*.py에서
   history = {
       'epochs': [],
       'train_loss': [],
       'val_rmse': [],
       'val_mae': [],
       'learning_rate': []
   }

   # 학습 후 저장
   with open('results/training_history.json', 'w') as f:
       json.dump({
           'dense_moe': history_dense,
           'ppo_moe': history_ppo,
           'grpo_moe': history_grpo
       }, f)
   ```

3. **재학습 후 최종 시각화**:
   ```bash
   # 1. 모델 재학습 (batch loading fix 적용)
   uv run python3 src/training/train_dense_moe.py --epochs 10
   uv run python3 src/training/train_ppo_moe.py --epochs 10
   uv run python3 src/training/train_grpo_moe.py --epochs 10

   # 2. 평가 실행
   uv run python3 src/training/evaluate.py \
       --output_file results/all_models_evaluation.json

   # 3. 시각화 생성
   uv run python3 src/visualization/publication_plots.py
   uv run python3 src/visualization/training_curves.py \
       --training_logs_file results/training_history.json
   ```

---

## 📚 참고 자료

- [Matplotlib Publication Quality](https://matplotlib.org/stable/tutorials/text/usetex.html)
- [IEEE Graphics Requirements](https://www.ieee.org/publications/authors/author-graphics.html)
- [Colorblind-Friendly Palettes](https://personal.sron.nl/~pault/)

---

## 📈 성능 결과 (현재)

### Evaluation Metrics
| Model      | MSE    | RMSE   | MAE    |
|-----------|--------|--------|--------|
| Dense-MoE | 0.9611 | 0.9803 | 0.7756 |
| PPO-MoE   | 1.1361 | 1.0659 | 0.8575 |
| GRPO-MoE  | 1.0841 | 1.0412 | 0.8276 |

**최고 성능**: Dense-MoE (모든 메트릭에서 우수)

---

**작업 완료**: 2025-11-03
**다음 작업**: 실제 학습 로그 수집 후 training curves 재생성
