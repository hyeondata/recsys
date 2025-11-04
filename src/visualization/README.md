# Visualization Tools for MoE Research

논문 제출용 고품질 시각화 도구

## 📁 파일 구성

1. **publication_plots.py** - 논문용 성능 비교 시각화
2. **paper_metrics_comparison.py** - 종합 지표 비교 (확장성, 효율성, 성능)
3. **training_curves.py** - 학습 곡선 시각화

## 🎨 Publication-Quality Plots

### 특징
- **고해상도**: PDF (600 DPI) + PNG (300 DPI)
- **LaTeX 스타일**: Times New Roman 폰트, 수식 지원
- **Colorblind-friendly**: 색각 이상자도 구분 가능한 색상
- **IEEE/ACM 스타일**: 논문 제출 기준 충족

### 생성되는 시각화

#### 1. 개별 메트릭 비교 (3개 파일)
- `mse_comparison.pdf/.png` - MSE 비교
- `rmse_comparison.pdf/.png` - RMSE 비교
- `mae_comparison.pdf/.png` - MAE 비교

#### 2. 통합 성능 비교
- `performance_combined.pdf/.png` - 3개 메트릭 통합

#### 3. Expert 분석
- `expert_distribution.pdf/.png` - Expert 사용 분포

#### 4. 비교 테이블
- `comparison_table.pdf/.png` - 시각화된 테이블
- `comparison_table.tex` - LaTeX 코드

## 🚀 사용법

### 1. 논문용 시각화 생성

```bash
# 기본 실행
uv run python3 src/visualization/publication_plots.py

# 커스텀 설정
uv run python3 src/visualization/publication_plots.py \
    --results_file results/all_models_evaluation.json \
    --output_dir results/publication
```

**출력**: `results/publication/` 폴더에 PDF와 PNG 파일 생성

### 2. 종합 지표 비교 (논문용) ⭐

```bash
# 확장성, 효율성, 성능 종합 비교
uv run python3 src/visualization/paper_metrics_comparison.py \
    --results_file results/all_models_evaluation.json \
    --output_dir results/publication
```

**출력**:
- `comprehensive_comparison.pdf/.png` - 4-panel 종합 비교
- `performance_vs_efficiency.pdf/.png` - 성능-효율성 산점도
- `comprehensive_table.tex` - LaTeX 테이블

### 3. 학습 곡선 시각화

```bash
# 더미 데이터로 실행 (데모)
uv run python3 src/visualization/training_curves.py

# 실제 학습 로그 사용
uv run python3 src/visualization/training_curves.py \
    --training_logs_file results/training_history.json \
    --output_dir results/publication
```

**출력**:
- `training_curves.pdf/.png` - 4개 서브플롯 (Loss, RMSE, MAE, LR)
- `convergence_comparison.pdf/.png` - 수렴 속도 비교

## 📊 입력 데이터 형식

### 평가 결과 (JSON)

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

### 학습 로그 (JSON, 선택적)

`results/training_history.json`:
```json
{
  "dense_moe": {
    "epochs": [1, 2, 3, ...],
    "train_loss": [0.0660, 0.0545, ...],
    "val_rmse": [1.0414, 0.9931, ...],
    "val_mae": [0.8382, 0.7951, ...],
    "learning_rate": [0.001, 0.001, ...]
  },
  "ppo_moe": { ... },
  "grpo_moe": { ... }
}
```

## 🎯 LaTeX 통합

### 1. 그림 삽입

```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.9\columnwidth]{figures/performance_combined.pdf}
\caption{Performance comparison of MoE models on MovieLens 100k dataset.
(a) MSE, (b) RMSE, (c) MAE. Lower values indicate better performance.}
\label{fig:performance}
\end{figure}
```

### 2. 테이블 삽입

생성된 `comparison_table.tex` 파일을 직접 include:

```latex
\input{tables/comparison_table.tex}
```

또는 PDF 이미지로 사용:

```latex
\begin{table}[t]
\centering
\includegraphics[width=0.9\columnwidth]{tables/comparison_table.pdf}
\caption{Performance comparison of MoE models.}
\label{tab:performance}
\end{table}
```

### 3. 서브플롯 참조

```latex
As shown in Figure~\ref{fig:performance}(a), Dense-MoE achieves
the lowest MSE of 0.9611.
```

## 🔧 커스터마이징

### 색상 변경

`publication_plots.py`의 `setup_publication_style()`:

```python
return {
    'dense': '#0173B2',  # Blue
    'ppo': '#DE8F05',    # Orange
    'grpo': '#029E73',   # Green (변경 가능)
}
```

### 폰트 변경

```python
'font.serif': ['Times New Roman', 'DejaVu Serif'],  # 원하는 폰트 추가
```

### DPI 조정

```python
'savefig.dpi': 600,  # 더 높은 해상도: 1200
```

## 📐 권장 크기

### Single-column (IEEE/ACM)
- Width: 3.5 inches (8.9 cm)
- DPI: 300-600

### Double-column
- Width: 7 inches (17.8 cm)
- DPI: 300-600

### Poster
- Width: 10-12 inches
- DPI: 300

현재 설정은 대부분의 학술지에 적합한 크기입니다.

## ✅ 체크리스트 (논문 제출 전)

- [ ] 모든 그림이 PDF 형식으로 저장되었는가?
- [ ] 폰트 크기가 8-10pt인가? (읽기 쉬운가?)
- [ ] 축 레이블과 범례가 명확한가?
- [ ] 색상이 흑백 인쇄에서도 구분 가능한가?
- [ ] 서브플롯 레이블 (a), (b), (c)가 올바른가?
- [ ] 캡션에 충분한 설명이 포함되어 있는가?
- [ ] 파일 크기가 적절한가? (<1MB per figure)

## 🐛 문제 해결

### "Font not found" 에러
```bash
# matplotlib 폰트 캐시 삭제
rm -rf ~/.cache/matplotlib
```

### PDF 파일이 크기가 너무 큼
```python
# publication_plots.py에서
plt.savefig(..., dpi=300)  # 600 대신 300 사용
```

### 그래프가 잘림
```python
plt.tight_layout()
plt.savefig(..., bbox_inches='tight', pad_inches=0.1)
```

## 📚 참고 자료

- [Matplotlib Publication Quality](https://matplotlib.org/stable/tutorials/text/usetex.html)
- [IEEE Graphics Requirements](https://www.ieee.org/publications/authors/author-graphics.html)
- [Nature Figure Guidelines](https://www.nature.com/nature/for-authors/final-submission)

## 🤝 기여

시각화 개선 제안이나 버그 리포트는 환영합니다!
