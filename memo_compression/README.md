# Memo Compression 폴더

압축된 프로젝트 문서 (토큰 절약용)

## 📋 파일 구성

### ⭐ 필수 참조
- **00_using.md** - 🔥 **전체 사용법 통합 문서** (최신 업데이트)
  - 모든 모델 학습 방법
  - 시각화 도구 사용법
  - 버전 히스토리
  - Quick Start 가이드

### 프로젝트 개요
1. **01_summary.md** - 프로젝트 전체 요약
2. **02_technical_details.md** - 기술적 세부사항
3. **03_known_issues.md** - 이슈 및 해결책
4. **04_usage.md** - 간단 참조용 (→ 00_using.md 참조)

## 원본 문서
상세한 내용은 `memo/` 폴더의 19개 파일 참조 (최신: 19_vectorization_optimization.md)

## 핵심 요약
- **목표**: MoE 기반 영화 추천 시스템
- **모델**: Dense, PPO, GRPO MoE
- **결과**: Dense가 최고 성능 (RMSE 0.9803)
- **주요 기능**:
  - Batch loading 방식 (메모리 300배 절약)
  - 종합 지표 비교 시각화
  - **Resume 기능** (중단된 학습 이어하기) v2.1
  - **벡터화 최적화** (학습 속도 10-50배 향상) ⭐ v2.3

## Quick Start
```bash
# 상세 사용법 확인
cat memo_compression/00_using.md

# 빠른 실험
uv run python3 src/training/train_dense_moe.py --epochs 10 --batch_size 1024

# 중단된 학습 이어하기 (v2.1)
uv run python3 src/training/train_ppo_moe.py --epochs 100 --batch_size 256 --resume
```
