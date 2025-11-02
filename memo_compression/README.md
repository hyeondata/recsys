# Memo Compression 폴더

압축된 프로젝트 문서 (토큰 절약용)

## 파일 구성
1. **01_summary.md** - 프로젝트 전체 요약
2. **02_technical_details.md** - 기술적 세부사항
3. **03_known_issues.md** - 이슈 및 해결책
4. **04_usage.md** - 사용법

## 원본 문서
상세한 내용은 `memo/` 폴더의 14개 파일 참조

## 핵심 요약
- **목표**: MoE 기반 영화 추천 시스템
- **모델**: Dense, PPO, GRPO MoE
- **결과**: Dense가 최고 성능 (RMSE 0.9803)
- **주요 수정**: Batch loading 방식 개선 (300배 메모리 절약)
