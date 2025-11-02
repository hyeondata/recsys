# 알려진 이슈 및 해결

## 1. CUDA 에러 (해결됨)
**문제**: PPO/GRPO 학습 중 `CUBLAS_STATUS_EXECUTION_FAILED`
**원인**: DataFrame 인덱싱 문제
**해결**: 입력 데이터를 episode 수집 시 함께 저장

## 2. Batch Loading 불공정 (해결됨)
**문제**: GRPO/PPO가 Dense보다 4배 많은 업데이트
**해결**: 배치 단위 학습으로 변경 (14_batch_loading_fix.md)

## 3. GPU 드라이버 크래시 (해결됨)
**문제**: batch_size 1024로 GRPO/PPO 학습 시 크래시
**해결**: batch_size 256으로 제한

## 4. 권장 설정
- Dense MoE: batch_size 1024 (빠름)
- PPO/GRPO: batch_size 256 (안정적)
- num_workers: 4
- GPU: RTX 3090 (24GB)
