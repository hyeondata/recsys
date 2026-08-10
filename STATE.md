# Learning State

> 이 파일은 Codex가 관리합니다. 사용자의 학습 서술은 `notes/`에 기록합니다.

## Current unit

- Unit: `01-evaluation`
- Topic: 전체 추천 흐름과 offline evaluation
- Status: `planned`
- Last session: `-`
- Dataset: MovieLens 1M 예정
- Protocol: 사용자별 temporal leave-two-out, 최소 interaction 5, test 시점의 train+validation 이력, full observed catalog, seed 42
- Primary metrics: Recall@10, NDCG@10

## Mastery evidence

- Lab: 환경 자체 검증 통과, MovieLens 1M popularity baseline 실행 확인
- Quiz: 실행 전
- Teach-back: 작성 전
- Result: [`labs/01-evaluation/result.json`](labs/01-evaluation/result.json) — 초기 환경 검증값이며 학습 완료 증거는 아님

## Misconceptions to revisit

- 아직 기록 없음

## Review queue

| Due | Unit | Focus | Status |
|---|---|---|---|
| - | - | - | - |

## Next action

1. 추천 파이프라인에서 offline evaluation의 위치를 설명한다.
2. `labs/01-evaluation/lab.py`의 toy example을 실행한다.
3. Precision@K, Recall@K, MRR@K, MAP@K, NDCG@K를 손으로 계산한다.
4. `labs/01-evaluation/quiz.md`에 답하고 `notes/01-evaluation.md`에 자신의 설명을 작성한다.
