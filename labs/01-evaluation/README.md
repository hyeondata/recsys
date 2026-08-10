# 01. Offline evaluation

## 목표

- 사용자별 시간 분할이 미래 정보 누수를 막는 이유를 설명한다.
- Precision@K, Recall@K, MRR@K, MAP@K, NDCG@K를 작은 예제로 계산한다.
- 학습 catalog 전체에서 popularity 추천을 만들고 이미 본 아이템을 제외한다.
- sampled candidate 평가가 full-catalog 평가와 다른 질문에 답한다는 점을 이해한다.

## 실행

외부 데이터 없이 손계산용 toy example과 자체 검증을 실행합니다.

```bash
uv run python labs/01-evaluation/lab.py
```

MovieLens 1M `ratings.dat`으로 실행합니다.

```bash
uv run python labs/01-evaluation/lab.py \
  --ratings data/ml-1m/ratings.dat \
  --output labs/01-evaluation/result.json
```

MovieLens의 모든 rating event를 implicit interaction으로 취급합니다. 사용자의 마지막 interaction은 test, 직전 interaction은 validation, 나머지는 train입니다. interaction이 5개 미만인 사용자는 제외합니다.

## 완료 조건

- 자체 검증이 통과한다.
- MovieLens 결과 JSON이 생성된다.
- `quiz.md`에서 5문제 중 4문제 이상을 설명한다.
- `notes/01-evaluation.md`를 자기 말로 작성한다.

