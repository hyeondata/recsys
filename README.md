# Recommender Systems Study

추천시스템을 **평가 지표부터 BERT4Rec과 최신 생성형 추천까지** 실무 관점으로 학습하는 저장소입니다.

현재 진도와 바로 다음 행동은 [STATE.md](STATE.md)에서 확인합니다. 대화 전문은 저장하지 않습니다. 사용자는 `notes/`에 배운 내용을 자기 말로 쓰고, Codex는 상태·오개념·실험 결과를 관리합니다.

## 학습 원칙

1. 전체 흐름에서 현재 주제의 위치를 먼저 확인합니다.

   `비즈니스 목표 → 로그 → 데이터 분할 → 후보 생성 → 랭킹 → 서빙 → 온라인 평가 → 피드백`

2. 필요한 수학은 모델을 배울 때 작은 숫자 예제로 익힙니다.
3. 모델은 `최소 직접 구현 → 동일 조건 비교 → RecBole 교차 확인` 순서로 다룹니다.
4. 단원 완료 조건은 실습 실행, 퀴즈 80% 이상, 사용자 설명 검토입니다.
5. 회사 데이터·스키마·수치는 이 공개 저장소에 저장하지 않습니다.

## 시작하기

Python 3.11과 [uv](https://docs.astral.sh/uv/)를 사용합니다.

```bash
uv sync
uv run python labs/01-evaluation/lab.py
```

MovieLens 1M을 사용할 때는 [GroupLens](https://grouplens.org/datasets/movielens/1m/)에서 내려받은 `ratings.dat`을 Git에서 제외되는 `data/ml-1m/` 아래에 둡니다.

```bash
uv run python labs/01-evaluation/lab.py \
  --ratings data/ml-1m/ratings.dat \
  --output labs/01-evaluation/result.json
```

## 8주 핵심 과정

| 주차 | 주제 | 결과물 |
|---|---|---|
| 1 | 전체 추천 흐름, temporal split, Precision/Recall/MRR/MAP/NDCG, coverage, offline/online 평가 | MovieLens popularity baseline |
| 2 | item/user kNN, co-visitation, MF, BPR | 고전 기준선 비교 |
| 3 | Word2Vec, Skip-gram, negative sampling, Item2Vec | 세션 기반 유사 아이템 후보 생성 |
| 4 | NCF, two-tower, feature embedding, ranking | 후보 생성기와 MLP ranker |
| 5 | GRU4Rec 개념, positional embedding, causal attention, SASRec | next-item SASRec |
| 6 | masked item modeling, bidirectional Transformer, BERT4Rec | SASRec/BERT4Rec 및 RecBole 비교 |
| 7 | point-in-time correctness, cold start, latency, feedback loop, A/B test | 배치 추천 CLI와 회사 적용 템플릿 |
| 8 | Amazon Reviews 2023 `All_Beauty` | 공개 상품 데이터 캡스톤 |

BERT4Rec 이후에는 LLM 기반 추천, semantic ID 기반 생성 추천, diffusion 추천을 월 1개 주제로 추가합니다. 구체적인 논문 순서는 [papers/reading-list.md](papers/reading-list.md)를 따릅니다.

## 저장소 규칙

- `STATE.md`: Codex 관리. 현재 상태, 학습 증거, 오개념, 복습 큐, 다음 행동.
- `notes/`: 사용자 관리. 자신의 설명, 수식 해석, 틀린 점, 회사 적용 아이디어.
- `labs/`: 실행 가능한 현재·완료 단원만 추가.
- `papers/`: 논문 링크와 읽기 질문. PDF 파일은 저장하지 않음.
- `data/`, 모델 체크포인트, 회사 자료, 비밀값은 커밋하지 않음.
- 실험은 같은 split, test 직전까지 관측된 이력, full-catalog 후보, seed 42를 기본으로 비교.

세션이 끝나면 Codex가 diff를 보여줍니다. 사용자 승인 후 `learn(<unit>): <outcome>` 형식으로 `main`에 commit하고 `origin/main`에 push합니다.
