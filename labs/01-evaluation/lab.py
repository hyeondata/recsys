"""Minimal full-catalog evaluation lab using only the Python standard library."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

Interaction = tuple[str, str, int]


def _check_k(k: int) -> None:
    if k <= 0:
        raise ValueError("k must be positive")


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    _check_k(k)
    hits = len(set(ranked[:k]) & relevant)
    return hits / k


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    _check_k(k)
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def reciprocal_rank_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    _check_k(k)
    return next((1.0 / rank for rank, item in enumerate(ranked[:k], 1) if item in relevant), 0.0)


def average_precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    _check_k(k)
    if not relevant:
        return 0.0

    hits = 0
    score = 0.0
    counted: set[str] = set()
    for rank, item in enumerate(ranked[:k], 1):
        if item in relevant and item not in counted:
            hits += 1
            score += hits / rank
            counted.add(item)
    return score / min(len(relevant), k)


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    _check_k(k)
    if not relevant:
        return 0.0

    counted: set[str] = set()
    dcg = 0.0
    for rank, item in enumerate(ranked[:k], 1):
        if item in relevant and item not in counted:
            dcg += 1.0 / math.log2(rank + 1)
            counted.add(item)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg


def temporal_leave_two_out(
    interactions: list[Interaction], min_interactions: int = 5
) -> tuple[list[Interaction], dict[str, str], dict[str, str]]:
    """Return train rows plus one validation and test item per eligible user."""
    if min_interactions < 3:
        raise ValueError("min_interactions must be at least 3")

    by_user: dict[str, list[Interaction]] = defaultdict(list)
    for row in interactions:
        by_user[row[0]].append(row)

    train: list[Interaction] = []
    validation: dict[str, str] = {}
    test: dict[str, str] = {}
    for user, rows in by_user.items():
        if len(rows) < min_interactions:
            continue
        ordered = sorted(rows, key=lambda row: row[2])
        train.extend(ordered[:-2])
        validation[user] = ordered[-2][1]
        test[user] = ordered[-1][1]
    return train, validation, test


def popularity_ranking(train: list[Interaction]) -> list[str]:
    counts = Counter(item for _, item, _ in train)
    return [item for item, _ in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))]


def recommend_unseen(popular_items: list[str], seen: set[str], k: int) -> list[str]:
    _check_k(k)
    return [item for item in popular_items if item not in seen][:k]


def evaluate_popularity(
    train: list[Interaction], test: dict[str, str], k: int
) -> dict[str, float]:
    popular_items = popularity_ranking(train)
    catalog = set(popular_items)
    seen_by_user: dict[str, set[str]] = defaultdict(set)
    for user, item, _ in train:
        seen_by_user[user].add(item)

    totals = Counter()
    recommended_items: set[str] = set()
    for user, target in test.items():
        ranked = recommend_unseen(popular_items, seen_by_user[user], k)
        relevant = {target}
        recommended_items.update(ranked)
        totals["precision"] += precision_at_k(ranked, relevant, k)
        totals["recall"] += recall_at_k(ranked, relevant, k)
        totals["mrr"] += reciprocal_rank_at_k(ranked, relevant, k)
        totals["map"] += average_precision_at_k(ranked, relevant, k)
        totals["ndcg"] += ndcg_at_k(ranked, relevant, k)

    users = len(test)
    if not users:
        raise ValueError("no eligible users to evaluate")
    return {
        f"precision@{k}": totals["precision"] / users,
        f"recall@{k}": totals["recall"] / users,
        f"mrr@{k}": totals["mrr"] / users,
        f"map@{k}": totals["map"] / users,
        f"ndcg@{k}": totals["ndcg"] / users,
        f"catalog_coverage@{k}": len(recommended_items) / len(catalog) if catalog else 0.0,
    }


def load_movielens_1m(path: Path) -> list[Interaction]:
    interactions: list[Interaction] = []
    with path.open(encoding="latin-1") as ratings:
        for line_number, line in enumerate(ratings, 1):
            fields = line.rstrip("\n").split("::")
            if len(fields) != 4:
                raise ValueError(f"invalid ratings.dat row {line_number}")
            user, item, _, timestamp = fields
            interactions.append((user, item, int(timestamp)))
    return interactions


def toy_interactions() -> list[Interaction]:
    return [
        ("u1", "a", 1), ("u1", "b", 2), ("u1", "c", 3), ("u1", "d", 4), ("u1", "e", 5),
        ("u2", "a", 1), ("u2", "c", 2), ("u2", "d", 3), ("u2", "e", 4), ("u2", "b", 5),
        ("u3", "b", 1), ("u3", "c", 2), ("u3", "a", 3), ("u3", "e", 4), ("u3", "d", 5),
    ]


def self_check() -> None:
    ranked = ["a", "b", "c"]
    relevant = {"b", "d"}
    assert math.isclose(precision_at_k(ranked, relevant, 3), 1 / 3)
    assert math.isclose(recall_at_k(ranked, relevant, 3), 1 / 2)
    assert math.isclose(reciprocal_rank_at_k(ranked, relevant, 3), 1 / 2)
    assert math.isclose(average_precision_at_k(ranked, relevant, 3), 1 / 4)
    expected_ndcg = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    assert math.isclose(ndcg_at_k(ranked, relevant, 3), expected_ndcg)
    assert ndcg_at_k([], set(), 10) == 0.0

    train, validation, test = temporal_leave_two_out(toy_interactions())
    assert len(train) == 9
    assert validation == {"u1": "d", "u2": "e", "u3": "e"}
    assert test == {"u1": "e", "u2": "b", "u3": "d"}
    assert recommend_unseen(["a", "b", "c"], {"a"}, 2) == ["b", "c"]


def run(interactions: list[Interaction], dataset: str, k: int, min_interactions: int) -> dict[str, object]:
    started = time.perf_counter()
    train, validation, test = temporal_leave_two_out(interactions, min_interactions)
    observed_before_test = train + [(user, item, 0) for user, item in validation.items()]
    metrics = evaluate_popularity(observed_before_test, test, k)
    return {
        "dataset": dataset,
        "split": {
            "strategy": "per-user temporal leave-two-out",
            "min_interactions": min_interactions,
            "test_history": "train + validation interactions",
            "candidates": "full observed catalog excluding user history",
        },
        "seed": 42,
        "model": "popularity",
        "params": {"k": k},
        "metrics": metrics,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings", type=Path, help="Path to MovieLens 1M ratings.dat")
    parser.add_argument("--output", type=Path, help="Optional result JSON path")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--min-interactions", type=int, default=5)
    args = parser.parse_args()

    self_check()
    interactions = load_movielens_1m(args.ratings) if args.ratings else toy_interactions()
    result = run(interactions, "MovieLens 1M" if args.ratings else "toy", args.k, args.min_interactions)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print("self-check: passed")
    print(rendered)


if __name__ == "__main__":
    main()
