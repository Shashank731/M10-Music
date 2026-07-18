"""Ranking metrics for recommender evaluation and MLflow logging."""

from __future__ import annotations

import math


def precision_at_k(actual: set[str], predicted: list[str], k: int) -> float:
    """Compute Precision@K."""
    if k <= 0:
        return 0.0
    return len(actual.intersection(predicted[:k])) / k


def recall_at_k(actual: set[str], predicted: list[str], k: int) -> float:
    """Compute Recall@K."""
    if not actual:
        return 0.0
    return len(actual.intersection(predicted[:k])) / len(actual)


def average_precision_at_k(actual: set[str], predicted: list[str], k: int) -> float:
    """Compute AP@K for a single user/query."""
    if not actual:
        return 0.0
    score = 0.0
    hits = 0
    for idx, item in enumerate(predicted[:k], start=1):
        if item in actual:
            hits += 1
            score += hits / idx
    return score / min(len(actual), k)


def ndcg_at_k(actual: set[str], predicted: list[str], k: int) -> float:
    """Compute NDCG@K for binary relevance."""
    dcg = 0.0
    for idx, item in enumerate(predicted[:k], start=1):
        if item in actual:
            dcg += 1.0 / math.log2(idx + 1)
    ideal_hits = min(len(actual), k)
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def ranking_metrics(
    actual_by_query: dict[str, set[str]],
    predicted_by_query: dict[str, list[str]],
    k: int,
) -> dict[str, float]:
    """Aggregate ranking metrics across users or seed songs."""
    if not actual_by_query:
        return {
            f"precision_at_{k}": 0.0,
            f"recall_at_{k}": 0.0,
            f"map_at_{k}": 0.0,
            f"ndcg_at_{k}": 0.0,
        }

    precisions = []
    recalls = []
    maps = []
    ndcgs = []
    for query, actual in actual_by_query.items():
        predicted = predicted_by_query.get(query, [])
        precisions.append(precision_at_k(actual, predicted, k))
        recalls.append(recall_at_k(actual, predicted, k))
        maps.append(average_precision_at_k(actual, predicted, k))
        ndcgs.append(ndcg_at_k(actual, predicted, k))

    count = len(actual_by_query)
    return {
        f"precision_at_{k}": sum(precisions) / count,
        f"recall_at_{k}": sum(recalls) / count,
        f"map_at_{k}": sum(maps) / count,
        f"ndcg_at_{k}": sum(ndcgs) / count,
    }
