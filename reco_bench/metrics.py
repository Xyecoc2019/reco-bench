"""Ranking metrics: MAP@k, Recall@k, NDCG@k, catalogue coverage and novelty."""

from __future__ import annotations

import numpy as np
import torch

__all__ = ["topk_recommend", "evaluate", "novelty"]


def topk_recommend(scores: torch.Tensor, train_seen, k: int = 100) -> torch.Tensor:
    """Select the top-k items per user, excluding already seen items.

    Args:
        scores: Score matrix of shape ``(n_users, n_items)``.
        train_seen: List of arrays — items each user has already been shown.
        k: Number of recommendations per user.

    Returns:
        Tensor of shape ``(n_users, min(k, n_items))`` with item indices.
    """
    scores = scores.clone()
    for u, items in enumerate(train_seen):
        if len(items):
            scores[u, torch.from_numpy(np.asarray(items).astype(np.int64))] = -np.inf
    k = min(k, scores.shape[1])
    return torch.topk(scores, k=k, dim=1).indices


def _precision_at_k(hits: int, k: int) -> float:
    return hits / k


def evaluate(recs: np.ndarray, truths: list[set[int]], ks=(10, 20, 50),
             n_items: int | None = None) -> dict:
    """Compute ranking metrics for a set of users.

    Args:
        recs: Array of shape ``(n_users, max_k)`` with recommended item indices.
        truths: Per user set of ground-truth item indices.
        ks: Cut-offs to evaluate.
        n_items: Size of the catalogue. ``Coverage@k`` is the share of the
            catalogue that appears in the recommendations. When omitted, the
            catalogue size is inferred from the largest item index in ``recs``.

    Returns:
        Dict with ``MAP@k``, ``Recall@k``, ``NDCG@k`` and ``Coverage@k`` per cut-off.
    """
    out: dict[str, float] = {}
    n = len(truths)
    if n == 0:
        return out
    for k in ks:
        aps, recalls, ndcgs = [], [], []
        recommended: set[int] = set()
        for row, truth in zip(recs, truths):
            if not truth:
                continue
            top = row[:k]
            recommended.update(int(x) for x in top)
            hits = 0
            score = 0.0
            dcg = 0.0
            idcg = 0.0
            n_true = min(len(truth), k)
            for rank, item in enumerate(top, start=1):
                if int(item) in truth:
                    hits += 1
                    score += hits / rank
                    dcg += 1.0 / np.log2(rank + 1)
                idcg += 1.0 / np.log2(rank + 1)
            aps.append(score / n_true)
            recalls.append(hits / n_true)
            ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
        out[f"MAP@{k}"] = float(np.mean(aps))
        out[f"Recall@{k}"] = float(np.mean(recalls))
        out[f"NDCG@{k}"] = float(np.mean(ndcgs))
        if n_items:
            out[f"Coverage@{k}"] = len(recommended) / n_items
        else:
            catalogue = max(int(recs.max()) + 1, 1) if recs.size else 1
            out[f"Coverage@{k}"] = len(recommended) / catalogue
    return out


def novelty(recs: np.ndarray, train_seen, n_items: int) -> float:
    """Share of recommendations the user has not been shown before.

    Args:
        recs: Array of recommended item indices.
        train_seen: List of arrays — items each user has already been shown.
        n_items: Size of the catalogue (kept for API symmetry).

    Returns:
        Fraction of recommended items that are new for the user.
    """
    total = 0
    new = 0
    for u, row in enumerate(recs):
        seen = {int(x) for x in train_seen[u]}
        for item in row:
            total += 1
            if int(item) not in seen:
                new += 1
    return new / total if total else 0.0
