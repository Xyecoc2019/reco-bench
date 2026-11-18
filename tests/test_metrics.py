"""Tests for the ranking metrics."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from reco_bench.metrics import evaluate, novelty, topk_recommend  # noqa: E402


def test_topk_excludes_seen_items():
    scores = torch.tensor([[0.1, 0.9, 0.5, 0.7]])
    seen = [np.array([1, 3], dtype=np.int64)]
    recs = topk_recommend(scores, seen, k=2).numpy()
    assert recs.shape == (1, 2)
    assert set(recs[0].tolist()).isdisjoint({1, 3})


def test_topk_handles_k_larger_than_catalogue():
    scores = torch.tensor([[0.1, 0.9]])
    recs = topk_recommend(scores, [np.array([], dtype=np.int64)], k=10).numpy()
    assert recs.shape == (1, 2)


def test_evaluate_perfect_ranking():
    recs = np.array([[0, 1, 2, 3]])
    truths = [{0, 1}]
    res = evaluate(recs, truths, ks=(2, 4))
    assert res["MAP@2"] == pytest.approx(1.0)
    assert res["Recall@2"] == pytest.approx(1.0)
    assert res["NDCG@2"] == pytest.approx(1.0)


def test_evaluate_worst_ranking():
    recs = np.array([[2, 3, 0, 1]])
    truths = [{0, 1}]
    res = evaluate(recs, truths, ks=(4,))
    assert res["MAP@4"] < 0.5
    assert res["Recall@4"] == pytest.approx(1.0)


def test_coverage_is_the_share_of_the_catalogue():
    recs = np.array([[0, 1], [2, 3]])
    truths = [{0}, {2}]
    res = evaluate(recs, truths, ks=(2, 4), n_items=10)
    assert res["Coverage@2"] == pytest.approx(0.4)
    assert res["Coverage@4"] == pytest.approx(0.4)


def test_coverage_without_catalogue_size_infers_it_from_the_indices():
    recs = np.array([[0, 1], [2, 3]])
    truths = [{0}, {2}]
    res = evaluate(recs, truths, ks=(2, 4))
    assert res["Coverage@2"] == pytest.approx(1.0)


def test_novelty_counts_unseen_items():
    recs = np.array([[0, 1], [2, 3]])
    seen = [np.array([0], dtype=np.int64), np.array([2], dtype=np.int64)]
    assert novelty(recs, seen, n_items=4) == pytest.approx(0.5)
