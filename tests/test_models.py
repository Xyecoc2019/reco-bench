"""Tests for the models: shapes, determinism and learning behaviour."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from reco_bench.models import ALS, BPR, ItemKNN, Popularity, PureSVD  # noqa: E402


def toy_problem(n_users=30, n_items=20, seed=0):
    rng = np.random.default_rng(seed)
    rows, cols, vals = [], [], []
    for u in range(n_users):
        for i in rng.choice(n_items, size=6, replace=False):
            rows.append(u)
            cols.append(int(i))
            vals.append(1.0)
    return (np.array(rows), np.array(cols), np.array(vals, dtype=np.float32)), n_users, n_items


def test_popularity_ranks_by_total_weight():
    cols = np.array([0, 0, 1, 2, 2, 2])
    vals = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    model = Popularity().fit(cols, vals, n_items=3, n_users=2)
    scores = model.score_matrix()
    assert scores.shape == (2, 3)
    assert int(scores[0].argmax()) == 2


@pytest.mark.parametrize("model_factory", [
    lambda Rw, nu, ni: ItemKNN(topk_neighbors=5).fit(Rw[0], Rw[1], Rw[2], nu, ni),
    lambda Rw, nu, ni: PureSVD(k=4).fit(Rw[0], Rw[1], Rw[2], nu, ni),
])
def test_baselines_produce_score_matrix(model_factory):
    Rw, n_users, n_items = toy_problem()
    model = model_factory(Rw, n_users, n_items)
    scores = model.score_matrix()
    assert scores.shape == (n_users, n_items)
    assert torch.isfinite(scores).all()


def test_als_is_deterministic_with_the_same_seed():
    Rw, n_users, n_items = toy_problem()
    a = ALS(k=4, iters=2, cg_iters=3, seed=7, verbose=False).fit(Rw, n_users, n_items)
    b = ALS(k=4, iters=2, cg_iters=3, seed=7, verbose=False).fit(Rw, n_users, n_items)
    assert torch.allclose(a.score_matrix(), b.score_matrix())


def test_als_reduces_reconstruction_error():
    Rw, n_users, n_items = toy_problem(seed=3)
    rows, cols, vals = Rw
    model = ALS(k=8, iters=6, cg_iters=20, alpha=10.0, reg=0.05, seed=0,
                verbose=False).fit(Rw, n_users, n_items)
    scores = model.score_matrix()
    before = float((vals ** 2).sum())
    after = float(((scores[rows, cols] - vals) ** 2).sum())
    assert after < before


def test_bpr_learns_and_returns_scores():
    Rw, n_users, n_items = toy_problem()
    rows, cols, _ = Rw
    model = BPR(k=4, epochs=3, seed=0, verbose=False).fit(rows, cols, n_users, n_items)
    scores = model.score_matrix()
    assert scores.shape == (n_users, n_items)
    assert torch.isfinite(scores).all()


def test_als_beats_popularity_on_latent_factor_data():
    """Regression test: the ALS half-step must pair weights with the right (user, item)."""
    from reco_bench import metrics

    rng = np.random.default_rng(0)
    n_users, n_items, k = 60, 40, 4
    user_factors = rng.normal(size=(n_users, k))
    item_factors = rng.normal(size=(n_items, k))

    rows, cols, vals = [], [], []
    seen, truths = [], []
    for u in range(n_users):
        order = np.argsort(-(item_factors @ user_factors[u]))
        train_items = order[:8]
        future_items = order[8:10]
        seen.append(train_items.astype(np.int64))
        truths.append({int(i) for i in future_items})
        for i in train_items:
            rows.append(u)
            cols.append(int(i))
            vals.append(1.0)
    Rw = (np.array(rows), np.array(cols), np.array(vals, dtype=np.float32))

    als = ALS(k=k, alpha=10.0, reg=0.05, iters=10, cg_iters=20, seed=0,
              verbose=False).fit(Rw, n_users, n_items)
    pop = Popularity().fit(Rw[1], Rw[2], n_items, n_users)

    als_recs = metrics.topk_recommend(als.score_matrix(), seen, k=10).numpy()
    pop_recs = metrics.topk_recommend(pop.score_matrix(), seen, k=10).numpy()
    als_map = metrics.evaluate(als_recs, truths, ks=(10,))["MAP@10"]
    pop_map = metrics.evaluate(pop_recs, truths, ks=(10,))["MAP@10"]
    assert als_map > pop_map, f"ALS MAP@10={als_map:.4f} <= Popularity {pop_map:.4f}"
