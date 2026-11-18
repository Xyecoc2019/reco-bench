"""End-to-end test of the experiment pipeline on a small synthetic dataset."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from reco_bench.experiment import run_experiment  # noqa: E402
from reco_bench.data import add_scores, build_matrix, split_holdout  # noqa: E402


def make_frame(n_users=40, n_items=30, per_user=25, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_users):
        for i in rng.choice(n_items, size=per_user, replace=False):
            rows.append({
                "user_id": u,
                "item_id": int(i),
                "timespent": float(rng.integers(5, 500)),
                "like": bool(rng.random() < 0.25),
                "dislike": bool(rng.random() < 0.05),
                "share": bool(rng.random() < 0.1),
                "bookmark": bool(rng.random() < 0.08),
                "click_on_author": bool(rng.random() < 0.2),
                "open_comments": bool(rng.random() < 0.15),
            })
    df = pd.DataFrame(rows)
    df["row_id"] = np.arange(len(df))
    return df


def test_split_and_matrix_are_consistent():
    df = make_frame()
    train, holdout = split_holdout(df, holdout_size=5)
    mat = build_matrix(df, train, scheme="rich")
    assert mat["n_users"] == 40
    assert mat["n_items"] == 30
    assert len(holdout) == 40 * 5


def test_run_experiment_produces_a_results_table():
    df = make_frame()
    table, results = run_experiment(
        df, scheme="rich", k=4, als_iters=2, cg_iters=3, bpr_epochs=2,
        holdout=5, topk=10, verbose=False,
    )
    assert set(results) == {"Popularity", "ItemKNN", "PureSVD", "ALS", "BPR-MF"}
    assert "MAP@10" in table.columns
    assert table["MAP@10"].notna().all()
    assert (table["MAP@10"] >= 0).all() and (table["MAP@10"] <= 1).all()
    # The table is sorted by MAP@10 in descending order.
    assert list(table["MAP@10"]) == sorted(table["MAP@10"], reverse=True)
