"""Tests for the data pipeline: weighting, splitting and matrix building."""

import numpy as np
import pandas as pd
import pytest

from reco_bench.data import add_scores, build_matrix, split_holdout  # noqa: E402


def make_frame(n_users=5, per_user=8):
    rows = []
    for u in range(n_users):
        for j in range(per_user):
            rows.append({
                "user_id": u,
                "item_id": u * 100 + j,
                "timespent": 10.0 * (j + 1),
                "like": j % 3 == 0,
                "dislike": j % 7 == 0,
                "share": j % 4 == 0,
                "bookmark": j % 5 == 0,
                "click_on_author": j % 2 == 0,
                "open_comments": j % 6 == 0,
            })
    df = pd.DataFrame(rows)
    df["row_id"] = np.arange(len(df))
    return df


def test_add_scores_like_scheme_only_likes():
    df = make_frame()
    scored = add_scores(df, scheme="like")
    assert scored["score"].max() <= 1.0
    expected = scored["like"].astype(bool) & ~scored["dislike"].astype(bool)
    assert (scored["score"] > 0).equals(expected)


def test_add_scores_dislike_is_always_zero():
    df = make_frame()
    scored = add_scores(df, scheme="rich")
    disliked = scored[scored["dislike"].astype(bool)]
    assert (disliked["score"] == 0.0).all()
    assert (~disliked["positive"]).all()


def test_add_scores_rejects_unknown_scheme():
    with pytest.raises(ValueError):
        add_scores(make_frame(), scheme="nonsense")


def test_split_holdout_keeps_last_interactions_per_user():
    df = make_frame(n_users=4, per_user=10)
    train, holdout = split_holdout(df, holdout_size=3)
    assert len(holdout) == 4 * 3
    assert len(train) == 4 * 7
    for u, group in holdout.groupby("user_id"):
        assert group["row_id"].max() == df[df["user_id"] == u]["row_id"].max()


def test_build_matrix_shapes_and_aggregation():
    df = make_frame(n_users=4, per_user=10)
    train, holdout = split_holdout(df, holdout_size=3)
    mat = build_matrix(df, train, scheme="binary")
    assert mat["n_users"] == 4
    assert mat["n_items"] == 40
    rows, cols, vals = mat["Rw"]
    assert len(rows) == len(cols) == len(vals)
    # No duplicate (user, item) pairs after aggregation.
    assert len(set(zip(rows.tolist(), cols.tolist()))) == len(rows)
    assert all(len(seen) > 0 for seen in mat["train_seen"])
    assert all(isinstance(x, np.ndarray) for x in mat["train_items"])
