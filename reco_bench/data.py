"""Dataset loading, implicit-feedback weighting and temporal holdout splitting.

The public dataset used during development is a clickstream of a content platform.
It is **not** redistributed in this repository (competition terms); the pipeline
below works with any parquet file that follows the same schema, and
``scripts/generate_synthetic_data.py`` produces a schema-compatible stand-in.

Schema (one row per user/item interaction):

===================  ====================================================
column               meaning
===================  ====================================================
``user_id``          user identifier
``item_id``          content identifier
``place``            surface the content was shown on
``platform``        client platform
``agent``            client build
``timespent``        seconds spent on the content
``like``             explicit like flag
``dislike``          explicit dislike flag
``share``            share flag
``bookmark``         bookmark flag
``click_on_author``  profile click flag
``open_comments``    comments opened flag
===================  ====================================================
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

__all__ = [
    "BEHAVIOR_COLS",
    "FEATURE_WEIGHTS",
    "default_data_dir",
    "load_train",
    "add_scores",
    "split_holdout",
    "build_matrix",
]

#: Signal columns that count as an explicit engagement.
BEHAVIOR_COLS = ["like", "bookmark", "share", "click_on_author", "open_comments"]

#: Weights of the individual signals in the ``rich`` weighting scheme.
FEATURE_WEIGHTS = {
    "like": 3.0,
    "bookmark": 2.0,
    "share": 1.5,
    "click_on_author": 0.5,
    "open_comments": 1.0,
}

#: Weight of the (log-scaled) dwell time in the ``rich`` weighting scheme.
TIME_WEIGHT = 0.5

#: Name of the environment variable that overrides the data directory.
DATA_DIR_ENV = "RECO_DATA_DIR"


def default_data_dir() -> Path:
    """Return the directory the datasets are read from.

    Resolution order:

    1. ``$RECO_DATA_DIR`` environment variable;
    2. the ``data/`` directory next to the repository root.

    Returns:
        Path to the directory that holds the parquet files.
    """
    env = os.environ.get(DATA_DIR_ENV)
    if env:
        return Path(env).expanduser()
    return Path(__file__).resolve().parents[1] / "data"


def load_train(path: Path | str | None = None) -> pd.DataFrame:
    """Read the interaction log from a parquet file.

    The file is read through :func:`pyarrow.parquet.read_table` on purpose:
    some snapshots of the dataset have a footer that ``pandas.read_parquet``
    refuses to parse, while the table itself is intact.

    Args:
        path: Path to the parquet file. Defaults to ``<data dir>/train.parquet``.

    Returns:
        DataFrame with the original columns plus a ``row_id`` column that keeps
        the original row order (used as the temporal proxy during splitting).
    """
    if path is None:
        path = default_data_dir() / "train.parquet"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. "
            f"Set ${DATA_DIR_ENV} or generate one with "
            f"`python scripts/generate_synthetic_data.py`."
        )
    df = pq.read_table(str(path)).to_pandas()
    df = df.reset_index(drop=True)
    df["row_id"] = np.arange(len(df))
    return df


def add_scores(df: pd.DataFrame, scheme: str = "rich") -> pd.DataFrame:
    """Turn raw interaction flags into a positive label and a confidence score.

    Args:
        df: Raw interaction frame.
        scheme: One of ``binary``, ``like`` or ``rich``.

            * ``binary`` — any engagement or an above-median dwell time is positive;
            * ``like`` — only explicit likes are positive;
            * ``rich`` — positive set as in ``binary``, but the score is a weighted
              combination of likes, bookmarks, shares, comments and dwell time.

    Returns:
        Copy of the frame with ``positive`` (bool) and ``score`` (float) columns.

    Raises:
        ValueError: If ``scheme`` is unknown.
    """
    df = df.copy()
    df["log_time"] = np.log1p(df["timespent"].astype(float))
    has_action = df[BEHAVIOR_COLS].any(axis=1)
    threshold = float(df["timespent"].median())

    if scheme == "binary":
        df["positive"] = has_action | (df["timespent"] >= threshold)
        df["score"] = np.where(df["positive"], 1.0, 0.0)
    elif scheme == "like":
        df["positive"] = df["like"].astype(bool)
        df["score"] = df["like"].astype(float)
    elif scheme == "rich":
        df["positive"] = has_action | (df["timespent"] >= threshold)
        score = sum(w * df[col].astype(float) for col, w in FEATURE_WEIGHTS.items())
        df["score"] = score + TIME_WEIGHT * df["log_time"]
    else:
        raise ValueError(f"Unknown weighting scheme: {scheme}")

    # A dislike always removes the interaction from the positive set.
    dislike = df["dislike"].astype(bool)
    df.loc[dislike, "positive"] = False
    df.loc[dislike, "score"] = 0.0
    return df


def split_holdout(df: pd.DataFrame, holdout_size: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the log into a training part and a per-user temporal holdout.

    The last ``holdout_size`` interactions of every user (in file order) become
    the holdout set, the rest is used for training. This mimics the real
    "recommend what the user consumes next" setting.

    Args:
        df: Interaction frame, ideally with a ``row_id`` column.
        holdout_size: Number of interactions per user kept for evaluation.

    Returns:
        Tuple ``(train_df, holdout_df)``.
    """
    if "row_id" not in df.columns:
        df = df.reset_index(drop=True)
        df["row_id"] = np.arange(len(df))
    ordered = df.sort_values(["user_id", "row_id"], kind="mergesort")
    rank = ordered.groupby("user_id").cumcount(ascending=False)
    holdout = ordered[rank < holdout_size].reset_index(drop=True)
    train = ordered[rank >= holdout_size].reset_index(drop=True)
    return train, holdout


def build_matrix(df_all, df_train, scheme: str = "rich") -> dict:
    """Build the sparse interaction matrix and the index mappings.

    The id -> index mapping is derived from ``df_all`` so that items that only
    appear in the holdout still receive an index (they can then be predicted).

    Args:
        df_all: All interactions (used to build the index space).
        df_train: Training interactions (used to compute the positive weights).
        scheme: Weighting scheme, see :func:`add_scores`.

    Returns:
        Dict with ``n_users``, ``n_items``, ``uidx``/``iidx`` (original ids per
        index), ``Rw`` (tuple of rows/cols/values of the positive-weight
        matrix), ``train_items`` (positive items per user), ``train_seen``
        (all items a user has already been shown) and the scored frames.
    """
    scored_all = add_scores(df_all, scheme=scheme)
    scored_train = add_scores(df_train, scheme=scheme)
    pos = scored_train[scored_train["positive"]].copy()

    uidx, uuniq = pd.factorize(scored_all["user_id"], sort=True)
    iidx, iuniq = pd.factorize(scored_all["item_id"], sort=True)
    n_users, n_items = len(uuniq), len(iuniq)

    # id -> index lookup tables (unique ids, so a plain reindex is safe)
    user_pos = pd.Series(np.arange(n_users), index=uuniq)
    item_pos = pd.Series(np.arange(n_items), index=iuniq)

    rows = user_pos.reindex(pos["user_id"]).to_numpy()
    cols = item_pos.reindex(pos["item_id"]).to_numpy()
    vals = pos["score"].to_numpy(dtype=np.float32)

    # The same (user, item) pair may appear several times — sum the weights.
    flat = rows * n_items + cols
    uniq_flat, inverse = np.unique(flat, return_inverse=True)
    agg = np.zeros(len(uniq_flat), dtype=np.float32)
    np.add.at(agg, inverse, vals)
    u_rows = (uniq_flat // n_items).astype(np.int64)
    i_cols = (uniq_flat % n_items).astype(np.int64)

    order = np.lexsort((i_cols, u_rows))
    u_rows, i_cols, agg = u_rows[order], i_cols[order], agg[order]

    counts = np.bincount(u_rows, minlength=n_users)
    bounds = np.cumsum(counts)[:-1]
    train_items = [
        np.split(i_cols, bounds)[u] if counts[u] else np.array([], dtype=np.int64)
        for u in range(n_users)
    ]

    # Everything the user has already been shown in the training part, negatives included.
    seen_u = user_pos.reindex(scored_train["user_id"]).to_numpy()
    seen_i = item_pos.reindex(scored_train["item_id"]).to_numpy()
    buckets: list[list[int]] = [[] for _ in range(n_users)]
    for u, i in zip(seen_u, seen_i):
        buckets[u].append(int(i))
    train_seen = [np.array(sorted(set(b)), dtype=np.int64) for b in buckets]

    return {
        "n_users": n_users,
        "n_items": n_items,
        "uidx": uuniq.to_numpy(),
        "iidx": iuniq.to_numpy(),
        "Rw": (u_rows, i_cols, agg),
        "train_items": train_items,
        "train_seen": train_seen,
        "df_all": scored_all,
        "df_train": scored_train,
    }
