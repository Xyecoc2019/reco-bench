"""Generate a schema-compatible synthetic clickstream.

The real dataset used during development comes from a competition and cannot be
redistributed. This script produces a stand-in with the same schema and the same
kind of structure (long-tail item popularity, latent user taste, noisy signals),
which is enough to run the whole pipeline, the tests and the CI end to end.

Usage::

    python scripts/generate_synthetic_data.py --out data/synthetic_train.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PLACES = ["feed", "search", "related", "profile", "notifications"]
PLATFORMS = ["android", "ios", "web"]
AGENTS = ["app-1.2.3", "app-1.3.0", "web-2024.11", "web-2025.01"]


def generate(
    n_users: int = 4000,
    n_items: int = 1500,
    n_topics: int = 12,
    interactions_per_user: int = 60,
    seed: int = 0,
) -> pd.DataFrame:
    """Generate a synthetic interaction log.

    Args:
        n_users: Number of users.
        n_items: Number of items.
        n_topics: Latent topic dimensionality.
        interactions_per_user: Average number of interactions per user.
        seed: Random seed.

    Returns:
        DataFrame with the competition schema.
    """
    rng = np.random.default_rng(seed)

    user_topics = rng.normal(size=(n_users, n_topics))
    item_topics = rng.normal(size=(n_items, n_topics))
    # Long-tail popularity: a few items are shown far more often than the rest.
    item_popularity = rng.pareto(1.5, size=n_items) + 1.0

    rows = []
    for u in range(n_users):
        n_inter = max(5, int(rng.poisson(interactions_per_user)))
        affinity = user_topics[u] @ item_topics.T
        probs = np.exp(affinity - affinity.max())
        probs = probs * item_popularity
        probs /= probs.sum()
        picked = rng.choice(n_items, size=n_inter, p=probs)

        for i in picked:
            match = affinity[i]
            dwell = float(np.clip(rng.lognormal(3.0 + 0.6 * match, 0.9), 0.0, 3600.0))
            engaged = match + rng.normal(0.0, 1.2)
            like = bool(engaged > 1.2)
            dislike = bool(engaged < -1.2)
            rows.append({
                "user_id": u,
                "item_id": int(i),
                "place": PLACES[rng.integers(len(PLACES))],
                "platform": PLATFORMS[rng.integers(len(PLATFORMS))],
                "agent": AGENTS[rng.integers(len(AGENTS))],
                "timespent": round(dwell, 2),
                "like": like,
                "dislike": dislike,
                "share": bool(rng.random() < 0.08 + 0.05 * max(match, 0)),
                "bookmark": bool(rng.random() < 0.05 + 0.04 * max(match, 0)),
                "click_on_author": bool(rng.random() < 0.25),
                "open_comments": bool(rng.random() < 0.18),
            })

    df = pd.DataFrame(rows)
    df["like"] = df["like"].astype(bool)
    df["dislike"] = df["dislike"].astype(bool)
    df["share"] = df["share"].astype(bool)
    df["bookmark"] = df["bookmark"].astype(bool)
    df["click_on_author"] = df["click_on_author"].astype(bool)
    df["open_comments"] = df["open_comments"].astype(bool)
    return df


def main() -> None:
    p = argparse.ArgumentParser(description="Synthetic clickstream generator")
    p.add_argument("--out", default="data/synthetic_train.parquet")
    p.add_argument("--users", type=int, default=4000)
    p.add_argument("--items", type=int, default=1500)
    p.add_argument("--topics", type=int, default=12)
    p.add_argument("--interactions", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    df = generate(
        n_users=args.users,
        n_items=args.items,
        n_topics=args.topics,
        interactions_per_user=args.interactions,
        seed=args.seed,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), str(out))
    print(f"written {out}  rows={len(df):,}  users={df['user_id'].nunique():,}  "
          f"items={df['item_id'].nunique():,}")


if __name__ == "__main__":
    main()
