"""End-to-end experiment: train every model, evaluate, write the results table.

CLI::

    python -m reco_bench.experiment --data data/synthetic_train.parquet --scheme rich
    python -m reco_bench.experiment --schemes rich,binary,like --k 32 --out results
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import data as data_mod
from . import metrics
from .models import ALS, BPR, ItemKNN, Popularity, PureSVD

__all__ = ["build_truths", "eval_users", "run_experiment", "main"]

RESULT_COLUMNS = [
    "model", "MAP@10", "Recall@10", "NDCG@10", "Coverage@10",
    "MAP@20", "Recall@20", "MAP@50", "Recall@50", "Novelty@10",
    "fit_seconds",
]


def build_truths(mat: dict, holdout: pd.DataFrame) -> dict[int, set[int]]:
    """Map the holdout interactions to the index space.

    Args:
        mat: Matrix bundle from :func:`reco_bench.data.build_matrix`.
        holdout: Holdout interactions.

    Returns:
        Dict ``user index -> set of item indices``.
    """
    item_index = pd.Index(mat["iidx"])
    user_index = pd.Index(mat["uidx"])
    h = holdout.copy()
    h["item_idx"] = item_index.get_indexer(h["item_id"])
    h["user_idx"] = user_index.get_indexer(h["user_id"])
    h = h[h["item_idx"] >= 0]
    truths: dict[int, set[int]] = {}
    for u, items in h.groupby("user_idx")["item_idx"]:
        truths[int(u)] = {int(x) for x in items}
    return truths


def eval_users(mat: dict, truths: dict[int, set[int]]) -> list[int]:
    """Users that have holdout data and at least one positive in the training part."""
    return [u for u in sorted(truths) if len(mat["train_items"][u]) > 0]


def run_experiment(
    df: pd.DataFrame,
    scheme: str = "rich",
    k: int = 32,
    als_iters: int = 8,
    cg_iters: int = 12,
    alpha: float = 10.0,
    reg: float = 0.05,
    bpr_epochs: int = 6,
    holdout: int = 10,
    topk: int = 50,
    skip: set[str] | None = None,
    seed: int = 0,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Train all models on the given data and evaluate them on the holdout.

    Args:
        df: Raw interaction frame.
        scheme: Weighting scheme (``rich``/``binary``/``like``).
        k: Latent dimensionality.
        als_iters: ALS iterations.
        cg_iters: Conjugate gradient iterations per ALS step.
        alpha: ALS confidence scaling.
        reg: ALS regularisation.
        bpr_epochs: BPR epochs.
        holdout: Holdout size per user.
        topk: Number of recommendations to produce.
        skip: Model names to skip.
        seed: Random seed.
        verbose: Print progress.

    Returns:
        Tuple ``(results table, raw results dict)``.
    """
    skip = skip or set()
    train_df, holdout_df = data_mod.split_holdout(df, holdout_size=holdout)
    mat = data_mod.build_matrix(df, train_df, scheme=scheme)
    n_users, n_items = mat["n_users"], mat["n_items"]
    rows, cols, vals = mat["Rw"]

    if verbose:
        print(f"  users: {n_users:,}  items: {n_items:,}  positives: {len(vals):,}",
              flush=True)

    truths = build_truths(mat, holdout_df)
    users = eval_users(mat, truths)
    if verbose:
        print(f"  evaluated users: {len(users):,}", flush=True)

    results: dict[str, dict] = {}

    def run_model(name, model):
        t0 = time.time()
        scores = model.score_matrix()
        recs = metrics.topk_recommend(scores, mat["train_seen"], k=topk).numpy()
        sub_recs = recs[users]
        sub_truths = [truths[u] for u in users]
        res = metrics.evaluate(sub_recs, sub_truths, n_items=n_items)
        res["Novelty@10"] = metrics.novelty(
            sub_recs, [mat["train_seen"][u] for u in users], n_items)
        res["fit_seconds"] = round(time.time() - t0, 1)
        results[name] = res
        if verbose:
            print(f"  {name:12s} MAP@10={res['MAP@10']:.4f}  "
                  f"Recall@10={res['Recall@10']:.4f}  NDCG@10={res['NDCG@10']:.4f}  "
                  f"Cov@10={res['Coverage@10']:.3f}  ({res['fit_seconds']} s)",
                  flush=True)
        return res

    if "popularity" not in skip:
        run_model("Popularity", Popularity().fit(cols, vals, n_items, n_users))

    if "itemknn" not in skip:
        run_model("ItemKNN", ItemKNN().fit(rows, cols, vals, n_users, n_items))

    if "puresvd" not in skip:
        run_model("PureSVD", PureSVD(k=k).fit(rows, cols, vals, n_users, n_items))

    if "als" not in skip:
        als = ALS(k=k, alpha=alpha, reg=reg, iters=als_iters, cg_iters=cg_iters,
                  seed=seed, verbose=verbose)
        als.fit((rows, cols, vals), n_users, n_items)
        run_model("ALS", als)

    if "bpr" not in skip:
        pos = mat["df_train"]
        pos = pos[pos["positive"]]
        u_index = pd.Index(mat["uidx"])
        i_index = pd.Index(mat["iidx"])
        bu = u_index.get_indexer(pos["user_id"])
        bi = i_index.get_indexer(pos["item_id"])
        bpr = BPR(k=k, epochs=bpr_epochs, seed=seed, verbose=verbose)
        bpr.fit(bu, bi, n_users, n_items)
        run_model("BPR-MF", bpr)

    table = pd.DataFrame(results).T.reset_index().rename(columns={"index": "model"})
    for col in RESULT_COLUMNS:
        if col not in table.columns:
            table[col] = np.nan
    table = table[RESULT_COLUMNS].sort_values("MAP@10", ascending=False)
    return table, results


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Implicit-feedback recommender benchmark")
    p.add_argument("--data", default=None,
                   help="Parquet file with the interactions "
                        "(default: $RECO_DATA_DIR/train.parquet)")
    p.add_argument("--schemes", default="rich",
                   help="Comma-separated weighting schemes: rich,binary,like")
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--als-iters", type=int, default=15)
    p.add_argument("--cg-iters", type=int, default=20)
    p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument("--reg", type=float, default=0.05)
    p.add_argument("--bpr-epochs", type=int, default=6)
    p.add_argument("--holdout", type=int, default=10)
    p.add_argument("--topk", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results")
    p.add_argument("--tag", default="", help="Optional suffix for the result files")
    p.add_argument("--skip", default="", help="Comma-separated model names to skip")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    skip = {s.strip().lower() for s in args.skip.split(",") if s.strip()}

    print("Loading data...", flush=True)
    df = data_mod.load_train(args.data)
    print(f"  rows: {len(df):,}  users: {df['user_id'].nunique():,}  "
          f"items: {df['item_id'].nunique():,}", flush=True)

    for scheme in [s.strip() for s in args.schemes.split(",") if s.strip()]:
        print(f"\n=== scheme: {scheme} ===", flush=True)
        table, results = run_experiment(
            df,
            scheme=scheme,
            k=args.k,
            als_iters=args.als_iters,
            cg_iters=args.cg_iters,
            alpha=args.alpha,
            reg=args.reg,
            bpr_epochs=args.bpr_epochs,
            holdout=args.holdout,
            topk=args.topk,
            skip=skip,
            seed=args.seed,
        )
        table.to_csv(out_dir / f"results_{scheme}_k{args.k}{args.tag}.csv",
                     index=False, encoding="utf-8-sig")
        with open(out_dir / f"results_{scheme}_k{args.k}{args.tag}.json", "w",
                  encoding="utf-8") as f:
            json.dump({"args": vars(args), "results": results}, f,
                      ensure_ascii=False, indent=2)
        print("\n" + table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
