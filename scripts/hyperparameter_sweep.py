"""Hyperparameter sweep for the ALS model.

Grid over ``k × alpha × reg`` for one or more weighting schemes. The result is
written as JSON so that the report can be regenerated from it.

Usage::

    python scripts/hyperparameter_sweep.py --data data/synthetic_train.parquet \
        --schemes rich,binary --ks 8,16,24,32 --alphas 1,10,40 --regs 0.01,0.05 \
        --out results/als_hyperparameter_sweep.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reco_bench import data as data_mod  # noqa: E402
from reco_bench import metrics  # noqa: E402
from reco_bench.experiment import build_truths, eval_users  # noqa: E402
from reco_bench.models import ALS  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="ALS hyperparameter sweep")
    p.add_argument("--data", default="data/synthetic_train.parquet")
    p.add_argument("--schemes", default="rich,binary")
    p.add_argument("--ks", default="8,16,24,32")
    p.add_argument("--alphas", default="1,10,40")
    p.add_argument("--regs", default="0.01,0.05")
    p.add_argument("--als-iters", type=int, default=15)
    p.add_argument("--cg-iters", type=int, default=20)
    p.add_argument("--holdout", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/als_hyperparameter_sweep.json")
    args = p.parse_args()

    df = data_mod.load_train(args.data)
    train_df, holdout_df = data_mod.split_holdout(df, holdout_size=args.holdout)

    ks = [int(x) for x in args.ks.split(",") if x]
    alphas = [float(x) for x in args.alphas.split(",") if x]
    regs = [float(x) for x in args.regs.split(",") if x]
    schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]

    results: dict[str, dict] = {}
    for scheme in schemes:
        mat = data_mod.build_matrix(df, train_df, scheme=scheme)
        rows, cols, vals = mat["Rw"]
        truths = build_truths(mat, holdout_df)
        users = eval_users(mat, truths)
        for k, alpha, reg in itertools.product(ks, alphas, regs):
            t0 = time.time()
            als = ALS(k=k, alpha=alpha, reg=reg, iters=args.als_iters,
                      cg_iters=args.cg_iters, seed=args.seed, verbose=False)
            als.fit((rows, cols, vals), mat["n_users"], mat["n_items"])
            recs = metrics.topk_recommend(als.score_matrix(), mat["train_seen"], k=50).numpy()
            res = metrics.evaluate(recs[users], [truths[u] for u in users],
                                   n_items=mat["n_items"])
            key = f"{scheme}_k{k}_a{alpha}_r{reg}"
            results[key] = {
                "MAP@10": res["MAP@10"],
                "Recall@10": res["Recall@10"],
                "NDCG@10": res["NDCG@10"],
                "Coverage@10": res["Coverage@10"],
                "fit_seconds": round(time.time() - t0, 1),
            }
            print(f"{key:34s} MAP@10={res['MAP@10']:.4f} "
                  f"NDCG@10={res['NDCG@10']:.4f} Cov@10={res['Coverage@10']:.3f}",
                  flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    best = max(results, key=lambda key: results[key]["MAP@10"])
    print(f"written {out}; best: {best} MAP@10={results[best]['MAP@10']:.4f}")


if __name__ == "__main__":
    main()
