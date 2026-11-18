"""reco_bench — a small, reproducible benchmark for implicit-feedback recommenders.

The package contains:

* :mod:`reco_bench.data` — dataset loading, signal weighting, temporal holdout split;
* :mod:`reco_bench.models` — Popularity / ItemKNN / PureSVD baselines, a from-scratch
  implicit ALS (PyTorch + conjugate gradient) and BPR-MF;
* :mod:`reco_bench.metrics` — MAP@k, Recall@k, NDCG@k, catalogue coverage and novelty;
* :mod:`reco_bench.experiment` — the end-to-end training / evaluation pipeline and CLI.

Everything runs on CPU and is deterministic given a seed.
"""

from .data import add_scores, build_matrix, load_train, split_holdout
from .metrics import evaluate, novelty, topk_recommend
from .models import ALS, BPR, ItemKNN, Popularity, PureSVD

__all__ = [
    "ALS",
    "BPR",
    "ItemKNN",
    "Popularity",
    "PureSVD",
    "add_scores",
    "build_matrix",
    "evaluate",
    "load_train",
    "novelty",
    "split_holdout",
    "topk_recommend",
]

__version__ = "0.1.0"
