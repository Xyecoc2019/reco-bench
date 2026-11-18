"""Build a markdown report from the result tables in ``results/``.

Usage::

    python scripts/make_report.py --results results --out docs/RESULTS.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

COLUMNS = ["model", "MAP@10", "Recall@10", "NDCG@10", "Coverage@10", "fit_seconds"]
ROUND = {"MAP@10": 4, "Recall@10": 4, "NDCG@10": 4, "Coverage@10": 3, "fit_seconds": 1}


def table_for(csv_path: Path) -> str:
    """Render one result CSV as a markdown table."""
    df = pd.read_csv(csv_path)
    for col in df.columns:
        if col == "model" or col not in COLUMNS:
            continue
        df[col] = df[col].astype(float).round(ROUND.get(col, 4))
    df = df[[c for c in COLUMNS if c in df.columns]]
    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep] + rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Markdown report from result tables")
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="docs/RESULTS.md")
    args = p.parse_args()

    results_dir = Path(args.results)
    files = sorted(results_dir.glob("results_*.csv"))
    if not files:
        raise SystemExit(f"No result files in {results_dir}")

    blocks = []
    for csv_path in files:
        title = csv_path.stem.replace("results_", "")
        blocks.append(f"### {title}\n\n{table_for(csv_path)}\n")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("# Результаты экспериментов\n\n" + "\n".join(blocks), encoding="utf-8")
    print(f"written {out} ({len(files)} tables)")


if __name__ == "__main__":
    main()
