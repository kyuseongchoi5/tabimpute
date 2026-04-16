#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate CPS causal benchmark results across draws."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing per_draw_metrics.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    per_draw_path = args.results_dir / "per_draw_metrics.csv"
    if not per_draw_path.exists():
        raise SystemExit(f"Missing per-draw metrics file: {per_draw_path}")

    df = pd.read_csv(per_draw_path)
    if df.empty:
        raise SystemExit("Per-draw metrics file is empty.")

    summary = (
        df.groupby("method", dropna=False)
        .agg(
            n_draws=("draw_id", "count"),
            bias_tau=("tau_hat", "mean"),
            rmse_tau=("tau_sq_error", lambda s: float(np.sqrt(np.mean(s)))),
            mae_tau=("tau_abs_error", "mean"),
            mean_treated_block_rmse=("treated_block_rmse", "mean"),
            mean_treated_block_mae=("treated_block_mae", "mean"),
            mean_observed_block_rmse=("observed_block_rmse", "mean"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
        )
        .reset_index()
        .sort_values("rmse_tau", ascending=True)
    )

    summary_path = args.results_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"\nWrote summary to {summary_path}")


if __name__ == "__main__":
    main()
