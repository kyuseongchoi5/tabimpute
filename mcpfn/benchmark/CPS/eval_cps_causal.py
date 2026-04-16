#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the CPS causal benchmark by imputing the treated-post block."
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        required=True,
        help="Directory created by create_cps_causal_benchmark.py.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="knn,tabimpute_hf_default,tabimpute_v2,softimpute",
        help="Comma-separated method ids.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--mcpfn-checkpoint", type=str, default=None)
    parser.add_argument("--tabimputev2-checkpoint", type=str, default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of draw folders to evaluate.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Overwrite existing per-method outputs.",
    )
    return parser.parse_args()


def fill_all_nan_columns(x_missing: np.ndarray) -> np.ndarray:
    x_processed = x_missing.copy()
    for col_idx in range(x_processed.shape[1]):
        if np.all(np.isnan(x_processed[:, col_idx])):
            x_processed[:, col_idx] = 0.0
    return x_processed


def fill_all_nan_rows(x_missing: np.ndarray) -> np.ndarray:
    x_processed = x_missing.copy()
    col_means = np.nanmean(x_processed, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    for row_idx in range(x_processed.shape[0]):
        if np.all(np.isnan(x_processed[row_idx, :])):
            x_processed[row_idx, :] = col_means
    return x_processed


def make_knn_imputer() -> Callable[[np.ndarray], np.ndarray]:
    model = KNNImputer(n_neighbors=5)
    return lambda x: model.fit_transform(fill_all_nan_columns(x.copy()))


def make_tabimpute_imputer(device: str, checkpoint_path: str | None) -> Callable[[np.ndarray], np.ndarray]:
    from tabimpute.interface import ImputePFN

    model = ImputePFN(device=device, checkpoint_path=checkpoint_path)
    return lambda x: model.impute(x.copy())


def make_tabimputev2_imputer(
    device: str, checkpoint_path: str | None
) -> Callable[[np.ndarray], np.ndarray]:
    from tabimpute.tabimpute_v2 import TabImputeV2

    model = TabImputeV2(device=device, checkpoint_path=checkpoint_path)
    return lambda x: model.impute(x.copy())


def make_hyperimpute_imputer(plugin_name: str) -> Callable[[np.ndarray], np.ndarray]:
    from hyperimpute.plugins.imputers import Imputers

    def _run(x: np.ndarray) -> np.ndarray:
        plugin = Imputers().get(plugin_name, random_state=0)
        out_df = plugin.fit_transform(pd.DataFrame(fill_all_nan_rows(fill_all_nan_columns(x.copy()))))
        return out_df.to_numpy(dtype=np.float32)

    return _run


def build_method_registry(
    args: argparse.Namespace, requested_methods: list[str]
) -> dict[str, Callable[[np.ndarray], np.ndarray]]:
    builders: dict[str, Callable[[], Callable[[np.ndarray], np.ndarray]]] = {
        "knn": make_knn_imputer,
        "tabimpute_hf_default": lambda: make_tabimpute_imputer(args.device, args.mcpfn_checkpoint),
        "tabimpute_v2": lambda: make_tabimputev2_imputer(args.device, args.tabimputev2_checkpoint),
        "softimpute": lambda: make_hyperimpute_imputer("softimpute"),
        "hyperimpute": lambda: make_hyperimpute_imputer("hyperimpute"),
        "missforest": lambda: make_hyperimpute_imputer("missforest"),
    }

    unknown = [m for m in requested_methods if m not in builders]
    if unknown:
        raise SystemExit(f"Unknown methods requested: {unknown}")

    return {method: builders[method]() for method in requested_methods}


def compute_metrics(true: np.ndarray, completed: np.ndarray, treatment_mask: np.ndarray) -> dict[str, float]:
    treated = treatment_mask.astype(bool)
    observed = ~treated

    treated_true = true[treated]
    treated_pred = completed[treated]

    treated_block_rmse = float(np.sqrt(np.mean((treated_true - treated_pred) ** 2)))
    treated_block_mae = float(np.mean(np.abs(treated_true - treated_pred)))
    tau_hat = float(np.mean(treated_true) - np.mean(treated_pred))

    observed_rmse = float(np.sqrt(np.mean((true[observed] - completed[observed]) ** 2)))

    return {
        "treated_block_rmse": treated_block_rmse,
        "treated_block_mae": treated_block_mae,
        "tau_hat": tau_hat,
        "tau_sq_error": float(tau_hat**2),
        "tau_abs_error": float(abs(tau_hat)),
        "observed_block_rmse": observed_rmse,
        "n_treated_cells": int(treated.sum()),
    }


def main() -> None:
    args = parse_args()

    if not args.benchmark_dir.exists():
        raise SystemExit(f"Benchmark directory does not exist: {args.benchmark_dir}")

    requested_methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    registry = build_method_registry(args, requested_methods)

    draw_dirs = sorted(
        p for p in args.benchmark_dir.iterdir() if p.is_dir() and p.name.startswith("draw_")
    )
    if args.limit is not None:
        draw_dirs = draw_dirs[: args.limit]

    results_dir = args.benchmark_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []

    for draw_dir in draw_dirs:
        true = np.load(draw_dir / "true.npy")
        observed = np.load(draw_dir / "observed.npy")
        treatment_mask = np.load(draw_dir / "treatment_mask.npy")
        metadata = json.loads((draw_dir / "metadata.json").read_text(encoding="utf-8"))

        missing_mask = np.isnan(observed)

        for method in requested_methods:
            out_path = draw_dir / f"{method}.npy"
            if out_path.exists() and not args.force_rerun:
                completed = np.load(out_path)
                runtime_seconds = np.nan
            else:
                start = time.time()
                try:
                    completed = registry[method](observed.copy()).astype(np.float32)
                except Exception as exc:
                    failures.append(
                        {"draw_name": draw_dir.name, "method": method, "error": repr(exc)}
                    )
                    continue
                runtime_seconds = time.time() - start
                completed[~missing_mask] = observed[~missing_mask]
                np.save(out_path, completed)

            metrics = compute_metrics(true, completed, treatment_mask)
            metrics_rows.append(
                {
                    "draw_name": draw_dir.name,
                    "draw_id": int(metadata["draw_id"]),
                    "method": method,
                    "runtime_seconds": runtime_seconds,
                    **metrics,
                }
            )

    per_draw_path = results_dir / "per_draw_metrics.csv"
    pd.DataFrame(metrics_rows).to_csv(per_draw_path, index=False)

    if failures:
        pd.DataFrame(failures).to_csv(results_dir / "failures.csv", index=False)

    print(f"Wrote per-draw metrics to {per_draw_path}")
    if failures:
        print(f"Encountered {len(failures)} method failures; see {results_dir / 'failures.csv'}")


if __name__ == "__main__":
    main()
