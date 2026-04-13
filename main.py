#!/usr/bin/env python3
"""
CS460 Recommender Systems – Main Experiment Script
===================================================
Runs KNN, SVD, ALS, and Hybrid recommenders on the MovieLens 100K and 1M
datasets and compares their RMSE against published benchmark targets.

Usage
-----
    python main.py [--datasets 100k 1m] [--no-hybrid] [--no-plot]

Benchmark targets
-----------------
    KNN  : RMSE ≈ 0.98
    SVD  : RMSE ≈ 0.93
    ALS  : (scalability-focused; competitive with SVD)
    Hybrid: best of all three
"""

import argparse
import time
import sys

import numpy as np
import pandas as pd

# ── project imports ──────────────────────────────────────────────────────────
from src.data_loader import load_ratings, train_test_split, dataset_summary
from src.knn_recommender import KNNRecommender
from src.svd_recommender import SVDRecommender
from src.als_recommender import ALSRecommender
from src.hybrid_recommender import HybridRecommender
from src.evaluation import train_and_evaluate, print_results_table

# ── published benchmark targets ──────────────────────────────────────────────
BENCHMARKS = {
    "KNN":  {"rmse": 0.98},
    "SVD":  {"rmse": 0.93},
}

# ── model factories ──────────────────────────────────────────────────────────

def build_models(include_hybrid: bool = True) -> dict:
    """Return a dict of {name: model_instance} to evaluate."""
    models = {
        "KNN":  KNNRecommender(k=40, sim_name="pearson", user_based=True),
        "SVD":  SVDRecommender(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02),
        "ALS":  ALSRecommender(n_factors=50, n_iter=20, reg=0.1, verbose=True),
    }
    if include_hybrid:
        models["Hybrid"] = HybridRecommender(
            als_params={"n_factors": 50, "n_iter": 15, "reg": 0.1, "verbose": False}
        )
    return models


# ── plotting ─────────────────────────────────────────────────────────────────

def plot_results(results: dict, output_path: str = "results.png"):
    """
    Save a bar-chart comparison of RMSE across models and datasets.

    Parameters
    ----------
    results : dict
        {model_name: {dataset_name: metrics_dict}}
    output_path : str
        File path for the saved figure.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available – skipping plot.")
        return

    models = list(results.keys())
    datasets = list(next(iter(results.values())).keys())

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#4C72B0", "#DD8452"]

    for idx, (dataset, color) in enumerate(zip(datasets, colors)):
        rmse_vals = [results[m].get(dataset, {}).get("rmse", 0) for m in models]
        offset = (idx - len(datasets) / 2 + 0.5) * width
        bars = ax.bar(x + offset, rmse_vals, width, label=f"ML-{dataset.upper()}", color=color, alpha=0.85)
        for bar, val in zip(bars, rmse_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=9,
            )

    # Draw benchmark lines
    ax.axhline(y=0.98, color="blue", linestyle="--", linewidth=1, label="KNN benchmark (0.98)")
    ax.axhline(y=0.93, color="orange", linestyle="--", linewidth=1, label="SVD benchmark (0.93)")

    ax.set_xlabel("Model")
    ax.set_ylabel("RMSE (lower is better)")
    ax.set_title("Recommender System Comparison – RMSE on MovieLens Datasets")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to {output_path}")
    plt.close()


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CS460 Recommender Systems experiment")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["100k", "1m"],
        choices=["100k", "1m"],
        help="Which MovieLens datasets to run (default: both)",
    )
    parser.add_argument(
        "--no-hybrid",
        action="store_true",
        help="Skip the Hybrid model (faster for quick iteration)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip saving the results plot",
    )
    args = parser.parse_args()

    # {model_name: {dataset_name: metrics}}
    all_results: dict = {}

    for dataset_name in args.datasets:
        print(f"\n{'='*60}")
        print(f"  Dataset: MovieLens {dataset_name.upper()}")
        print(f"{'='*60}")

        # Load & summarise
        df = load_ratings(dataset_name)
        dataset_summary(df, name=dataset_name)

        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        print(f"  Train: {len(train_df):,}  |  Test: {len(test_df):,}")

        models = build_models(include_hybrid=not args.no_hybrid)

        for model_name, model in models.items():
            print(f"\n--- {model_name} on ML-{dataset_name.upper()} ---")
            metrics = train_and_evaluate(model, train_df, test_df)

            # Store
            all_results.setdefault(model_name, {})[dataset_name] = metrics

            rmse_val = metrics["rmse"]
            mae_val = metrics["mae"]
            train_t = metrics["train_time_s"]
            print(
                f"  RMSE={rmse_val:.4f}  MAE={mae_val:.4f}  "
                f"train={train_t:.1f}s"
            )

            # Report vs benchmark
            if model_name in BENCHMARKS:
                target = BENCHMARKS[model_name]["rmse"]
                diff = rmse_val - target
                status = "✓ beats target" if diff < 0 else f"Δ +{diff:.4f} vs target"
                print(f"  Benchmark target (RMSE ≤ {target}): {status}")

    # Summary table
    print_results_table(all_results, benchmark=BENCHMARKS)

    # Plot
    if not args.no_plot and all_results:
        plot_results(all_results, output_path="results.png")


if __name__ == "__main__":
    main()
