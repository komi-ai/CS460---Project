"""
Evaluation utilities: RMSE, MAE, and timing helpers.
"""

import time
import numpy as np
import pandas as pd


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(actual - predicted)))


def evaluate(model, test_df: pd.DataFrame) -> dict:
    """
    Evaluate a fitted recommender on a test DataFrame.

    Parameters
    ----------
    model : object with predict_batch(df) method
    test_df : pd.DataFrame with columns userId, movieId, rating

    Returns
    -------
    dict with keys 'rmse', 'mae', 'inference_time_s'
    """
    t0 = time.time()
    preds = model.predict_batch(test_df)
    elapsed = time.time() - t0
    actuals = test_df["rating"].to_numpy()
    return {
        "rmse": rmse(actuals, preds),
        "mae": mae(actuals, preds),
        "inference_time_s": elapsed,
    }


def train_and_evaluate(model, train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """
    Fit a model on train_df, then evaluate on test_df.

    Returns
    -------
    dict with keys 'rmse', 'mae', 'train_time_s', 'inference_time_s'
    """
    t0 = time.time()
    model.fit(train_df)
    train_time = time.time() - t0

    metrics = evaluate(model, test_df)
    metrics["train_time_s"] = train_time
    return metrics


def print_results_table(results: dict, benchmark: dict = None):
    """
    Pretty-print a comparison table of results.

    Parameters
    ----------
    results : dict
        {model_name: {dataset_name: metrics_dict}}
    benchmark : dict, optional
        {model_name: {'rmse': float}} — benchmark targets to display.
    """
    print("\n" + "=" * 72)
    print(f"{'Model':<20}{'Dataset':<12}{'RMSE':>8}{'MAE':>8}{'Train(s)':>10}")
    print("-" * 72)

    for model_name, datasets in results.items():
        for dataset_name, metrics in datasets.items():
            rmse_val = metrics.get("rmse", float("nan"))
            mae_val = metrics.get("mae", float("nan"))
            train_t = metrics.get("train_time_s", float("nan"))
            print(
                f"{model_name:<20}{dataset_name:<12}"
                f"{rmse_val:>8.4f}{mae_val:>8.4f}{train_t:>10.1f}"
            )

    if benchmark:
        print("-" * 72)
        print("Benchmark targets:")
        for name, vals in benchmark.items():
            rmse_target = vals.get("rmse", "—")
            print(f"  {name:<18} RMSE ≤ {rmse_target}")

    print("=" * 72)
