import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def calculate_metrics(actual, predictions):
    errors = predictions - actual

    mse = np.mean(errors**2)
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(mse)

    residual_sum = np.sum(errors**2)
    total_sum = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - residual_sum / total_sum

    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2),
    }


def save_metrics(metrics):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_DIR / "metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)


def save_predictions(actual, predictions):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    errors = predictions - actual

    results = pd.DataFrame({
        "actual": actual,
        "predicted": predictions,
        "error": errors,
        "absolute_error": np.abs(errors),
        "actual_dollars": actual * 100_000,
        "predicted_dollars": predictions * 100_000,
    })

    results.to_csv(
        RESULTS_DIR / "prediction.csv",
        index=False,
    )