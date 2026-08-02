

import numpy as np


def find_best_threshold(
    actual,
    probabilities,
    thresholds=None,
):
    actual = np.asarray(actual).reshape(-1)
    probabilities = np.asarray(
        probabilities
    ).reshape(-1)

    if actual.shape != probabilities.shape:
        raise ValueError(
            f"Shape mismatch: actual={actual.shape}, "
            f"probabilities={probabilities.shape}"
        )

    if thresholds is None:
        thresholds = np.arange(
            0.05,
            0.55,
            0.05,
        )

    best_threshold = 0.5
    best_metrics = None
    best_f1 = -1.0
    all_results = []

    for threshold in thresholds:
        predictions = (
            probabilities >= threshold
        ).astype(int)

        metrics = classification_metrics(
            actual,
            predictions,
        )

        all_results.append({
            "threshold": float(threshold),
            **metrics,
        })

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = threshold
            best_metrics = metrics

    return (
        float(best_threshold),
        best_metrics,
        all_results,
    )

from pathlib import Path
import json
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

def save_metrics(metrics):
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "metrics.json", "w") as file:
        json.dump(metrics, file,indent=4)

import pandas as pd

def save_predictions(actual, predictions):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    actual = np.asarray(actual).reshape(-1)
    predictions = np.asarray(predictions).reshape(-1)

    if actual.shape != predictions.shape:
        raise ValueError(
            f"Shape mismatch: actual={actual.shape}, "
            f"predictions={predictions.shape}"
        )

    prediction_results = pd.DataFrame({
        "actual": actual,
        "prediction": predictions,
    })

    prediction_results.to_csv(
        RESULTS_DIR / "spam_predictions.csv",
        index=False,
    )

    print("Predictions saved successfully.")


def classification_metrics(actual,prediction):
    actual = np.array(actual)
    prediction = np.array(prediction)

    true_positive = np.sum(
        (actual == 1) & (prediction == 1)
    )
    true_negative = np.sum(
        (actual == 0) & (prediction  == 0)
    )

    false_positive = np.sum(
        (actual == 0) & (prediction  == 1)
    )
    false_negative = np.sum(
        (actual == 1) & (prediction  == 0)
    )

    total = len(actual)
    accuracy  = (true_positive + true_negative) / total
    precision_denominator = true_positive + false_positive

    precision = (
        true_positive / precision_denominator
        if precision_denominator > 0
        else 0.0
    )
    recall_denominator = (
            true_positive + false_negative
    )

    recall = (
        true_positive / recall_denominator
        if recall_denominator > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return{
        "true_positive" :int(true_positive),
        "true_negative":int(true_negative),
        "false_positive":int(false_positive),
        "false_negative":int(false_negative),
        "accuracy":accuracy,
        "precision":precision,
        "recall":recall,
        "f1":f1

    }


