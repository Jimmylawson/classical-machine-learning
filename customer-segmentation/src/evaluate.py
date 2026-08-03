import json
from pathlib import Path

import numpy as np

from src.kmean import KMeans


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def calculate_elbow(X, k_values=range(1, 11)):
    """Fit one model per K and return each model's inertia."""
    k_values = list(k_values)
    inertias = []

    for k in k_values:
        model = KMeans(k=k)
        model.fit(X)
        inertias.append(model.inertia)

    return k_values, inertias


def cluster_sizes(labels):
    """Return the number of customers assigned to each cluster."""
    cluster_numbers, counts = np.unique(
        labels,
        return_counts=True,
    )

    return {
        str(int(cluster)): int(count)
        for cluster, count in zip(cluster_numbers, counts)
    }


def save_results(
    clustered_customers,
    centroid_table,
    model,
    k_values,
    inertias,
):
    """Save customer assignments, centroids, and model metrics."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    clustered_customers.to_csv(
        RESULTS_DIR / "clustered_customers.csv",
        index=False,
    )
    centroid_table.to_csv(
        RESULTS_DIR / "centroids.csv",
        index=False,
    )

    metrics = {
        "selected_k": int(model.k),
        "inertia": float(model.inertia),
        "iterations": int(model.n_iterations),
        "cluster_sizes": cluster_sizes(model.labels),
        "elbow_results": {
            str(k): float(inertia)
            for k, inertia in zip(k_values, inertias)
        },
    }

    with open(
        RESULTS_DIR / "metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=4)

    return metrics
