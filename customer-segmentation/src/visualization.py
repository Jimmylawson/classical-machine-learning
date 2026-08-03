from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def plot_elbow(k_values, inertias):
    """Save the inertia-versus-K elbow graph."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(k_values, inertias, marker="o")
    axis.set_xlabel("Number of clusters (K)")
    axis.set_ylabel("Inertia")
    axis.set_title("Elbow Method")
    axis.set_xticks(k_values)
    axis.grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(
        RESULTS_DIR / "elbow_plot.png",
        dpi=150,
    )
    plt.close(figure)


def plot_clusters(X, labels, centroids):
    """Save the customer clusters in their original feature units."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    centroids = np.asarray(centroids, dtype=float)

    figure, axis = plt.subplots(figsize=(9, 6))

    for cluster_index in np.unique(labels):
        cluster_customers = X[labels == cluster_index]
        axis.scatter(
            cluster_customers[:, 0],
            cluster_customers[:, 1],
            label=f"Cluster {cluster_index}",
            alpha=0.75,
        )

    axis.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="X",
        s=220,
        color="black",
        edgecolor="white",
        linewidth=1,
        label="Centroids",
    )

    axis.set_xlabel("Annual Income (k$)")
    axis.set_ylabel("Spending Score (1-100)")
    axis.set_title("Mall Customer Segments")
    axis.legend()
    axis.grid(alpha=0.2)

    figure.tight_layout()
    figure.savefig(
        RESULTS_DIR / "cluster_plot.png",
        dpi=150,
    )
    plt.close(figure)
