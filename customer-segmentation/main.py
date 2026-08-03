import pandas as pd

from src.data import load_data
from src.evaluate import calculate_elbow, save_results
from src.kmean import KMeans
from src.preprocessing import (
    FEATURE_COLUMNS,
    prepare_features,
    restore_original_scale,
)
from src.visualization import plot_clusters, plot_elbow


def main():
    data = load_data()

    X, X_scaled, mu, sigma = prepare_features(data)

    # Compare possible values of K. The graph is inspected to find
    # the point where decreases in inertia begin to slow down.
    k_values, inertias = calculate_elbow(
        X_scaled,
        k_values=range(1, 11),
    )
    plot_elbow(k_values, inertias)

    # The elbow for these two mall-customer features is around K=5.
    selected_k = 5
    model = KMeans(k=selected_k)
    model.fit(X_scaled)

    original_centroids = restore_original_scale(
        model.centroids,
        mu,
        sigma,
    )

    centroid_table = pd.DataFrame(
        original_centroids,
        columns=FEATURE_COLUMNS,
    )
    centroid_table.insert(
        0,
        "Cluster",
        range(selected_k),
    )

    clustered_customers = data.copy()
    clustered_customers["Cluster"] = model.labels

    metrics = save_results(
        clustered_customers,
        centroid_table,
        model,
        k_values,
        inertias,
    )

    plot_clusters(
        X.to_numpy(),
        model.labels,
        original_centroids,
    )

    print("Final centroids in original units:")
    print(centroid_table)

    print("\nCluster sizes:")
    for cluster, count in metrics["cluster_sizes"].items():
        print(f"Cluster {cluster}: {count} customers")

    print(f"\nSelected K: {model.k}")
    print(f"Final inertia: {model.inertia:.4f}")
    print(f"Iterations: {model.n_iterations}")
    print("Results saved successfully.")


if __name__ == "__main__":
    main()
