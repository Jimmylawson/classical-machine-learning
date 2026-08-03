import numpy as np


FEATURE_COLUMNS = [
    "Annual Income (k$)",
    "Spending Score (1-100)",
]


def prepare_features(data):
    """Select and standardize the features used by K-means."""
    X = data[FEATURE_COLUMNS].copy()

    if X.isna().any().any():
        missing_counts = X.isna().sum()
        raise ValueError(
            "Selected features contain missing values:\n"
            f"{missing_counts[missing_counts > 0]}"
        )

    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma = sigma.mask(sigma == 0, 1.0)

    X_scaled = ((X - mu) / sigma).to_numpy(dtype=float)

    return X, X_scaled, mu, sigma


def restore_original_scale(standardized_values, mu, sigma):
    """Convert standardized values back to their original units."""
    standardized_values = np.asarray(
        standardized_values,
        dtype=float,
    )

    return (
        standardized_values * sigma.to_numpy()
        + mu.to_numpy()
    )
