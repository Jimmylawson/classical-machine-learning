"""Prepare user, movie, and rating data for training."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


REQUIRED_RATING_COLUMNS = {
    "user_id",
    "movie_id",
    "rating",
}


def validate_rating_columns(ratings):
    """Verify that the rating data contains the required columns."""
    missing_columns = REQUIRED_RATING_COLUMNS - set(ratings.columns)

    if missing_columns:
        raise ValueError(
            "Missing required rating columns: "
            f"{sorted(missing_columns)}"
        )

    if ratings.empty:
        raise ValueError("The ratings DataFrame cannot be empty.")


def create_id_mappings(ratings):
    """Create forward and reverse user/movie ID mappings."""
    validate_rating_columns(ratings)

    user_ids = np.sort(ratings["user_id"].unique())
    movie_ids = np.sort(ratings["movie_id"].unique())

    user_to_index = {
        user_id: index
        for index, user_id in enumerate(user_ids)
    }
    movie_to_index = {
        movie_id: index
        for index, movie_id in enumerate(movie_ids)
    }

    index_to_user = {
        index: user_id
        for user_id, index in user_to_index.items()
    }

    index_to_movie = {
        index: movie_id
        for movie_id, index in movie_to_index.items()
    }

    return (
        user_to_index,
        movie_to_index,
        index_to_user,
        index_to_movie,
    )


def apply_id_mapping(
    ratings,
    user_to_index,
    movie_to_index,
):
    """Add zero-based user and movie indexes to each rating."""
    validate_rating_columns(ratings)

    processed_ratings = ratings.copy()

    processed_ratings["user_index"] = (
        processed_ratings["user_id"].map(
            user_to_index
        )
    )

    processed_ratings["movie_index"] = (
        processed_ratings["movie_id"].map(
            movie_to_index
        )
    )

    missing_user_indexes = processed_ratings["user_index"].isna()
    missing_movie_indexes = processed_ratings["movie_index"].isna()

    if missing_user_indexes.any():
        unknown_users = (
            processed_ratings.loc[
                missing_user_indexes,
                "user_id",
            ]
            .unique()
            .tolist()
        )
        raise ValueError(
            f"Unknown user IDs found: {unknown_users}"
        )

    if missing_movie_indexes.any():
        unknown_movies = (
            processed_ratings.loc[
                missing_movie_indexes,
                "movie_id",
            ]
            .unique()
            .tolist()
        )
        raise ValueError(
            f"Unknown movie IDs found: {unknown_movies}"
        )

    processed_ratings["user_index"] = (
        processed_ratings["user_index"].astype(int)
    )
    processed_ratings["movie_index"] = (
        processed_ratings["movie_index"].astype(int)
    )

    return processed_ratings


def split_ratings(processed_ratings, random_state=42):
    """Create train/validation/test splits with known users and movies."""
    required_columns = {
        "user_index",
        "movie_index",
        "rating",
    }
    missing_columns = required_columns - set(processed_ratings.columns)

    if missing_columns:
        raise ValueError(
            "Ratings must be mapped before splitting. Missing: "
            f"{sorted(missing_columns)}"
        )

    if processed_ratings.empty:
        raise ValueError("The processed ratings cannot be empty.")

    train_ratings, temporary_ratings = train_test_split(
        processed_ratings,
        test_size=0.20,
        random_state=random_state,
        shuffle=True,
        stratify=processed_ratings["user_index"],
    )

    # A movie factor can only be learned if that movie occurs in training.
    training_movie_indexes = set(train_ratings["movie_index"])
    unseen_movie_mask = ~temporary_ratings["movie_index"].isin(
        training_movie_indexes
    )

    if unseen_movie_mask.any():
        train_ratings = pd.concat(
            [
                train_ratings,
                temporary_ratings[unseen_movie_mask],
            ],
            ignore_index=True,
        )
        temporary_ratings = temporary_ratings[
            ~unseen_movie_mask
        ].copy()

    ratings_per_user = temporary_ratings.groupby(
        "user_index"
    ).size()

    if len(ratings_per_user) != processed_ratings["user_index"].nunique():
        raise ValueError(
            "At least one user has no ratings left for validation/test."
        )

    if (ratings_per_user < 2).any():
        raise ValueError(
            "Each user needs at least two temporary ratings "
            "to create validation and test sets."
        )

    validation_ratings, test_ratings = train_test_split(
        temporary_ratings,
        test_size=0.50,
        random_state=random_state,
        shuffle=True,
        stratify=temporary_ratings["user_index"],
    )

    train_ratings = train_ratings.reset_index(drop=True)
    validation_ratings = validation_ratings.reset_index(drop=True)
    test_ratings = test_ratings.reset_index(drop=True)

    training_users = set(train_ratings["user_index"])
    training_movies = set(train_ratings["movie_index"])

    for split_name, split in (
        ("validation", validation_ratings),
        ("test", test_ratings),
    ):
        unknown_users = set(split["user_index"]) - training_users
        unknown_movies = set(split["movie_index"]) - training_movies

        if unknown_users or unknown_movies:
            raise ValueError(
                f"The {split_name} split contains unknown users "
                "or movies."
            )

    return train_ratings, validation_ratings, test_ratings
