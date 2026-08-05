"""Run the movie recommendation pipeline."""

from src.data import load_movies, load_ratings
from src.preprocessing import (
    apply_id_mapping,
    create_id_mappings,
    split_ratings,
)


def main():
    ratings = load_ratings()
    movies = load_movies()

    rating_details = ratings[
        ["user_id", "movie_id", "rating"]
    ].copy()

    movie_details = movies[
        ["movie_id", "title"]
    ].copy()

    (
        user_to_index,
        movie_to_index,
        index_to_user,
        index_to_movie,
    ) = create_id_mappings(rating_details)

    processed_ratings = apply_id_mapping(
        rating_details,
        user_to_index,
        movie_to_index,
    )

    (
        train_ratings,
        validation_ratings,
        test_ratings,
    ) = split_ratings(processed_ratings)

    print("Number of users:", len(user_to_index))
    print("Number of movies:", len(movie_to_index))

    print("\nTraining shape:")
    print(train_ratings.shape)

    print("\nValidation shape:")
    print(validation_ratings.shape)

    print("\nTest shape:")
    print(test_ratings.shape)

    print("\nProcessed ratings:")
    print(processed_ratings.head())

    print("\nMovie details:")
    print(movie_details.head())


if __name__ == "__main__":
    main()