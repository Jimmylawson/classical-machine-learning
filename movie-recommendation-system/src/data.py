from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data"/"raw"/ "ml-100k"
RATING_PATH = RAW_DATA_DIR / "u.data"
MOVIE_PATH = RAW_DATA_DIR / "u.item"
import pandas as pd


MOVIE_COLUMNS = [
    "movie_id",
    "title",
    "release_date",
    "video_release_date",
    "imdb_url",
    "unknown",
    "action",
    "adventure",
    "animation",
    "children",
    "comedy",
    "crime",
    "documentary",
    "drama",
    "fantasy",
    "film_noir",
    "horror",
    "musical",
    "mystery",
    "romance",
    "sci_fi",
    "thriller",
    "war",
    "western",
]


def load_ratings():
    # Return the ratings as a DataFrame
    if not RATING_PATH.exists():
        raise FileNotFoundError(
            f"Ratings file not found: {RATING_PATH}"
        )

    rating = pd.read_csv(
        RATING_PATH,
        sep="\t",
        header=None,
        names = [
            "user_id",
            "movie_id", "rating",
            "timestamp"
        ]
    )
    return rating


def load_movies():
    # Return movie information as a DataFrame
    if not MOVIE_PATH.exists():
        raise FileNotFoundError(
            f"Movies file not found: {MOVIE_PATH}"
        )
    movies = pd.read_csv(
        MOVIE_PATH,
        sep="|",
        header=None,
        names= MOVIE_COLUMNS,
        encoding="latin-1"
    )