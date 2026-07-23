from sklearn.datasets import fetch_california_housing
from pathlib import Path
import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw"/"california_housing.csv"


def download_raw_data():
    """Download and save the dataset if it does not already exist."""
    housing = fetch_california_housing(as_frame=True)
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    housing.frame.to_csv(RAW_DATA_PATH, index=False)

    print(f'Dataset saved to: {RAW_DATA_PATH}')


def load_data():
    """Load the raw dataset as a pandas DataFrame"""

    if not RAW_DATA_PATH.exists():
        download_raw_data()

    return pd.read_csv(RAW_DATA_PATH)

def clean_data(data):
    """Clean the dataset by removing duplicates and handling missing values."""
    cleaned_data = data.copy()
    #remove duplicate rows
    cleaned_data = cleaned_data.drop_duplicates()
    #remove rows with missing target values
    cleaned_data = cleaned_data.dropna(subset=["MedHouseVal"])

    return cleaned_data


if __name__ == "__main__":
    download_raw_data()
