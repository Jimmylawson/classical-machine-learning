from pathlib import Path
import pandas as pd
import kagglehub

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data"/ "raw"
RAW_DATA_PATH = RAW_DATA_DIR/ "Mall_Customers.csv"


def download_raw_data():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    kagglehub.dataset_download(
        "vjchoudhary7/customer-segmentation-tutorial-in-python",
        path="Mall_Customers.csv",
        output_dir=str(RAW_DATA_DIR),
    )

    print(f"Dataset downloaded to: {RAW_DATA_PATH}")


def load_data():
    if not RAW_DATA_PATH.exists():
        download_raw_data()

    return pd.read_csv(RAW_DATA_PATH)

