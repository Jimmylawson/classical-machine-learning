import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw"/ "SMSSpamCollection"
PROCESSED_DATA_DIR = PROJECT_ROOT/"data"/"processed"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "spam_detection.csv"
PROCESSED_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)
def load_data():
    data = pd.read_csv(
        RAW_DATA_PATH, #tell pandas which file to open
        sep="\t", #it says the file's columns are separated by a tab character
        header=None, #tell pandas there is no header
        names=["label", "message"] #because the file has no header, they are assign names to the two columns
    )

    data.to_csv(PROCESSED_DATA_PATH, index=False)

    return data