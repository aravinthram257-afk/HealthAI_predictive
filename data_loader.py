from pathlib import Path
import pandas as pd
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "raw"
def load_los():
    return pd.read_csv(DATA_DIR / "health_los.csv")
def load_reviews():
    return pd.read_csv(DATA_DIR / "patient_feedback_clean.csv")
