from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "housing.csv"
MODEL_PATH = ROOT / "models" / "best_model.joblib"
OUT_DIR = ROOT / "outputs"

TARGET = "price"
NUMERIC = ["area", "bedrooms", "bathrooms", "age", "distance_city"]
CATEGORICAL = ["city"]
TEST_SIZE = 0.2
RANDOM_STATE = 42
