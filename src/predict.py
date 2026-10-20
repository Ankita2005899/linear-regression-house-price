"""Load the saved model and predict for new data.  Usage: python -m src.predict"""
import joblib
import pandas as pd
from . import config

def predict(records):
    model = joblib.load(config.MODEL_PATH)
    return model.predict(pd.DataFrame(records))

if __name__ == "__main__":
    sample = [{"area": 1800, "bedrooms": 3, "bathrooms": 2, "age": 5, "distance_city": 8, "city": "Pune"}]
    print(f"Predicted price: {predict(sample)[0]:,.0f}")
