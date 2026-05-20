import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

root = Path(__file__).resolve().parents[1]
sys.path.append(str(root / "src"))

model = joblib.load(root / "models" / "best_house_price_model.joblib")

sample = pd.DataFrame([{
    "date": "2014-05-02 00:00:00",
    "bedrooms": 3,
    "bathrooms": 2.0,
    "sqft_living": 1800,
    "sqft_lot": 5000,
    "floors": 1.0,
    "waterfront": 0,
    "view": 0,
    "condition": 3,
    "sqft_above": 1600,
    "sqft_basement": 200,
    "yr_built": 1995,
    "yr_renovated": 0,
    "street": "Example Street",
    "city": "Seattle",
    "statezip": "WA 98133",
    "country": "USA"
}])

predicted_price = np.expm1(model.predict(sample))[0]
print(f"Predicted house price: ${predicted_price:,.2f}")
