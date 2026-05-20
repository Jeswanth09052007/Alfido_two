import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if "date" in X.columns:
            dt = pd.to_datetime(X["date"], errors="coerce")
            X["sale_year"] = dt.dt.year
            X["sale_month"] = dt.dt.month
            X["sale_dayofweek"] = dt.dt.dayofweek
            X = X.drop(columns=["date"])
        if "yr_built" in X.columns:
            X["house_age"] = 2014 - X["yr_built"]
        if "yr_renovated" in X.columns:
            X["was_renovated"] = (X["yr_renovated"] > 0).astype(int)
            X["renovation_age"] = np.where(X["yr_renovated"] > 0, 2014 - X["yr_renovated"], 0)
        if "statezip" in X.columns:
            X["zipcode"] = X["statezip"].astype(str).str.extract(r"(\d{5})", expand=False)
        if "street" in X.columns:
            X = X.drop(columns=["street"])
        if "country" in X.columns and X["country"].nunique(dropna=False) <= 1:
            X = X.drop(columns=["country"])
        return X
