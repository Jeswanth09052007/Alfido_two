import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib, pickle
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from preprocessing import FeatureEngineer


def build_preprocessor(X):
    X_fe = FeatureEngineer().fit_transform(X)
    num_cols = X_fe.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_fe.select_dtypes(exclude=[np.number]).columns.tolist()
    log_cols = [c for c in ["sqft_living", "sqft_lot", "sqft_above", "sqft_basement"] if c in num_cols]
    normal_num_cols = [c for c in num_cols if c not in log_cols]
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer([
        ("log_numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")), ("scaler", StandardScaler())]), log_cols),
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), normal_num_cols),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", ohe)]), cat_cols),
    ])


def main():
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / "data" / "house_prices.csv").drop_duplicates()
    df = df[df["price"] > 0].copy()
    X = df.drop(columns=["price"])
    y = df["price"].astype(float)
    y_log = np.log1p(y)

    X_train, X_test, y_train_log, y_test_log, y_train, y_test = train_test_split(X, y_log, y, test_size=0.2, random_state=42)
    preprocessor = build_preprocessor(X_train)
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=80, random_state=42, min_samples_leaf=2, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=120, learning_rate=0.05, max_depth=3, random_state=42),
    }
    rows, fitted = [], {}
    for name, model in models.items():
        pipe = Pipeline([("features", FeatureEngineer()), ("preprocess", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train_log)
        pred = np.expm1(pipe.predict(X_test))
        rows.append({"Model": name, "RMSE": np.sqrt(mean_squared_error(y_test, pred)), "MAE": mean_absolute_error(y_test, pred), "R2": r2_score(y_test, pred)})
        fitted[name] = pipe
    metrics = pd.DataFrame(rows).sort_values("RMSE")
    (root / "outputs").mkdir(exist_ok=True)
    (root / "models").mkdir(exist_ok=True)
    metrics.to_csv(root / "outputs" / "model_comparison_metrics.csv", index=False)
    best_name = metrics.iloc[0]["Model"]
    joblib.dump(fitted[best_name], root / "models" / "best_house_price_model.joblib")
    with open(root / "models" / "best_house_price_model.pkl", "wb") as f:
        pickle.dump(fitted[best_name], f)
    print(metrics)
    print(f"Best model saved: {best_name}")

if __name__ == "__main__":
    main()
