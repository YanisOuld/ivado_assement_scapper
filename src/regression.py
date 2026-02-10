from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def train_regression(
    in_csv: str,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    df = pd.read_csv(in_csv)

    required = {"city_population", "annual_visitors"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}. Got: {set(df.columns)}")

    # Keep only rows with needed features
    df = df.dropna(subset=["city_population", "annual_visitors"]).copy()

    X = df[["city_population"]].astype(float).values
    y = df["annual_visitors"].astype(float).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))

    return {
        "n_rows": int(len(df)),
        "test_size": float(test_size),
        "coef": float(model.coef_[0]),
        "intercept": float(model.intercept_),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def _default_input_path() -> str:
    default_path = "data/processed/museums_processed.csv"
    try:
        from src import config as cfg  # type: ignore

        return getattr(cfg, "PROCESSED_MUSEUMS_CSV", default_path)
    except Exception:
        return default_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a baseline linear regression model.")
    parser.add_argument("--in", dest="in_csv", default=_default_input_path(), help="Input CSV path")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    results = train_regression(
        in_csv=args.in_csv,
        test_size=args.test_size,
        random_state=args.seed,
    )

    # Minimal console output (clean)
    print(
        "Regression results\n"
        f"- rows: {results['n_rows']}\n"
        f"- test_size: {results['test_size']}\n"
        f"- coef(log_city_population): {results['coef']:.4f}\n"
        f"- intercept: {results['intercept']:.4f}\n"
        f"- RMSE: {results['rmse']:.4f}\n"
        f"- MAE: {results['mae']:.4f}\n"
        f"- R^2: {results['r2']:.4f}\n"
    )


if __name__ == "__main__":
    main()
