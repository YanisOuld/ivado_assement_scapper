from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def _default_processed_path() -> str:
    default_path = "data/processed/museums_processed.csv"
    try:
        from src import config as cfg  # type: ignore

        return getattr(cfg, "PROCESSED_MUSEUMS_CSV", default_path)
    except Exception:
        return default_path


def _default_models_dir() -> str:
    default_dir = "models"
    try:
        from src import config as cfg  # type: ignore

        return getattr(cfg, "MODELS_DIR", default_dir)
    except Exception:
        return default_dir


def train_and_save(
    in_csv: str,
    out_dir: str,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    df = pd.read_csv(in_csv)

    required = {"museum_name", "city", "city_population", "annual_visitors"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}. Got: {set(df.columns)}")

    df = df.dropna(subset=["city_population", "annual_visitors"]).copy()

    X = df[["city_population"]].astype(float).values
    y = df["annual_visitors"].astype(float).values

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y,
        df.index.values,
        test_size=test_size,
        random_state=random_state,
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Timestamped run folder (keeps history)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = out_path / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = run_dir / "linear_regression.joblib"
    joblib.dump(model, model_path)

    # Save metrics + config
    payload = {
        "run_id": run_id,
        "input": str(in_csv),
        "n_rows_used": int(len(df)),
        "test_size": float(test_size),
        "random_state": int(random_state),
        "features": ["city_population"],
        "target": "annual_visitors",
        "coef": float(model.coef_[0]),
        "intercept": float(model.intercept_),
        "metrics": {"rmse": rmse, "mae": mae, "r2": r2},
        "artifacts": {
            "model_path": str(model_path),
            "metrics_path": str(run_dir / "metrics.json"),
            "predictions_path": str(run_dir / "predictions_test.csv"),
        },
    }

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Save test predictions for inspection
    pred_df = df.loc[idx_test, ["museum_name", "city", "city_population", "annual_visitors"]].copy()
    pred_df["pred_annual_visitors"] = preds
    pred_df.to_csv(run_dir / "predictions_test.csv", index=False)

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and save a baseline regression model.")
    parser.add_argument("--in", dest="in_csv", default=_default_processed_path(), help="Input processed CSV path")
    parser.add_argument("--out", dest="out_dir", default=_default_models_dir(), help="Output models directory")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    result = train_and_save(
        in_csv=args.in_csv,
        out_dir=args.out_dir,
        test_size=args.test_size,
        random_state=args.seed,
    )

    # Clean summary
    m = result["metrics"]
    print(
        "Model trained & saved\n"
        f"- run: {result['run_id']}\n"
        f"- rows: {result['n_rows_used']}\n"
        f"- coef: {result['coef']:.4f}\n"
        f"- intercept: {result['intercept']:.4f}\n"
        f"- RMSE: {m['rmse']:.4f}\n"
        f"- MAE: {m['mae']:.4f}\n"
        f"- R^2: {m['r2']:.4f}\n"
        f"- saved to: {result['artifacts']['model_path']}\n"
        f"- metrics: {result['artifacts']['metrics_path']}\n"
        f"- preds: {result['artifacts']['predictions_path']}\n"
    )


if __name__ == "__main__":
    main()
