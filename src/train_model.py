from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

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


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def train_and_save(
    in_csv: str,
    out_dir: str,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    log_eps: float = 1e-9,
) -> dict:
    df = pd.read_csv(in_csv)

    required = {"museum_name", "city", "city_population", "annual_visitors"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}. Got: {set(df.columns)}")

    df = df.dropna(subset=["city_population", "annual_visitors"]).copy()

    # Ensure numeric
    df["city_population"] = pd.to_numeric(df["city_population"], errors="coerce")
    df["annual_visitors"] = pd.to_numeric(df["annual_visitors"], errors="coerce")
    df = df.dropna(subset=["city_population", "annual_visitors"]).copy()

    X = df[["city_population"]].astype(float).values
    y = df["annual_visitors"].astype(float).values

    # Log target (only for diagnostics + optional log-model)
    y_log = np.log(y + float(log_eps))

    X_train, X_test, y_train, y_test, ylog_train, ylog_test, idx_train, idx_test = train_test_split(
        X,
        y,
        y_log,
        df.index.values,
        test_size=test_size,
        random_state=random_state,
    )

    # -------------------------
    # Model 1: train on normal scale
    # -------------------------
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics_normal = _metrics(y_test, preds)

    # Also compute log-metrics for the SAME predictions (diagnostic)
    # Note: log(pred) requires positive predictions
    log_diag_ok = bool(np.all(preds + log_eps > 0)) and bool(np.all(y_test + log_eps > 0))
    metrics_log_diagnostic: Dict[str, float] | None = None
    if log_diag_ok:
        metrics_log_diagnostic = _metrics(np.log(y_test + log_eps), np.log(preds + log_eps))

    # -------------------------
    # Model 2: train on log(target) scale
    # -------------------------
    model_log = LinearRegression()
    model_log.fit(X_train, ylog_train)
    preds_log = model_log.predict(X_test)

    # Metrics in log-space (directly what model optimizes)
    metrics_log_space = _metrics(ylog_test, preds_log)

    # Convert log-preds back to original scale for interpretability
    preds_from_log = np.exp(preds_log)  # this predicts exp(E[log(y)]) ~ median-ish
    metrics_from_log_back = _metrics(y_test, preds_from_log)

    # Optional: "smearing" correction to reduce exp-bias (often improves mean prediction)
    # Compute residuals in log-space on train, then apply average exp(residual)
    train_resid_log = ylog_train - model_log.predict(X_train)
    smear = float(np.mean(np.exp(train_resid_log)))
    preds_from_log_smear = preds_from_log * smear
    metrics_from_log_back_smear = _metrics(y_test, preds_from_log_smear)

    # -------------------------
    # Save outputs
    # -------------------------
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = out_path / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save both models
    model_path = run_dir / "linear_regression.joblib"
    model_log_path = run_dir / "linear_regression_log_target.joblib"
    joblib.dump(model, model_path)
    joblib.dump(model_log, model_log_path)

    payload: Dict[str, Any] = {
        "run_id": run_id,
        "input": str(in_csv),
        "n_rows_used": int(len(df)),
        "test_size": float(test_size),
        "random_state": int(random_state),
        "features": ["city_population"],
        "target": "annual_visitors",
        "log_target": {"enabled": True, "eps": float(log_eps)},
        "normal_model": {
            "model_file": "linear_regression.joblib",
            "coef": float(model.coef_[0]),
            "intercept": float(model.intercept_),
            "metrics": metrics_normal,
            "log_metrics_diagnostic": metrics_log_diagnostic,  # may be None if invalid
        },
        "log_target_model": {
            "model_file": "linear_regression_log_target.joblib",
            "coef": float(model_log.coef_[0]),
            "intercept": float(model_log.intercept_),
            "smearing_factor": smear,
            "metrics_log_space": metrics_log_space,
            "metrics_back_to_original": metrics_from_log_back,
            "metrics_back_to_original_smear": metrics_from_log_back_smear,
        },
        "artifacts": {
            "model_path": str(model_path),
            "model_log_path": str(model_log_path),
            "metrics_path": str(run_dir / "metrics.json"),
            "predictions_path": str(run_dir / "predictions_test.csv"),
        },
    }

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Save test predictions (both versions)
    pred_df = df.loc[idx_test, ["museum_name", "city", "city_population", "annual_visitors"]].copy()

    # Normal model predictions
    pred_df["pred_annual_visitors"] = preds
    pred_df["residual"] = pred_df["annual_visitors"].astype(float) - pred_df["pred_annual_visitors"].astype(float)

    # Log diagnostics (always defined, but log(pred) can be invalid)
    pred_df["log_annual_visitors"] = np.log(pred_df["annual_visitors"].astype(float).values + log_eps)

    pred_df["pred_log_from_normal"] = np.where(
        preds + log_eps > 0,
        np.log(preds + log_eps),
        np.nan,
    )

    # Log-target model predictions
    pred_df["pred_log_annual_visitors"] = preds_log
    pred_df["pred_annual_visitors_from_log"] = preds_from_log
    pred_df["pred_annual_visitors_from_log_smear"] = preds_from_log_smear

    pred_df.to_csv(run_dir / "predictions_test.csv", index=False)

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and save baseline regression models (normal + log target).")
    parser.add_argument("--in", dest="in_csv", default=_default_processed_path(), help="Input processed CSV path")
    parser.add_argument("--out", dest="out_dir", default=_default_models_dir(), help="Output models directory")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-eps", type=float, default=1e-9, help="Epsilon added before log to avoid log(0)")
    args = parser.parse_args()

    result = train_and_save(
        in_csv=args.in_csv,
        out_dir=args.out_dir,
        test_size=args.test_size,
        random_state=args.seed,
        log_eps=args.log_eps,
    )

    n = result["normal_model"]
    l = result["log_target_model"]

    print("Models trained & saved")
    print(f"- run: {result['run_id']}")
    print(f"- rows: {result['n_rows_used']}")
    print(f"- saved (normal): {result['artifacts']['model_path']}")
    print(f"- saved (log target): {result['artifacts']['model_log_path']}")
    print(f"- metrics: {result['artifacts']['metrics_path']}")
    print(f"- preds: {result['artifacts']['predictions_path']}")

    print("\nNormal model (trained on annual_visitors)")
    print(f"- coef: {n['coef']:.4f}")
    print(f"- intercept: {n['intercept']:.4f}")
    print(f"- RMSE: {n['metrics']['rmse']:.4f}")
    print(f"- MAE: {n['metrics']['mae']:.4f}")
    print(f"- R^2: {n['metrics']['r2']:.4f}")

    if n["log_metrics_diagnostic"] is not None:
        lm = n["log_metrics_diagnostic"]
        print("  Log diagnostic (log(y) vs log(ŷ) from normal model)")
        print(f"  - RMSE: {lm['rmse']:.4f}")
        print(f"  - MAE: {lm['mae']:.4f}")
        print(f"  - R^2: {lm['r2']:.4f}")

    print("\nLog-target model (trained on log(annual_visitors))")
    print(f"- coef: {l['coef']:.4f}")
    print(f"- intercept: {l['intercept']:.4f}")
    print(f"- smearing_factor: {l['smearing_factor']:.6f}")

    print("  In log space (direct)")
    print(f"  - RMSE: {l['metrics_log_space']['rmse']:.4f}")
    print(f"  - MAE: {l['metrics_log_space']['mae']:.4f}")
    print(f"  - R^2: {l['metrics_log_space']['r2']:.4f}")

    print("  Back to original scale (exp)")
    print(f"  - RMSE: {l['metrics_back_to_original']['rmse']:.4f}")
    print(f"  - MAE: {l['metrics_back_to_original']['mae']:.4f}")
    print(f"  - R^2: {l['metrics_back_to_original']['r2']:.4f}")

    print("  Back to original scale (exp + smearing)")
    print(f"  - RMSE: {l['metrics_back_to_original_smear']['rmse']:.4f}")
    print(f"  - MAE: {l['metrics_back_to_original_smear']['mae']:.4f}")
    print(f"  - R^2: {l['metrics_back_to_original_smear']['r2']:.4f}")


if __name__ == "__main__":
    main()
