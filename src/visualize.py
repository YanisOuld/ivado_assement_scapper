from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import joblib  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Missing dependency: joblib. Install it with `pip install joblib`."
    ) from e


def _default_processed_path() -> str:
    default_path = "data/processed/museums_processed.csv"
    try:
        from src import config as cfg  # type: ignore

        return getattr(cfg, "PROCESSED_MUSEUMS_CSV", default_path)
    except Exception:
        return default_path


def _default_figures_dir() -> str:
    default_dir = "reports/figures"
    try:
        from src import config as cfg  # type: ignore

        return getattr(cfg, "FIGURES_DIR", default_dir)
    except Exception:
        return default_dir


def _default_runs_dir() -> str:
    default_dir = "outputs/models"
    try:
        from src import config as cfg  # type: ignore

        return getattr(cfg, "MODELS_DIR", default_dir)
    except Exception:
        return default_dir


def _pick_latest_run(runs_dir: Path) -> Path:
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    candidates = [p for p in runs_dir.iterdir() if p.is_dir() and p.name.startswith("run_")]
    if not candidates:
        raise FileNotFoundError(f"No run_* folders found in: {runs_dir}")

    # Pick most recently modified run folder
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _resolve_run_dir(runs_dir: Path, run_id: Optional[str]) -> Path:
    if run_id:
        run_dir = runs_dir / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run folder not found: {run_dir}")
        return run_dir
    return _pick_latest_run(runs_dir)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float_series(s: pd.Series) -> pd.Series:
    # Coerce to numeric safely
    return pd.to_numeric(s, errors="coerce")


def visualize(
    in_csv: str,
    out_dir: str,
    runs_dir: str,
    run_id: Optional[str] = None,
) -> dict:
    # Load dataset
    df = pd.read_csv(in_csv)

    # Resolve which trained model run to use
    runs_path = Path(runs_dir)
    run_dir = _resolve_run_dir(runs_path, run_id)

    metrics_path = run_dir / "metrics.json"
    model_path = run_dir / "linear_regression.joblib"
    preds_test_path = run_dir / "predictions_test.csv"

    run_info = _read_json(metrics_path)

    features = run_info.get("features")
    target = run_info.get("target")
    metrics = run_info.get("metrics", {})

    if not isinstance(features, list) or not features:
        raise ValueError(f"Invalid or missing `features` in {metrics_path}")
    if not isinstance(target, str) or not target.strip():
        raise ValueError(f"Invalid or missing `target` in {metrics_path}")

    required = set(features) | {target}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}. Got: {set(df.columns)}")

    # Clean / coerce to numeric
    for col in features + [target]:
        df[col] = _safe_float_series(df[col])

    df = df.dropna(subset=features + [target]).copy()

    # Output dir
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # 1) Basic plots (keep yours)
    # -------------------------
    # If your project always has these columns, keep the original visuals too:
    if "city_population" in df.columns and "annual_visitors" in df.columns:
        plt.figure()
        plt.scatter(df["city_population"], df["annual_visitors"])
        plt.xlabel("city_population")
        plt.ylabel("annual_visitors")
        plt.title("Museums: city population vs annual visitors")
        scatter_path = out_path / "scatter_pop_vs_visitors.png"
        plt.savefig(scatter_path, dpi=200, bbox_inches="tight")
        plt.close()

        plt.figure()
        plt.hist(df["annual_visitors"].astype(float), bins=30)
        plt.xlabel("annual_visitors")
        plt.ylabel("count")
        plt.title("Distribution of annual visitors")
        hist_path = out_path / "hist_visitors.png"
        plt.savefig(hist_path, dpi=200, bbox_inches="tight")
        plt.close()
    else:
        scatter_path = None
        hist_path = None

    # -------------------------
    # 2) Model-based visuals
    # -------------------------
    model = joblib.load(model_path)

    X = df[features].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=float)

    # Predict on the full dataset (for visualization)
    try:
        y_pred = model.predict(X)
    except Exception as e:
        raise RuntimeError(
            f"Model prediction failed. Ensure your dataset features match the training features.\n"
            f"features={features}, target={target}, model={model_path}"
        ) from e

    # a) Predicted vs Actual
    plt.figure()
    plt.scatter(y, y_pred)
    plt.xlabel(f"Actual: {target}")
    plt.ylabel(f"Predicted: {target}")
    plt.title("Predicted vs Actual")

    # Annotate with test metrics from metrics.json (most honest)
    r2 = metrics.get("r2")
    mae = metrics.get("mae")
    rmse = metrics.get("rmse")

    lines = []
    if r2 is not None:
        lines.append(f"R² (test) = {float(r2):.3f}")
    if mae is not None:
        lines.append(f"MAE (test) = {float(mae):,.0f}")
    if rmse is not None:
        lines.append(f"RMSE (test) = {float(rmse):,.0f}")
    lines.append(f"run = {run_dir.name}")

    plt.gca().text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=plt.gca().transAxes,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    pred_vs_actual_path = out_path / "predicted_vs_actual.png"
    plt.savefig(pred_vs_actual_path, dpi=200, bbox_inches="tight")
    plt.close()

    # b) Residual plot
    residuals = y - y_pred
    plt.figure()
    plt.scatter(y_pred, residuals)
    plt.axhline(0)
    plt.xlabel("Predicted")
    plt.ylabel("Residual (y - ŷ)")
    plt.title("Residuals vs Predicted")
    residuals_path = out_path / "residuals.png"
    plt.savefig(residuals_path, dpi=200, bbox_inches="tight")
    plt.close()

    # c) Regression line on scatter (only if 1 feature)
    regline_path = None
    if len(features) == 1:
        x_name = features[0]
        x = df[x_name].to_numpy(dtype=float)

        # Sort by x for a clean line
        order = np.argsort(x)
        x_sorted = x[order]
        y_sorted = y[order]
        y_pred_sorted = y_pred[order]

        plt.figure()
        plt.scatter(x_sorted, y_sorted)
        plt.plot(x_sorted, y_pred_sorted)

        plt.xlabel(x_name)
        plt.ylabel(target)
        plt.title(f"{target} vs {x_name} (model line)")

        plt.gca().text(
            0.02,
            0.98,
            "\n".join(lines),
            transform=plt.gca().transAxes,
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )

        regline_path = out_path / "scatter_with_model_line.png"
        plt.savefig(regline_path, dpi=200, bbox_inches="tight")
        plt.close()

    # d) Optional: if predictions_test.csv exists, plot test residuals too
    test_plot_path = None
    if preds_test_path.exists():
        preds_df = pd.read_csv(preds_test_path)

        # Try common column names
        y_true_col = None
        y_hat_col = None
        for cand in ["y_true", "actual", "target", "annual_visitors"]:
            if cand in preds_df.columns:
                y_true_col = cand
                break
        for cand in ["y_pred", "pred", "prediction", "y_hat"]:
            if cand in preds_df.columns:
                y_hat_col = cand
                break

        if y_true_col and y_hat_col:
            y_true_t = _safe_float_series(preds_df[y_true_col]).dropna().to_numpy(dtype=float)
            y_hat_t = _safe_float_series(preds_df[y_hat_col]).dropna().to_numpy(dtype=float)

            n = min(len(y_true_t), len(y_hat_t))
            if n > 0:
                y_true_t = y_true_t[:n]
                y_hat_t = y_hat_t[:n]
                res_t = y_true_t - y_hat_t

                plt.figure()
                plt.scatter(y_hat_t, res_t)
                plt.axhline(0)
                plt.xlabel("Predicted (test)")
                plt.ylabel("Residual (test)")
                plt.title("Test residuals (from predictions_test.csv)")
                test_plot_path = out_path / "residuals_test.png"
                plt.savefig(test_plot_path, dpi=200, bbox_inches="tight")
                plt.close()

    return {
        "rows_used": int(len(df)),
        "run_dir": str(run_dir),
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "scatter_path": str(scatter_path) if scatter_path else None,
        "hist_path": str(hist_path) if hist_path else None,
        "pred_vs_actual_path": str(pred_vs_actual_path),
        "residuals_path": str(residuals_path),
        "regline_path": str(regline_path) if regline_path else None,
        "residuals_test_path": str(test_plot_path) if test_plot_path else None,
        "features": features,
        "target": target,
        "metrics_test": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create plots for the dataset + model-based visuals.")
    parser.add_argument("--in", dest="in_csv", default=_default_processed_path(), help="Input processed CSV path")
    parser.add_argument("--out", dest="out_dir", default=_default_figures_dir(), help="Output directory for figures")

    # New: choose which training run to use
    parser.add_argument(
        "--runs-dir",
        dest="runs_dir",
        default=_default_runs_dir(),
        help="Directory containing run_* folders (default: outputs/models)",
    )
    parser.add_argument(
        "--run-id",
        dest="run_id",
        default=None,
        help="Specific run folder name to use (example: run_20260206_232705). If omitted, uses latest run.",
    )

    args = parser.parse_args()

    result = visualize(
        in_csv=args.in_csv,
        out_dir=args.out_dir,
        runs_dir=args.runs_dir,
        run_id=args.run_id,
    )

    print("Figures generated")
    print(f"- rows: {result['rows_used']}")
    print(f"- run: {result['run_dir']}")
    if result["scatter_path"]:
        print(f"- scatter: {result['scatter_path']}")
    if result["hist_path"]:
        print(f"- hist: {result['hist_path']}")
    print(f"- predicted vs actual: {result['pred_vs_actual_path']}")
    print(f"- residuals: {result['residuals_path']}")
    if result["regline_path"]:
        print(f"- scatter + model line: {result['regline_path']}")
    if result["residuals_test_path"]:
        print(f"- residuals (test file): {result['residuals_test_path']}")


if __name__ == "__main__":
    main()
