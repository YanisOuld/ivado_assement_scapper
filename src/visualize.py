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
    raise ImportError("Missing dependency: joblib. Install it with `pip install joblib`.") from e


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
    return pd.to_numeric(s, errors="coerce")


def _add_metric_box(lines: list[str]) -> None:
    plt.gca().text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=plt.gca().transAxes,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )


def _plot_residuals_with_smooth(
    x: np.ndarray,
    residuals: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    out_file: Path,
) -> None:
    # Sort by x for a clean smooth line
    order = np.argsort(x)
    xs = x[order]
    rs = residuals[order]

    plt.figure()
    plt.scatter(xs, rs)
    plt.axhline(0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # Rolling mean to reveal structure
    window = max(10, len(rs) // 30)
    smooth = (
        pd.Series(rs)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )
    plt.plot(xs, smooth)

    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()


def _plot_pred_vs_actual(
    y_true: np.ndarray,
    y_hat: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    out_file: Path,
    lines: list[str],
) -> None:
    plt.figure()
    plt.scatter(y_true, y_hat)

    # Ideal line y = x
    mn = float(np.min([y_true.min(), y_hat.min()]))
    mx = float(np.max([y_true.max(), y_hat.max()]))
    plt.plot([mn, mx], [mn, mx])
    plt.xlim(mn, mx)
    plt.ylim(mn, mx)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    _add_metric_box(lines)

    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()


def visualize(
    in_csv: str,
    out_dir: str,
    runs_dir: str,
    run_id: Optional[str] = None,
) -> dict:
    df = pd.read_csv(in_csv)

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

    # Coerce to numeric
    for col in features + [target]:
        df[col] = _safe_float_series(df[col])

    df = df.dropna(subset=features + [target]).copy()

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # 1) Basic plots
    # -------------------------
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

    # Optional safety check if model exposes expected feature names
    if hasattr(model, "feature_names_in_"):
        trained = list(getattr(model, "feature_names_in_"))
        if trained != features:
            raise ValueError(f"Model expects {trained} but metrics.json has {features}")

    X = df[features].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=float)

    try:
        y_pred = model.predict(X)
    except Exception as e:
        raise RuntimeError(
            "Model prediction failed. Ensure your dataset features match the training features.\n"
            f"features={features}, target={target}, model={model_path}"
        ) from e

    # Metrics box (from test metrics)
    r2 = metrics.get("r2")
    mae = metrics.get("mae")
    rmse = metrics.get("rmse")

    lines: list[str] = []
    if r2 is not None:
        lines.append(f"R² (test) = {float(r2):.3f}")
    if mae is not None:
        lines.append(f"MAE (test) = {float(mae):,.0f}")
    if rmse is not None:
        lines.append(f"RMSE (test) = {float(rmse):,.0f}")
    lines.append(f"run = {run_dir.name}")

    # a) Predicted vs Actual (original scale) + ideal line
    pred_vs_actual_path = out_path / "predicted_vs_actual.png"
    _plot_pred_vs_actual(
        y_true=y,
        y_hat=y_pred,
        xlabel=f"Actual: {target}",
        ylabel=f"Predicted: {target}",
        title="Predicted vs Actual",
        out_file=pred_vs_actual_path,
        lines=lines,
    )

    # b) Residuals vs Predicted (original scale) + smooth line
    residuals = y - y_pred
    residuals_path = out_path / "residuals.png"
    _plot_residuals_with_smooth(
        x=y_pred,
        residuals=residuals,
        xlabel="Predicted",
        ylabel="Residual (y - ŷ)",
        title="Residuals vs Predicted",
        out_file=residuals_path,
    )

    # -------------------------
    # 3) Log-scale diagnostics (if target positive-ish)
    # -------------------------
    eps = 1e-9
    log_ok = bool(np.all(np.isfinite(y))) and bool(np.all(y + eps > 0)) and bool(np.all(y_pred + eps > 0))

    pred_vs_actual_log_path = None
    residuals_log_path = None

    if log_ok:
        y_log = np.log(y + eps)
        y_pred_log = np.log(y_pred + eps)

        pred_vs_actual_log_path = out_path / "predicted_vs_actual_log.png"
        _plot_pred_vs_actual(
            y_true=y_log,
            y_hat=y_pred_log,
            xlabel=f"Actual: log({target})",
            ylabel=f"Predicted: log({target})",
            title="Predicted vs Actual (log scale)",
            out_file=pred_vs_actual_log_path,
            lines=lines,
        )

        residuals_log = y_log - y_pred_log
        residuals_log_path = out_path / "residuals_log.png"
        _plot_residuals_with_smooth(
            x=y_pred_log,
            residuals=residuals_log,
            xlabel=f"Predicted log({target})",
            ylabel="Residual (log(y) - log(ŷ))",
            title="Residuals vs Predicted (log scale)",
            out_file=residuals_log_path,
        )

    # c) Regression line on scatter (only if 1 feature)
    regline_path = None
    if len(features) == 1:
        x_name = features[0]
        x = df[x_name].to_numpy(dtype=float)

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
        _add_metric_box(lines)

        regline_path = out_path / "scatter_with_model_line.png"
        plt.savefig(regline_path, dpi=200, bbox_inches="tight")
        plt.close()

    # d) Optional: predictions_test.csv residuals (fix alignment by dropping NA jointly)
    test_plot_path = None
    if preds_test_path.exists():
        preds_df = pd.read_csv(preds_test_path)

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
            tmp = pd.DataFrame(
                {
                    "y_true": _safe_float_series(preds_df[y_true_col]),
                    "y_hat": _safe_float_series(preds_df[y_hat_col]),
                }
            ).dropna()

            if len(tmp) > 0:
                y_true_t = tmp["y_true"].to_numpy(dtype=float)
                y_hat_t = tmp["y_hat"].to_numpy(dtype=float)
                res_t = y_true_t - y_hat_t

                test_plot_path = out_path / "residuals_test.png"
                _plot_residuals_with_smooth(
                    x=y_hat_t,
                    residuals=res_t,
                    xlabel="Predicted (test)",
                    ylabel="Residual (test)",
                    title="Test residuals (from predictions_test.csv)",
                    out_file=test_plot_path,
                )

    # e) Optional: coefficients plot if linear model
    coef_path = None
    if hasattr(model, "coef_"):
        try:
            coefs = np.array(getattr(model, "coef_"), dtype=float).reshape(-1)
            if len(coefs) == len(features):
                order = np.argsort(np.abs(coefs))[::-1]
                plt.figure()
                plt.bar(np.array(features)[order], coefs[order])
                plt.xticks(rotation=45, ha="right")
                plt.xlabel("Feature")
                plt.ylabel("Coefficient")
                plt.title("Linear model coefficients")
                coef_path = out_path / "coefficients.png"
                plt.savefig(coef_path, dpi=200, bbox_inches="tight")
                plt.close()
        except Exception:
            coef_path = None

    return {
        "rows_used": int(len(df)),
        "run_dir": str(run_dir),
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "scatter_path": str(scatter_path) if scatter_path else None,
        "hist_path": str(hist_path) if hist_path else None,
        "pred_vs_actual_path": str(pred_vs_actual_path),
        "residuals_path": str(residuals_path),
        "pred_vs_actual_log_path": str(pred_vs_actual_log_path) if pred_vs_actual_log_path else None,
        "residuals_log_path": str(residuals_log_path) if residuals_log_path else None,
        "regline_path": str(regline_path) if regline_path else None,
        "residuals_test_path": str(test_plot_path) if test_plot_path else None,
        "coefficients_path": str(coef_path) if coef_path else None,
        "features": features,
        "target": target,
        "metrics_test": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create plots for the dataset + model-based visuals.")
    parser.add_argument("--in", dest="in_csv", default=_default_processed_path(), help="Input processed CSV path")
    parser.add_argument("--out", dest="out_dir", default=_default_figures_dir(), help="Output directory for figures")

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

    if result["pred_vs_actual_log_path"]:
        print(f"- predicted vs actual (log): {result['pred_vs_actual_log_path']}")
    if result["residuals_log_path"]:
        print(f"- residuals (log): {result['residuals_log_path']}")

    if result["regline_path"]:
        print(f"- scatter + model line: {result['regline_path']}")
    if result["residuals_test_path"]:
        print(f"- residuals (test file): {result['residuals_test_path']}")
    if result["coefficients_path"]:
        print(f"- coefficients: {result['coefficients_path']}")


if __name__ == "__main__":
    main()
