from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


def preprocess_dataset(
    in_csv: str,
    out_csv: str,
    *,
    drop_unmatched: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    df = pd.read_csv(in_csv)

    required = {"museum_name", "city", "annual_visitors"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in input: {missing}. Got: {set(df.columns)}")

    df = df.copy()

    # Ensure numeric
    df["annual_visitors"] = pd.to_numeric(df["annual_visitors"], errors="coerce")

    if "city_population" in df.columns:
        df["city_population"] = pd.to_numeric(df["city_population"], errors="coerce")
    else:
        df["city_population"] = pd.NA

    # Optional geo
    if "Latitude" in df.columns and "Longitude" in df.columns:
        df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
        df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    else:
        df["Latitude"] = pd.NA
        df["Longitude"] = pd.NA

    # Basic cleanup text
    df["museum_name"] = df["museum_name"].astype(str).str.strip()
    df["city"] = df["city"].astype(str).str.strip()

    if "country" in df.columns:
        df["country"] = df["country"].astype(str).str.strip()
    else:
        df["country"] = ""

    # Drop invalid target
    df = df.dropna(subset=["annual_visitors"]).copy()
    df = df[df["annual_visitors"] > 0].copy()

    # Drop unmatched (optional)
    if drop_unmatched:
        df = df.dropna(subset=["city_population"]).copy()
        df = df[df["city_population"] > 0].copy()

    # Feature engineering
    # log1p handles 0 safely; we already filtered >0 when drop_unmatched=True
    df["log_city_population"] = df["city_population"].apply(lambda x: math.log1p(float(x)) if pd.notna(x) else pd.NA)
    df["log_annual_visitors"] = df["annual_visitors"].apply(lambda x: math.log1p(float(x)))

    # Keep only useful columns for modeling + traceability
    keep_cols = [
        "museum_name",
        "city",
        "country",
        "annual_visitors",
        "log_annual_visitors",
        "city_population",
        "log_city_population",
        "Latitude",
        "Longitude",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    # Drop rows where engineered numeric features are missing
    df = df.dropna(subset=["log_annual_visitors"]).copy()
    if drop_unmatched and "log_city_population" in df.columns:
        df = df.dropna(subset=["log_city_population"]).copy()

    # Deduplicate (same museum/city)
    df = df.drop_duplicates(subset=["museum_name", "city", "country"]).reset_index(drop=True)

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    if verbose:
        matched_pop = int(df["city_population"].notna().sum()) if "city_population" in df.columns else 0
        print(f"Rows saved: {len(df)}")
        print(f"Rows with population: {matched_pop}")

    return df


def _default_paths() -> tuple[str, str]:
    in_csv = "data/processed/museums_joined.csv"
    out_csv = "data/processed/museums_processed.csv"

    try:
        from src import config as cfg  # type: ignore

        in_csv = getattr(cfg, "PROCESSED_JOINED_MUSEUMS_CSV", in_csv)
        out_csv = getattr(cfg, "PROCESSED_MUSEUMS_CSV", out_csv)
    except Exception:
        pass

    return in_csv, out_csv


def main() -> None:
    default_in, default_out = _default_paths()

    parser = argparse.ArgumentParser(description="Preprocess museums dataset for modeling.")
    parser.add_argument("--in", dest="in_csv", default=default_in, help="Input joined CSV path")
    parser.add_argument("--out", dest="out_csv", default=default_out, help="Output processed CSV path")
    parser.add_argument("--keep-unmatched", action="store_true", help="Keep rows without population match")
    parser.add_argument("--verbose", action="store_true", help="Print stats")

    args = parser.parse_args()

    preprocess_dataset(
        in_csv=args.in_csv,
        out_csv=args.out_csv,
        drop_unmatched=not args.keep_unmatched,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
