from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd


def _normalize_text(s: object) -> str:
    if s is None:
        return ""
    txt = str(s).strip().lower()
    if not txt:
        return ""

    txt = unicodedata.normalize("NFKD", txt)
    txt = "".join(ch for ch in txt if not unicodedata.combining(ch))

    txt = txt.replace("&", " and ")
    txt = re.sub(r"[/,_\-]+", " ", txt)
    txt = re.sub(r"[^a-z0-9\s]", "", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def _load_museums(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"museum_name", "city", "annual_visitors"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in museums file: {missing}. Got: {set(df.columns)}")

    df = df.copy()
    df["city_norm"] = df["city"].apply(_normalize_text)

    if "country" in df.columns:
        df["country_norm"] = df["country"].apply(_normalize_text)
    else:
        df["country_norm"] = ""

    return df


def _load_population(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Accept either raw format (Country/City/Population) or processed (country/city/population)
    if {"Country", "City", "Population"}.issubset(df.columns):
        df = df.rename(
            columns={"Country": "country", "City": "city", "Population": "population"}
        )

    required = {"country", "city", "population"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in population file: {missing}. Got: {set(df.columns)}")

    df = df.copy()
    df["city_norm"] = df["city"].apply(_normalize_text)
    df["country_norm"] = df["country"].apply(_normalize_text)

    # Optional columns
    if "Latitude" not in df.columns:
        df["Latitude"] = pd.NA
    if "Longitude" not in df.columns:
        df["Longitude"] = pd.NA

    df["population"] = pd.to_numeric(df["population"], errors="coerce")
    return df


def join_museums_population(
    museums_csv: str,
    population_csv: str,
    out_csv: str,
    *,
    drop_unmatched: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    museums = _load_museums(museums_csv)
    pop = _load_population(population_csv)

    has_country = museums["country_norm"].astype(bool).any()

    if has_country:
        merged = museums.merge(
            pop[["country_norm", "city_norm", "population", "Latitude", "Longitude", "country"]],
            on=["country_norm", "city_norm"],
            how="left",
            suffixes=("", "_pop"),
        )
        merged["match_type"] = merged["population"].notna().map(lambda ok: "country_city" if ok else "unmatched")
    else:
        pop_city = pop.sort_values("population", ascending=False).drop_duplicates(subset=["city_norm"])
        merged = museums.merge(
            pop_city[["city_norm", "population", "Latitude", "Longitude", "country"]],
            on="city_norm",
            how="left",
            suffixes=("", "_pop"),
        )
        merged["match_type"] = merged["population"].notna().map(lambda ok: "city_only" if ok else "unmatched")

        # infer country if museums doesn't have it
        if "country" not in merged.columns or merged["country"].isna().all():
            merged["country"] = merged["country_pop"] if "country_pop" in merged.columns else merged.get("country")

    # Make final target column
    merged["city_population"] = merged["population"]
    merged["city_population"] = pd.to_numeric(merged["city_population"], errors="coerce")

    # Optionally drop unmatched
    if drop_unmatched:
        merged = merged.dropna(subset=["city_population"]).copy()

    if verbose:
        total = len(merged)
        matched = int(merged["city_population"].notna().sum())
        unmatched = total - matched
        print(f"Total museums: {total}")
        print(f"Matched: {matched} ({matched/total:.1%})")
        print(f"Unmatched: {unmatched} ({unmatched/total:.1%})")
        print(merged["match_type"].value_counts(dropna=False).to_string())

    # Save
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    return merged


def _default_paths():
    museums_csv = "data/raw/museums_raw.csv"
    population_csv = "data/processed/population.csv"
    out_csv = "data/processed/museums_joined.csv"

    try:
        from src import config as cfg
        museums_csv = getattr(cfg, "RAW_MUSEUMS_CSV", museums_csv)
        population_csv = getattr(cfg, "PROCESSED_POPULATION_CSV", population_csv)
        out_csv = getattr(cfg, "JOINED_MUSEUMS_POPULATION_CSV", out_csv)
    except Exception:
        pass

    return museums_csv, population_csv, out_csv


def main() -> None:
    default_museums, default_pop, default_out = _default_paths()

    parser = argparse.ArgumentParser(description="Join museums dataset with city population dataset.")
    parser.add_argument("--museums", default=default_museums, help="Path to museums_raw.csv")
    parser.add_argument("--population", default=default_pop, help="Path to population.csv")
    parser.add_argument("--out", default=default_out, help="Output CSV path")
    parser.add_argument("--drop-unmatched", action="store_true", help="Drop rows with no population match")
    parser.add_argument("--verbose", action="store_true", help="Print matching stats")

    args = parser.parse_args()

    join_museums_population(
        museums_csv=args.museums,
        population_csv=args.population,
        out_csv=args.out,
        drop_unmatched=args.drop_unmatched,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
