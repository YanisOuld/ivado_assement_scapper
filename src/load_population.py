from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import POPULATION_RAW_CSV, POPULATION_PROCESSED_CSV


def _strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )


def _clean_text(x: object) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _parse_population(value: object) -> Optional[int]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    s = str(value).strip().lower()
    if s in {"", "-", "—", "n/a", "na"}:
        return None

    s = re.sub(r"\[[^\]]*\]", "", s).strip()

    digits = re.sub(r"[^\d]", "", s)
    if digits.isdigit():
        return int(digits)

    return None


def load_population_dataset(raw_path: str = POPULATION_RAW_CSV) -> pd.DataFrame:
    raw_file = Path(raw_path)
    if not raw_file.exists():
        raise FileNotFoundError(f"Population raw CSV not found: {raw_file}")

    # Some datasets are latin-1; try utf-8 first, fallback if needed.
    try:
        df = pd.read_csv(raw_file)
    except UnicodeDecodeError:
        df = pd.read_csv(raw_file, encoding="latin-1")

    required_cols = {"Country", "City", "Population"}
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns {missing}. Found: {list(df.columns)}")

    out = df[["Country", "City", "Population"]].copy()
    out.columns = ["country", "city", "population_raw"]

    out["country"] = out["country"].apply(_clean_text)
    out["city"] = out["city"].apply(_clean_text)
    out["population"] = out["population_raw"].apply(_parse_population)

    out = out.dropna(subset=["country", "city", "population"])
    out = out[(out["country"] != "") & (out["city"] != "") & (out["population"] > 0)].copy()

    # Normalized keys for merges
    out["country_norm"] = out["country"].apply(lambda x: _strip_accents(x).lower())
    out["city_norm"] = out["city"].apply(lambda x: _strip_accents(x).lower())

    # Deduplicate (keep max population per city/country)
    out = (
        out.groupby(["country_norm", "city_norm"], as_index=False)
        .agg(country=("country", "first"), city=("city", "first"), population=("population", "max"))
        .sort_values(["country", "city"])
        .reset_index(drop=True)
    )

    return out


def main() -> None:
    df = load_population_dataset(POPULATION_RAW_CSV)

    out_path = Path(POPULATION_PROCESSED_CSV)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
