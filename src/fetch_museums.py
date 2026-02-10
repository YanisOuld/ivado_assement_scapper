from __future__ import annotations

import re
from io import StringIO
from typing import Optional
from pathlib import Path

import pandas as pd
import requests

from src.config import MIN_VISITORS, RAW_MUSEUMS_CSV, WIKIPEDIA_URL, USER_AGENT_HEADERS


def _parse_visits(value: object) -> Optional[int]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    s = str(value).strip().lower()
    if s in {"-", "—", "n/a", "na", ""}:
        return None

    s = re.sub(r"\[[^\]]*\]", "", s).strip()

    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:million|m)\b", s)
    if m:
        return int(round(float(m.group(1)) * 1_000_000))

    m = re.search(r"\d{1,3}(?:,\d{3})+", s)
    if m:
        return int(m.group(0).replace(",", ""))

    m = re.search(r"\b\d+\b", s)
    if m:
        return int(m.group(0))

    return None


def fetch_museums_from_wikipedia(url: str = WIKIPEDIA_URL) -> pd.DataFrame:
    resp = requests.get(url, headers=USER_AGENT_HEADERS, timeout=30)
    resp.raise_for_status()

    tables = pd.read_html(StringIO(resp.text), header=0)

    best_table = None
    best_score = -1

    for t in tables:
        if isinstance(t.columns, pd.MultiIndex):
            t.columns = [" ".join(map(str, col)).strip() for col in t.columns]

        cols = [str(c).lower() for c in t.columns]
        score = 0

        if any("museum" in c or "name" in c for c in cols):
            score += 2
        if any("visitor" in c or "attendance" in c for c in cols):
            score += 2
        if any("city" in c or "location" in c for c in cols):
            score += 1

        if score > best_score:
            best_score = score
            best_table = t

    if best_table is None or best_score == 0:
        raise RuntimeError("No suitable museum table found on Wikipedia page.")

    df = best_table.copy()

    col_map = {}
    museum_col = visitor_col = city_col = False

    for c in df.columns:
        cl = str(c).lower()

        if not museum_col and ("museum" in cl or "name" in cl):
            col_map[c] = "museum_name"
            museum_col = True
        elif not visitor_col and ("visitor" in cl or "attendance" in cl):
            col_map[c] = "annual_visitors_raw"
            visitor_col = True
        elif not city_col and ("city" in cl or "location" in cl):
            col_map[c] = "city"
            city_col = True

    df = df.rename(columns=col_map)

    required = {"museum_name", "city", "annual_visitors_raw"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"Expected columns {required}, got {set(df.columns)}")

    df = df[["museum_name", "city", "annual_visitors_raw"]].copy()
    df["annual_visitors"] = df["annual_visitors_raw"].apply(_parse_visits)

    df = df.dropna(subset=["annual_visitors"])
    df = df[df["annual_visitors"] > MIN_VISITORS]

    df["museum_name"] = df["museum_name"].astype(str).str.strip()
    df["city"] = df["city"].astype(str).str.strip()

    df = df.drop_duplicates(subset=["museum_name", "city"])
    df = df[["museum_name", "city", "annual_visitors"]].reset_index(drop=True)

    return df


def main() -> None:
    df = fetch_museums_from_wikipedia(WIKIPEDIA_URL)

    output_path = Path(RAW_MUSEUMS_CSV)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
