"""Fetch ACS 5-Year estimates by ZIP Code Tabulation Area, cached per year.

Requires a Census API key (the API rejects keyless requests with X-DataWebAPI-KeyError):
    https://api.census.gov/data/key_signup.html

    export CENSUS_API_KEY=your_key_here
    python scripts/fetch_acs.py

Years 2011-2015 pull population only -- they exist solely to compute the non-overlapping
5-year population growth for base years 2016-2020. Full variables are pulled for the base
years themselves plus 2024 (the forecast input for Deliverable 2).
"""
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw" / "acs"
BASE_YEARS = [2016, 2017, 2018, 2019, 2020]
POP_ONLY_YEARS = [2011, 2012, 2013, 2014, 2015]
FORECAST_YEARS = [2019, 2024]

FULL_VARS = [
    "B19013_001E",  # median household income
    "B25064_001E",  # median gross rent
    "B01003_001E",  # total population
    "B25001_001E",  # total housing units
    "B23025_003E",  # civilian labor force
    "B23025_005E",  # unemployed
    "B15003_001E",  # population 25+
    "B15003_022E",  # bachelor's
    "B15003_023E",  # master's
    "B15003_024E",  # professional degree
    "B15003_025E",  # doctorate
]
POP_VARS = ["B01003_001E"]

# The Census uses these as null sentinels. Left in place they silently destroy every statistic.
NULL_SENTINELS = [-666666666, -999999999, -888888888, -222222222, -333333333, -555555555]


def fetch_year(year: int, variables: list[str], key: str) -> pd.DataFrame:
    """Pull one ACS 5-year vintage for all ZCTAs nationally.

    ZCTAs stopped being nestable within a state in the API from 2019 on, so we always pull
    nationally and filter to California ZIPs downstream against the Zillow ZIP list.
    """
    resp = requests.get(
        f"https://api.census.gov/data/{year}/acs/acs5",
        params={
            "get": ",".join(variables),
            "for": "zip code tabulation area:*",
            "key": key,
        },
        timeout=180,
    )
    resp.raise_for_status()
    if not resp.text.lstrip().startswith("["):
        raise RuntimeError(f"{year}: non-JSON response (first 200 chars): {resp.text[:200]}")

    rows = resp.json()
    df = pd.DataFrame(rows[1:], columns=rows[0])

    zcta_col = next(c for c in df.columns if "zip code" in c.lower())
    df = df.rename(columns={zcta_col: "zip_code"})
    df["zip_code"] = df["zip_code"].astype(str).str.zfill(5)

    for v in variables:
        df[v] = pd.to_numeric(df[v], errors="coerce")
        df.loc[df[v].isin(NULL_SENTINELS), v] = pd.NA

    df["acs_year"] = year
    return df[["zip_code", "acs_year"] + variables]


def main() -> int:
    key = os.environ.get("CENSUS_API_KEY", "").strip()
    if not key:
        print(
            "ERROR: CENSUS_API_KEY is not set.\n"
            "  Get a free key: https://api.census.gov/data/key_signup.html\n"
            "  Then: export CENSUS_API_KEY=your_key_here",
            file=sys.stderr,
        )
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    plan = {y: POP_VARS for y in POP_ONLY_YEARS}
    for y in sorted(set(BASE_YEARS + FORECAST_YEARS)):
        plan[y] = FULL_VARS

    for year in sorted(plan):
        dest = OUT / f"acs5_{year}.csv"
        if dest.exists():
            print(f"  {year}  cached, skipping")
            continue
        try:
            df = fetch_year(year, plan[year], key)
        except Exception as exc:  # noqa: BLE001 - report and continue to the next vintage
            print(f"  {year}  FAILED: {exc}", file=sys.stderr)
            continue
        df.to_csv(dest, index=False)
        print(f"  {year}  {len(df):>6} ZCTAs, {len(plan[year])} vars -> {dest.name}")
        time.sleep(1)  # be polite to the API

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
