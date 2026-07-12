"""Build and clean the California ZIP-year housing panel (Deliverable 1).

One row = one California ZIP code observed in one base year, with the realized five-year
home-price growth that followed. See ../DATASET_DESIGN.md for the full specification.

    python scripts/build_dataset.py                  # base years 2016-2020 -> raw + clean panel
    python scripts/build_dataset.py --base-years 2020

Outputs
    data/processed/ca_zip_housing_panel.csv         raw panel, before cleaning
    data/processed/ca_zip_housing_panel_clean.csv   the deliverable; D2/D3/D4 load this

Requires the ACS files from scripts/fetch_acs.py and the Zillow/FRED files in data/raw
(see README for download instructions).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

ZHVI = RAW / "Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
ZORI = RAW / "Zip_zori_uc_sfrcondomfr_sm_month.csv"
FRED = RAW / "MORTGAGE30US.csv"

LISTINGS = {
    "for_sale_inventory": "Zip_invt_fs_uc_sfrcondo_sm_month.csv",
    "new_listings": "Zip_new_listings_uc_sfrcondo_sm_month.csv",
    "price_cut_pct": "Zip_perc_listings_price_cut_uc_sfrcondo_sm_month.csv",
    "median_list_price": "Zip_mlp_uc_sfrcondo_sm_month.csv",
}
# Every listings series begins 2018-03. For base years 2016-2017 these columns are
# structurally absent, not missing -- they are left null and never imputed.
LISTINGS_START_YEAR = 2018

# ACS features that get median-imputed within base year. The derived ratios
# (price_to_income, price_to_rent) are recomputed *after* imputation rather than imputed
# directly, so a ratio is always consistent with the income/rent it was built from.
ACS_IMPUTE_COLS = [
    "median_household_income",
    "median_gross_rent",
    "population",
    "housing_units",
    "unemployment_rate",
    "bachelor_degree_pct",
    "population_growth_pct",
]

CLEAN_COLUMNS = [
    # identifiers / context
    "zip_code", "base_year", "city", "county",
    # core model features (12)
    "current_home_value", "home_value_1yr_growth_pct", "home_value_3yr_growth_pct",
    "median_household_income", "median_gross_rent", "population", "population_growth_pct",
    "unemployment_rate", "bachelor_degree_pct", "mortgage_rate",
    "price_to_income_ratio", "price_to_rent_ratio",
    # supplementary
    "housing_units", "zori_rent", "zori_rent_available",
    # listings block (base years 2018-2020 only)
    "inventory_per_1k_housing_units", "new_listings_per_1k_housing_units",
    "price_cut_pct", "list_to_value_ratio", "listings_available",
    # outlier flags -- rows are flagged, never dropped
    "is_growth_outlier", "is_value_outlier",
    # targets (future_5yr_home_value is the target restated: drop before modeling)
    "future_5yr_home_value", "future_5yr_growth_pct",
]


def dec(year: int) -> str:
    return f"{year}-12-31"


def load_zillow_ca(path: Path) -> pd.DataFrame:
    # dtype=str on RegionName preserves leading zeros. Irrelevant in CA, correct everywhere else.
    df = pd.read_csv(path, dtype={"RegionName": str})
    df = df[df["State"] == "CA"].copy()
    df["RegionName"] = df["RegionName"].str.zfill(5)
    return df.set_index("RegionName")


def mortgage_rate_by_year() -> pd.Series:
    """Annual mean of the weekly national 30-year fixed rate."""
    f = pd.read_csv(FRED, parse_dates=["observation_date"])
    f = f.rename(columns={"observation_date": "date", "MORTGAGE30US": "rate"})
    return f.groupby(f["date"].dt.year)["rate"].mean()


def load_acs(year: int) -> pd.DataFrame:
    path = RAW / "acs" / f"acs5_{year}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: CENSUS_API_KEY=... python scripts/fetch_acs.py"
        )
    return pd.read_csv(path, dtype={"zip_code": str}).set_index("zip_code")


def build_base_year(year: int, zhvi, zori, listings, rates) -> pd.DataFrame:
    """Assemble every California ZIP for a single base year."""
    t, prev, t3, fut = dec(year), dec(year - 1), dec(year - 3), dec(year + 5)

    for col, label in [(t, "base"), (prev, "prior"), (fut, "outcome")]:
        if col not in zhvi.columns:
            raise ValueError(f"base_year {year}: ZHVI missing {label} month {col}")

    # A row exists only if we can compute both the base value and the realized outcome.
    # Only the anchor months are used -- we never melt all 300+ monthly columns.
    keep = zhvi[t].notna() & zhvi[prev].notna() & zhvi[fut].notna()
    z = zhvi[keep]

    df = pd.DataFrame(index=z.index)
    df.index.name = "zip_code"
    df["base_year"] = year
    df["city"] = z["City"]
    df["county"] = z["CountyName"]

    df["current_home_value"] = z[t]
    df["future_5yr_home_value"] = z[fut]
    df["home_value_1yr_growth_pct"] = (z[t] / z[prev] - 1) * 100
    df["home_value_3yr_growth_pct"] = (
        (z[t] / z[t3] - 1) * 100 if t3 in z.columns else np.nan
    )
    df["future_5yr_growth_pct"] = (z[fut] / z[t] - 1) * 100

    # ZORI: supplementary only. Coverage is ~28% in 2020 and ~15% in 2016, and the
    # missingness is not random (it skews to large, liquid rental markets), so it is never
    # imputed and never used as a core model feature. ACS median gross rent fills that role.
    df["zori_rent"] = zori[t].reindex(df.index) if t in zori.columns else np.nan
    df["zori_rent_available"] = df["zori_rent"].notna().astype(int)

    # ACS: levels at t; population also at t-5, for a non-overlapping 5-year growth window.
    acs = load_acs(year).reindex(df.index)
    acs_lag = load_acs(year - 5).reindex(df.index)

    df["median_household_income"] = acs["B19013_001E"]
    df["median_gross_rent"] = acs["B25064_001E"]
    df["population"] = acs["B01003_001E"]
    df["housing_units"] = acs["B25001_001E"]

    labor_force = acs["B23025_003E"].replace(0, np.nan)
    df["unemployment_rate"] = acs["B23025_005E"] / labor_force * 100

    adults = acs["B15003_001E"].replace(0, np.nan)
    bachelor_plus = acs[["B15003_022E", "B15003_023E", "B15003_024E", "B15003_025E"]].sum(
        axis=1, min_count=1
    )
    df["bachelor_degree_pct"] = bachelor_plus / adults * 100

    pop_lag = acs_lag["B01003_001E"].replace(0, np.nan)
    df["population_growth_pct"] = (acs["B01003_001E"] / pop_lag - 1) * 100

    df["mortgage_rate"] = rates.get(year, np.nan)

    # Listings block: 2018+ only. Counts are normalized by housing units, otherwise they
    # merely proxy for how large the ZIP code is.
    per_1k = df["housing_units"].replace(0, np.nan) / 1000
    listing_cols = [
        "for_sale_inventory", "new_listings", "price_cut_pct", "median_list_price",
        "inventory_per_1k_housing_units", "new_listings_per_1k_housing_units",
        "list_to_value_ratio",
    ]
    if year >= LISTINGS_START_YEAR:
        for name, lf in listings.items():
            df[name] = lf[t].reindex(df.index) if t in lf.columns else np.nan
        df["inventory_per_1k_housing_units"] = df["for_sale_inventory"] / per_1k
        df["new_listings_per_1k_housing_units"] = df["new_listings"] / per_1k
        df["list_to_value_ratio"] = df["median_list_price"] / df["current_home_value"]
    else:
        for name in listing_cols:
            df[name] = np.nan

    return df.reset_index()


def build_raw_panel(base_years: list[int]) -> pd.DataFrame:
    zhvi = load_zillow_ca(ZHVI)
    zori = load_zillow_ca(ZORI)
    listings = {k: load_zillow_ca(RAW / v) for k, v in LISTINGS.items()}
    rates = mortgage_rate_by_year()

    frames = [build_base_year(y, zhvi, zori, listings, rates) for y in sorted(base_years)]
    panel = pd.concat(frames, ignore_index=True)

    # A duplicate (zip, year) can only come from a join bug, so fail loudly rather than
    # silently double-weighting a ZIP code.
    dupes = panel.duplicated(subset=["zip_code", "base_year"]).sum()
    if dupes:
        raise AssertionError(f"{dupes} duplicate (zip_code, base_year) rows -- join bug")

    return panel


def flag_outliers(s: pd.Series) -> pd.Series:
    """IQR flag, computed within a base year.

    Within-year, because the panel pools five different market regimes: a 40% five-year gain
    is unremarkable for a 2020 base year (pandemic boom) and extreme for 2016. A pooled bound
    would flag an entire base year rather than genuinely unusual ZIP codes.
    """
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)


def clean_panel(panel: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Apply the D1 cleaning strategy. Every decision is justified in the README."""
    df = panel.copy()
    n0 = len(df)

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    # 1. Census null sentinels. fetch_acs.py already converts these, but assert it here too:
    #    a surviving -666666666 would silently wreck every mean, correlation and box plot.
    sentinels = [-666666666, -999999999, -888888888, -222222222, -333333333, -555555555]
    num = df.select_dtypes(include="number")
    n_sent = int(num.isin(sentinels).sum().sum())
    if n_sent:
        raise AssertionError(f"{n_sent} Census null sentinels survived into the panel")
    log(f"1. Census null sentinels found: {n_sent}")

    # 2. Impossible values -> NaN, so they are handled as missing instead of being modeled.
    impossible = {
        "current_home_value": df["current_home_value"] <= 0,
        "median_household_income": df["median_household_income"] <= 0,
        "median_gross_rent": df["median_gross_rent"] <= 0,
        "population": df["population"] < 0,
        "unemployment_rate": (df["unemployment_rate"] < 0) | (df["unemployment_rate"] > 100),
        "bachelor_degree_pct": (df["bachelor_degree_pct"] < 0) | (df["bachelor_degree_pct"] > 100),
    }
    for col, mask in impossible.items():
        n = int(mask.sum())
        if n:
            df.loc[mask, col] = np.nan
        log(f"2. impossible values nulled in {col:<26} {n}")

    # 3. Drop rows with no target and no base value. The target is never imputed -- imputing
    #    the thing you predict manufactures the answer. current_home_value is the target's
    #    denominator, so a row without it has no recoverable target either.
    before = len(df)
    df = df.dropna(subset=["future_5yr_growth_pct", "current_home_value"])
    log(f"3. rows dropped (missing target or current_home_value): {before - len(df)}")

    # 4. ACS features: median-impute *within base year*. A 2016 median income is not a 2020
    #    median income; imputing from a pooled median would inject the pandemic-era level
    #    into 2016 rows and bias the year-over-year comparison.
    for col in ACS_IMPUTE_COLS:
        n = int(df[col].isna().sum())
        df[col] = df.groupby("base_year")[col].transform(lambda s: s.fillna(s.median()))
        log(f"4. median-imputed within base year: {col:<26} {n}")

    # 4b. home_value_3yr_growth_pct is missing (~2%) wherever Zillow's index for that ZIP does
    #     not reach back three years -- an artifact of when the series starts, unrelated to the
    #     housing market, so it is close to missing-at-random. It is a core feature, so it gets
    #     the same within-base-year median imputation rather than being left for D2 to trip on.
    n = int(df["home_value_3yr_growth_pct"].isna().sum())
    df["home_value_3yr_growth_pct"] = df.groupby("base_year")["home_value_3yr_growth_pct"].transform(
        lambda s: s.fillna(s.median())
    )
    log(f"4b. median-imputed within base year: home_value_3yr_growth_pct  {n}")

    # 5. Derived ratios recomputed after imputation, so each ratio is consistent with the
    #    income/rent actually stored in the row.
    df["price_to_income_ratio"] = (
        df["current_home_value"] / df["median_household_income"].replace(0, np.nan)
    )
    df["price_to_rent_ratio"] = (
        df["current_home_value"] / (df["median_gross_rent"].replace(0, np.nan) * 12)
    )

    # 6. Listings: structurally absent before 2018. The availability flag exists for
    #    diagnostics and filtering only -- DATASET_DESIGN.md §6.9 forbids using it as a
    #    predictor, since it measures Zillow's reporting behaviour, not the housing market.
    df["listings_available"] = (
        df["base_year"].ge(LISTINGS_START_YEAR)
        & df["inventory_per_1k_housing_units"].notna()
    ).astype(int)

    # 7. Outliers: FLAGGED, NOT DROPPED. A $4M Atherton ZIP is not a data error, it is
    #    Atherton. Deleting legitimate high-value markets destroys real signal, so every row
    #    is kept and D2 decides what to do with the flag.
    df["is_growth_outlier"] = (
        df.groupby("base_year")["future_5yr_growth_pct"].transform(flag_outliers).astype(int)
    )
    df["is_value_outlier"] = (
        df.groupby("base_year")["current_home_value"].transform(flag_outliers).astype(int)
    )
    log(f"7. growth outliers flagged: {int(df['is_growth_outlier'].sum())}  "
        f"(rows dropped: 0)")
    log(f"   value  outliers flagged: {int(df['is_value_outlier'].sum())}  "
        f"(rows dropped: 0)")

    df = df[CLEAN_COLUMNS].reset_index(drop=True)

    # Post-conditions -- the definition of done, enforced in code.
    assert df["future_5yr_growth_pct"].notna().all(), "null target survived"
    assert df["current_home_value"].notna().all(), "null current_home_value survived"
    assert not df.duplicated(["zip_code", "base_year"]).any(), "duplicate (zip, year)"
    core = ACS_IMPUTE_COLS + [
        "price_to_income_ratio", "price_to_rent_ratio",
        "home_value_1yr_growth_pct", "home_value_3yr_growth_pct", "mortgage_rate",
    ]
    assert df[core].notna().all().all(), "null survived in a core feature"
    log(f"\nclean panel: {len(df)} rows x {df.shape[1]} columns  (from {n0} raw rows)")
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-years", nargs="+", type=int, default=[2016, 2017, 2018, 2019, 2020])
    args = ap.parse_args()

    PROC.mkdir(parents=True, exist_ok=True)

    panel = build_raw_panel(args.base_years)
    raw_out = PROC / "ca_zip_housing_panel.csv"
    panel.to_csv(raw_out, index=False)
    print(f"Wrote {raw_out.relative_to(ROOT)}  ({len(panel)} rows x {panel.shape[1]} cols)\n")

    print("Cleaning")
    print("-" * 60)
    clean = clean_panel(panel)
    clean_out = PROC / "ca_zip_housing_panel_clean.csv"
    clean.to_csv(clean_out, index=False)
    print(f"\nWrote {clean_out.relative_to(ROOT)}")
    print(f"  rows per base year:\n{clean['base_year'].value_counts().sort_index().to_string()}")

    remaining = clean.isna().sum()
    remaining = remaining[remaining > 0].sort_values(ascending=False)
    print("\n  columns still holding nulls (by design):")
    for c, n in remaining.items():
        print(f"    {c:<36} {n:>6}  ({n / len(clean) * 100:5.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
