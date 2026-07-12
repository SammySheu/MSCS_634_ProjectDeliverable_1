# MSCS 634 — Project Deliverable 1

## Data Collection, Cleaning, and Exploration
### Factors Associated with Five-Year Home Price Growth in California

**Name:** Chun-Shuo Sheu
**Course:** MSCS 634 — Big Data and Data Mining
**Deliverable:** 1 of 4 — Data Collection, Cleaning, and Exploration

---

## What this deliverable produces

One artifact:

> **`data/processed/ca_zip_housing_panel_clean.csv` — 7,487 rows × 28 columns**

Deliverables 2, 3 and 4 load this file and **never re-clean it**. Every cleaning decision is made
once, here, and justified in writing.

| | |
|---|---|
| **Unit of analysis** | One California ZIP code observed in one base year |
| **Records** | **7,487** (5 base years × ~1,500 ZIP codes) |
| **Core model features** | **12** (28 columns total, incl. identifiers, targets, supplementary columns and flags) |
| **Target** | `future_5yr_growth_pct` — % change in home value over the *following five years* |
| **Base years** | 2016, 2017, 2018, 2019, 2020 |
| **Outcome windows** | 2021, 2022, 2023, 2024, 2025 — **all fully elapsed** |

Because every outcome window has already happened, the target is a **measured outcome, not a
forecast**. The analysis can be validated against what actually occurred.

### Relationship to Lab 1

Lab 1 built the `base_year = 2020` slice (1,528 rows). D1 generalizes that into a loop over base
years 2016–2020 — the same logic, five times, stacked. **Lab 1 is not a separate dataset; it is a
single-year slice of this panel.**

---

## Unit of analysis and base-year design

Each row asks: *given what this ZIP code looked like at the end of `base_year`, what happened to its
home prices over the next five years?*

| zip_code | base_year | current_home_value | … | future_5yr_growth_pct |
|---|---|---|---|---|
| 94533 | 2016 | 310,000 | … | 41.2 |
| 94533 | 2020 | 402,000 | … | 38.6 |

**December is the anchor month.** Zillow publishes monthly; December is an unambiguous year-end that
avoids depending on the most recent, still-revising month.

For each base year *t*:

| Block | Anchors |
|---|---|
| Home values | ZHVI at Dec(t−3), Dec(t−1), Dec(t), Dec(t+5) → level, 1-yr and 3-yr momentum, and the target |
| Demographics | ACS 5-Year at *t* and *t−5* → income, rent, population growth, unemployment, education, housing units |
| Macro | FRED annual mean of the weekly 30-year fixed rate for year *t* |
| Derived | `price_to_income_ratio`, `price_to_rent_ratio` |
| Listings | attached for t ∈ {2018, 2019, 2020}; **null for 2016–2017** |

A row exists only if ZHVI is present at Dec(t−1), Dec(t) **and** Dec(t+5) — i.e. only if both the base
value and the realized outcome can actually be computed.

### Records per base year

| Base year | Rows |
|---|---|
| 2016 | 1,452 |
| 2017 | 1,496 |
| 2018 | 1,500 |
| 2019 | 1,511 |
| 2020 | 1,528 |
| **Total** | **7,487** |

---

## Target variable

```python
future_5yr_growth_pct = (future_5yr_home_value / current_home_value - 1) * 100
```

**Why growth rather than the future price.** If the target were the future price,
`current_home_value` would dominate every model — a $1M home is still expensive in five years. The
model would post a high R² while explaining nothing. Predicting *growth* forces the features to
explain **why** prices move. The dollar forecast is recovered afterwards:

```python
predicted_future_value = current_home_value * (1 + predicted_5yr_growth_pct / 100)
```

> ⚠️ `future_5yr_home_value` is retained for reference but **must be dropped before modeling.**
> It is the target restated, and leaving it in the feature matrix is direct leakage.

---

## Data sources

| Source | Contributes | Access |
|---|---|---|
| **Zillow ZHVI** (ZIP, smoothed, seasonally adj.) | Home values, monthly, 2000-01 → 2026-05 | Manual CSV download |
| **Zillow ZORI** (ZIP, smoothed) | Observed asking rent — supplementary only | Manual CSV download |
| **Zillow listings** (inventory, new listings, price cuts, median list price) | Market liquidity, 2018-03 → 2026-05 | `files.zillowstatic.com` |
| **US Census ACS 5-Year** (ZCTA) | Income, rent, population, housing units, unemployment, education | API — **requires a free key** |
| **FRED `MORTGAGE30US`** | 30-year fixed mortgage rate, weekly | Public CSV, no key |

### Features

**Core model features (12)**

| # | Feature | Source |
|---|---|---|
| 1 | `current_home_value` | ZHVI Dec(t) |
| 2 | `home_value_1yr_growth_pct` | ZHVI Dec(t−1) → Dec(t) |
| 3 | `home_value_3yr_growth_pct` | ZHVI Dec(t−3) → Dec(t) |
| 4 | `median_household_income` | ACS `B19013_001E` |
| 5 | `median_gross_rent` | ACS `B25064_001E` |
| 6 | `population` | ACS `B01003_001E` |
| 7 | `population_growth_pct` | ACS `B01003_001E`, t−5 → t |
| 8 | `unemployment_rate` | ACS `B23025_005E / B23025_003E` |
| 9 | `bachelor_degree_pct` | ACS `B15003` (022+023+024+025) / 001 |
| 10 | `mortgage_rate` | FRED, annual mean of weekly |
| 11 | `price_to_income_ratio` | derived — affordability |
| 12 | `price_to_rent_ratio` | derived — buy-vs-rent |

**Supplementary:** `zori_rent`, `zori_rent_available`, `housing_units`, `city`, `county`.
**Listings block (2018–2020 only):** `inventory_per_1k_housing_units`,
`new_listings_per_1k_housing_units`, `price_cut_pct`, `list_to_value_ratio`, `listings_available`.
**Flags:** `is_growth_outlier`, `is_value_outlier`.

Raw inventory and new-listing *counts* are normalized by housing units — otherwise they merely proxy
for how big the ZIP code is.

---

## Cleaning decisions — and the reason for each

| Issue | Decision | Why |
|---|---|---|
| ZIP codes read as int | Force `dtype=str` | Preserves leading zeros |
| Wide Zillow format | Melt **only** the anchor months | Melting 300+ month columns creates millions of rows we would discard |
| Duplicate ZIP-year rows | Assert none exist; fail loudly | A duplicate means a join bug, not a data quirk |
| Impossible values | Home value ≤ 0, income ≤ 0, unemployment outside 0–100% → null | A negative rate is not a measurement; treat as missing rather than model it |
| ACS `-666666666` | Convert to `NaN` | Census's null sentinel. Left in place it silently destroys every statistic |
| **Missing target** | **Drop the row** (0 rows) | **Never impute the thing you are predicting** |
| Missing `current_home_value` | Drop the row (0 rows) | It is the target's *denominator*, so the target is unrecoverable without it |
| Missing ACS features | Median-impute **within base year** | A 2016 median income is not a 2020 median income |
| Missing 3-yr momentum (2.2%) | Median-impute within base year | Missing only where Zillow's index doesn't reach back 3 years — a series-start artifact |
| Derived ratios | Recompute **after** imputation | Keeps each ratio consistent with the income/rent stored in the row |
| ZORI (78% missing) | Keep as-is + availability flag | Do not impute a column that is mostly absent, and non-randomly so |
| Listings, 2016–2017 | Leave null, never impute | **Structurally absent, not missing** |
| Outliers | **Flag, do not delete** | See below |

### Missing values — three different problems, three different responses

Treating these identically would be the most damaging mistake available here.

1. **Structurally absent — never impute.** The listings columns are 54–77% "missing" in the raw
   panel, but mostly because **Zillow's listings series does not begin until 2018-03.** For a 2016
   row there is no value to estimate. Imputing across that boundary would not fill a gap; it would
   *invent 2016 listings data that never existed.*
2. **Non-random and severe — flag, never impute.** ZORI rent is **78% missing**, and the missingness
   skews to large, liquid rental markets — exactly the markets that also grew fastest. Imputing it
   would smear big-city rent levels across rural ZIPs and manufacture a correlation. It is kept with
   a `zori_rent_available` flag, and **ACS `median_gross_rent` is the rent variable the models use.**
3. **Genuinely missing — median-impute within base year.** ACS gaps are small and scattered:
   `median_gross_rent` 409 rows, `median_household_income` 172, everything else ~35.

**Imputation is done within base year, never pooled.** A pooled median would inject pandemic-era
levels into 2016 rows.

Even within 2018–2020, listings coverage is thin: `inventory` reaches 75% of rows but `new_listings`
only **38%**.

> **Zero missing values in `future_5yr_growth_pct` and `current_home_value`.** The two columns that
> must never be missing are not.

### Outliers — flagged, not deleted

**This is the single most important judgment call in D1.**

- **236 growth outliers (3.2%)** and **450 value outliers (6.0%)** flagged.
- **0 rows deleted.**

A $6.5M Atherton ZIP code is **not a data error. It is Atherton** — the most expensive ZIP code in
the United States. The value-outlier group averages **$2.02M vs $543k** for everyone else. These are
legitimate high-value coastal markets, and they are *precisely* the markets the affordability
hypothesis is about. **Deleting them would delete the signal the project is trying to measure.**

Growth outliers also sit at **both** extremes — crashes *and* booms — and are spread across the full
price range. They are unusual **outcomes**, not bad **data**.

So the pipeline computes IQR bounds, adds `is_growth_outlier` / `is_value_outlier`, keeps every row,
and lets D2 make an informed modeling choice **and report the difference**. The bounds are computed
**within base year**, because a 40% five-year gain is unremarkable from a 2020 base and extreme from
a 2016 base; a pooled bound would flag an entire base year rather than genuinely unusual ZIPs.

---

## EDA findings

Ten figures, each with a written interpretation, are in the notebook and in `figures/`.

### 1. Affordability constrains growth — the strongest signal in the dataset

`price_to_rent_ratio` (**r = −0.305**) and `price_to_income_ratio` (**r = −0.286**) are the two
features most correlated with future growth, both **negative**. A market already priced beyond what
local incomes can carry has less room left to run.

Crucially, the effect **holds inside every single base year** (−0.19 to −0.41), so it is a genuine
cross-sectional relationship, not a macro coincidence. Both are *engineered* features that beat their
own ingredients (income alone: −0.14; home value alone: −0.23) — the best argument in this dataset for
ratio-building over raw columns.

Honest caveat: r ≈ −0.3 is a **moderate** effect, explaining ~9% of variance on its own.

### 2. Income tells you where homes are expensive — not where they will appreciate

corr(income, home **value**) = **+0.73**. corr(income, future **growth**) = **−0.14**.
Prosperity is a *level* effect, not a *growth* effect.

### 3. Price momentum is nearly useless at a five-year horizon

`home_value_1yr_growth_pct` r = **+0.08**; `home_value_3yr_growth_pct` r = **+0.10**. The most
intuitive feature in real estate — "it's been going up, so it'll keep going up" — has almost no
five-year predictive power.

### 4. The affluence features are severely collinear — which is why D2 uses Ridge

| Pair | r |
|---|---|
| `median_household_income` ↔ `median_gross_rent` | **+0.84** |
| `bachelor_degree_pct` ↔ `median_household_income` | **+0.76** |
| `current_home_value` ↔ `bachelor_degree_pct` | **+0.75** |

Income, rent, education and home value all measure one latent thing: *how affluent this ZIP is*.
Under plain OLS the coefficients become unstable and can flip sign under resampling — the
*explanation* becomes worthless even when predictions are fine. **Ridge's L2 penalty shares weight
across correlated predictors instead of arbitrarily crowning one.** This is an empirical response to
the +0.84, not a stylistic preference.

### 5. The target does not need a log transform — and that is a finding

Skew is only **+0.31**, and mean (41.2%) ≈ median (42.1%). The conventional log transform for a growth
variable would buy essentially nothing. The real distributional issue is **fat tails (excess kurtosis
3.81)**, and the response to fat tails is *robustness*, not a transform. The **features**, however,
are strongly right-skewed (`current_home_value`, `median_household_income`, `population`) — a log
transform on the feature side is worth testing in D2.

### 6. Every base year is a different regime — and this is the leakage trap

Median five-year growth falls monotonically: **55% → 49% → 40% → 40% → 26%** from base year 2016 to
2020. Each successive base year captures *less* of the 2020–2022 pandemic surge. `base_year` is a
first-order variable, not an index column.

### 7. No single dominant predictor

Nothing exceeds |r| = 0.31. Five-year growth is genuinely multivariate. A realistic D2 expectation is
**R² ≈ 0.3–0.5**; an R² near 0.9 should be read as **evidence of leakage** (most likely
`future_5yr_home_value` left in the feature matrix), not as success.

---

## Known limitations

State these plainly. They are not disclaimers — they change how the results must be read.

1. **Overlapping outcome windows.** Base years 2016–2020 produce outcome windows 2021–2025 that
   overlap heavily; the 2016 and 2017 rows for a given ZIP share four years of price history.
   **7,487 rows are not 7,487 independent observations.** A random train/test split would leak badly.
   **Mitigation: D2 must split by base year, never randomly.**
2. **The mortgage rate is national.** Every ZIP in a given year receives the identical value, so it
   cannot explain why one ZIP outgrew another *within* a year — only why one year differed from
   another. With five base years it is effectively **five data points**, perfectly confounded with
   `base_year`. Its r = +0.158 with the target is a `base_year` effect in disguise. Treat it as a
   market-cycle control, never as a star predictor.
3. **All five outcome windows contain the pandemic housing boom.** The model can only ever learn
   "what predicted growth *during a boom*." This constrains how far the D2 2029 forecast can be
   trusted, and no amount of cross-validation fixes it.
4. **The 2029 forecast requires extrapolation.** Training mortgage rates span **3.11–4.54%**; the 2024
   forecast input is ~**6.7%** — outside the training range entirely. **Random Forests cannot
   extrapolate**; they silently clamp to the edge of what they have seen. Mitigation: D2 produces the
   forecast both with and without the rate feature and discusses the divergence.
5. **ZIP ≠ ZCTA.** Zillow uses postal ZIP codes; the Census uses ZIP Code Tabulation Areas. These are
   similar but not identical geographies. Additionally, ACS 2011–2019 use 2010-vintage ZCTAs while
   2020+ use 2020-vintage ZCTAs, so a small number of boundaries shift mid-panel.
6. **ACS 5-year estimates are 5-year averages**, carrying sampling error and a built-in lag. Growth
   computed between adjacent releases is heavily smoothed — which is why population growth uses
   non-overlapping t−5 → t windows.
7. **ZORI rent is severely and non-randomly missing** (78% of panel rows). Coverage skews toward
   large, liquid rental markets. This is why ACS median gross rent is the modeling variable.
8. **ZHVI is an index, not a transaction record.** It estimates *typical* home value; it is not the
   actual sale price of any specific house.
9. **For-sale listings coverage is non-random and correlated with the outcome.** ZIPs with complete
   listings data have ~4× the median population and grew ~10 points faster — Zillow reports listings
   for large, active markets. Any listings-based analysis is therefore conditional on big-city
   markets, and a listings-*availability* flag must **never** be used as a predictor: it measures
   Zillow's reporting behaviour, not the housing market, and is leakage by proxy. Use `population` to
   capture the real urban-scale effect instead.

---

## Repository layout

```
notebooks/Project_Deliverable_1.ipynb     the deliverable notebook (run top to bottom)
scripts/fetch_acs.py                      pulls + caches ACS 5-Year by ZCTA
scripts/build_dataset.py                  builds the raw panel, then cleans it
scripts/make_notebook.py                  regenerates the notebook from source
data/raw/                                 gitignored — see "Reproducing" below
data/processed/ca_zip_housing_panel.csv        raw panel, pre-cleaning
data/processed/ca_zip_housing_panel_clean.csv  ** the deliverable **
figures/                                  the 10 EDA figures
```

**Raw Zillow files are gitignored** (ZHVI alone is 117 MB). The processed California subset is
committed. Re-download instructions are below.

---

## Reproducing

```bash
pip install -r requirements.txt

# 1. Download the raw files into data/raw/
#    Zillow (https://www.zillow.com/research/data/):
#      Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv   (ZHVI, ZIP, smoothed + seas. adj.)
#      Zip_zori_uc_sfrcondomfr_sm_month.csv                  (ZORI, ZIP, smoothed)
#      Zip_invt_fs_uc_sfrcondo_sm_month.csv                  (for-sale inventory)
#      Zip_new_listings_uc_sfrcondo_sm_month.csv             (new listings)
#      Zip_perc_listings_price_cut_uc_sfrcondo_sm_month.csv  (% listings with a price cut)
#      Zip_mlp_uc_sfrcondo_sm_month.csv                      (median list price)
#    FRED (https://fred.stlouisfed.org/series/MORTGAGE30US):
#      MORTGAGE30US.csv

# 2. Fetch the ACS data (a free Census API key is REQUIRED --
#    the API rejects keyless requests with X-DataWebAPI-KeyError)
export CENSUS_API_KEY=your_key      # https://api.census.gov/data/key_signup.html
python scripts/fetch_acs.py

# 3. Build + clean the panel
python scripts/build_dataset.py

# 4. Run the notebook
jupyter notebook notebooks/Project_Deliverable_1.ipynb
```
