"""
NFHS-5 Loader & District Health Vulnerability Index (DHVI)
------------------------------------------------------------
Loads the two real NFHS-5 datasets (district-level and state factsheet-level)
and produces a cleaned, analysis-ready table with one composite score per
district: the District Health Vulnerability Index (DHVI).

Why this matters for the hackathon:
NFHS-5 does NOT contain live operational data (no stock levels, no daily
footfall, no bed occupancy, no doctor attendance logs) - it's a household
health survey. So it can't directly power the real-time dashboard.

What it CAN do, and what we use it for:
  - Give every district a real, defensible "health burden" baseline
  - Seed the operational data simulator so simulated PHC/CHC stress levels
    are grounded in actual survey indicators, not arbitrary randomness
  - Act as a static prior feature in the risk-scoring / flagging model
    (a district with historically poor institutional-birth rates and high
    anaemia burden is a priori more likely to need intervention)

NFHS data quirks handled here:
  - '*'  -> suppressed (based on <25 unweighted cases) -> treated as NaN
  - negative numbers on percentage columns (e.g. -92.6) -> NFHS's raw export
    uses a leading '-' to flag "based on 25-49 unweighted cases" (small but
    non-zero sample). The magnitude is still the real estimate, so we take
    the absolute value and additionally emit a `_low_confidence` flag.
"""

import pandas as pd
import os
import numpy as np
import re

DISTRICT_CSV = os.path.join(os.path.dirname(__file__), "nfhs5_district.csv")
FACTSHEET_CSV = "nfhs5_factsheet.csv"

# Columns from the DISTRICT file relevant to health-centre operations.
# (Exact header strings copied from the source file.)
RELEVANT_COLS = {
    "District Names": "district",
    "State/UT": "state",
    "Population living in households with electricity (%)": "electricity_pct",
    "Population living in households with an improved drinking-water source1 (%)": "water_pct",
    "Households with any usual member covered under a health insurance/financing scheme (%)": "insurance_pct",
    "Mothers who had at least 4 antenatal care visits  (for last birth in the 5 years before the survey) (%)": "anc4_pct",
    "Institutional births (in the 5 years before the survey) (%)": "institutional_births_pct",
    "Institutional births in public facility (in the 5 years before the survey) (%)": "public_institutional_births_pct",
    "Children age 12-23 months fully vaccinated based on information from either vaccination card or mother's recall11 (%)": "full_immunization_pct",
    "Average out-of-pocket expenditure per delivery in a public health facility (for last birth in the 5 years before the survey) (Rs.)": "oop_delivery_cost_rs",
    "Prevalence of diarrhoea in the 2 weeks preceding the survey (Children under age 5 years) (%) ": "diarrhoea_prevalence_pct",
    "All women age 15-49 years who are anaemic22 (%)": "women_anaemia_pct",
    "Children age 6-59 months who are anaemic (<11.0 g/dl)22 (%)": "child_anaemia_pct",
}

# Indicators where a HIGHER value means BETTER health access
# (used with inverted sign in the vulnerability index)
POSITIVE_INDICATORS = [
    "electricity_pct", "water_pct", "insurance_pct", "anc4_pct",
    "institutional_births_pct", "public_institutional_births_pct",
    "full_immunization_pct",
]

# Indicators where a HIGHER value means WORSE health burden
NEGATIVE_INDICATORS = [
    "oop_delivery_cost_rs", "diarrhoea_prevalence_pct",
    "women_anaemia_pct", "child_anaemia_pct",
]


def _clean_numeric(series: pd.Series):
    """Convert an NFHS raw column to numeric, handling '*' and small-sample
    negative-sign flags. Returns (clean_values, low_confidence_flag)."""
    s = series.astype(str).str.strip()

    def is_low_conf(v):
        v = v.strip()
        if not v.startswith("-"):
            return False
        rest = v[1:].replace(".", "", 1)
        return rest.isdigit()

    low_conf = s.apply(is_low_conf)

    def parse(v):
        v = v.strip()
        if v in ("*", "", "nan", "NaN", "-"):
            return np.nan
        v = re.sub(r"[^0-9.\-]", "", v)
        if v in ("", "-", "."):
            return np.nan
        try:
            return abs(float(v))  # take magnitude; sign is a sample-size flag, not a real negative
        except ValueError:
            return np.nan

    clean = s.apply(parse)
    return clean, low_conf


def load_district_data(path: str = DISTRICT_CSV) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw.columns = [c.strip() for c in raw.columns]

    df = pd.DataFrame()
    for src_col, new_col in RELEVANT_COLS.items():
        src_col_stripped = src_col.strip()
        matches = [c for c in raw.columns if c.strip() == src_col_stripped]
        if not matches:
            matches = [c for c in raw.columns if c.strip()[:40] == src_col_stripped[:40]]
        if not matches:
            print(f"WARNING: column not found, skipping: {src_col[:60]}...")
            continue
        col = matches[0]
        if new_col in ("district", "state"):
            df[new_col] = raw[col].astype(str).str.strip()
        else:
            clean, low_conf = _clean_numeric(raw[col])
            df[new_col] = clean
            df[f"{new_col}_low_confidence"] = low_conf

    return df


def compute_dhvi(df: pd.DataFrame) -> pd.DataFrame:
    """Compute District Health Vulnerability Index (0-100, higher = more
    vulnerable / more in need of intervention & resources)."""
    df = df.copy()
    z_components = []

    for col in POSITIVE_INDICATORS:
        if col not in df.columns:
            continue
        mu, sigma = df[col].mean(), df[col].std()
        if sigma and not np.isnan(sigma):
            z = (df[col] - mu) / sigma
            z_components.append(-z)

    for col in NEGATIVE_INDICATORS:
        if col not in df.columns:
            continue
        mu, sigma = df[col].mean(), df[col].std()
        if sigma and not np.isnan(sigma):
            z = (df[col] - mu) / sigma
            z_components.append(z)

    if not z_components:
        raise ValueError("No indicators available to compute DHVI")

    stacked = pd.concat(z_components, axis=1)
    raw_score = stacked.mean(axis=1, skipna=True)

    df["dhvi_raw"] = raw_score
    rmin, rmax = raw_score.min(), raw_score.max()
    df["dhvi_score"] = ((raw_score - rmin) / (rmax - rmin) * 100).round(1)

    df["dhvi_tier"] = pd.cut(
        df["dhvi_score"],
        bins=[-0.1, 25, 50, 75, 100.1],
        labels=["Low", "Moderate", "High", "Critical"],
    )
    return df


def load_and_score(path: str = DISTRICT_CSV) -> pd.DataFrame:
    df = load_district_data(path)
    df = df.dropna(subset=["district"])
    df = df[df["district"].str.lower() != "nan"]
    df = compute_dhvi(df)
    return df.sort_values("dhvi_score", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    result = load_and_score()
    print(f"Loaded {len(result)} districts")
    print("\nTop 10 most vulnerable districts (highest DHVI):")
    print(result[["district", "state", "dhvi_score", "dhvi_tier"]].head(10).to_string(index=False))
    print("\nTop 10 least vulnerable districts (lowest DHVI):")
    print(result[["district", "state", "dhvi_score", "dhvi_tier"]].tail(10).to_string(index=False))

    out_path = os.path.join(os.path.dirname(__file__), "district_health_vulnerability_index.csv")
    result.to_csv(out_path, index=False)
    print(f"\nSaved full scored dataset -> {out_path}")
