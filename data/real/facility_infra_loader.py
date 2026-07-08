"""
PHC/CHC/SDH/DH Facility Infrastructure Loader
------------------------------------------------
Loads real state-level public health facility counts and total bed capacity
(source: national health infrastructure statistics) and derives two things
the simulator needs to stop guessing:

1. Real state-wise facility-type mix (PHC : CHC : SDH : DH ratio) — so our
   simulated district facility rosters reflect the real skew (e.g. Bihar has
   ~32x more PHCs than CHCs; Kerala's CHC network is proportionally much
   denser).

2. Fitted average beds-per-facility-type — India doesn't publish a clean
   "beds per PHC" figure standalone (the source only gives total beds across
   ALL facility types per state). We recover it with a non-negative least
   squares fit:

        total_beds_s ≈ b_phc * n_phc_s + b_chc * n_chc_s
                        + b_sdh * n_sdh_s + b_dh * n_dh_s   for each state s

   solved across all 36 states simultaneously. This is a legitimate,
   defensible estimate grounded in real data — not a guess — and it should
   land close to the known IPHS (Indian Public Health Standards) benchmarks
   of ~6 beds/PHC and ~30 beds/CHC, which we can sanity-check against.
"""

import pandas as pd
import numpy as np
from scipy.optimize import nnls
import os

BEDS_CSV = os.path.join(os.path.dirname(__file__), "phc_chc_beds.csv")

FACILITY_COLS = {
    "No. of Public facilities - PHC": "n_phc",
    "No. of Public facilities - CHC": "n_chc",
    "No. of Public facilities - SDH": "n_sdh",
    "No. of Public facilities - DH": "n_dh",
    "No. of beds available in public facilities": "total_beds",
}


def load_facility_data(path: str = BEDS_CSV) -> pd.DataFrame:
    import sys
    sys.path.append(os.path.dirname(__file__))
    from state_utils import canonical_state

    df = pd.read_csv(path)
    df = df.rename(columns=FACILITY_COLS)
    df = df.rename(columns={"States/UTs": "state"})
    df = df[df["state"].str.lower() != "all india"].copy()
    for col in ["n_phc", "n_chc", "n_sdh", "n_dh", "total_beds"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["state"] = df["state"].apply(canonical_state)
    df = df.groupby("state", as_index=False)[["n_phc", "n_chc", "n_sdh", "n_dh", "total_beds"]].sum()
    return df.reset_index(drop=True)


def fit_bed_norms(df: pd.DataFrame) -> dict:
    """Non-negative least squares fit of average beds per facility type,
    using all states simultaneously."""
    A = df[["n_phc", "n_chc", "n_sdh", "n_dh"]].values
    b = df["total_beds"].values
    coeffs, residual = nnls(A, b)
    return {
        "beds_per_phc": round(coeffs[0], 1),
        "beds_per_chc": round(coeffs[1], 1),
        "beds_per_sdh": round(coeffs[2], 1),
        "beds_per_dh": round(coeffs[3], 1),
        "fit_residual": round(residual, 1),
    }


def state_facility_mix(df: pd.DataFrame) -> pd.DataFrame:
    """Real PHC:CHC ratio and facility density per state, for realistic
    district-level facility roster generation."""
    out = df.copy()
    out["phc_to_chc_ratio"] = (out["n_phc"] / out["n_chc"].replace(0, np.nan)).round(1)
    out["total_facilities"] = out["n_phc"] + out["n_chc"] + out["n_sdh"] + out["n_dh"]
    out["beds_per_facility_avg"] = (out["total_beds"] / out["total_facilities"].replace(0, np.nan)).round(1)
    return out[["state", "n_phc", "n_chc", "n_sdh", "n_dh", "total_beds",
                "phc_to_chc_ratio", "beds_per_facility_avg"]]


if __name__ == "__main__":
    df = load_facility_data()
    norms = fit_bed_norms(df)

    print("Fitted national average beds per facility type (NNLS regression across 36 states):")
    for k, v in norms.items():
        print(f"  {k}: {v}")
    print("\n  (Sanity check vs IPHS official norms: PHC=6 beds, CHC=30 beds - "
          "fitted values should be in a plausible neighborhood of these, since "
          "real facilities vary around the standard.)")

    mix = state_facility_mix(df)
    print("\nState-wise facility mix (sample):")
    print(mix.sort_values("phc_to_chc_ratio", ascending=False).head(10).to_string(index=False))

    out_dir = os.path.dirname(__file__)
    mix.to_csv(os.path.join(out_dir, "state_facility_mix.csv"), index=False)
    pd.Series(norms).to_csv(os.path.join(out_dir, "fitted_bed_norms.csv"))
    print("\nSaved -> state_facility_mix.csv, fitted_bed_norms.csv")
