"""
PHC/CHC Operational Data Simulator
------------------------------------
NFHS-5 gives us a real, per-district Health Vulnerability Index (DHVI), but
no live operational data. This module generates realistic facility-level
time series (stock, footfall, beds, doctor attendance, test availability)
seeded by DHVI so the simulation is grounded in real district health burden
instead of pure randomness.

Design principle: higher DHVI (more vulnerable district) -> higher baseline
footfall pressure, higher stock-out probability, lower doctor attendance
reliability, lower bed availability, more test-equipment downtime. This
mirrors what NFHS's own components (poor institutional birth rates, high
anaemia, low insurance coverage) actually imply about facility strain.

Usage:
    python simulator.py --days 90 --districts 15 --facilities-per-district 4
"""

import numpy as np
import pandas as pd
import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "real"))
from nfhs_loader import load_and_score  # noqa: E402
from who_eml_loader import load_formulary  # noqa: E402
from facility_infra_loader import load_facility_data, fit_bed_norms, state_facility_mix  # noqa: E402
from state_utils import infra_lookup_state  # noqa: E402
from census_geo_loader import load_district_centroids, resolve_district_name  # noqa: E402

RNG_SEED = 42
DEFAULT_FORMULARY_SIZE = 40  # keep the demo dataset a manageable, realistic PHC formulary size


def sample_formulary(tier: str, n: int, rng) -> pd.DataFrame:
    """Sample a realistic-sized formulary, weighted toward higher-criticality
    medicines (mirrors how a real PHC's core stock list is dominated by
    high-priority items with a smaller tail of supportive medicines)."""
    full = load_formulary(
        os.path.join(os.path.dirname(__file__), "..", "real", "who_essential_medicines.xlsx"), tier=tier
    )
    weights = full["criticality"] ** 2  # criticality 3 items ~9x more likely than criticality 1
    idx = rng.choice(full.index, size=min(n, len(full)), replace=False, p=weights / weights.sum())
    return full.loc[idx].reset_index(drop=True)


def pick_facilities(dhvi_df: pd.DataFrame, n_districts: int, facilities_per_district: int, rng,
                     bed_norms: dict, mix_df: pd.DataFrame, geo_df: pd.DataFrame = None):
    """Sample a mix of districts across vulnerability tiers so the demo dataset
    isn't all-critical or all-low (more realistic and better for showing the
    redistribution/flagging logic working across a spectrum).

    Facility type mix and bed counts are drawn from each district's real
    state-level PHC:CHC ratio and fitted bed-per-facility-type norms, rather
    than a fixed 1-CHC-per-district assumption."""
    tiers = dhvi_df["dhvi_tier"].unique()
    per_tier = max(1, n_districts // len(tiers))
    sampled = []
    for t in tiers:
        subset = dhvi_df[dhvi_df["dhvi_tier"] == t]
        take = min(per_tier, len(subset))
        sampled.append(subset.sample(take, random_state=rng.integers(0, 1e6)))
    districts = pd.concat(sampled).head(n_districts).reset_index(drop=True)

    mix_lookup = mix_df.set_index("state")
    national_phc_chc_ratio = (mix_df["n_phc"].sum() / mix_df["n_chc"].sum())

    geo_lookup = geo_df.set_index("district") if geo_df is not None else None

    facilities = []
    for _, row in districts.iterrows():
        infra_state = infra_lookup_state(row["state"])
        if infra_state in mix_lookup.index:
            ratio = mix_lookup.loc[infra_state, "phc_to_chc_ratio"]
            ratio = ratio if pd.notna(ratio) else national_phc_chc_ratio
        else:
            ratio = national_phc_chc_ratio  # no matching state infra data -> fall back to national average

        # expected number of CHCs in a facilities_per_district-sized roster, given the real ratio
        n_chc = max(1, round(facilities_per_district / (1 + ratio)))
        n_chc = min(n_chc, facilities_per_district - 1)  # always leave room for at least one PHC

        # real district centroid (Census 2011), if available -> facilities are
        # placed at small realistic offsets around it rather than left un-located
        district_lat, district_lon = None, None
        resolved_name = resolve_district_name(row["district"])
        if geo_lookup is not None and resolved_name in geo_lookup.index:
            geo_row = geo_lookup.loc[resolved_name]
            district_lat, district_lon = geo_row["centroid_lat"], geo_row["centroid_lon"]

        for i in range(facilities_per_district):
            ftype = "CHC" if i < n_chc else "PHC"
            mean_beds = bed_norms["beds_per_chc"] if ftype == "CHC" else bed_norms["beds_per_phc"]
            total_beds = max(2, int(round(rng.normal(mean_beds, mean_beds * 0.2))))

            lat, lon = None, None
            if district_lat is not None:
                # jitter within ~0-6km of the real district centroid so facilities
                # within a district are distinct points, not stacked on one pin
                jitter_deg = 0.03  # ~3km at these latitudes
                lat = round(district_lat + rng.uniform(-jitter_deg, jitter_deg), 5)
                lon = round(district_lon + rng.uniform(-jitter_deg, jitter_deg), 5)

            facilities.append({
                "facility_id": f"{row['district'][:4].upper().replace(' ', '')}-{ftype}-{i+1}",
                "facility_type": ftype,
                "district": row["district"],
                "state": row["state"],
                "dhvi_score": row["dhvi_score"],
                "dhvi_tier": row["dhvi_tier"],
                "total_beds": total_beds,
                "lat": lat,
                "lon": lon,
            })
    return pd.DataFrame(facilities)


def simulate_stock(facilities: pd.DataFrame, formulary: pd.DataFrame, days: int, rng) -> pd.DataFrame:
    records = []
    for _, f in facilities.iterrows():
        vuln = f["dhvi_score"] / 100.0
        for _, med_row in formulary.iterrows():
            med = med_row["medicine_name"]
            crit = med_row["criticality"]  # 1-3, from real WHO EML-derived criticality
            crit_factor = crit / 2.0  # normalize around 1.0 for criticality=2

            base_stock = rng.integers(200, 600)
            daily_consumption_mean = rng.uniform(5, 20) * (1 + vuln) * crit_factor
            stock = base_stock
            resupply_cycle = rng.integers(10, 21)
            for day in range(days):
                consumption = max(0, rng.normal(daily_consumption_mean, daily_consumption_mean * (0.3 + vuln * 0.4)))
                stock -= consumption
                if day % resupply_cycle == 0 and day > 0:
                    # higher-criticality medicines get prioritized resupply even in vulnerable districts
                    resupply_reliability = 1 - (vuln * 0.5) * (1 - 0.15 * (crit - 1))
                    if rng.random() < resupply_reliability:
                        stock += rng.integers(150, 400)
                stock = max(0, stock)
                records.append({
                    "facility_id": f["facility_id"], "day": day, "medicine": med,
                    "criticality": crit, "stock_level": round(stock, 1),
                    "daily_consumption": round(consumption, 1),
                })
    return pd.DataFrame(records)


def simulate_footfall(facilities: pd.DataFrame, days: int, rng) -> pd.DataFrame:
    records = []
    for _, f in facilities.iterrows():
        vuln = f["dhvi_score"] / 100.0
        base_footfall = rng.integers(40, 90) * (1 + vuln * 0.6)
        for day in range(days):
            weekday = day % 7
            weekend_dip = 0.7 if weekday >= 5 else 1.0
            seasonal = 1 + 0.15 * np.sin(2 * np.pi * day / 30)  # monthly wave (e.g. outbreak cycles)
            outbreak_spike = 1.8 if rng.random() < (0.02 + vuln * 0.03) else 1.0
            footfall = max(0, rng.poisson(base_footfall * weekend_dip * seasonal * outbreak_spike))
            records.append({"facility_id": f["facility_id"], "day": day, "patient_footfall": int(footfall)})
    return pd.DataFrame(records)


def simulate_beds(facilities: pd.DataFrame, days: int, rng) -> pd.DataFrame:
    records = []
    for _, f in facilities.iterrows():
        vuln = f["dhvi_score"] / 100.0
        total_beds = int(f["total_beds"])  # from real state-calibrated bed norms (see pick_facilities)
        base_occupancy_rate = min(0.95, 0.4 + vuln * 0.45)
        for day in range(days):
            occ_rate = np.clip(rng.normal(base_occupancy_rate, 0.1), 0, 1)
            occupied = int(round(occ_rate * total_beds))
            records.append({
                "facility_id": f["facility_id"], "day": day, "total_beds": total_beds,
                "occupied_beds": occupied, "occupancy_rate": round(occupied / total_beds, 2),
            })
    return pd.DataFrame(records)


def simulate_doctor_attendance(facilities: pd.DataFrame, days: int, rng) -> pd.DataFrame:
    records = []
    for _, f in facilities.iterrows():
        vuln = f["dhvi_score"] / 100.0
        rostered_doctors = rng.integers(2, 5) if f["facility_type"] == "CHC" else rng.integers(1, 3)
        base_attendance_rate = max(0.5, 0.95 - vuln * 0.35)  # more vulnerable -> more absenteeism
        for day in range(days):
            weekday = day % 7
            weekend_factor = 0.85 if weekday >= 5 else 1.0
            attendance_rate = np.clip(rng.normal(base_attendance_rate * weekend_factor, 0.08), 0, 1)
            present = int(round(attendance_rate * rostered_doctors))
            records.append({
                "facility_id": f["facility_id"], "day": day, "rostered_doctors": rostered_doctors,
                "doctors_present": present, "attendance_rate": round(present / rostered_doctors, 2),
            })
    return pd.DataFrame(records)


def simulate_test_availability(facilities: pd.DataFrame, days: int, rng) -> pd.DataFrame:
    tests = ["Blood Glucose", "Hemoglobin", "Malaria RDT", "Pregnancy Test", "TB Sputum", "X-Ray"]
    records = []
    for _, f in facilities.iterrows():
        vuln = f["dhvi_score"] / 100.0
        for test in tests:
            base_functional_prob = max(0.3, 0.95 - vuln * 0.5)
            for day in range(0, days, 7):  # weekly audit cadence
                functional = rng.random() < base_functional_prob
                records.append({
                    "facility_id": f["facility_id"], "day": day, "test_name": test,
                    "functional": bool(functional),
                })
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--districts", type=int, default=16)
    parser.add_argument("--facilities-per-district", type=int, default=4)
    parser.add_argument("--formulary-size", type=int, default=DEFAULT_FORMULARY_SIZE)
    parser.add_argument("--out-dir", type=str, default=".")
    args = parser.parse_args()

    rng = np.random.default_rng(RNG_SEED)
    real_data_dir = os.path.join(os.path.dirname(__file__), "..", "real")

    dhvi_df = load_and_score(os.path.join(real_data_dir, "nfhs5_district.csv"))

    infra_df = load_facility_data(os.path.join(real_data_dir, "phc_chc_beds.csv"))
    bed_norms = fit_bed_norms(infra_df)
    mix_df = state_facility_mix(infra_df)
    print(f"Using fitted bed norms: PHC={bed_norms['beds_per_phc']}, CHC={bed_norms['beds_per_chc']} "
          f"(NNLS-fit from real state facility data)")

    formulary = sample_formulary(tier="PHC", n=args.formulary_size, rng=rng)
    print(f"Using {len(formulary)} medicines from WHO EML-derived PHC formulary "
          f"(criticality-weighted sample)")

    geo_df = load_district_centroids()
    print(f"Loaded real district centroids (Census 2011) for {len(geo_df)} districts")

    facilities = pick_facilities(dhvi_df, args.districts, args.facilities_per_district, rng, bed_norms, mix_df, geo_df)
    n_located = facilities["lat"].notna().sum()
    print(f"{n_located}/{len(facilities)} facilities got real-geo-anchored coordinates "
          f"({len(facilities) - n_located} districts had no shapefile match and were left unlocated)")

    print(f"Simulating {len(facilities)} facilities across {facilities['district'].nunique()} districts for {args.days} days...")
    print(f"Facility type mix: {facilities['facility_type'].value_counts().to_dict()}")

    stock_df = simulate_stock(facilities, formulary, args.days, rng)
    footfall_df = simulate_footfall(facilities, args.days, rng)
    beds_df = simulate_beds(facilities, args.days, rng)
    doctors_df = simulate_doctor_attendance(facilities, args.days, rng)
    tests_df = simulate_test_availability(facilities, args.days, rng)

    os.makedirs(args.out_dir, exist_ok=True)
    facilities.to_csv(os.path.join(args.out_dir, "facilities.csv"), index=False)
    stock_df.to_csv(os.path.join(args.out_dir, "stock_timeseries.csv"), index=False)
    footfall_df.to_csv(os.path.join(args.out_dir, "footfall_timeseries.csv"), index=False)
    beds_df.to_csv(os.path.join(args.out_dir, "beds_timeseries.csv"), index=False)
    doctors_df.to_csv(os.path.join(args.out_dir, "doctor_attendance_timeseries.csv"), index=False)
    tests_df.to_csv(os.path.join(args.out_dir, "test_availability.csv"), index=False)
    formulary.to_csv(os.path.join(args.out_dir, "active_formulary.csv"), index=False)

    print("Saved: facilities.csv, stock_timeseries.csv, footfall_timeseries.csv, "
          "beds_timeseries.csv, doctor_attendance_timeseries.csv, test_availability.csv, "
          "active_formulary.csv")
    print(f"\nFacility sample:\n{facilities.head(8).to_string(index=False)}")


if __name__ == "__main__":
    main()
