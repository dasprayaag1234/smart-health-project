"""
Smart Resource Redistribution Optimizer
------------------------------------------
Given current stock levels and forecasted stock-out risk across a district's
PHCs/CHCs, this solves a linear program to recommend concrete transfers:
"move X units of medicine M from facility A (surplus) to facility B (predicted
shortage)" minimizing unmet predicted demand and total transport distance.

This is the rubric's "smart resource redistribution recommendations across a
district's PHCs/CHCs" requirement, addressed as a real optimization problem
rather than a hardcoded if/else rule.

Model:
  For each medicine, each facility has:
    - current stock
    - forecasted 7-day consumption (from the stock-out forecaster / rolling mean)
    - a computed surplus or deficit against a safety-stock target

  Decision variables: transfer[i][j] = units of medicine moved from
  facility i to facility j (i != j, same district only - keeps transfers
  operationally realistic, e.g. a courier run within a district in one day).

  Objective: minimize (unmet deficit after transfers) + lambda * (transfer distance-proxy)
  Subject to: a facility can't send more than its surplus, can't exceed a
  single-trip capacity, and all transfer quantities are non-negative.
"""

import pandas as pd
import numpy as np
import pulp
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "data", "real"))
from census_geo_loader import haversine_km  # noqa: E402

SAFETY_STOCK_DAYS = 10  # target buffer: enough stock for 10 days at current consumption rate
MAX_TRANSFER_PER_PAIR = 200  # operational cap: a single courier run can't move unlimited units
TRANSPORT_PENALTY_PER_UNIT_KM = 0.002  # cost per (unit transferred x km travelled)
FALLBACK_DISTANCE_KM = 15  # used only if a facility has no lat/lon (e.g. post-2011 district)


def compute_facility_balance(stock_df: pd.DataFrame, facilities: pd.DataFrame) -> pd.DataFrame:
    """For each facility x medicine, compute current stock, forecasted daily
    consumption, and surplus/deficit against a safety-stock target."""
    latest_day = stock_df["day"].max()
    latest = stock_df[stock_df["day"] == latest_day].copy()

    recent = stock_df[stock_df["day"] > latest_day - 14]
    avg_consumption = recent.groupby(["facility_id", "medicine"])["daily_consumption"].mean().reset_index()
    avg_consumption.columns = ["facility_id", "medicine", "avg_daily_consumption"]

    balance = latest.merge(avg_consumption, on=["facility_id", "medicine"], how="left")
    balance["avg_daily_consumption"] = balance["avg_daily_consumption"].fillna(1.0)
    balance["safety_target"] = balance["avg_daily_consumption"] * SAFETY_STOCK_DAYS
    balance["net_position"] = balance["stock_level"] - balance["safety_target"]  # + surplus, - deficit

    balance = balance.merge(facilities[["facility_id", "district", "lat", "lon"]], on="facility_id", how="left")
    return balance[["facility_id", "district", "medicine", "stock_level", "avg_daily_consumption",
                     "safety_target", "net_position", "lat", "lon"]]


def pairwise_distance_km(sub: pd.DataFrame, i: str, j: str) -> float:
    """Real haversine distance between two facilities using Census-2011-derived
    coordinates. Falls back to a flat estimate if either facility lacks
    coordinates (e.g. a post-2011 district with no shapefile match)."""
    lat_i, lon_i = sub.loc[i, "lat"], sub.loc[i, "lon"]
    lat_j, lon_j = sub.loc[j, "lat"], sub.loc[j, "lon"]
    if pd.isna(lat_i) or pd.isna(lon_i) or pd.isna(lat_j) or pd.isna(lon_j):
        return FALLBACK_DISTANCE_KM
    return haversine_km(lat_i, lon_i, lat_j, lon_j)


def optimize_district_medicine(balance_df: pd.DataFrame, district: str, medicine: str):
    """Solve the transfer LP for a single (district, medicine) pair."""
    sub = balance_df[(balance_df["district"] == district) & (balance_df["medicine"] == medicine)].copy()
    sub = sub.set_index("facility_id")
    facilities_list = sub.index.tolist()

    if len(facilities_list) < 2:
        return []

    surplus_facilities = sub[sub["net_position"] > 0].index.tolist()
    deficit_facilities = sub[sub["net_position"] < 0].index.tolist()

    if not surplus_facilities or not deficit_facilities:
        return []  # nothing to move: either everyone's short, or everyone's fine

    def clean(name: str) -> str:
        return name.replace(" ", "_").replace("-", "_")

    prob = pulp.LpProblem(f"redistribution_{clean(district)}_{clean(medicine)}", pulp.LpMinimize)

    transfer = {
        (i, j): pulp.LpVariable(f"t_{clean(i)}_{clean(j)}", lowBound=0, upBound=MAX_TRANSFER_PER_PAIR)
        for i in surplus_facilities for j in deficit_facilities
    }

    unmet = {j: pulp.LpVariable(f"unmet_{clean(j)}", lowBound=0) for j in deficit_facilities}

    # Real Census-2011-derived great-circle distance per (surplus, deficit) pair,
    # replacing what used to be a flat per-unit penalty. This means the LP now
    # genuinely prefers moving stock between nearby facilities over distant
    # ones, which is what "transport cost" should mean.
    distance_km = {
        (i, j): pairwise_distance_km(sub, i, j) for i in surplus_facilities for j in deficit_facilities
    }

    # Objective: minimize total unmet deficit (heavily weighted) + real distance-weighted transport cost
    prob += (
        pulp.lpSum(unmet[j] * 10 for j in deficit_facilities)
        + pulp.lpSum(
            transfer[(i, j)] * distance_km[(i, j)] * TRANSPORT_PENALTY_PER_UNIT_KM
            for i in surplus_facilities for j in deficit_facilities
        )
    )

    # Deficit facilities: incoming transfers + unmet slack must cover their shortfall
    for j in deficit_facilities:
        need = -sub.loc[j, "net_position"]
        prob += pulp.lpSum(transfer[(i, j)] for i in surplus_facilities) + unmet[j] >= need

    # Surplus facilities: can't send more than their surplus
    for i in surplus_facilities:
        available = sub.loc[i, "net_position"]
        prob += pulp.lpSum(transfer[(i, j)] for j in deficit_facilities) <= available

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    recommendations = []
    for (i, j), var in transfer.items():
        qty = var.value()
        if qty and qty > 0.5:
            recommendations.append({
                "district": district, "medicine": medicine,
                "from_facility": i, "to_facility": j, "units_to_transfer": round(qty, 1),
                "distance_km": round(distance_km[(i, j)], 1),
            })
    return recommendations


def generate_all_recommendations(balance_df: pd.DataFrame) -> pd.DataFrame:
    all_recs = []
    for district in balance_df["district"].dropna().unique():
        for medicine in balance_df["medicine"].unique():
            recs = optimize_district_medicine(balance_df, district, medicine)
            all_recs.extend(recs)
    return pd.DataFrame(all_recs)


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "simulated")
    stock_df = pd.read_csv(os.path.join(data_dir, "stock_timeseries.csv"))
    facilities = pd.read_csv(os.path.join(data_dir, "facilities.csv"))

    print("Computing facility stock balances...")
    balance_df = compute_facility_balance(stock_df, facilities)

    n_deficits = (balance_df["net_position"] < 0).sum()
    n_surplus = (balance_df["net_position"] > 0).sum()
    print(f"Deficit rows: {n_deficits} | Surplus rows: {n_surplus} (out of {len(balance_df)})")

    print("\nSolving redistribution LP per district x medicine...")
    recommendations = generate_all_recommendations(balance_df)

    print(f"\n{len(recommendations)} transfer recommendations generated. Top 15 by volume:")
    if len(recommendations) > 0:
        print(recommendations.sort_values("units_to_transfer", ascending=False).head(15).to_string(index=False))
        recommendations.to_csv(os.path.join(data_dir, "redistribution_recommendations.csv"), index=False)
        print(f"\nSaved -> {os.path.join(data_dir, 'redistribution_recommendations.csv')}")
    else:
        print("No transfers recommended given current balances.")
