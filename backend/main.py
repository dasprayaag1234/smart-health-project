"""
Smart Health API
-----------------
Thin FastAPI layer over the precomputed pipeline outputs (facility risk
scores, stock-out alerts, redistribution recommendations, facility/district
data). This is what turns "a set of scripts" into "a working prototype link"
for the hackathon submission and gives the frontend something real to call.

Run locally:
    uvicorn main:app --reload --port 8080

The Dockerfile in this folder deploys the same app to Cloud Run.
"""

import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Locally, data lives one level up at ../data/simulated. In the container,
# DATA_DIR_OVERRIDE points at wherever the Dockerfile copied it instead.
DATA_DIR = os.environ.get(
    "DATA_DIR_OVERRIDE",
    os.path.join(os.path.dirname(__file__), "..", "data", "simulated"),
)

app = FastAPI(
    title="Smart Health API",
    description="AI-driven PHC/CHC stock, risk, and redistribution data for district health administrators.",
    version="1.0.0",
)

# Allow the frontend (any origin during hackathon demo) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def df_to_records(df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame to JSON-safe records. Plain json.dumps (used by
    Starlette's default response) rejects NaN, which shows up here for any
    facility missing real coordinates (e.g. post-2011 districts) or any
    other legitimately-missing value — so NaN must become null, not crash
    the endpoint."""
    return df.astype(object).where(pd.notnull(df), None).to_dict(orient="records")


def series_to_dict(s: pd.Series) -> dict:
    """Same NaN -> None fix as df_to_records, for a single row (Series)."""
    return {k: (None if pd.isna(v) else v) for k, v in s.to_dict().items()}


def read_csv_safe(filename: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(
            status_code=503,
            detail=f"{filename} not found — run the data pipeline (see README) before starting the API.",
        )
    return pd.read_csv(path)


@app.get("/")
def root():
    return {
        "service": "Smart Health API",
        "status": "ok",
        "endpoints": [
            "/facilities", "/facilities/geojson", "/districts",
            "/alerts/stockout", "/redistribution", "/risk-scores",
            "/facility/{facility_id}",
        ],
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/facilities")
def get_facilities(district: str | None = None):
    df = read_csv_safe("facilities.csv")
    if district:
        df = df[df["district"].str.lower() == district.lower()]
    return df_to_records(df)


@app.get("/facilities/geojson")
def get_facilities_geojson():
    """GeoJSON FeatureCollection for the map view — one Point feature per
    facility, using the Census-2011-derived coordinates from the simulator."""
    df = read_csv_safe("facilities.csv")
    risk_path = os.path.join(DATA_DIR, "facility_risk_scores.csv")
    risk_df = pd.read_csv(risk_path)[["facility_id", "facility_risk_score", "risk_tier", "top_contributing_factor"]] \
        if os.path.exists(risk_path) else None

    if risk_df is not None:
        df = df.merge(risk_df, on="facility_id", how="left")

    features = []
    for _, row in df.iterrows():
        if pd.isna(row.get("lat")) or pd.isna(row.get("lon")):
            continue  # skip facilities with no real coordinate rather than fake one
        props = series_to_dict(row.drop(labels=["lat", "lon"]))
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [row["lon"], row["lat"]]},
            "properties": props,
        })
    return {"type": "FeatureCollection", "features": features}


@app.get("/districts")
def get_districts():
    df = read_csv_safe("facilities.csv")
    grouped = (
        df.groupby(["district", "state", "dhvi_score", "dhvi_tier"])
        .size()
        .reset_index(name="facility_count")
    )
    return df_to_records(grouped)


@app.get("/alerts/stockout")
def get_stockout_alerts(min_risk: float = 0.5, limit: int = 100):
    df = read_csv_safe("current_stockout_alerts.csv")
    df = df[df["stockout_risk_score"] >= min_risk].sort_values("stockout_risk_score", ascending=False)
    return df_to_records(df.head(limit))


@app.get("/redistribution")
def get_redistribution(district: str | None = None):
    df = read_csv_safe("redistribution_recommendations.csv")
    if district:
        df = df[df["district"].str.lower() == district.lower()]
    return df_to_records(df.sort_values("units_to_transfer", ascending=False))


@app.get("/risk-scores")
def get_risk_scores(tier: str | None = None):
    df = read_csv_safe("facility_risk_scores.csv")
    if tier:
        df = df[df["risk_tier"].str.lower() == tier.lower()]
    return df_to_records(df.sort_values("facility_risk_score", ascending=False))


@app.get("/facility/{facility_id}")
def get_facility_detail(facility_id: str):
    """Everything about one facility: profile, current alerts, risk score,
    and any redistribution moves involving it — the detail view a district
    admin lands on after clicking a map pin or a flagged-facility row."""
    facilities = read_csv_safe("facilities.csv")
    match = facilities[facilities["facility_id"] == facility_id]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Facility '{facility_id}' not found")

    result = series_to_dict(match.iloc[0])

    alerts_path = os.path.join(DATA_DIR, "current_stockout_alerts.csv")
    if os.path.exists(alerts_path):
        alerts = pd.read_csv(alerts_path)
        result["stockout_alerts"] = df_to_records(alerts[alerts["facility_id"] == facility_id])

    risk_path = os.path.join(DATA_DIR, "facility_risk_scores.csv")
    if os.path.exists(risk_path):
        risk = pd.read_csv(risk_path)
        row = risk[risk["facility_id"] == facility_id]
        if not row.empty:
            result["risk"] = series_to_dict(row.iloc[0])

    redis_path = os.path.join(DATA_DIR, "redistribution_recommendations.csv")
    if os.path.exists(redis_path):
        redis = pd.read_csv(redis_path)
        involved = redis[(redis["from_facility"] == facility_id) | (redis["to_facility"] == facility_id)]
        result["redistribution_moves"] = df_to_records(involved)

    return result
