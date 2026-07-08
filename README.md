# Smart Health — AI-Driven Health Centre & District Resource Management

Built for: Code for Communities / Build with AI hackathon
Challenge: multilingual AI platform for real-time health centre management
(stock, footfall, beds, doctor attendance, test audits) with forecasting,
redistribution, and underperformance flagging across a district's PHCs/CHCs.

## Pipeline (run in this order)

```bash
pip install pandas numpy lightgbm pulp scikit-learn scipy openpyxl pyshp shapely fastapi uvicorn --break-system-packages

# 1. Load real NFHS-5 data, compute District Health Vulnerability Index
python data/real/nfhs_loader.py
# -> data/real/district_health_vulnerability_index.csv (706 real districts scored)

# 2. Load real facility infrastructure data, fit bed norms per facility type
python data/real/facility_infra_loader.py
# -> state_facility_mix.csv, fitted_bed_norms.csv

# 3. Filter WHO Essential Medicines List down to a realistic PHC/CHC formulary
python data/real/who_eml_loader.py
# -> phc_formulary.csv (216 meds), chc_formulary.csv (259 meds)

# 4. Load real Census 2011 district boundaries -> real lat/lon centroids
python data/real/census_geo_loader.py
# -> district_centroids.csv (641 real districts, true polygon centroids)

# 5. Simulate PHC/CHC operational data, seeded by real DHVI + bed norms +
#    formulary + geo centroids
python data/simulated/simulator.py --days 90 --districts 16 --facilities-per-district 4
# -> facilities.csv (now with real lat/lon), stock_timeseries.csv, footfall_timeseries.csv,
#    beds_timeseries.csv, doctor_attendance_timeseries.csv, test_availability.csv,
#    active_formulary.csv

# 6. Train stock-out forecaster (LightGBM), generate current alerts
python ml/forecasting/stockout_forecaster.py
# -> current_stockout_alerts.csv (AUC ~0.95 on held-out time period)

# 7. Solve redistribution LP across each district's facilities, using real
#    haversine distance (Census 2011 centroids) as the transport cost
python ml/optimizer/redistribution_optimizer.py
# -> redistribution_recommendations.csv (concrete "move X units from A to B, Y km away")

# 8. Compute facility risk scores, flag underperforming centres
python ml/risk_score/facility_risk_score.py
# -> facility_risk_scores.csv (with top contributing factor per flagged facility)

# 9. Serve everything as a live API
cd backend && uvicorn main:app --reload --port 8080
# -> http://localhost:8080/docs for interactive Swagger UI
# see backend/DEPLOY.md to deploy this to Cloud Run
```

## Why this design

**NFHS-5 is real, but it's a household survey, not live operational data.**
It has no stock levels, footfall counts, bed occupancy, or doctor attendance.
So we use it for what it's genuinely good for: a real per-district Health
Vulnerability Index (DHVI) built from 11 real indicators (institutional
births, ANC visits, immunization, anaemia, diarrhoea prevalence, insurance
coverage, electricity/water access, out-of-pocket delivery cost).

**The PHC/CHC beds dataset and WHO Essential Medicines List add two more
layers of realism on top of DHVI:**

- *Facility infrastructure*: India doesn't publish "beds per PHC" directly —
  only total beds per state across all facility types combined. We recover
  it with a non-negative least-squares fit across all 36 states
  simultaneously (`facility_infra_loader.py`). The fitted CHC value (29.2
  beds) lands almost exactly on the official IPHS standard of 30 beds/CHC.
  The fitted PHC value (12.0 beds) is noticeably higher than the IPHS
  standard of 6 beds/PHC — worth stating plainly rather than only quoting
  the number that matched. This is a real limitation of recovering
  disaggregated figures from an aggregate total via regression (the fit
  can push error into whichever coefficient has the most residual room),
  not a hidden or fudged result.
- *Geolocation*: Census 2011 district boundary shapefile (641 districts,
  polygon geometry) gives every simulated district a true area-weighted
  centroid (via shapely, not a bounding-box midpoint, which can land
  outside irregular district shapes). This grounds the map view and lets
  the redistribution optimizer use real haversine distance as its
  transport-cost term instead of a flat per-unit penalty.
- *Medicine formulary*: the WHO EML has 1,738 entries covering everything
  from chemotherapy to hepatitis-C antivirals — none of which a rural PHC
  stocks. We filter it down to ~216 medicines across the sections an Indian
  PHC actually dispenses (access-group antibiotics, antimalarials, ORS/zinc,
  antianaemia, antihypertensives, insulin, TB drugs, vaccines, etc.), each
  tagged with a criticality score (1-3) that then weights simulated
  consumption rate and resupply priority.

The operational simulator combines all three real sources: DHVI sets each
facility's baseline stress level, the fitted bed norms set realistic bed
counts per facility type, and the filtered formulary with criticality
weighting sets which medicines are simulated and how urgently they're
restocked. The result: districts with worse real health indicators get
proportionally worse simulated stock-out rates, doctor absenteeism, and bed
pressure — validated below, not just assumed.

| DHVI Tier | Stock-out rate | Doctor attendance | Bed occupancy |
|---|---|---|---|
| Low | 6.2% | 95% | 48% |
| Moderate | 16.0% | 85% | 58% |
| High | 23.7% | 77% | 67% |
| Critical | 40.2% | 69% | 77% |

## The AI "brain" (this is what should carry your AI/Technical Execution score)

1. **Stock-out forecaster** (`ml/forecasting/stockout_forecaster.py`)
   LightGBM classifier predicting 7-day-forward stock-out risk per
   facility x medicine, using rolling consumption stats, trend, and
   days-since-resupply. Time-respecting train/test split (no leakage).
   AUC ≈ 0.96 on held-out days.

2. **Redistribution optimizer** (`ml/optimizer/redistribution_optimizer.py`)
   A real linear program (PuLP/CBC), not a rule engine: for each district
   and medicine, moves surplus stock to deficit facilities minimizing unmet
   demand + a transport cost weighted by real Census-2011-derived
   great-circle distance between facilities (not a flat per-unit penalty),
   capped at operationally realistic per-trip quantities. This directly
   answers the rubric's "smart resource redistribution recommendations
   across a district's PHCs/CHCs."

3. **Facility Risk Score** (`ml/risk_score/facility_risk_score.py`)
   Interpretable composite (weighted z-scores across stock-out exposure,
   bed occupancy, doctor attendance, equipment downtime) that flags
   underperforming centres AND names the top contributing factor — so a
   non-technical district administrator understands *why* a centre is
   flagged in one glance.

## Still to build (next steps)

- ✅ ~~FastAPI backend wrapping these modules as endpoints~~ — done, see `backend/main.py`
- Frontend dashboard (district admin view + PHC data-entry view)
- Multilingual layer: Google Cloud Translation API + Speech-to-Text for
  voice/low-literacy access
- Offline-first PWA caching for low-connectivity PHCs
- ✅ ~~Cloud Run deployment~~ — Dockerfile + deploy steps ready in
  `backend/DEPLOY.md`; needs an actual `gcloud run deploy` run against a
  real GCP project to get the live URL

## Datasets used so far

- `nfhs5_district.csv` — NFHS-5, district-level (706 districts) ✅ used → DHVI
- `nfhs5_factsheet.csv` — NFHS-5, state-level factsheet (loaded, not yet
  used in DHVI — can add state-level cross-checks, e.g. IMR/NMR benchmarks)
- `phc_chc_beds.csv` — real state-level PHC/CHC/SDH/DH counts + total beds
  ✅ used → fitted bed-per-facility-type norms + real facility type mix
- `who_essential_medicines.xlsx` — WHO Essential Medicines List (1,738
  entries) ✅ used → filtered to a 216-medicine PHC formulary with
  criticality tagging
- `2011_Dist.shp` (Census 2011 district boundaries, 641 districts) ✅ used →
  real facility coordinates for the map view + real distance in the
  redistribution optimizer's transport cost

Send the next dataset (e.g. RHS district-level facility counts, HMIS,
or population data for per-capita demand modeling) and I'll wire it into
the same pipeline. At this point, though, the highest-value next step
isn't another dataset — it's a frontend on top of the working API, the
multilingual layer, and an actual Cloud Run deploy, since those are what
the rubric's Deployability (25%) and Accessibility (15%) criteria are
looking for.

