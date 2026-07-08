# Deploying to Cloud Run

Run these from the **repo root** (`smart-health-project/`), not from inside `backend/`,
because the Docker build needs both `backend/` and `data/simulated/` in its context.

## One-time setup
```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com artifactregistry.googleapis.com
```

## Deploy
```bash
gcloud run deploy smart-health-api \
  --source . \
  --dockerfile backend/Dockerfile \
  --region asia-south1 \
  --allow-unauthenticated \
  --port 8080 \
  --memory 512Mi
```

`gcloud` will build the container (via Cloud Build) and give you a live
`https://smart-health-api-xxxxx.run.app` URL — that's your submission's
"working prototype link" if you're demoing the API directly, or the base
URL your frontend calls.

## Before you deploy: regenerate fresh data
The API serves whatever CSVs are sitting in `data/simulated/` at build time.
Re-run the pipeline first so the image ships current data:
```bash
python data/real/nfhs_loader.py
python data/real/facility_infra_loader.py
python data/real/who_eml_loader.py
python data/real/census_geo_loader.py
python data/simulated/simulator.py --days 90 --districts 16 --facilities-per-district 4 --out-dir data/simulated
python ml/forecasting/stockout_forecaster.py
python ml/optimizer/redistribution_optimizer.py
python ml/risk_score/facility_risk_score.py
```

## Test locally first (no GCP needed)
```bash
cd backend
pip install -r requirements.txt --break-system-packages
uvicorn main:app --reload --port 8080
# then visit http://localhost:8080/docs for interactive Swagger UI
```

## Quick smoke test after deploying
```bash
curl https://YOUR_CLOUD_RUN_URL/health
curl https://YOUR_CLOUD_RUN_URL/risk-scores | head -c 500
curl https://YOUR_CLOUD_RUN_URL/facilities/geojson | head -c 500
```

---

# Alternative: Deploying to Render.com (no GCP account needed)

Render is a simpler option if you don't have a Google Cloud account set up —
free tier, no billing/CLI setup, deploys straight from GitHub.

## One-time setup
1. Push this whole `smart-health-project/` folder (including `data/` — the
   API reads CSVs from `data/simulated/` at request time, so they must be
   in the repo) to a GitHub repository.
2. Go to https://render.com → sign up (free, no card required) → **New +** → **Web Service**.
3. Connect your GitHub repo.

## Configure the service
- **Root Directory:** leave blank (repo root), since the Dockerfile needs both `backend/` and `data/` in its build context.
- **Runtime:** Docker
- **Dockerfile Path:** `backend/Dockerfile`
- **Instance Type:** Free

Render will detect the Dockerfile automatically and build from it — no extra
build/start command needed since that's already defined in `backend/Dockerfile`.

## Deploy
Click **Create Web Service**. Render builds and deploys automatically. You'll
get a URL like `https://smart-health-api-xxxx.onrender.com`.

## After deploying
1. Smoke-test it the same way as the Cloud Run steps above, substituting your Render URL.
2. Paste that URL into the `LIVE_API_URL` constant near the bottom of the
   dashboard's `<script>` section — it will auto-connect on page load and
   the status badge will switch from "DEMO SNAPSHOT" to "LIVE" automatically.

## Known Render free-tier quirk
Free services "sleep" after inactivity and take 30-60 seconds to wake up on
the first request. If demoing live, hit the URL once a minute or two before
presenting so it's already warm.

