# Cloud Run deployment for the Smart Health API.
#
# IMPORTANT: build context must be the repo root (smart-health-project/),
# not the backend/ folder, since this Dockerfile also copies data/simulated.
# See backend/DEPLOY.md for exact commands.

FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/main.py ./main.py
COPY data/simulated /app/data/simulated

ENV DATA_DIR_OVERRIDE=/app/data/simulated
EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
