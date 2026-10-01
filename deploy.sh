#!/bin/bash
# deploy.sh — Deploy to GCP Cloud Run
# Builds locally if Docker is running, otherwise falls back to Cloud Build.
# Usage: ./deploy.sh
set -euo pipefail

export PATH="/Users/techsayan/google-cloud-sdk/google-cloud-sdk/bin:$PATH"

PROJECT_ID="personal-workspace-490012"
REGION="asia-south1"
SERVICE_NAME="festival-reviewer"
IMAGE="asia-south1-docker.pkg.dev/$PROJECT_ID/cloud-run-source-deploy/$SERVICE_NAME"

gcloud auth configure-docker asia-south1-docker.pkg.dev --quiet 2>/dev/null

if docker info >/dev/null 2>&1; then
  echo "Building image locally..."
  docker build --platform linux/amd64 -t "$IMAGE" .
  echo "Pushing image..."
  docker push "$IMAGE"
else
  echo "Docker not running — using Cloud Build..."
  gcloud builds submit --tag "$IMAGE" --project "$PROJECT_ID" .
fi

# ── COST NOTICE — read before changing the flags below ────────────────────────
# This service is billed ONLY while a request is in flight. That is deliberate
# and the whole cost model depends on it:
#
#   --min-instances 0   nothing is billed when idle
#   (no --no-cpu-throttling)  request-based billing, so no idle instance charge
#
# Film processing runs inside a request (Cloud Tasks -> /internal/process), so
# the instance lives exactly as long as the work. Setting --min-instances above
# 0, or re-adding --no-cpu-throttling, reinstates a 24/7 charge: that cost
# Rs ~10k/month in Sept 2026 (a flat 24.0 instance-hours/day whether or not any
# film was submitted) and the invoice is what took the project offline.
#
# Memory stays at 4Gi: Cloud Run's /tmp is RAM-backed and process_video pulls
# 1 GB+ videos into it. Cloud Run also requires >=2 vCPU for 4Gi. The Cloud
# Tasks queue caps concurrency at 2 so two jobs cannot exhaust that RAM.
#
# There is deliberately NO VPC connector / Cloud NAT. That stack cost
# Rs ~4,325/month and existed only to give Atlas a fixed IP to allowlist;
# Atlas is now reached over the public internet, protected by TLS + SCRAM auth.
# Re-adding it means paying that again.
#
# Before changing cpu/memory/min-instances, compute: rate x 730 h = monthly idle cost.
echo "Deploying to Cloud Run ($REGION)..."
gcloud run deploy "$SERVICE_NAME" \
  --image "$IMAGE" \
  --platform managed \
  --region "$REGION" \
  --project "$PROJECT_ID" \
  --memory 4Gi \
  --cpu 2 \
  --timeout 3600 \
  --concurrency 5 \
  --min-instances 0 \
  --max-instances 5 \
  --allow-unauthenticated \
  --set-env-vars "HTTPS=true,GCS_UPLOAD_BUCKET=festival-reviewer-uploads,\
GCP_PROJECT=$PROJECT_ID,TASKS_QUEUE=festival-reviewer-jobs,TASKS_LOCATION=$REGION,\
SERVICE_URL=https://festival-reviewer-e53sualg4a-el.a.run.app" \
  --set-secrets \
    "FLASK_SECRET=flask-secret:latest,\
GEMINI_API_KEY=gemini-api-key:latest,\
MONGODB_URI=mongodb-uri:latest,\
ADMIN_1_EMAIL=admin-1-email:latest,\
ADMIN_1_PASS=admin-1-pass:latest,\
ADMIN_2_EMAIL=admin-2-email:latest,\
ADMIN_2_PASS=admin-2-pass:latest,\
INTERNAL_TOKEN=internal-token:latest,\
YT_PROXY=yt-proxy:latest"

echo ""
echo "Deployed: $(gcloud run services describe $SERVICE_NAME --region $REGION --project $PROJECT_ID --format 'value(status.url)')"
