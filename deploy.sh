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
# --min-instances 1 + --no-cpu-throttling means one instance is billed 24/7,
# whether or not anyone submits a film. Measured Sept 2026: a constant
# 24.0 instance-hours/day (~746 h/month). At 4 vCPU that was ~$215/month
# (~Rs 18k) and the invoice is what took the project offline.
#
# CPU dominates that bill (~90%); memory is ~$21/month. Hence cpu=2, not 4 —
# the workload is I/O-bound (GCS download, Gemini upload, waiting on Gemini),
# so 2 vCPU is ample. Memory stays at 4Gi on purpose: Cloud Run's /tmp is
# RAM-backed and process_video writes 1 GB+ videos there, so lowering it
# risks OOM on feature-length films. Note Cloud Run also requires >=2 vCPU
# for 4Gi of memory.
#
# min-instances CANNOT go to 0 until film processing moves out of the detached
# background threads in review_app.py and into a request boundary (Cloud Run
# Jobs / Cloud Tasks) — scaling to zero today would kill in-flight reviews.
# That change is what takes idle cost to ~zero.
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
  --no-cpu-throttling \
  --timeout 3600 \
  --concurrency 5 \
  --min-instances 1 \
  --max-instances 5 \
  --allow-unauthenticated \
  --vpc-connector fr-connector \
  --vpc-egress all-traffic \
  --set-env-vars "HTTPS=true,GCS_UPLOAD_BUCKET=festival-reviewer-uploads" \
  --set-secrets \
    "FLASK_SECRET=flask-secret:latest,\
GEMINI_API_KEY=gemini-api-key:latest,\
MONGODB_URI=mongodb-uri:latest,\
ADMIN_1_EMAIL=admin-1-email:latest,\
ADMIN_1_PASS=admin-1-pass:latest,\
ADMIN_2_EMAIL=admin-2-email:latest,\
ADMIN_2_PASS=admin-2-pass:latest,\
YT_PROXY=yt-proxy:latest"

echo ""
echo "Deployed: $(gcloud run services describe $SERVICE_NAME --region $REGION --project $PROJECT_ID --format 'value(status.url)')"
