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

echo "Deploying to Cloud Run ($REGION)..."
gcloud run deploy "$SERVICE_NAME" \
  --image "$IMAGE" \
  --platform managed \
  --region "$REGION" \
  --project "$PROJECT_ID" \
  --memory 4Gi \
  --cpu 4 \
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
