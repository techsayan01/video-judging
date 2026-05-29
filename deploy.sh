#!/bin/bash
# deploy.sh — One command GCP Cloud Run deployment
# Usage: ./deploy.sh
#
# Secret Manager keys required (create once with gcloud secrets create):
#   flask-secret          — random 32-byte hex string
#   gemini-api-key        — global Gemini API key (festival-specific ones added in admin UI)
#   mongodb-uri           — full MongoDB Atlas connection string
#   admin-1-email         — admin-sayan's email
#   admin-1-pass          — admin-sayan's password (hashed on first boot, never stored plain)
#   admin-2-email         — admin-joyi's email
#   admin-2-pass          — admin-joyi's password
#
# Optional (only if using GCS streaming for large videos):
#   gcs-bucket            — GCS bucket name
#
# Create a secret:
#   echo -n "value" | gcloud secrets create secret-name --data-file=-
# Update a secret:
#   echo -n "new-value" | gcloud secrets versions add secret-name --data-file=-

set -euo pipefail

PROJECT_ID="personal-workspace-490012"
REGION="asia-south1"
SERVICE_NAME="festival-reviewer"
IMAGE="asia-south1-docker.pkg.dev/$PROJECT_ID/cloud-run-source-deploy/$SERVICE_NAME"

echo "Building image via Cloud Build..."
gcloud builds submit --tag "$IMAGE" --project "$PROJECT_ID" .

echo "Deploying to Cloud Run ($REGION)..."
gcloud run deploy "$SERVICE_NAME" \
  --image "$IMAGE" \
  --platform managed \
  --region "$REGION" \
  --project "$PROJECT_ID" \
  --memory 2Gi \
  --cpu 2 \
  --timeout 600 \
  --concurrency 10 \
  --min-instances 0 \
  --max-instances 3 \
  --allow-unauthenticated \
  --set-env-vars "HTTPS=true" \
  --set-secrets \
    "FLASK_SECRET=flask-secret:latest,\
GEMINI_API_KEY=gemini-api-key:latest,\
MONGODB_URI=mongodb-uri:latest,\
ADMIN_1_EMAIL=admin-1-email:latest,\
ADMIN_1_PASS=admin-1-pass:latest,\
ADMIN_2_EMAIL=admin-2-email:latest,\
ADMIN_2_PASS=admin-2-pass:latest"

echo ""
echo "Deployed: $(gcloud run services describe $SERVICE_NAME --region $REGION --project $PROJECT_ID --format 'value(status.url)')"
