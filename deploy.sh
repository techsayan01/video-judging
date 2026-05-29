#!/bin/bash
# deploy.sh — One command GCP Cloud Run deployment
# Usage: ./deploy.sh

PROJECT_ID="your-gcp-project-id"       # change this
REGION="asia-south1"                    # Mumbai — closest to Kolkata
SERVICE_NAME="festival-review-app"
IMAGE="gcr.io/$PROJECT_ID/$SERVICE_NAME"

echo "🚀 Deploying $SERVICE_NAME to Cloud Run ($REGION)"

# Build and push
gcloud builds submit --tag $IMAGE .

# Deploy to Cloud Run
gcloud run deploy $SERVICE_NAME \
  --image $IMAGE \
  --platform managed \
  --region $REGION \
  --memory 2Gi \
  --cpu 2 \
  --timeout 600 \
  --concurrency 10 \
  --min-instances 0 \
  --max-instances 3 \
  --no-allow-unauthenticated \
  --set-env-vars "FESTIVAL_NAME=ElegantIFF" \
  --set-secrets \
    "GEMINI_API_KEY=gemini-api-key:latest,\
     USER1_EMAIL=user1-email:latest,\
     USER1_PASS=user1-pass:latest,\
     USER2_EMAIL=user2-email:latest,\
     USER2_PASS=user2-pass:latest,\
     FLASK_SECRET=flask-secret:latest"

echo ""
echo "✅ Deployed. Grant employee access:"
echo "gcloud run services add-iam-policy-binding $SERVICE_NAME \\"
echo "  --region=$REGION \\"
echo "  --member='user:employee@email.com' \\"
echo "  --role='roles/run.invoker'"
