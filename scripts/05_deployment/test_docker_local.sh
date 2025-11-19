#!/bin/bash
# Test Training in Docker Locally
# Run this before deploying to GCS to catch issues early

set -e  # Exit on error

echo "🐳 Testing Docker Training Locally"
echo "================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${RED}❌ Error: .env file not found${NC}"
    echo "Please create .env file with GCP_PROJECT_ID"
    exit 1
fi

# Get project ID from .env
PROJECT_ID=$(grep GCP_PROJECT_ID .env | cut -d '=' -f2 | tr -d '"' | tr -d "'" | xargs)

if [ -z "$PROJECT_ID" ]; then
    echo -e "${RED}❌ Error: GCP_PROJECT_ID not set in .env${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Project ID: $PROJECT_ID"
echo ""

# Build Docker image
echo -e "${YELLOW}Step 1/3:${NC} Building Docker image..."
echo "This may take a few minutes on first run..."
echo ""

docker build --platform linux/amd64 \
  -f scripts/05_deployment/Dockerfile.vertex \
  -t gcr.io/$PROJECT_ID/model-trainer:test \
  . || {
    echo -e "${RED}❌ Docker build failed${NC}"
    exit 1
}

echo ""
echo -e "${GREEN}✓${NC} Docker image built successfully"
echo ""

# Check if data exists
if [ ! -d "data/processed" ] || [ -z "$(ls -A data/processed 2>/dev/null)" ]; then
    echo -e "${YELLOW}⚠️  Warning: data/processed/ is empty${NC}"
    echo "Run this first: python scripts/02_features/tft/tft_data_loader.py"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Run training in Docker
echo -e "${YELLOW}Step 2/3:${NC} Running training in Docker..."
echo "Mounting local directories:"
echo "  - data/ (read/write)"
echo "  - models/ (read/write)"
echo "  - logs/ (read/write)"
echo ""

docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  --env-file .env \
  gcr.io/$PROJECT_ID/model-trainer:test \
  --gcs_bucket $PROJECT_ID-models \
  --dataset_version local \
  --config configs/model_tft_config.yaml || {
    echo ""
    echo -e "${RED}❌ Docker training failed${NC}"
    echo "Check the error messages above"
    exit 1
}

echo ""
echo -e "${GREEN}✓${NC} Docker training completed successfully"
echo ""

# Summary
echo -e "${YELLOW}Step 3/3:${NC} Test Summary"
echo "================================="
echo -e "${GREEN}✅ All tests passed!${NC}"
echo ""
echo "Your code works in the same environment as GCS."
echo "You can now safely deploy to Vertex AI:"
echo ""
echo -e "  ${YELLOW}# Push Docker image${NC}"
echo "  docker tag gcr.io/$PROJECT_ID/model-trainer:test gcr.io/$PROJECT_ID/model-trainer:latest"
echo "  docker push gcr.io/$PROJECT_ID/model-trainer:latest"
echo ""
echo -e "  ${YELLOW}# Submit training job${NC}"
echo "  python scripts/05_deployment/submit_job.py --dataset-version v1"
echo ""
echo "================================="