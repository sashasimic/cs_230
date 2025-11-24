#!/bin/bash
set -e

# Load environment variables from .env
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/.env"

if [ ! -f "$ENV_FILE" ]; then
    echo "❌ Error: .env file not found at $ENV_FILE"
    echo "Please create .env file in project root with:"
    echo "  GCP_PROJECT_ID=your-project-id"
    echo "  GCP_REGION=us-central1"
    exit 1
fi

echo "📄 Loading configuration from .env..."
source "$ENV_FILE"

# Configuration from .env
export PROJECT_ID="${GCP_PROJECT_ID}"
export REGION="${GCP_REGION:-us-central1}"  # Default to us-central1 if not set
export GCS_BUCKET="${PROJECT_ID}-models"
export SA_NAME="vertex-model-trainer"

# Validate required variables
if [ -z "$PROJECT_ID" ]; then
    echo "❌ Error: GCP_PROJECT_ID not set in .env file"
    exit 1
fi

echo "✅ Using PROJECT_ID: $PROJECT_ID"
echo "✅ Using REGION: $REGION"

echo "🚀 Setting up GCP for Vertex AI Training..."
echo ""

# ============================================================================
# Enable APIs
# ============================================================================
echo "📦 Checking and enabling APIs..."
APIS=(
  "aiplatform.googleapis.com"
  "containerregistry.googleapis.com"
  "artifactregistry.googleapis.com"
  "storage-api.googleapis.com"
  "bigquery.googleapis.com"
  "compute.googleapis.com"
)

for api in "${APIS[@]}"; do
  if gcloud services list --enabled --filter="name:$api" --format="value(name)" 2>/dev/null | grep -q "$api"; then
    echo "  ✅ $api (already enabled)"
  else
    echo "  🔄 Enabling $api..."
    gcloud services enable "$api" --quiet
    echo "  ✅ $api (enabled)"
  fi
done

echo ""

# ============================================================================
# Create Service Account
# ============================================================================
export SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

echo "👤 Checking service account..."
if gcloud iam service-accounts describe "$SA_EMAIL" --project="$PROJECT_ID" &>/dev/null; then
  echo "  ✅ Service account exists: $SA_EMAIL"
else
  echo "  🔄 Creating service account: $SA_EMAIL"
  gcloud iam service-accounts create "$SA_NAME" \
    --display-name="Vertex AI Model Trainer" \
    --project="$PROJECT_ID" \
    --quiet
  echo "  ✅ Service account created"
fi

echo ""

# ============================================================================
# Grant IAM Roles
# ============================================================================
echo "🔐 Configuring IAM roles..."
ROLES=(
  "roles/aiplatform.user"
  "roles/storage.objectAdmin"
  "roles/storage.objectViewer"      # Needed to pull images from GCR
  "roles/artifactregistry.reader"   # Needed for Artifact Registry
  "roles/bigquery.dataViewer"
  "roles/bigquery.jobUser"
  "roles/logging.logWriter"
)

for role in "${ROLES[@]}"; do
  # Check if binding already exists
  if gcloud projects get-iam-policy "$PROJECT_ID" \
      --flatten="bindings[].members" \
      --filter="bindings.role:$role AND bindings.members:serviceAccount:$SA_EMAIL" \
      --format="value(bindings.role)" 2>/dev/null | grep -q "$role"; then
    echo "  ✅ $role (already granted)"
  else
    echo "  🔄 Granting $role..."
    gcloud projects add-iam-policy-binding "$PROJECT_ID" \
      --member="serviceAccount:${SA_EMAIL}" \
      --role="$role" \
      --condition=None \
      --quiet 2>/dev/null
    echo "  ✅ $role (granted)"
  fi
done

echo ""

# ============================================================================
# Create GCS Bucket
# ============================================================================
echo "🪣 Setting up GCS bucket..."
if gsutil ls -p "$PROJECT_ID" "gs://$GCS_BUCKET" &>/dev/null; then
  echo "  ✅ Bucket exists: gs://$GCS_BUCKET"
else
  echo "  🔄 Creating bucket: gs://$GCS_BUCKET"
  gsutil mb -p "$PROJECT_ID" -l "$REGION" -c STANDARD "gs://$GCS_BUCKET"
  echo "  ✅ Bucket created"
fi

# Create folder structure (idempotent - won't fail if exists)
echo "  🔄 Creating folder structure..."
for folder in models/ checkpoints/ logs/ artifacts/; do
  gsutil -q stat "gs://$GCS_BUCKET/$folder" 2>/dev/null || \
    echo "" | gsutil cp - "gs://$GCS_BUCKET/${folder}.keep" 2>/dev/null
done
echo "  ✅ Folder structure ready"

echo ""

# ============================================================================
# Configure Default Compute Service Account (for TensorBoard)
# ============================================================================
echo "🔧 Configuring default compute service account for TensorBoard..."

# Get project number
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format="value(projectNumber)")
COMPUTE_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

echo "  📋 Project number: $PROJECT_NUMBER"
echo "  👤 Compute SA: $COMPUTE_SA"

# Grant GCS bucket access (storage.admin includes storage.buckets.get needed by TensorBoard)
echo "  🔄 Granting GCS bucket access..."
gsutil iam ch "serviceAccount:${COMPUTE_SA}:roles/storage.admin" "gs://${GCS_BUCKET}" 2>/dev/null
echo "  ✅ roles/storage.admin on gs://${GCS_BUCKET}"

# Grant Vertex AI access for TensorBoard
if gcloud projects get-iam-policy "$PROJECT_ID" \
    --flatten="bindings[].members" \
    --filter="bindings.role:roles/aiplatform.user AND bindings.members:serviceAccount:$COMPUTE_SA" \
    --format="value(bindings.role)" 2>/dev/null | grep -q "roles/aiplatform.user"; then
  echo "  ✅ roles/aiplatform.user (already granted)"
else
  echo "  🔄 Granting roles/aiplatform.user..."
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${COMPUTE_SA}" \
    --role="roles/aiplatform.user" \
    --condition=None \
    --quiet 2>/dev/null
  echo "  ✅ roles/aiplatform.user (granted)"
fi

# Grant logging access for TensorBoard events
if gcloud projects get-iam-policy "$PROJECT_ID" \
    --flatten="bindings[].members" \
    --filter="bindings.role:roles/logging.logWriter AND bindings.members:serviceAccount:$COMPUTE_SA" \
    --format="value(bindings.role)" 2>/dev/null | grep -q "roles/logging.logWriter"; then
  echo "  ✅ roles/logging.logWriter (already granted)"
else
  echo "  🔄 Granting roles/logging.logWriter..."
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${COMPUTE_SA}" \
    --role="roles/logging.logWriter" \
    --condition=None \
    --quiet 2>/dev/null
  echo "  ✅ roles/logging.logWriter (granted)"
fi

echo ""

# ============================================================================
# Configure Docker
# ============================================================================
echo "🐳 Configuring Docker for GCR..."
if grep -q "gcr.io" ~/.docker/config.json 2>/dev/null; then
  echo "  ✅ Docker already configured for GCR"
else
  echo "  🔄 Configuring Docker..."
  gcloud auth configure-docker --quiet
  echo "  ✅ Docker configured"
fi

echo ""

# ============================================================================
# Summary
# ============================================================================
echo "="*80
echo "✅ Setup complete!"
echo "="*80
echo ""
echo "Configuration:"
echo "  Project ID:      $PROJECT_ID"
echo "  Region:          $REGION"
echo "  GCS Bucket:      gs://$GCS_BUCKET"
echo "  Service Account: $SA_EMAIL"
echo "  Compute SA:      $COMPUTE_SA (used for TensorBoard)"
echo ""
echo "Next steps:"
echo "  1. Build Docker image:"
echo "     docker build --platform linux/amd64 -f scripts/05_deployment/Dockerfile.vertex -t gcr.io/${PROJECT_ID}/model-trainer ."
echo ""
echo "  2. Push to Container Registry:"
echo "     docker push gcr.io/${PROJECT_ID}/model-trainer"
echo ""
echo "  3. Submit training jobs:"
echo "     python scripts/05_deployment/submit_job.py"
echo "     python scripts/05_deployment/submit_parallel.py"
echo ""
echo "Monitor jobs at:"
echo "  https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=$PROJECT_ID"
echo ""