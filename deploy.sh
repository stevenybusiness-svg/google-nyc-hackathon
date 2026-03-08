#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 在一起 — Cloud Run Deploy Script
# Usage:  ./deploy.sh [project-id] [region]
# Defaults: project from gcloud config, region us-central1
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Find gcloud ──────────────────────────────────────────────────────────────
GCLOUD=""
for candidate in \
    "$(which gcloud 2>/dev/null)" \
    "/usr/local/google-cloud-sdk/bin/gcloud" \
    "$HOME/google-cloud-sdk/bin/gcloud" \
    "/opt/homebrew/bin/gcloud" \
    "/usr/local/bin/gcloud" \
    "/opt/google-cloud-sdk/bin/gcloud"; do
  if [ -x "$candidate" ]; then
    GCLOUD="$candidate"
    break
  fi
done

# Also check Caskroom paths
if [ -z "$GCLOUD" ]; then
  for d in /usr/local/Caskroom/google-cloud-sdk /opt/homebrew/Caskroom/google-cloud-sdk; do
    g=$(find "$d" -name "gcloud" -type f 2>/dev/null | head -1)
    if [ -n "$g" ] && [ -x "$g" ]; then GCLOUD="$g"; break; fi
  done
fi

if [ -z "$GCLOUD" ]; then
  echo "❌  gcloud not found. Install with:"
  echo "    brew install --cask google-cloud-sdk"
  echo "    Then open a new terminal and re-run this script."
  exit 1
fi

echo "✓ Using gcloud: $GCLOUD"

# ── Config ───────────────────────────────────────────────────────────────────
PROJECT="${1:-$($GCLOUD config get-value project 2>/dev/null)}"
REGION="${2:-us-central1}"
SERVICE_NAME="zaiyiqi"
IMAGE="gcr.io/${PROJECT}/${SERVICE_NAME}"

if [ -z "$PROJECT" ]; then
  echo "❌  No GCP project set. Run:"
  echo "    $GCLOUD config set project YOUR_PROJECT_ID"
  echo "Or pass project as first argument: ./deploy.sh my-project-id"
  exit 1
fi

echo ""
echo "┌─────────────────────────────────────────────┐"
echo "│  在一起 — Cloud Run Deployment               │"
echo "│  Project : $PROJECT"
echo "│  Region  : $REGION"
echo "│  Image   : $IMAGE"
echo "└─────────────────────────────────────────────┘"
echo ""

# ── Auth check ───────────────────────────────────────────────────────────────
ACCOUNT=$($GCLOUD auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null | head -1)
if [ -z "$ACCOUNT" ]; then
  echo "⚠  Not logged in. Running: gcloud auth login"
  $GCLOUD auth login
fi
echo "✓ Authenticated as: $ACCOUNT"

# ── Enable required APIs ──────────────────────────────────────────────────────
echo "→ Enabling required APIs (this takes ~30s on first run)…"
$GCLOUD services enable \
  run.googleapis.com \
  containerregistry.googleapis.com \
  cloudbuild.googleapis.com \
  --project "$PROJECT" 2>/dev/null || true

# ── Load API key from .env ────────────────────────────────────────────────────
GOOGLE_API_KEY=""
if [ -f ".env" ]; then
  GOOGLE_API_KEY=$(grep -E "^GOOGLE_API_KEY=" .env | cut -d= -f2- | tr -d '"' | tr -d "'")
fi
if [ -z "$GOOGLE_API_KEY" ]; then
  read -rp "Enter your GOOGLE_API_KEY: " GOOGLE_API_KEY
fi

ELEVENLABS_API_KEY=""
if [ -f ".env" ]; then
  ELEVENLABS_API_KEY=$(grep -E "^ELEVENLABS_API_KEY=" .env | cut -d= -f2- | tr -d '"' | tr -d "'" || true)
fi

GRADIUM_API_KEY=""
if [ -f ".env" ]; then
  GRADIUM_API_KEY=$(grep -E "^GRADIUM_API_KEY=" .env | cut -d= -f2- | tr -d '"' | tr -d "'" || true)
fi

# ── Build & push with Cloud Build (no local Docker needed) ───────────────────
echo ""
echo "→ Building image via Cloud Build…"
$GCLOUD builds submit . \
  --tag "$IMAGE" \
  --project "$PROJECT" \
  --timeout=10m

# ── Deploy to Cloud Run ───────────────────────────────────────────────────────
echo ""
echo "→ Deploying to Cloud Run…"
$GCLOUD run deploy "$SERVICE_NAME" \
  --image "$IMAGE" \
  --platform managed \
  --region "$REGION" \
  --project "$PROJECT" \
  --allow-unauthenticated \
  --port 8080 \
  --memory 1Gi \
  --cpu 1 \
  --min-instances 1 \
  --max-instances 3 \
  --timeout 300 \
  --concurrency 200 \
  --session-affinity \
  --set-env-vars "GOOGLE_API_KEY=${GOOGLE_API_KEY}" \
  $([ -n "$ELEVENLABS_API_KEY" ] && echo "--set-env-vars ELEVENLABS_API_KEY=${ELEVENLABS_API_KEY}" || true) \
  $([ -n "$GRADIUM_API_KEY" ] && echo "--set-env-vars GRADIUM_API_KEY=${GRADIUM_API_KEY}" || true)

# ── Get the URL ───────────────────────────────────────────────────────────────
URL=$($GCLOUD run services describe "$SERVICE_NAME" \
  --platform managed \
  --region "$REGION" \
  --project "$PROJECT" \
  --format "value(status.url)" 2>/dev/null)

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  ✅  Deployed!                                           ║"
echo "║                                                          ║"
echo "║  Public URL: $URL"
echo "║                                                          ║"
echo "║  Live demo URL:    $URL/demo/live                        ║"
echo "║  Memory demo URL:  $URL/demo/memory                      ║"
echo "║  Seed/reset demo:  $URL/api/demo/reseed (POST)           ║"
echo "║  Open root page:   $URL/                                  ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Share the URL with your teammate — they join the same room name."
echo ""
echo "Custom domain (optional):"
echo "  $GCLOUD beta run domain-mappings create --service $SERVICE_NAME --domain YOUR_DOMAIN --region $REGION --project $PROJECT"
echo "Then add the DNS records shown by:"
echo "  $GCLOUD beta run domain-mappings describe --domain YOUR_DOMAIN --region $REGION --project $PROJECT"
