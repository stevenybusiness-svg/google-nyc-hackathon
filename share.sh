#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 在一起 — Quick Share (Cloudflare Tunnel)
# Exposes your local server at a public HTTPS URL in seconds.
# No account or login needed — perfect for hackathon demos.
#
# Usage:  ./share.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PORT=8080

# ── Ensure server is running ──────────────────────────────────────────────────
if ! curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
  echo "⚠  Local server not running on port $PORT."
  echo "   Start it first:"
  echo "   source .venv/bin/activate && uvicorn server:app --port $PORT --reload"
  exit 1
fi
echo "✓ Local server is running on port $PORT"

# ── Install cloudflared if needed ─────────────────────────────────────────────
if ! command -v cloudflared &>/dev/null; then
  echo "→ Installing cloudflared (Cloudflare Tunnel)…"
  brew install cloudflare/cloudflare/cloudflared
fi

echo ""
echo "┌─────────────────────────────────────────────────────────────────┐"
echo "│  在一起 — Public Share via Cloudflare Tunnel                    │"
echo "│                                                                  │"
echo "│  A public HTTPS URL will appear below in ~5 seconds.            │"
echo "│  Share it with your teammate — they open the same room name.    │"
echo "│                                                                  │"
echo "│  Demo storybook:  <public-url>/demo                             │"
echo "│  Live call room:  <public-url>/room/ROOMNAME                    │"
echo "│                                                                  │"
echo "│  Press Ctrl+C to stop.                                          │"
echo "└─────────────────────────────────────────────────────────────────┘"
echo ""

cloudflared tunnel --url "http://localhost:$PORT" 2>&1
