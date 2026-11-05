#!/bin/bash

# ChurnZero Customer Churn Intelligence Platform
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

VENV="${DIR}/.ml-venv/bin/streamlit"
if [[ ! -x "$VENV" ]]; then
  VENV="streamlit"
fi

echo "════════════════════════════════════════════════════════════════"
echo "🛡️  ChurnZero - Customer Retention & Churn Intelligence Platform"
echo "════════════════════════════════════════════════════════════════"
echo "Starting application at: http://localhost:8501"
echo "Press Ctrl+C to stop."
echo "════════════════════════════════════════════════════════════════"

exec "$VENV" run app.py --server.headless false --server.port 8501
