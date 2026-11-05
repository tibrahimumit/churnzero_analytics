#!/usr/bin/env bash
# Müşteri Kaybı (Churn) İş Paneli — ChurnZero tarzı
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$DIR/.." && pwd)"
VENV="${ROOT}/.ml-venv/bin/streamlit"
if [[ ! -x "$VENV" ]]; then
  VENV="streamlit"
fi
cd "$DIR"
exec "$VENV" run client_app.py --server.headless true --server.port 8503
