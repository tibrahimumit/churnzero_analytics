#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$DIR/.." && pwd)"
STREAMLIT="${ROOT}/.ml-venv/bin/streamlit"
if [[ ! -x "$STREAMLIT" ]]; then
  STREAMLIT="streamlit"
fi

cd "$DIR"
echo "Starting Data Science & MLOps Platform at http://localhost:8502"
exec "$STREAMLIT" run app.py --server.headless true --server.port 8502
