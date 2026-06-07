#!/usr/bin/env bash
# InsureIQ — launch Streamlit locally. Assumes setup.sh has been run.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Make sure Ollama is up
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "🚀 Starting Ollama..."
    nohup ollama serve >/tmp/ollama.log 2>&1 &
    sleep 3
fi

# Activate venv if present
if [ -d ".venv" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

exec streamlit run app.py \
    --server.port 8501 \
    --server.headless true \
    --browser.gatherUsageStats false
