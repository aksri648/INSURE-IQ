#!/usr/bin/env bash
# InsureIQ — one-shot local setup. Installs Ollama, pulls models sized to VRAM,
# creates a Python venv, and installs Python dependencies.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "📦 InsureIQ setup starting in: $PROJECT_ROOT"

# ── 1. Ollama ───────────────────────────────────────────────────
if ! command -v ollama >/dev/null 2>&1; then
    echo "⬇️  Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "✅ Ollama already installed: $(ollama --version 2>/dev/null || echo present)"
fi

# Start Ollama in background if it isn't responding
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "🚀 Starting Ollama server in background..."
    nohup ollama serve >/tmp/ollama.log 2>&1 &
    for i in $(seq 1 10); do
        sleep 2
        if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
            echo "✅ Ollama is running on port 11434"
            break
        fi
        echo "⏳ Waiting for Ollama... ($i/10)"
    done
else
    echo "✅ Ollama already running on port 11434"
fi

# ── 2. Pick model sizes by VRAM ─────────────────────────────────
VRAM=0
if command -v nvidia-smi >/dev/null 2>&1; then
    VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 || echo 0)
fi
echo "🔍 Detected VRAM: ${VRAM} MiB"

if [ "${VRAM:-0}" -gt 35000 ]; then
    OCR_MODEL="llava:13b"
    ANALYST_MODEL="deepseek-r1:14b"
else
    OCR_MODEL="llava:7b"
    ANALYST_MODEL="deepseek-r1:7b"
fi
EMBED_MODEL="nomic-embed-text"

echo "📥 Pulling embedding model: $EMBED_MODEL"
ollama pull "$EMBED_MODEL"

echo "📥 Pulling OCR model: $OCR_MODEL"
ollama pull "$OCR_MODEL"

echo "📥 Pulling analyst model: $ANALYST_MODEL"
ollama pull "$ANALYST_MODEL"

# Persist the resolved model names so the Python app picks them up
CFG="${INSUREIQ_MODEL_CONFIG:-/tmp/model_config.env}"
mkdir -p "$(dirname "$CFG")"
{
    echo "OCR_MODEL=$OCR_MODEL"
    echo "ANALYST_MODEL=$ANALYST_MODEL"
    echo "EMBED_MODEL=$EMBED_MODEL"
} > "$CFG"
echo "📝 Wrote $CFG"

# ── 3. Python venv + deps ───────────────────────────────────────
if [ ! -d ".venv" ]; then
    echo "🐍 Creating Python venv at .venv"
    python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# ── 4. .env scaffold ────────────────────────────────────────────
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "📝 Created .env — add your TAVILY_API_KEY to enable web research."
fi

echo ""
echo "✅ Setup complete."
echo "   Next: ./run.sh   (or:  source .venv/bin/activate && streamlit run app.py)"
