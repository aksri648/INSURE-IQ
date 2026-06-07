# InsureIQ — AI Insurance Policy Analyst (Local)

Multi-agent RAG pipeline built on LangGraph + Ollama + ChromaDB + Streamlit.
Local fork of the Colab deployment described in `insureiq_colab_deployment.md`.

## Project Layout

```
ai_insurance/
├── agents/
│   ├── state.py
│   ├── ocr_agent.py
│   ├── rag_agent.py
│   ├── web_research_agent.py
│   ├── analyst_agent.py
│   └── compiler_agent.py
├── utils/
│   └── model_config.py
├── graph.py
├── app.py
├── requirements.txt
├── setup.sh
├── run.sh
└── .env.example
```

## Run It

### 1. One-time setup

```bash
cd /home/akshat/Projects/ai_insurance
./setup.sh
```

This installs Ollama, pulls models sized to your VRAM
(`llava` + `deepseek-r1` + `nomic-embed-text`), creates a Python venv,
installs Python deps, and scaffolds `.env`.

### 2. (Optional) Add Tavily key for web research

Edit `.env`:

```
TAVILY_API_KEY=tvly-xxxxxxxxxxxx
```

Without a key the Web Research agent silently skips external lookups.

### 3. Launch

```bash
./run.sh
```

Open <http://localhost:8501>, upload a policy PDF, click **Analyze Policy**.

## Manual Setup (if you skip `setup.sh`)

```bash
# 1. Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull nomic-embed-text
ollama pull llava:7b
ollama pull deepseek-r1:7b

# 2. Python
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```

## Notes

- Sequential model loading: OCR model is offloaded before the analyst model
  loads, so the pipeline fits on a single consumer GPU.
- ChromaDB is in-memory and per-session — re-upload to re-analyze.
- 100% local inference; only Tavily (if enabled) sees outbound network
  requests, and only short non-sensitive query strings are sent.
