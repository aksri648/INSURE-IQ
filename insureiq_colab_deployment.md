# InsureIQ — Google Colab + Cloudflare Tunnel Deployment Guide

> Run the full multi-agent pipeline on Colab's free/Pro GPU and expose Streamlit publicly via Cloudflare Tunnel — no server, no domain, no cost.

---

## ARCHITECTURE CHANGE FOR COLAB

```
┌─────────────────────────────────────────────────────────┐
│              GOOGLE COLAB (T4 / A100 GPU)               │
│                                                         │
│  ┌─────────────┐   ┌──────────────┐   ┌─────────────┐  │
│  │  Ollama     │   │  ChromaDB    │   │  Streamlit  │  │
│  │  (port 11434│   │  (in-memory) │   │  (port 8501)│  │
│  └─────────────┘   └──────────────┘   └──────┬──────┘  │
│                                               │         │
└───────────────────────────────────────────────┼─────────┘
                                                │
                                    ┌───────────▼──────────┐
                                    │  cloudflared tunnel  │
                                    │  (trycloudflare.com) │
                                    └───────────┬──────────┘
                                                │ Public HTTPS URL
                                                ▼
                                         Anyone on internet
```

---

## COLAB RUNTIME RECOMMENDATIONS

| Plan | GPU | VRAM | Suitability |
|------|-----|------|-------------|
| Free | T4 | 15 GB | ✅ Works — use 7B models |
| Pro | A100 | 40 GB | ✅ Best — use 14B+ models |
| Pro+ | A100 | 80 GB | ✅ Overkill but smooth |

**Model selection by VRAM:**

| VRAM | OCR Model | Analyst Model |
|------|-----------|---------------|
| 15 GB (T4) | `llava:7b` | `deepseek-r1:7b` |
| 40 GB (A100) | `llava:13b` | `deepseek-r1:14b` |
| 80 GB (A100) | `llava:34b` | `deepseek-r1:32b` |

---

## FULL COLAB NOTEBOOK

### Cell 1 — GPU Check & System Info

```python
# ── Cell 1: Verify GPU ──────────────────────────────────────────
import subprocess

gpu_info = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
     "--format=csv,noheader"],
    capture_output=True, text=True
).stdout.strip()

print("🖥  GPU Info:")
print(gpu_info)

ram_info = subprocess.run(
    ["free", "-h"], capture_output=True, text=True
).stdout
print("\n💾 RAM:")
print(ram_info)
```

---

### Cell 2 — Install All Dependencies

```python
# ── Cell 2: Install Dependencies ────────────────────────────────
%%bash
set -e

echo "📦 Installing system packages..."
apt-get install -qq -y curl libgl1 libglib2.0-0 > /dev/null

echo "🐍 Installing Python packages..."
pip install -q \
    streamlit \
    langgraph \
    langchain \
    langchain-community \
    langchain-ollama \
    chromadb \
    pymupdf \
    tavily-python \
    python-dotenv \
    Pillow \
    reportlab \
    nest-asyncio \
    pyngrok

echo "✅ All packages installed"
```

---

### Cell 3 — Install & Start Ollama

```python
# ── Cell 3: Install and Start Ollama ────────────────────────────
%%bash
set -e

echo "⬇️  Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh > /dev/null 2>&1

echo "🚀 Starting Ollama server in background..."
nohup ollama serve > /tmp/ollama.log 2>&1 &

# Wait for Ollama to be ready
sleep 5
for i in {1..10}; do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✅ Ollama is running on port 11434"
        break
    fi
    echo "⏳ Waiting for Ollama... ($i/10)"
    sleep 3
done
```

---

### Cell 4 — Pull Models (Sequential, Memory-Safe)

```python
# ── Cell 4: Pull Models ─────────────────────────────────────────
%%bash

# Detect available VRAM and set model sizes
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
echo "🔍 Detected VRAM: ${VRAM} MiB"

if [ "$VRAM" -gt 35000 ]; then
    OCR_MODEL="llava:13b"
    ANALYST_MODEL="deepseek-r1:14b"
    EMBED_MODEL="nomic-embed-text"
elif [ "$VRAM" -gt 14000 ]; then
    OCR_MODEL="llava:7b"
    ANALYST_MODEL="deepseek-r1:7b"
    EMBED_MODEL="nomic-embed-text"
else
    OCR_MODEL="llava:7b"
    ANALYST_MODEL="deepseek-r1:7b"
    EMBED_MODEL="nomic-embed-text"
fi

echo "📥 Pulling embedding model: $EMBED_MODEL"
ollama pull $EMBED_MODEL

echo "📥 Pulling OCR model: $OCR_MODEL (this may take a few minutes...)"
ollama pull $OCR_MODEL

echo "📥 Pulling analyst model: $ANALYST_MODEL (this may take a few minutes...)"
ollama pull $ANALYST_MODEL

echo "✅ All models ready"
echo "OCR_MODEL=$OCR_MODEL" > /tmp/model_config.env
echo "ANALYST_MODEL=$ANALYST_MODEL" >> /tmp/model_config.env
echo "EMBED_MODEL=$EMBED_MODEL" >> /tmp/model_config.env
```

---

### Cell 5 — Write Project Files

```python
# ── Cell 5: Write Project Structure ─────────────────────────────
import os

os.makedirs("/content/insureiq/agents", exist_ok=True)
os.makedirs("/content/insureiq/utils", exist_ok=True)
```

```python
# ── agents/state.py ─────────────────────────────────────────────
%%writefile /content/insureiq/agents/state.py

from typing import TypedDict, Optional

class PolicyState(TypedDict):
    session_id: str
    pdf_path: str
    ocr_text: list          # [{page, text, tables}]
    chunks: list            # [{chunk_id, text, metadata}]
    insurer_name: str
    external_research: dict
    section_analyses: dict
    final_report: dict
    citations: list
    error: Optional[str]
    status: str             # current pipeline stage
```

```python
# ── agents/ocr_agent.py ─────────────────────────────────────────
%%writefile /content/insureiq/agents/ocr_agent.py

import fitz          # PyMuPDF
import ollama
import base64
import json
import os
from agents.state import PolicyState

# Read model from env file written in Cell 4
def get_ocr_model():
    config = {}
    if os.path.exists("/tmp/model_config.env"):
        with open("/tmp/model_config.env") as f:
            for line in f:
                k, v = line.strip().split("=")
                config[k] = v
    return config.get("OCR_MODEL", "llava:7b")

OCR_SYSTEM_PROMPT = """You are a precise insurance document digitizer.
Extract ALL text from this policy page. Preserve:
- Section headings and hierarchy
- Clause numbers (e.g. 4.2.1)
- Table structures as markdown tables
- Bold/emphasized terms

Return ONLY valid JSON:
{
  "section": "<section title or empty string>",
  "text": "<full verbatim text>",
  "tables": ["<table as markdown>"],
  "clause_numbers": ["4.1", "4.2"]
}
Do NOT summarize. Do NOT paraphrase."""

def page_to_base64(doc, page_num: int) -> str:
    page = doc[page_num]
    mat = fitz.Matrix(2, 2)   # 2x zoom for better OCR
    pix = page.get_pixmap(matrix=mat)
    return base64.b64encode(pix.tobytes("png")).decode()

def ocr_agent_node(state: PolicyState) -> PolicyState:
    print(f"[OCR Agent] Loading {get_ocr_model()}...")
    ocr_model = get_ocr_model()

    doc = fitz.open(state["pdf_path"])
    total_pages = len(doc)
    ocr_results = []

    for i in range(total_pages):
        print(f"[OCR Agent] Processing page {i+1}/{total_pages}")
        img_b64 = page_to_base64(doc, i)

        try:
            response = ollama.chat(
                model=ocr_model,
                messages=[{
                    "role": "user",
                    "content": OCR_SYSTEM_PROMPT,
                    "images": [img_b64]
                }]
            )
            raw = response["message"]["content"]
            # Strip markdown fences if present
            raw = raw.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(raw)
        except Exception as e:
            parsed = {"section": "", "text": f"OCR Error page {i+1}: {e}",
                      "tables": [], "clause_numbers": []}

        ocr_results.append({"page": i + 1, **parsed})

    doc.close()

    # ── CRITICAL: Offload OCR model to free VRAM ──
    print(f"[OCR Agent] Offloading {ocr_model} from VRAM...")
    try:
        ollama.delete(ocr_model)
    except Exception:
        pass  # Model may not support direct delete; Ollama will evict on next load

    print(f"[OCR Agent] Done. Extracted {total_pages} pages.")
    return {**state, "ocr_text": ocr_results, "status": "ocr_complete"}
```

```python
# ── agents/rag_agent.py ─────────────────────────────────────────
%%writefile /content/insureiq/agents/rag_agent.py

import chromadb
import ollama
import os
from agents.state import PolicyState

def get_embed_model():
    config = {}
    if os.path.exists("/tmp/model_config.env"):
        with open("/tmp/model_config.env") as f:
            for line in f:
                k, v = line.strip().split("=")
                config[k] = v
    return config.get("EMBED_MODEL", "nomic-embed-text")

_chroma_client = chromadb.Client()

def semantic_chunk(ocr_results: list, chunk_size=400, overlap=50) -> list:
    """Clause-aware chunking with metadata."""
    chunks = []
    chunk_id = 0

    for page_data in ocr_results:
        text = page_data.get("text", "")
        page = page_data.get("page", 0)
        section = page_data.get("section", "")

        # Split by double newline (paragraph/clause boundaries)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        buffer = ""
        for para in paragraphs:
            if len(buffer) + len(para) < chunk_size:
                buffer += " " + para
            else:
                if buffer.strip():
                    chunks.append({
                        "chunk_id": f"chunk_{chunk_id:04d}",
                        "text": buffer.strip(),
                        "metadata": {
                            "page": page,
                            "section": section,
                            "chunk_index": chunk_id
                        }
                    })
                    chunk_id += 1
                buffer = para  # start new chunk

        if buffer.strip():  # flush remaining
            chunks.append({
                "chunk_id": f"chunk_{chunk_id:04d}",
                "text": buffer.strip(),
                "metadata": {"page": page, "section": section, "chunk_index": chunk_id}
            })
            chunk_id += 1

    return chunks

def embed_and_store_node(state: PolicyState) -> PolicyState:
    print("[RAG] Chunking policy document...")
    chunks = semantic_chunk(state["ocr_text"])
    embed_model = get_embed_model()

    # Create per-session ChromaDB collection
    collection_name = f"policy_{state['session_id'][:8]}"
    try:
        _chroma_client.delete_collection(collection_name)
    except Exception:
        pass
    collection = _chroma_client.create_collection(collection_name)

    print(f"[RAG] Embedding {len(chunks)} chunks with {embed_model}...")
    for chunk in chunks:
        embedding = ollama.embeddings(
            model=embed_model,
            prompt=chunk["text"]
        )["embedding"]

        collection.add(
            ids=[chunk["chunk_id"]],
            embeddings=[embedding],
            documents=[chunk["text"]],
            metadatas=[chunk["metadata"]]
        )

    print(f"[RAG] Stored {len(chunks)} chunks in ChromaDB.")
    return {**state, "chunks": chunks, "status": "rag_complete"}

def retrieve_chunks(session_id: str, query: str, n_results=8) -> list:
    embed_model = get_embed_model()
    collection_name = f"policy_{session_id[:8]}"
    collection = _chroma_client.get_collection(collection_name)

    query_embedding = ollama.embeddings(
        model=embed_model,
        prompt=query
    )["embedding"]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas"]
    )
    return list(zip(results["documents"][0], results["metadatas"][0]))
```

```python
# ── agents/web_research_agent.py ────────────────────────────────
%%writefile /content/insureiq/agents/web_research_agent.py

import os
from tavily import TavilyClient
from agents.state import PolicyState

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

def extract_insurer_name(ocr_results: list) -> str:
    """Extract insurer name from first few pages."""
    for page in ocr_results[:3]:
        text = page.get("text", "")
        # Look for common insurer indicators
        for line in text.split("\n"):
            if any(kw in line.lower() for kw in
                   ["insurance", "insurer", "company", "ltd", "limited"]):
                return line.strip()[:80]
    return "Unknown Insurer"

def web_research_node(state: PolicyState) -> PolicyState:
    if not TAVILY_API_KEY:
        print("[Web Research] No Tavily API key. Skipping external research.")
        return {**state, "external_research": {}, "insurer_name": "Unknown",
                "status": "web_research_complete"}

    client = TavilyClient(api_key=TAVILY_API_KEY)
    insurer_name = extract_insurer_name(state["ocr_text"])
    print(f"[Web Research] Researching: {insurer_name}")

    research = {}

    # Insurer claim settlement ratio
    try:
        result = client.search(
            f"{insurer_name} claim settlement ratio IRDAI annual report",
            max_results=3
        )
        research["claim_settlement"] = {
            "data": result["results"],
            "source": "EXTERNAL",
            "query": f"{insurer_name} claim settlement ratio"
        }
    except Exception as e:
        research["claim_settlement"] = {"error": str(e), "source": "EXTERNAL"}

    # Insurer solvency/regulatory status
    try:
        result = client.search(
            f"{insurer_name} IRDAI registration solvency ratio",
            max_results=3
        )
        research["regulatory"] = {
            "data": result["results"],
            "source": "EXTERNAL"
        }
    except Exception as e:
        research["regulatory"] = {"error": str(e), "source": "EXTERNAL"}

    print("[Web Research] Done.")
    return {**state,
            "external_research": research,
            "insurer_name": insurer_name,
            "status": "web_research_complete"}
```

```python
# ── agents/analyst_agent.py ─────────────────────────────────────
%%writefile /content/insureiq/agents/analyst_agent.py

import ollama
import json
import os
from agents.state import PolicyState
from agents.rag_agent import retrieve_chunks

def get_analyst_model():
    config = {}
    if os.path.exists("/tmp/model_config.env"):
        with open("/tmp/model_config.env") as f:
            for line in f:
                k, v = line.strip().split("=")
                config[k] = v
    return config.get("ANALYST_MODEL", "deepseek-r1:7b")

ANALYST_SYSTEM_PROMPT = """You are a senior insurance policy analyst with 20 years experience.
You protect consumers by finding hidden exclusions, risks, and unfair terms.

STRICT RULES:
1. ONLY make claims supported by the provided policy text chunks.
2. Every claim MUST cite: [Source: chunk_id, Page X, Section Y]
3. External research claims: [EXTERNAL: description]
4. If not found in policy text, state: "NOT FOUND IN POLICY"
5. Think step by step before concluding.
6. Output ONLY valid JSON — no preamble, no markdown fences."""

SECTION_QUERIES = {
    "executive_summary":  "policy name insurer coverage type sum insured premium",
    "coverage_benefits":  "covered benefits hospitalization treatment claims payable",
    "exclusions":         "exclusions not covered excluded conditions limitations",
    "waiting_periods":    "waiting period days months pre-existing disease",
    "premium_analysis":   "premium amount payment frequency loading renewal",
    "claims_process":     "claim filing process documents required cashless reimbursement",
    "risk_assessment":    "risk void cancellation lapse forfeiture penalty",
    "policy_conditions":  "conditions terms obligations policyholder duties",
}

def build_section_prompt(section: str, chunks: list, external: dict) -> str:
    chunk_text = "\n\n".join([
        f"[{meta['chunk_id'] if 'chunk_id' in meta else 'chunk'} | "
        f"Page {meta.get('page','?')} | {meta.get('section','?')}]\n{doc}"
        for doc, meta in chunks
    ])

    ext_summary = ""
    if external:
        ext_summary = "\n\nEXTERNAL RESEARCH (mark all as [EXTERNAL]):\n"
        for k, v in external.items():
            if isinstance(v, dict) and "data" in v:
                for item in v["data"][:2]:
                    ext_summary += f"- {item.get('title','')}: {item.get('content','')[:200]}\n"

    return f"""Analyze the {section.upper()} section of this insurance policy.

POLICY TEXT CHUNKS:
{chunk_text}
{ext_summary}

Return a JSON object for the {section} section with:
- findings: list of key findings, each with "claim", "citation", "severity" (for exclusions/risks)
- summary: 2-3 sentence plain language summary
- score: 1-10 rating for this section (10=excellent for consumer)
- red_flags: list of concerning items

Cite every finding: [Source: chunk_XXXX, Page N, Section Title]"""

def analyst_agent_node(state: PolicyState) -> PolicyState:
    analyst_model = get_analyst_model()
    print(f"[Analyst] Loading {analyst_model}...")

    analyses = {}
    citations = []
    cit_counter = 1

    for section, query in SECTION_QUERIES.items():
        print(f"[Analyst] Analyzing: {section}")
        chunks = retrieve_chunks(state["session_id"], query)
        prompt = build_section_prompt(section, chunks, state.get("external_research", {}))

        try:
            response = ollama.chat(
                model=analyst_model,
                messages=[
                    {"role": "system", "content": ANALYST_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ]
            )
            raw = response["message"]["content"]
            raw = raw.replace("```json", "").replace("```", "").strip()

            # DeepSeek R1 wraps reasoning in <think> tags — strip them
            if "<think>" in raw:
                raw = raw[raw.rfind("</think>") + 8:].strip()

            parsed = json.loads(raw)
            analyses[section] = parsed

            # Extract citations
            for finding in parsed.get("findings", []):
                citations.append({
                    "citation_id": f"CIT-{cit_counter:03d}",
                    "section": section,
                    "claim": finding.get("claim", ""),
                    "citation_text": finding.get("citation", ""),
                    "source_type": "EXTERNAL" if "[EXTERNAL]" in finding.get("citation", "") else "POLICY_DOCUMENT"
                })
                cit_counter += 1

        except Exception as e:
            analyses[section] = {"error": str(e), "findings": [], "summary": "Analysis failed", "score": 0}

    # ── CRITICAL: Offload analyst model ──
    print(f"[Analyst] Offloading {analyst_model} from VRAM...")
    try:
        ollama.delete(analyst_model)
    except Exception:
        pass

    print(f"[Analyst] Done. {len(citations)} citations extracted.")
    return {**state, "section_analyses": analyses, "citations": citations, "status": "analysis_complete"}
```

```python
# ── agents/compiler_agent.py ────────────────────────────────────
%%writefile /content/insureiq/agents/compiler_agent.py

from agents.state import PolicyState
from datetime import datetime

def score_policy(analyses: dict) -> int:
    """Compute overall analyst score 0-100."""
    scores = [v.get("score", 5) for v in analyses.values() if isinstance(v, dict)]
    if not scores:
        return 50
    avg = sum(scores) / len(scores)
    return round(avg * 10)

def get_verdict(score: int) -> str:
    if score >= 75: return "GOOD_VALUE"
    if score >= 55: return "BUY_WITH_CAUTION"
    if score >= 35: return "REVIEW_NEEDED"
    return "AVOID"

def report_compiler_node(state: PolicyState) -> PolicyState:
    print("[Compiler] Building final report...")
    analyses = state.get("section_analyses", {})
    score = score_policy(analyses)

    red_flags = []
    for section, data in analyses.items():
        if isinstance(data, dict):
            red_flags.extend(data.get("red_flags", []))

    report = {
        "report_metadata": {
            "session_id": state["session_id"],
            "generated_at": datetime.now().isoformat(),
            "policy_file": state["pdf_path"].split("/")[-1],
            "total_pages_analyzed": len(state.get("ocr_text", [])),
            "total_citations": len(state.get("citations", [])),
        },
        "executive_summary": {
            "insurer_name": state.get("insurer_name", "Unknown"),
            "analyst_score": score,
            "verdict": get_verdict(score),
            "key_red_flags": red_flags[:5],
            "summary": analyses.get("executive_summary", {}).get("summary", ""),
        },
        "sections": analyses,
        "insurer_profile": state.get("external_research", {}),
        "all_citations": state.get("citations", []),
    }

    print(f"[Compiler] Report complete. Score: {score}/100. Verdict: {get_verdict(score)}")
    return {**state, "final_report": report, "status": "complete"}
```

```python
# ── graph.py — LangGraph StateGraph ─────────────────────────────
%%writefile /content/insureiq/graph.py

from langgraph.graph import StateGraph, END
from agents.state import PolicyState
from agents.ocr_agent import ocr_agent_node
from agents.rag_agent import embed_and_store_node
from agents.web_research_agent import web_research_node
from agents.analyst_agent import analyst_agent_node
from agents.compiler_agent import report_compiler_node

def build_graph():
    g = StateGraph(PolicyState)

    g.add_node("ocr",          ocr_agent_node)
    g.add_node("embed_store",  embed_and_store_node)
    g.add_node("web_research", web_research_node)
    g.add_node("analyst",      analyst_agent_node)
    g.add_node("compiler",     report_compiler_node)

    g.set_entry_point("ocr")
    g.add_edge("ocr",          "embed_store")
    g.add_edge("embed_store",  "web_research")
    g.add_edge("web_research", "analyst")
    g.add_edge("analyst",      "compiler")
    g.add_edge("compiler",     END)

    return g.compile()

pipeline = build_graph()
```

```python
# ── app.py — Streamlit Frontend ──────────────────────────────────
%%writefile /content/insureiq/app.py

import streamlit as st
import uuid
import os
import json
import tempfile
import sys

sys.path.insert(0, "/content/insureiq")
from graph import pipeline

st.set_page_config(
    page_title="InsureIQ — AI Policy Analyst",
    page_icon="⬡",
    layout="wide"
)

# ── Styles ──────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background: #0a0e1a; color: #e2e8f0; }
  .verdict-GOOD_VALUE      { color: #10b981; font-weight: 700; }
  .verdict-BUY_WITH_CAUTION{ color: #f59e0b; font-weight: 700; }
  .verdict-REVIEW_NEEDED   { color: #ef4444; font-weight: 700; }
  .verdict-AVOID           { color: #dc2626; font-weight: 900; }
  .citation-badge {
    background: #1e2d45; border: 1px solid #00d4ff33;
    border-radius: 4px; padding: 2px 8px;
    font-size: 11px; color: #00d4ff; font-family: monospace;
  }
  .red-flag {
    background: #ef444410; border-left: 3px solid #ef4444;
    padding: 8px 12px; margin: 4px 0; border-radius: 2px;
  }
</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────────
st.markdown("## ⬡ InsureIQ — AI Insurance Policy Analyst")
st.caption("Upload any insurance policy PDF for a full cited analysis · Powered by local AI")

# ── Upload ──────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload Insurance Policy PDF",
    type=["pdf"],
    help="Your PDF stays on this server and is never sent to external AI services"
)

if uploaded_file:
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.info(f"📄 {uploaded_file.name} — {uploaded_file.size // 1024} KB")
    with col3:
        if st.button("🔍 Analyze Policy", type="primary", use_container_width=True):
            # Save uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            session_id = str(uuid.uuid4())

            # Run pipeline with progress
            progress_bar = st.progress(0)
            status_text = st.empty()

            stages = [
                ("ocr_complete",           "👁  OCR Extraction",        25),
                ("rag_complete",           "🗄  Building Vector Store",  50),
                ("web_research_complete",  "🌐  External Research",      65),
                ("analysis_complete",      "🤖  Deep Analysis",          85),
                ("complete",               "📋  Compiling Report",      100),
            ]

            stage_map = {s[0]: (s[1], s[2]) for s in stages}

            initial_state = {
                "session_id": session_id,
                "pdf_path": tmp_path,
                "ocr_text": [],
                "chunks": [],
                "insurer_name": "",
                "external_research": {},
                "section_analyses": {},
                "final_report": {},
                "citations": [],
                "error": None,
                "status": "starting"
            }

            status_text.text("🚀 Starting analysis pipeline...")

            try:
                # Stream events for live progress
                for event in pipeline.stream(initial_state):
                    for node_name, node_state in event.items():
                        current_status = node_state.get("status", "")
                        if current_status in stage_map:
                            label, pct = stage_map[current_status]
                            progress_bar.progress(pct)
                            status_text.text(f"{label}...")

                    # Store final state
                    final_state = node_state

                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")
                st.session_state["report"] = final_state.get("final_report", {})

            except Exception as e:
                st.error(f"Pipeline error: {e}")
            finally:
                os.unlink(tmp_path)

# ── Report Display ───────────────────────────────────────────────
if "report" in st.session_state and st.session_state["report"]:
    report = st.session_state["report"]
    summary = report.get("executive_summary", {})
    sections = report.get("sections", {})

    st.divider()
    st.markdown("### 📊 Analysis Report")

    # Score + Verdict
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        score = summary.get("analyst_score", 0)
        color = "#10b981" if score >= 75 else "#f59e0b" if score >= 55 else "#ef4444"
        st.markdown(f"<h2 style='color:{color};margin:0'>{score}<span style='font-size:14px'>/100</span></h2>", unsafe_allow_html=True)
        st.caption("Analyst Score")
    with col2:
        verdict = summary.get("verdict", "UNKNOWN")
        st.markdown(f"<p class='verdict-{verdict}'>{verdict.replace('_',' ')}</p>", unsafe_allow_html=True)
        st.caption("Verdict")
    with col3:
        st.metric("Insurer", summary.get("insurer_name", "—")[:25])
    with col4:
        st.metric("Citations", len(report.get("all_citations", [])))

    # Red flags
    flags = summary.get("key_red_flags", [])
    if flags:
        st.markdown("#### ⚠️ Key Red Flags")
        for flag in flags:
            st.markdown(f"<div class='red-flag'>⚠️ {flag}</div>", unsafe_allow_html=True)

    # Section tabs
    section_labels = {
        "executive_summary": "📋 Summary",
        "coverage_benefits": "✅ Benefits",
        "exclusions":        "❌ Exclusions",
        "waiting_periods":   "⏳ Waiting",
        "premium_analysis":  "💰 Premium",
        "claims_process":    "📝 Claims",
        "risk_assessment":   "⚠️ Risks",
        "policy_conditions": "📜 Conditions",
    }

    tabs = st.tabs([section_labels.get(k, k) for k in sections.keys()])
    for tab, (section_key, section_data) in zip(tabs, sections.items()):
        with tab:
            if isinstance(section_data, dict):
                if "summary" in section_data:
                    st.markdown(section_data["summary"])
                if "findings" in section_data:
                    for finding in section_data["findings"]:
                        with st.expander(f"📌 {finding.get('claim','Finding')[:80]}"):
                            st.write(finding.get("claim", ""))
                            if "citation" in finding:
                                st.markdown(
                                    f"<span class='citation-badge'>{finding['citation']}</span>",
                                    unsafe_allow_html=True
                                )

    # Download JSON report
    st.divider()
    st.download_button(
        label="⬇️ Download Full Report (JSON)",
        data=json.dumps(report, indent=2),
        file_name=f"insureiq_report_{session_id[:8]}.json",
        mime="application/json"
    )
```

---

### Cell 6 — Set API Keys

```python
# ── Cell 6: Configure API Keys ──────────────────────────────────
import os

# Get your free Tavily key at: https://tavily.com
os.environ["TAVILY_API_KEY"] = "tvly-YOUR_KEY_HERE"   # ← paste your key

print("✅ Environment configured")
```

---

### Cell 7 — Launch Streamlit + Cloudflare Tunnel

```python
# ── Cell 7: Launch App with Cloudflare Tunnel ───────────────────
import subprocess
import threading
import time
import requests

def start_streamlit():
    subprocess.Popen([
        "streamlit", "run", "/content/insureiq/app.py",
        "--server.port", "8501",
        "--server.headless", "true",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false",
        "--browser.gatherUsageStats", "false"
    ])

def start_cloudflare_tunnel():
    """
    cloudflared tunnel — no account needed.
    trycloudflare.com gives a random public HTTPS URL.
    """
    # Download cloudflared binary
    subprocess.run([
        "wget", "-q", "-O", "/usr/local/bin/cloudflared",
        "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    ], check=True)
    subprocess.run(["chmod", "+x", "/usr/local/bin/cloudflared"])

    # Start tunnel — prints public URL to stdout
    process = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", "http://localhost:8501"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    print("⏳ Waiting for Cloudflare tunnel URL...")
    for line in process.stdout:
        if "trycloudflare.com" in line or "https://" in line:
            # Extract URL
            for word in line.split():
                if word.startswith("https://"):
                    print(f"\n{'='*60}")
                    print(f"  🌐 PUBLIC URL: {word}")
                    print(f"{'='*60}")
                    print("  Share this link with anyone!")
                    print("  ⚠️  URL is valid while this Colab session runs")
                    print(f"{'='*60}\n")
                    return

# Start Streamlit in background thread
print("🚀 Starting Streamlit...")
t = threading.Thread(target=start_streamlit)
t.daemon = True
t.start()

# Wait for Streamlit to be ready
time.sleep(8)

# Check Streamlit is up
try:
    r = requests.get("http://localhost:8501", timeout=10)
    print(f"✅ Streamlit running (status: {r.status_code})")
except Exception as e:
    print(f"⚠️ Streamlit check: {e} — it may still be starting up")

# Start Cloudflare tunnel (blocking — will print URL)
start_cloudflare_tunnel()
```

---

## COLAB LIMITATIONS & WORKAROUNDS

| Limitation | Impact | Workaround |
|------------|--------|------------|
| Session timeout (free: ~90 min) | Pipeline interrupted | Use Colab Pro; add keep-alive cell |
| VRAM shared with system | OOM on large models | Use 7B models on T4; sequential offload |
| No persistent storage | ChromaDB lost on restart | Re-upload and re-analyze |
| Tunnel URL changes on restart | Users lose link | Restart Cell 7 and reshare URL |
| Network egress limits | Slow model pull | Pre-pull models at session start |

### Keep-Alive Cell (add before Cell 7 if needed)

```python
# Prevent Colab from timing out during long analyses
import IPython
js_code = """
function ClickConnect(){
    console.log("Keeping Colab alive...");
    document.querySelector("#top-toolbar > colab-connect-button")
            .shadowRoot.querySelector("#connect").click()
}
setInterval(ClickConnect, 60000)
"""
IPython.display.display(IPython.display.Javascript(js_code))
print("✅ Keep-alive activated")
```

---

## CLOUDFLARE TUNNEL — HOW IT WORKS

```
Colab Machine                Cloudflare Edge         User's Browser
     │                            │                        │
     │  cloudflared tunnel        │                        │
     │ ──────────────────────────►│                        │
     │  registers random subdomain│                        │
     │                            │◄───── HTTPS request ───│
     │◄── forwards to :8501 ──────│                        │
     │                            │                        │
  Streamlit                       │──── response ─────────►│
```

- **No account required** — uses `trycloudflare.com` free tier
- **HTTPS by default** — Cloudflare handles SSL termination
- **No port forwarding** — works behind any NAT/firewall
- **URL is ephemeral** — new random URL each session start

### Optional: Named Tunnel (stable URL)

If you want a stable subdomain across restarts:
1. Create free Cloudflare account at cloudflare.com
2. Run `cloudflared login` in Colab terminal
3. Create named tunnel: `cloudflared tunnel create insureiq`
4. Use: `cloudflared tunnel run --url http://localhost:8501 insureiq`

---

## QUICK REFERENCE — ALL COLAB CELLS IN ORDER

```
Cell 1  →  GPU check
Cell 2  →  pip install all dependencies
Cell 3  →  Install + start Ollama server
Cell 4  →  Pull models (auto-sized to VRAM)
Cell 5  →  Write all project files
Cell 6  →  Set TAVILY_API_KEY
Cell 7  →  Launch Streamlit + Cloudflare tunnel → get public URL
```

Total setup time: ~15–25 minutes (mostly model download)
Analysis time per policy: ~5–15 minutes depending on PDF length and GPU

---

*InsureIQ · Colab + Cloudflare Deployment Guide · v1.0*
