# ⬡ InsureIQ — AI Insurance Policy Analyst

**Local-first, multi-agent RAG pipeline that turns any insurance policy PDF into a fully cited consumer report — plus a Tavily-powered company profile of the insurer.**

Built on **LangGraph · Ollama · ChromaDB · Tavily · Streamlit**.

- 100 % local LLM inference (your PDF never leaves the machine).
- 6 LangGraph nodes, sequential VRAM-safe model loading.
- 18-section consumer report rendered as Markdown + structured JSON.
- New ★ **Company Profile** node uses Tavily to research the insurer's claim settlement ratio, recent disputes, customer reviews, ratings, market share, and overall credibility.

📓 Want to run it on Google Colab + Cloudflare Tunnel instead?
**[Open `insureiq_colab.ipynb` in Colab](https://colab.research.google.com/github/aksri648/INSURE-IQ/blob/main/insureiq_colab.ipynb)** — full guide in [`insureiq_colab_deployment.md`](./insureiq_colab_deployment.md).

---

## Pipeline at a Glance

```
PDF
 ▼
[OCR Agent]            llava → JSON per page → offload
 ▼
[RAG Agent]            clause chunk → nomic-embed → ChromaDB
 ▼
[Web Research]         Tavily light lookups (insurer, regulatory)
 ▼
[Analyst Agent]        deepseek-r1 → 18 grounded section JSONs → offload
 ▼
[Company Profile] ★    Tavily ×7 facets → LLM synthesis → trust score
 ▼
[Report Compiler]      Markdown + structured JSON
 ▼
Streamlit UI           tabs · downloads (.md, .json)
```

For the full architecture see [`insureiq_architecture_diagram.md`](./insureiq_architecture_diagram.md).

---

## Project Layout

```
ai_insurance/
├── agents/
│   ├── state.py                    # PolicyState TypedDict
│   ├── ocr_agent.py                # LLaVA → JSON per page → offload
│   ├── rag_agent.py                # Chunker + ChromaDB + retrieval
│   ├── web_research_agent.py       # Heuristic insurer + light Tavily
│   ├── analyst_agent.py            # 18-section grounded prompts
│   ├── company_profile_agent.py    # Tavily ×7 facets + LLM synthesis  ★
│   └── compiler_agent.py           # Markdown report + structured JSON
├── utils/
│   └── model_config.py             # VRAM-aware model selection
├── graph.py                        # LangGraph wiring (6 nodes)
├── app.py                          # Streamlit UI (3 tabs + downloads)
├── requirements.txt
├── setup.sh                        # One-shot local setup
├── run.sh                          # Launch Streamlit
├── insureiq_colab.ipynb            # Colab + Cloudflare Tunnel notebook
├── insureiq_colab_deployment.md    # Colab deployment guide
├── insureiq_architecture_diagram.md
└── README.md  (this file)
```

---

## Run It Locally

Requires: Linux, Python 3.10+, NVIDIA GPU (T4-class or better) with recent driver, `curl`, `git`.

### 1. One-time setup

```bash
git clone https://github.com/aksri648/INSURE-IQ.git
cd INSURE-IQ
./setup.sh
```

`setup.sh` will:
- Install Ollama and start the server on `:11434`.
- Detect VRAM and pull the right model sizes:
  - `> 35 GiB`: `llava:13b` + `deepseek-r1:14b` + `nomic-embed-text`
  - otherwise: `llava:7b`  + `deepseek-r1:7b`  + `nomic-embed-text`
- Create a Python `.venv`, install `requirements.txt`.
- Write `/tmp/model_config.env` with the resolved model names.
- Copy `.env.example` to `.env`.

### 2. (Recommended) Add your Tavily key

Edit `.env`:

```
TAVILY_API_KEY=tvly-xxxxxxxxxxxx
```

Without a key the **Web Research** and **Company Profile** agents short-circuit cleanly (pipeline still completes; profile tab shows "unavailable").

Get a free key at <https://tavily.com>.

### 3. Launch

```bash
./run.sh
```

Open <http://localhost:8501>, drop a policy PDF, click **Analyze Policy**.

---

## Manual Setup (skip `setup.sh`)

```bash
# 1. Ollama
curl -fsSL https://ollama.com/install.sh | sh
nohup ollama serve >/tmp/ollama.log 2>&1 &
ollama pull nomic-embed-text
ollama pull llava:7b
ollama pull deepseek-r1:7b

# 2. Python
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Optional: Tavily
echo "TAVILY_API_KEY=tvly-..." > .env

# 4. Run
streamlit run app.py
```

---

## The Report

Every analysis produces the same Markdown structure:

```
# Your Insurance Policy Review
## Quick Summary  → At a Glance · Key Takeaways
# Basic Policy Information
# What Is Covered?
# How Much Protection Do You Actually Get?
# What Is NOT Covered?
# When Does Coverage Start?            (waiting periods)
# What Costs Will You Still Pay Yourself?
# Is the Premium Worth It?
# How Does the Claim Process Work?
# Can the Policy Be Renewed or Cancelled?
# Important Legal Terms You Should Know
# Important Definitions That Could Affect Claims
# Potential Risks and Concerns
# How Likely Is a Claim to Be Rejected?
# Real-Life Examples
# Hidden Surprises We Found
# How This Policy Compares to Others
# Plain English Summary
# Final Verdict
   Detailed Scores (8 axes, each /10)
   Overall Rating /100 · Confidence · Recommendation
# Company Profile  ★  (Tavily research)
   Overview · Claim Settlement Ratio · Recent Disputes ·
   Customer Reviews · Ratings · Market Share · Credibility ·
   Overall Assessment · Trust Score /100 · Sources Consulted
```

### Citations

Every analyst claim must cite a chunk:

```
[Source: chunk_0042, Page 12, General Exclusions]
[EXTERNAL: CRISIL AAA financial strength rating]
```

If a piece of information is not in the PDF the analyst is required to write exactly `"Not specified in the policy"` — no fabrication.

### Verdict bands

| Overall Rating | Recommendation     | Risk Level |
|---|---|---|
| 75–100 | `GOOD_VALUE`        | `LOW`    |
| 55–74  | `BUY_WITH_CAUTION`  | `MEDIUM` |
| 35–54  | `REVIEW_NEEDED`     | `HIGH`   |
| 0–34   | `AVOID`             | `HIGH`   |

---

## The Streamlit UI

Three tabs:

- **📄 Full Report** — renders the Markdown report verbatim.
- **🏢 Company Profile** — structured Tavily research with collapsible source URLs.
- **🧾 Raw JSON** — the full structured report (for programmatic consumers).

Two download buttons: `.md` (consumer report) and `.json` (structured report with citations).

---

## Configuration

| Env var | Purpose | Default |
|---|---|---|
| `TAVILY_API_KEY` | Enables Web Research + Company Profile | unset → both skipped |
| `OCR_MODEL` | Override OCR model | auto from VRAM |
| `ANALYST_MODEL` | Override analyst model | auto from VRAM |
| `EMBED_MODEL` | Override embedding model | `nomic-embed-text` |
| `INSUREIQ_MODEL_CONFIG` | Path to resolved-model config file | `/tmp/model_config.env` |

`.env` is loaded by `app.py` via `python-dotenv`.

---

## Design Notes

- **Sequential model loading.** OCR offloads (`ollama.generate(..., keep_alive=0)`) before the analyst loads. The analyst model is reused by the Company Profile node for snippet synthesis, then offloaded once. Fits on a single 15 GB T4.
- **Per-section retrieval.** The analyst defines 18 distinct queries — exclusions retrieve "exclusions not covered excluded conditions limitations", waiting periods retrieve "waiting period initial cooling pre-existing disease PED maternity", etc. This keeps the prompt context tight and grounded.
- **Strict per-section JSON schemas.** Each section returns a fixed shape that maps 1-to-1 to a markdown heading. The compiler walks the keys and fills the template — no string-glue templating from LLM prose.
- **Tavily company profile is a real agent.** It issues 7 advanced searches, preserves source URLs, and runs a second LLM pass for synthesis. The trust score and "Recent Disputes" bullets are derived from public sources, not the policy.
- **Graceful degradation.** Missing Tavily key → skip cleanly. Missing chunks → "Not specified in the policy". Malformed LLM JSON → preserved error string + empty schema.

---

## Privacy

- Policy PDFs never leave the host.
- ChromaDB is in-memory and per-session.
- Tavily receives only the heuristically extracted insurer name + canned query templates — never policy text.

---

## Repo

<https://github.com/aksri648/INSURE-IQ>
