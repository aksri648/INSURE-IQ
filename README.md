# ⬡ InsureIQ — AI Insurance Policy Analyst

**Local-first, hallucination-proof multi-agent RAG pipeline. Upload an insurance policy PDF, get a fully-cited LaTeX-compiled PDF report with every finding tagged TRUSTED or NEEDS HUMAN REVIEW by a deterministic validator agent.**

Built on **LangGraph · Ollama · ChromaDB · Tavily · Streamlit · Tectonic (LaTeX→PDF)**.

- 100% local LLM inference; PDF never leaves the host.
- **7 LangGraph nodes**, sequential VRAM-safe model loading.
- **Validator agent** does a deterministic substring check on every analyst claim against the policy text — no LLM in the verification loop, so the validator itself cannot hallucinate.
- Final report is rendered as **LaTeX → PDF**. The Streamlit UI shows the source on the right and exposes a single **Download Report PDF** button.
- Tavily-powered **Company Profile** agent: claim settlement ratio, recent disputes, customer reviews, ratings, market share, credibility, trust score.

📓 Run it on Colab + Cloudflare Tunnel:
**[Open `insureiq_colab.ipynb` in Colab](https://colab.research.google.com/github/aksri648/INSURE-IQ/blob/main/insureiq_colab.ipynb)** — full guide in [`insureiq_colab_deployment.md`](./insureiq_colab_deployment.md).

---

## Pipeline at a Glance

```
PDF
 ▼
[OCR Agent]              llava → per-page JSON → offload
 ▼
[RAG Indexer]            clause chunk → nomic-embed → ChromaDB + chunk_index
 ▼
[Web Research]           Tavily light lookups (insurer, regulatory)
 ▼
[Analyst Agent]          deepseek-r1 → verifiable findings only
                         each finding ships a verbatim_quote + chunk_id
 ▼
[Company Profile] ★      Tavily ×7 facets → LLM synthesis → trust score
 ▼
[Validator Agent] ★      Deterministic substring check on every claim.
                         Tags each finding TRUSTED or NEEDS_HUMAN_REVIEW.
                         Drops findings whose quote isn't in the policy.
 ▼
[Report Compiler]        LaTeX source → tectonic → PDF bytes
 ▼
Streamlit UI             Left: upload · Middle: live agent flowchart
                         Right: LaTeX viewer + Download Report PDF
```

For the full architecture see [`insureiq_architecture_diagram.md`](./insureiq_architecture_diagram.md).

---

## Hallucination Defense

The system is designed so that no LLM output reaches the report unless it can be tied to a real piece of policy text:

| Stage | Constraint |
|---|---|
| Analyst prompt | Returns findings only. Each finding **must** include `verbatim_quote` (an exact substring of the cited chunk) + `chunk_id` + `page`. No free-form prose fields. |
| Analyst parsing | Findings missing `chunk_id`, `verbatim_quote`, or with an unknown label are dropped before they reach validation. |
| Validator | Pure Python. Normalises whitespace and case, then checks if `verbatim_quote` is literally inside the cited chunk. **No LLM call here** — the verifier itself cannot hallucinate. |
| Tagging | **TRUSTED**: exact substring match + page consistency. **NEEDS_HUMAN_REVIEW**: only fuzzy word-overlap match (≥70%), or page mismatch. **Dropped**: quote not in the cited chunk at all. |
| Compiler | Only consumes validated findings. Each finding renders the original verbatim quote in italics under the consumer-facing claim, plus a coloured TRUSTED / NEEDS HUMAN REVIEW tag and the chunk/page reference. |

The result: any line in the PDF either matches a real run of characters in the policy, or it has a visible amber tag warning the reader to verify it manually.

---

## Project Layout

```
ai_insurance/
├── agents/
│   ├── state.py                    # PolicyState TypedDict
│   ├── ocr_agent.py                # LLaVA → per-page JSON → offload
│   ├── rag_agent.py                # Chunker + ChromaDB + chunk_index
│   ├── web_research_agent.py       # Insurer name + light Tavily
│   ├── analyst_agent.py            # Verifiable-only findings
│   ├── company_profile_agent.py    # Tavily ×7 facets + LLM synthesis
│   ├── validator_agent.py          # ★ Deterministic verification
│   └── compiler_agent.py           # LaTeX builder + tectonic
├── utils/
│   ├── model_config.py             # VRAM-aware model selection
│   └── pdf_builder.py              # ★ tectonic / pdflatex / reportlab fallback
├── graph.py                        # LangGraph wiring (7 nodes)
├── app.py                          # Streamlit UI (3 columns + live flowchart)
├── requirements.txt
├── setup.sh                        # Ollama + tectonic + venv + deps
├── run.sh                          # Launch Streamlit
├── insureiq_colab.ipynb            # Colab + Cloudflare Tunnel notebook
├── insureiq_colab_deployment.md
├── insureiq_architecture_diagram.md
└── README.md
```

---

## Streamlit UI

Three columns:

| Column | Contents |
|---|---|
| **Left** | PDF upload + **Analyze Policy** button. Once complete, shows insurer / rating / risk / verdict + the validator counts. |
| **Middle** | Vertical agent flowchart — one rounded box per node. The currently-executing node has a **glowing animated cyan border**. Completed nodes turn green. Pending nodes stay grey. |
| **Right** | Read-only **LaTeX source viewer** + a single **⬇️ Download Report PDF** button below it. No markdown view, no JSON view — just the compiled PDF you'll actually share. |

---

## Run It Locally

Requires: Linux, Python 3.10+, NVIDIA GPU with recent driver, `curl`, `git`.

### 1. One-time setup

```bash
git clone https://github.com/aksri648/INSURE-IQ.git
cd INSURE-IQ
./setup.sh
```

`setup.sh` will:
- Install **tectonic** (self-contained LaTeX engine) — required to render the PDF.
- Install Ollama and start the server on `:11434`.
- Pull VRAM-sized models:
  - `> 35 GiB`: `llava:13b` + `deepseek-r1:14b` + `nomic-embed-text`
  - otherwise: `llava:7b`  + `deepseek-r1:7b`  + `nomic-embed-text`
- Create `.venv`, install `requirements.txt`, write `/tmp/model_config.env`.
- Scaffold `.env` from `.env.example`.

### 2. (Recommended) Add your Tavily key

```
# .env
TAVILY_API_KEY=tvly-xxxxxxxxxxxx
```

Without it, the Web Research and Company Profile agents short-circuit cleanly.

### 3. Launch

```bash
./run.sh
```

Open <http://localhost:8501> → drop a PDF → click **Analyze Policy** → watch the flowchart light up → press **Download Report PDF**.

---

## Manual Setup (skip `setup.sh`)

```bash
# 1. tectonic (LaTeX → PDF)
curl -fsSL https://drop-sh.fullyjustified.net | sh
sudo mv tectonic /usr/local/bin/

# 2. Ollama
curl -fsSL https://ollama.com/install.sh | sh
nohup ollama serve >/tmp/ollama.log 2>&1 &
ollama pull nomic-embed-text
ollama pull llava:7b
ollama pull deepseek-r1:7b

# 3. Python
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 4. Optional: Tavily
echo "TAVILY_API_KEY=tvly-..." > .env

# 5. Run
streamlit run app.py
```

If `tectonic` is unavailable the compiler falls back to `pdflatex`, and if neither is available it produces a notice PDF (via reportlab) that contains the LaTeX source — the Download button always returns something.

---

## The Report

Every analysis produces the same structured LaTeX document, rendered identically to a markdown viewer:

```
Your Insurance Policy Review
  Quick Summary  → At a Glance · Key Takeaways
Basic Policy Information
What Is Covered?
How Much Protection Do You Actually Get?
What Is NOT Covered?
When Does Coverage Start?            (waiting periods)
What Costs Will You Still Pay Yourself?
Is the Premium Worth It?
How Does the Claim Process Work?
Can the Policy Be Renewed or Cancelled?
Important Legal Terms You Should Know
Important Definitions That Could Affect Claims
Potential Risks and Concerns
How Likely Is a Claim to Be Rejected?
Hidden Surprises We Found
Final Verdict
  Detailed Scores · Overall Rating · Risk · Confidence · Recommendation
Company Profile  ★  (Tavily research)
  Overview · CSR · Disputes · Reviews · Ratings · Market Share ·
  Credibility · Overall Assessment · Trust Score · Sources Consulted
```

### Inline tag styling

Each finding renders like:

```
• Pre-existing diseases are excluded for 48 months from inception.
  [TRUSTED] [CIT-007 • chunk_0042 • Page 12 • General Exclusions]
  "pre-existing diseases excluded for 48 months from inception"
```

vs. an unverifiable one:

```
• Maternity benefits are available after 24 months.
  [NEEDS HUMAN REVIEW] [CIT-018 • chunk_0061 • Page 21]
  "maternity benefit after twenty four month wait"
  Reviewer note: Quote paraphrased (62% word overlap)
```

The PDF's "How to Read This Report" section explains the legend to the consumer.

### Verdict bands

| Overall Rating | Recommendation     | Risk Level |
|---|---|---|
| 75–100 | `GOOD VALUE`        | `LOW`    |
| 55–74  | `BUY WITH CAUTION`  | `MEDIUM` |
| 35–54  | `REVIEW NEEDED`     | `HIGH`   |
| 0–34   | `AVOID`             | `HIGH`   |

Confidence is computed from the validator's TRUSTED ratio:

| Trusted/Total | Confidence |
|---|---|
| ≥ 0.8 | `HIGH` |
| ≥ 0.5 | `MEDIUM` |
| < 0.5 | `LOW` |

The overall rating is also penalised (up to −15) when many findings need review.

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

## Privacy

- Policy PDFs never leave the host.
- ChromaDB is in-memory and per-session.
- Tavily receives only the heuristically extracted insurer name + canned query templates — never policy text.

---

## Repo

<https://github.com/aksri648/INSURE-IQ>
