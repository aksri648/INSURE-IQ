# ⬡ InsureIQ — AI Policy Analyst SaaS
### SYSTEM ARCHITECTURE · v1.0
**Multi-Agent RAG · Sequential Ollama Loading · Grounded Citation Engine**

---

## TAB 1 — ARCHITECTURE

### System Layer Stack

```
┌──────────────────────────────────────────────────────────────────┐
│  ⬡  PRESENTATION LAYER                                  3 comps  │
│     Streamlit UI · Session State Manager · Report Renderer       │
└──────────────────────────────┬───────────────────────────────────┘
                               ↓ ↓ ↓
┌──────────────────────────────────────────────────────────────────┐
│  ◈  ORCHESTRATION LAYER — LangGraph Multi-Agent         5 comps  │
│     Supervisor · OCR Agent · Analyst Agent · Web Research ·      │
│     Report Compiler                                              │
└──────────────────────────────┬───────────────────────────────────┘
                               ↓ ↓ ↓
┌──────────────────────────────────────────────────────────────────┐
│  ⬟  RAG & MEMORY LAYER                                  4 comps  │
│     Document Chunker · ChromaDB · Embedding Model ·              │
│     Citation Tracker                                             │
└──────────────────────────────┬───────────────────────────────────┘
                               ↓ ↓ ↓
┌──────────────────────────────────────────────────────────────────┐
│  ⬠  MODEL LAYER — Sequential Loading via Ollama         3 comps  │
│     OCR Model (LLaVA) · DeepSeek R1 · Ollama Runtime            │
└──────────────────────────────┬───────────────────────────────────┘
                               ↓ ↓ ↓
┌──────────────────────────────────────────────────────────────────┐
│  ◇  EXTERNAL SERVICES                                   2 comps  │
│     Tavily Search API · PDF Processor (PyMuPDF)                  │
└──────────────────────────────────────────────────────────────────┘
```

---

### Layer 1 — Presentation Layer

| Component | Description |
|-----------|-------------|
| 🖥 **Streamlit UI** | Policy Upload · Analysis Dashboard · Report Viewer |
| 🔁 **Session State Manager** | Multi-session isolation · Upload progress · Result caching |
| 📄 **Cited Report Renderer** | Inline citations · Section highlights · PDF export |

---

### Layer 2 — Orchestration Layer (LangGraph Multi-Agent)

| Component | Description |
|-----------|-------------|
| 🧠 **Supervisor Agent** | Routes tasks · Manages agent lifecycle · Aggregates results |
| 👁 **OCR Agent** | Loads LLaVA/Mistral-OCR via Ollama · Extracts structured text from PDF pages · Offloads model after run |
| 🔍 **Analyst Agent** | Loads DeepSeek-R1 via Ollama · Grounded RAG reasoning · Offloads model after run |
| 🌐 **Web Research Agent** | Tavily Search for company profile · Industry norms · Complex term definitions |
| 📋 **Report Compiler Agent** | Assembles structured JSON report · Formats citations · Validates grounding |

---

### Layer 3 — RAG & Memory Layer

| Component | Description |
|-----------|-------------|
| ✂️ **Document Chunker** | Semantic splitting · Clause-aware boundaries · Metadata tagging |
| 🗄 **Vector Store (ChromaDB)** | Per-session policy embeddings · Similarity search · Source-chunk retrieval |
| 🔢 **Embedding Model** | nomic-embed-text via Ollama · Local, no data leakage |
| 🔗 **Citation Tracker** | Maps every claim → source chunk → PDF page |

---

### Layer 4 — Model Layer (Sequential Loading via Ollama)

| Component | Description |
|-----------|-------------|
| 📸 **OCR Model (LLaVA / Mistral-OCR)** | STEP 1: Load → PDF page images → structured text extraction → Offload |
| 🤖 **DeepSeek R1 (Reasoner)** | STEP 2: Load → RAG context + policy text → deep analytical reasoning → Offload |
| ⚙️ **Ollama Runtime** | Sequential model orchestration · GPU VRAM management · API abstraction |

---

### Layer 5 — External Services

| Component | Description |
|-----------|-------------|
| 🔎 **Tavily Search API** | Insurer company profile · Claim settlement ratios · Regulatory standing |
| 📑 **PDF Processor (PyMuPDF)** | Page-to-image rendering · Text layer extraction · Metadata parsing |

---

### Key Design Decisions

#### 🟡 Sequential Model Loading
- OCR model loads → runs → unloads from VRAM
- DeepSeek R1 loads fresh → reasons → unloads
- Prevents VRAM overflow on consumer GPUs
- Ollama API handles lifecycle management

#### 🟢 Grounding Strategy
- Every claim maps to source PDF chunk
- Chunk → page number → sentence range stored
- Tavily results clearly flagged as external
- Hallucination gate: reject uncited claims

#### 🟣 LangGraph Agent Design
- StateGraph with typed state schema
- Conditional edges for model routing
- Retry logic per agent node
- Human-in-loop checkpoint support

#### 🔵 Privacy & Security
- 100% local inference — no cloud LLM calls
- Per-session ChromaDB isolation
- PDF never leaves local server
- Tavily only receives non-sensitive terms

---

## TAB 2 — AGENT FLOW

### End-to-End Pipeline · Sequential Execution

```
 ①  USER
     │  trigger
     ▼
 ②  SUPERVISOR AGENT
     │  dispatch
     ▼
 ③  OCR AGENT
     │  store chunks
     ▼
 ④  RAG PIPELINE (ChromaDB)
     │  RAG retrieval
     ▼
 ⑤  WEB RESEARCH AGENT
     │  enrichment
     ▼
 ⑥  ANALYST AGENT
     │  merge
     ▼
 ⑦  REPORT COMPILER AGENT
     │  cited report
     ▼
 ⑧  STREAMLIT UI
```

---

### Step-by-Step Detail

---

**`01` — USER**
> **Upload Insurance PDF**

Streamlit file uploader accepts PDF. Session ID generated. File written to temp storage.

`OUTPUT: PDF file path + session_id`

---

**`02` — SUPERVISOR AGENT**
> **Initialize LangGraph StateGraph**

Reads PDF metadata. Builds initial state object. Dispatches to OCR Agent node.

`OUTPUT: Initialized graph state with pdf_path, session_id`

---

**`03` — OCR AGENT**
> **Load OCR Model → Extract Text → Offload**

`ollama.pull('llava')` → Convert PDF pages to images (PyMuPDF) → Send each image to LLaVA → Collect structured text per page → `ollama.delete('llava')` frees VRAM.

`OUTPUT: List[{page_num, raw_text, tables, clauses}]`

---

**`04` — RAG PIPELINE**
> **Chunk → Embed → Store**

Semantic chunking by clause boundaries. nomic-embed-text generates embeddings. ChromaDB stores with metadata: `{page, section, clause_type}`.

`OUTPUT: Populated vector store, chunk_ids[]`

---

**`05` — WEB RESEARCH AGENT**
> **Tavily Search — Insurer & Term Enrichment**

Extract insurer name + complex terms from OCR text. Tavily queries: claim settlement ratio, regulatory status, term definitions. Results labeled `[EXTERNAL]`.

`OUTPUT: insurer_profile{}, term_definitions{}`

---

**`06` — ANALYST AGENT**
> **Load DeepSeek R1 → RAG + Reason → Offload**

`ollama.pull('deepseek-r1')` → For each report section, retrieve top-k chunks from ChromaDB → Build grounded prompt with chunks as context → DeepSeek R1 reasons step-by-step → Extracts benefits, exclusions, risks, verdicts → `ollama.delete('deepseek-r1')`.

`OUTPUT: section_analyses[] with inline citations`

---

**`07` — REPORT COMPILER AGENT**
> **Assemble Structured Report**

Merges all section analyses. Validates every claim has a source citation. Integrates Tavily external data with `[EXTERNAL]` tags. Generates confidence scores. Builds final JSON report.

`OUTPUT: PolicyReport{} JSON with full citations`

---

**`08` — STREAMLIT UI**
> **Render Cited Analysis Report**

Renders each section with expandable citation panels. Clicking a citation highlights the source PDF page. Overall Analyst Score displayed. PDF export available.

`OUTPUT: Interactive report + downloadable PDF`

---

## TAB 3 — REPORT SCHEMA

### Generated Report Structure · 9 Analysis Sections

---

**01 · EXECUTIVE SUMMARY**
- `policy_name`
- `insurer`
- `policy_type`
- `analyst_score /100`
- `one_line_verdict`
- `key_flags[]`

---

**02 · POLICY OVERVIEW**
- `coverage_type`
- `sum_insured`
- `policy_tenure`
- `premium_amount`
- `renewal_terms`
- `citations[]`

---

**03 · COVERAGE & BENEFITS**
- `core_benefits[]`
- `add_on_riders[]`
- `benefit_limits{}`
- `waiting_periods{}`
- `citations[]`

---

**04 · EXCLUSIONS & LIMITATIONS**
- `permanent_exclusions[]`
- `temp_exclusions[]`
- `sub_limits{}`
- `hidden_clauses[]`
- `citations[]`

---

**05 · PREMIUM ANALYSIS**
- `premium_breakdown{}`
- `value_for_money_score`
- `comparable_market_rate`
- `loading_factors[]`
- `citations[]`

---

**06 · CLAIMS PROCESS**
- `claim_filing_steps[]`
- `documents_required[]`
- `settlement_timeline`
- `cashless_hospitals[]`
- `citations[]`

---

**07 · RISK ASSESSMENT**
- `high_risks[]`
- `medium_risks[]`
- `low_risks[]`
- `risk_score /10`
- `citations[]`

---

**08 · INSURER PROFILE**
- `claim_settlement_ratio` ⚠️ `[EXTERNAL]`
- `solvency_ratio` ⚠️ `[EXTERNAL]`
- `regulatory_status` ⚠️ `[EXTERNAL]`
- `customer_reviews_summary` ⚠️ `[EXTERNAL]`

---

**09 · ANALYST VERDICT**
- `suitable_for[]`
- `not_suitable_for[]`
- `buy_recommendation`
- `negotiate_points[]`
- `alternatives_to_consider`

---

### Citation Object Schema

```
Citation {
  citation_id:    "CIT-042"
  claim:          "Pre-existing diseases excluded for 4 years"
  source_type:    "POLICY_DOCUMENT" | "EXTERNAL"
  chunk_id:       "chunk_017"
  page_number:    12
  section_title:  "General Exclusions"
  verbatim_text:  "...diseases diagnosed 48 months prior to..."
  confidence:     0.94
  tavily_url:     null  // only for EXTERNAL type
}
```

---

*INSUREIQ · MULTI-AGENT RAG ARCHITECTURE*
*OLLAMA · LANGGRAPH · CHROMADB · TAVILY · STREAMLIT*
