# ⬡ InsureIQ — AI Policy Analyst SaaS
### SYSTEM ARCHITECTURE · v3.0
**Multi-Agent RAG · Deterministic Validator · Hallucination-Proof · LaTeX-Compiled PDF**

---

## TAB 1 — ARCHITECTURE

### System Layer Stack

```
┌──────────────────────────────────────────────────────────────────┐
│  ⬡  PRESENTATION LAYER                                  3 comps  │
│     Streamlit UI (3 columns) · Live Agent Flowchart ·            │
│     LaTeX viewer + Download Report PDF (single button)           │
└──────────────────────────────┬───────────────────────────────────┘
                               ↓ ↓ ↓
┌──────────────────────────────────────────────────────────────────┐
│  ◈  ORCHESTRATION LAYER — LangGraph Multi-Agent         7 nodes  │
│     OCR · RAG · Web Research · Analyst · Company Profile ·       │
│     Validator · Report Compiler                                  │
└──────────────────────────────┬───────────────────────────────────┘
                               ↓ ↓ ↓
┌──────────────────────────────────────────────────────────────────┐
│  ⬟  RAG & MEMORY LAYER                                  4 comps  │
│     Clause-aware Chunker · ChromaDB · nomic-embed-text ·         │
│     chunk_index (verbatim store for validator)                   │
└──────────────────────────────┬───────────────────────────────────┘
                               ↓ ↓ ↓
┌──────────────────────────────────────────────────────────────────┐
│  ⬠  MODEL LAYER — Sequential Loading via Ollama         3 comps  │
│     OCR (LLaVA) · DeepSeek R1 (Analyst + Synth) · Ollama         │
└──────────────────────────────┬───────────────────────────────────┘
                               ↓ ↓ ↓
┌──────────────────────────────────────────────────────────────────┐
│  ◇  EXTERNAL SERVICES & RENDERING                       3 comps  │
│     Tavily Search API (2 + 7 queries) · PyMuPDF (OCR) ·          │
│     Tectonic (LaTeX → PDF)                                       │
└──────────────────────────────────────────────────────────────────┘
```

---

### Layer 1 — Presentation Layer

Three-column Streamlit layout.

| Column | Components | Description |
|---|---|---|
| **Left** | Upload + Analyze button | PDF file uploader, primary "Analyze Policy" button, post-run insurer/rating/risk/verdict summary + validator counts. |
| **Middle** | Live agent flowchart | Seven rounded-rectangle nodes stacked vertically with downward arrows. Borders animate cyan around the currently running node; completed nodes turn green; pending stay grey. State driven by a background-thread event queue that drains on each Streamlit rerun. |
| **Right** | LaTeX viewer + PDF button | Read-only LaTeX source viewer + a single **⬇️ Download Report PDF** button below it. No markdown / JSON views exposed — the PDF is the sole deliverable. |

---

### Layer 2 — Orchestration Layer (LangGraph StateGraph)

| Node | File | Description |
|------|------|-------------|
| 👁 **OCR Agent** | `agents/ocr_agent.py` | Loads `llava` via Ollama → 2× zoom PDF→PNG via PyMuPDF → per-page structured JSON (`section`, `text`, `tables`, `clause_numbers`) → offloads model (`keep_alive=0`). |
| ⬟ **RAG Indexer** | `agents/rag_agent.py` | Clause-aware chunking → `nomic-embed-text` embeddings → per-session ChromaDB collection. Also writes a deterministic `chunk_index: {chunk_id → text, page, section}` into state — used by the validator. |
| 🌐 **Web Research Agent** | `agents/web_research_agent.py` | Heuristic insurer name extraction from first OCR pages + 2 light Tavily lookups. Feeds context, never a citation source. |
| 🤖 **Analyst Agent** | `agents/analyst_agent.py` | Loads `deepseek-r1`. For each of **14 sections**, retrieves top-k chunks and asks for **findings only**. Each finding must include `label`, `claim`, `chunk_id`, `verbatim_quote` (exact substring), `page`, `section`. No prose summaries. Offloads model when done. |
| 🏢 **Company Profile Agent** | `agents/company_profile_agent.py` | **Tavily-powered** insurer intelligence: 7 advanced-search facets → LLM synthesis → trust score. |
| 🛡 **Validator Agent** ★ | `agents/validator_agent.py` | **Deterministic, no LLM.** For each finding: (a) confirm `chunk_id` exists in `chunk_index`; (b) check `verbatim_quote` is literally a substring of the chunk (whitespace + case normalised); (c) check page consistency. Tags each finding **TRUSTED** (exact + page match), **NEEDS_HUMAN_REVIEW** (fuzzy ≥70% word overlap, or page mismatch), or drops it entirely. Emits the global `citations[]` array. |
| 📋 **Report Compiler** | `agents/compiler_agent.py` | Walks validated findings into a LaTeX document mirroring the markdown structure. Inlines coloured TRUSTED / NEEDS HUMAN REVIEW boxes via `tcolorbox`. Calls `utils/pdf_builder.py` to compile via `tectonic` → `pdflatex` → reportlab notice. |

Edges: `ocr → embed_store → web_research → analyst → company_profile → validator → compiler → END`.

Each node returns `active_node: <its_name>` and a `status` string. The Streamlit worker thread streams `pipeline.stream(...)` events into a queue; the UI thread drains the queue on each rerun and updates the flowchart classes.

---

### Layer 3 — RAG & Memory Layer

| Component | Description |
|-----------|-------------|
| ✂️ **Clause-Aware Chunker** | Paragraph-boundary splitting · `chunk_size=400` · preserves `{page, section, chunk_index}` metadata. |
| 🗄 **ChromaDB (in-memory)** | One collection per session (`policy_<session_id[:8]>`). Used for semantic retrieval. |
| 🔢 **Embedding Model** | `nomic-embed-text` via Ollama — local, no data leakage. |
| 🔗 **`chunk_index`** ★ | Dict `chunk_id → {text, page, section}` of **verbatim** chunk text, used by the validator for substring matching. Does not pass through Chroma — kept in raw form precisely so the validator can do exact string comparisons without embedding noise. |

---

### Layer 4 — Model Layer (Sequential Loading via Ollama)

| Component | Description |
|-----------|-------------|
| 📸 **OCR Model** (`llava:7b` / `llava:13b`) | **STEP 1:** Load → digitize each PDF page to JSON → offload (`ollama.generate(..., keep_alive=0)`). |
| 🤖 **DeepSeek R1** (`deepseek-r1:7b` / `:14b`) | **STEP 2:** Load → 14-section analyst findings → reused for Company Profile synthesis → offload. |
| ⚙️ **Ollama Runtime** | Sequential model orchestration · VRAM auto-eviction · single HTTP API at `:11434`. |

`utils/model_config.py` picks model sizes by detected VRAM. Override via `OCR_MODEL`, `ANALYST_MODEL`, `EMBED_MODEL` env vars.

---

### Layer 5 — External Services & Rendering

| Component | Description |
|---|---|
| 🔎 **Tavily Search API** | Used by **two** nodes: Web Research (2 queries), Company Profile (7 advanced-search facet queries). |
| 📑 **PyMuPDF** | Page-to-image rendering for OCR · page metadata. |
| 📄 **Tectonic** ★ | Self-contained LaTeX engine. Compiles `compiler_agent.py`'s output to PDF bytes. Falls back to `pdflatex` if tectonic missing; falls back further to a reportlab notice PDF that embeds the LaTeX source so the Download button always returns content. |

---

### Key Design Decisions

#### 🟡 Sequential Model Loading
- OCR runs to completion, offloads via `keep_alive=0`.
- DeepSeek R1 loads once for analyst + company-profile synthesis, then offloads.
- Embedding model (`nomic-embed-text`) is lightweight and stays resident.

#### 🟢 Hallucination Proofing
- Analyst is constrained to emit **only structured findings** (no free-form summaries, narratives, or assessments).
- Every finding **must** carry a `verbatim_quote` field — an exact substring of the cited chunk.
- The validator does its check **in Python**, not via another LLM. A regex-normalised substring match returns a boolean. No model can fool the validator with fluent text.
- Findings whose quote is not present in the cited chunk are **dropped before the report is built**. The reader never sees them.
- Findings with weaker matches (fuzzy word overlap ≥ 70%, or correct chunk but wrong page) are kept but flagged `NEEDS_HUMAN_REVIEW` in amber.

#### 🟣 LangGraph Design
- Single `StateGraph(PolicyState)` with typed state including `chunk_index`, `validated_sections`, `validation_report`, `latex_source`, `pdf_bytes`.
- Each node returns `active_node` so the UI knows which one is running.
- The Streamlit worker uses `pipeline.stream(...)` and pushes events to a `queue.Queue` drained on each rerun.

#### 🔵 LaTeX as the Sole Output
- The Markdown report is no longer materialised. The compiler emits LaTeX directly, structured to render exactly like the markdown spec when viewed.
- `tcolorbox` macros provide the inline coloured TRUSTED / NEEDS HUMAN REVIEW tags.
- Tectonic produces a single PDF — that's the only thing the user downloads.

#### ⚪ Privacy & Security
- 100% local inference — no cloud LLM calls.
- ChromaDB is in-memory, per-session.
- PDF never leaves the host; tectonic runs locally.
- Tavily receives only the heuristically extracted insurer name + canned query templates, never policy text.

---

## TAB 2 — AGENT FLOW

### End-to-End Pipeline · 7 Sequential Nodes

```
 ①  USER
     │  upload PDF (left panel)
     ▼
 ②  OCR AGENT             ── load llava → per-page JSON → offload
     │
     ▼
 ③  RAG INDEXER           ── chunk + embed → ChromaDB + chunk_index
     │
     ▼
 ④  WEB RESEARCH          ── heuristic insurer name + light Tavily
     │
     ▼
 ⑤  ANALYST AGENT         ── load deepseek-r1 → 14 sections of FINDINGS
     │                       (each with verbatim_quote + chunk_id) → offload
     ▼
 ⑥  COMPANY PROFILE       ── Tavily ×7 facets → LLM synthesis → trust score
     │
     ▼
 ⑦  VALIDATOR AGENT  ★    ── deterministic substring check
     │                       TRUSTED / NEEDS_HUMAN_REVIEW / drop
     ▼
 ⑧  REPORT COMPILER       ── LaTeX → tectonic → PDF bytes
     │
     ▼
 ⑨  STREAMLIT UI          ── flowchart highlights live · single PDF download
```

---

### Step-by-Step Detail

---

**`01` — USER**
> Upload Insurance PDF (left column)

A UUID `session_id` is generated and the file is written to a temp path. The pipeline runs in a background thread; the UI polls a queue and re-renders the flowchart.

---

**`02` — OCR AGENT**
> Load LLaVA → per-page JSON → Offload

PyMuPDF rasterises each page at 2× zoom → base64 PNG → sent to `llava` with a strict "return JSON only" prompt → parsed into `{section, text, tables, clause_numbers}` → model offloaded via `keep_alive=0`.

---

**`03` — RAG INDEXER**
> Chunk → Embed → Store + chunk_index

Paragraph-boundary chunking with `chunk_size=400`. `nomic-embed-text` embeddings → ChromaDB. The **verbatim chunk_index** is built here and threaded into state — this is the validator's source of truth.

---

**`04` — WEB RESEARCH AGENT**
> Heuristic insurer detect + 2 Tavily lookups

Scans the first 3 OCR pages for an insurer-name line. Two Tavily queries (claim settlement, regulatory) provide minor side context. No-ops cleanly if `TAVILY_API_KEY` is missing.

---

**`05` — ANALYST AGENT**
> Load DeepSeek R1 → 14 sections of findings → Offload

The 14 sections (policy_basics, coverage, coverage_limits, exclusions, waiting_periods, out_of_pocket, premium_analysis, claims_process, renewal_cancellation, legal_terms, definitions, risks_concerns, claim_rejection_risk, hidden_surprises) each have:

- a tailored retrieval query (top-k chunks),
- a fixed list of allowed `label` slots (mapped 1-to-1 to the markdown headings),
- a strict per-section prompt demanding only verifiable findings.

System prompt requires every finding to carry a `verbatim_quote` that is an exact substring of the cited chunk. No summaries, no overall assessments, no narratives — those are not verifiable so they aren't requested.

Findings missing required fields or referencing unknown labels are silently dropped here, before validation.

---

**`06` — COMPANY PROFILE AGENT**
> Tavily ×7 facets → LLM synthesis → trust score

Issues 7 advanced-search Tavily queries (templated on `insurer_name`): company overview, claim settlement ratio, recent disputes, customer reviews, ratings (CRISIL/ICRA/AM Best/Moody's), market share, credibility/solvency. Trimmed snippets are synthesised by DeepSeek R1 into a structured JSON profile + `trust_score` (0–100).

Falls back to concatenated raw snippets if the LLM call fails. Source URLs preserved per facet.

---

**`07` — VALIDATOR AGENT** ★
> Deterministic verification, no LLM

For each finding emitted by the analyst:

```python
def _validate_finding(finding, chunk_index):
    chunk = chunk_index.get(finding["chunk_id"])
    if not chunk:
        return NEEDS_HUMAN_REVIEW            # cited a non-existent chunk
    if normalize(quote) in normalize(chunk["text"]):
        return TRUSTED if page_matches else NEEDS_HUMAN_REVIEW
    if word_overlap_ratio(quote, chunk["text"]) >= 0.70:
        return NEEDS_HUMAN_REVIEW            # paraphrased, possibly OK
    return DROP                              # quote not in the policy
```

Aggregates `validation_report = {counts: {trusted, review, dropped, total}, trusted_ratio}` and emits the global `citations[]` array. Confidence in the final report is derived from `trusted_ratio`.

---

**`08` — REPORT COMPILER**
> LaTeX → tectonic → PDF bytes

Walks each section's validated findings into a LaTeX document. Heading hierarchy mirrors the markdown spec exactly. Each finding renders as:

```latex
\item Plain-English claim. \tagTrusted  [CIT-007 · chunk_0042 · Page 12 · Section]
      \\\textit{``verbatim quote here''}
```

Headline card uses `tcolorbox` with the verdict-band colour. `utils/pdf_builder.py` runs:

```
tectonic --keep-logs --outdir <tmp> report.tex
   ↓ if missing
pdflatex -interaction=nonstopmode -halt-on-error report.tex (run twice)
   ↓ if missing
reportlab fallback: a notice PDF containing the LaTeX source
```

Returns `pdf_bytes` and `latex_source` in state.

---

**`09` — STREAMLIT UI**
> Flowchart highlights + single PDF download

The Streamlit thread drains the event queue, updates `completed_nodes` and `active_node`, and re-renders. The right column's textarea shows the (read-only) LaTeX source; the **⬇️ Download Report PDF** button serves the compiled bytes.

---

## TAB 3 — REPORT SCHEMA (v3)

The PDF mirrors this markdown structure exactly:

```
Your Insurance Policy Review
  Quick Summary
    At a Glance       (headline card)
    Key Takeaways     (best things / concerns / hidden / not covered / claim ease)

Basic Policy Information
What Is Covered?
How Much Protection Do You Actually Get?
What Is NOT Covered?
When Does Coverage Start?
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
Company Profile  ★  (Tavily external research)
  Overview · CSR · Disputes · Reviews · Ratings · Market Share ·
  Credibility · Overall Assessment · Trust Score · Sources Consulted
```

Every finding ships with a coloured tag and a citation badge:

```
[TRUSTED]            CIT-007 · chunk_0042 · Page 12 · General Exclusions
[NEEDS HUMAN REVIEW] CIT-018 · chunk_0061 · Page 21
```

The first page of the PDF includes a **"How to Read This Report"** legend explaining the tags so non-technical consumers understand them.

### Verdict bands

| Overall Rating | Recommendation     | Risk Level |
|---|---|---|
| 75–100 | `GOOD VALUE`        | `LOW`    |
| 55–74  | `BUY WITH CAUTION`  | `MEDIUM` |
| 35–54  | `REVIEW NEEDED`     | `HIGH`   |
| 0–34   | `AVOID`             | `HIGH`   |

### Confidence

Derived from validator output:

| trusted / total | Confidence |
|---|---|
| ≥ 0.8 | `HIGH` |
| ≥ 0.5 | `MEDIUM` |
| < 0.5 | `LOW` |

The compiler also penalises the overall rating (up to −15 points) proportional to the share of findings that needed human review.

---

## PROJECT LAYOUT

```
ai_insurance/
├── agents/
│   ├── state.py                    # PolicyState
│   ├── ocr_agent.py
│   ├── rag_agent.py                # populates chunk_index for validator
│   ├── web_research_agent.py
│   ├── analyst_agent.py            # findings-only, verifiable
│   ├── company_profile_agent.py
│   ├── validator_agent.py          ★ deterministic verification
│   └── compiler_agent.py           # LaTeX builder
├── utils/
│   ├── model_config.py
│   └── pdf_builder.py              ★ tectonic / pdflatex / reportlab
├── graph.py                        # 7 nodes + NODE_ORDER/NODE_LABELS exports
├── app.py                          # 3-column UI with live flowchart
├── requirements.txt
├── setup.sh                        # installs tectonic + ollama + venv + deps
├── run.sh
├── insureiq_colab.ipynb
├── insureiq_colab_deployment.md
├── insureiq_architecture_diagram.md  (this file)
└── README.md
```

---

*INSUREIQ · MULTI-AGENT RAG ARCHITECTURE · v3.0*
*OLLAMA · LANGGRAPH · CHROMADB · TAVILY · STREAMLIT · TECTONIC*
