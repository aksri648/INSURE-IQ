# ⬡ InsureIQ — AI Policy Analyst SaaS
### SYSTEM ARCHITECTURE · v2.0
**Multi-Agent RAG · Sequential Ollama Loading · 18-Section Consumer Report · Tavily Company Intelligence**

---

## TAB 1 — ARCHITECTURE

### System Layer Stack

```
┌──────────────────────────────────────────────────────────────────┐
│  ⬡  PRESENTATION LAYER                                  3 comps  │
│     Streamlit UI · Tabbed Report View · MD + JSON Export         │
└──────────────────────────────┬───────────────────────────────────┘
                               ↓ ↓ ↓
┌──────────────────────────────────────────────────────────────────┐
│  ◈  ORCHESTRATION LAYER — LangGraph Multi-Agent         6 nodes  │
│     OCR · RAG · Web Research · Analyst · Company Profile ·       │
│     Report Compiler                                              │
└──────────────────────────────┬───────────────────────────────────┘
                               ↓ ↓ ↓
┌──────────────────────────────────────────────────────────────────┐
│  ⬟  RAG & MEMORY LAYER                                  4 comps  │
│     Clause-aware Chunker · ChromaDB · nomic-embed-text ·         │
│     Citation Tracker                                             │
└──────────────────────────────┬───────────────────────────────────┘
                               ↓ ↓ ↓
┌──────────────────────────────────────────────────────────────────┐
│  ⬠  MODEL LAYER — Sequential Loading via Ollama         3 comps  │
│     OCR (LLaVA) · DeepSeek R1 (Analyst + Synthesizer) · Ollama   │
└──────────────────────────────┬───────────────────────────────────┘
                               ↓ ↓ ↓
┌──────────────────────────────────────────────────────────────────┐
│  ◇  EXTERNAL SERVICES                                   2 comps  │
│     Tavily Search API (7 facet queries) · PyMuPDF                │
└──────────────────────────────────────────────────────────────────┘
```

---

### Layer 1 — Presentation Layer

| Component | Description |
|-----------|-------------|
| 🖥 **Streamlit UI** | PDF Upload · Progress bar with 6 stages · Headline metrics (Rating / Risk / Recommendation / Insurer) |
| 🗂 **Tabbed Report View** | `📄 Full Report` (rendered markdown) · `🏢 Company Profile` (Tavily) · `🧾 Raw JSON` |
| ⬇️ **Export** | Download `.md` (consumer report) and `.json` (full structured report with citations) |

---

### Layer 2 — Orchestration Layer (LangGraph StateGraph)

| Node | File | Description |
|------|------|-------------|
| 👁 **OCR Agent** | `agents/ocr_agent.py` | Loads `llava` via Ollama → 2× zoom PDF→PNG via PyMuPDF → per-page structured JSON (`section`, `text`, `tables`, `clause_numbers`) → offloads model with `keep_alive=0`. |
| ⬟ **RAG Agent** | `agents/rag_agent.py` | Clause-aware chunking → `nomic-embed-text` embeddings → per-session ChromaDB collection. |
| 🌐 **Web Research Agent** | `agents/web_research_agent.py` | Heuristic insurer-name extraction + lightweight Tavily lookups (claim settlement, regulatory). Feeds context into analyst prompts. |
| 🤖 **Analyst Agent** | `agents/analyst_agent.py` | Loads `deepseek-r1` → runs **18 grounded section prompts**, each with its own retrieval query + strict JSON schema → offloads. |
| 🏢 **Company Profile Agent** | `agents/company_profile_agent.py` | **Tavily-powered** insurer intelligence: 7 advanced-search facets → LLM synthesis → trust score. |
| 📋 **Report Compiler** | `agents/compiler_agent.py` | Stitches all section JSON + company profile into the consumer markdown template. Computes overall rating + risk + recommendation. |

---

### Layer 3 — RAG & Memory Layer

| Component | Description |
|-----------|-------------|
| ✂️ **Clause-Aware Chunker** | Paragraph-boundary splitting · `chunk_size=400` · preserves `{page, section, chunk_index}` metadata. |
| 🗄 **ChromaDB (in-memory)** | One collection per session (`policy_<session_id[:8]>`). |
| 🔢 **Embedding Model** | `nomic-embed-text` via Ollama — local, no data leakage. |
| 🔗 **Citation Tracker** | Citations emitted as `[Source: chunk_XXXX, Page N, Section Title]` inside each finding, also aggregated into a top-level `all_citations[]`. |

---

### Layer 4 — Model Layer (Sequential Loading via Ollama)

| Component | Description |
|-----------|-------------|
| 📸 **OCR Model (`llava:7b` / `llava:13b`)** | **STEP 1:** Load → digitize each PDF page to JSON → offload (`ollama.generate(..., keep_alive=0)`). |
| 🤖 **DeepSeek R1 (`deepseek-r1:7b` / `deepseek-r1:14b`)** | **STEP 2:** Load → run 18 analyst sections → re-used by Company Profile node for Tavily synthesis → offload. |
| ⚙️ **Ollama Runtime** | Sequential model orchestration · VRAM auto-eviction · single HTTP API at `:11434`. |

`utils/model_config.py` picks model sizes from detected VRAM:
- `>35 GiB` → `llava:13b` + `deepseek-r1:14b`
- otherwise → `llava:7b` + `deepseek-r1:7b`

---

### Layer 5 — External Services

| Component | Description |
|-----------|-------------|
| 🔎 **Tavily Search API** | Used by **two** nodes: (a) light-touch Web Research, (b) deep Company Profile (7 facet queries, `search_depth="advanced"`). |
| 📑 **PyMuPDF** | Page-to-image rendering for OCR · text layer extraction · page metadata. |

---

### Key Design Decisions

#### 🟡 Sequential Model Loading
- OCR runs to completion, then explicitly offloads via `keep_alive=0`.
- DeepSeek R1 loads fresh for analyst + company-profile synthesis, then offloads.
- Embeddings run on the lightweight `nomic-embed-text` (always resident).
- Prevents VRAM overflow on consumer GPUs (T4 15 GB works).

#### 🟢 Grounding Strategy
- Every analyst section is fed top-k chunks from ChromaDB scoped to that section's query.
- System prompt mandates `[Source: chunk_id, Page X, Section Y]` for every factual claim.
- Missing data must be reported verbatim as `"Not specified in the policy"` — no fabrication.
- Tavily results are tagged `[EXTERNAL]` everywhere they appear.

#### 🟣 LangGraph Design
- Single `StateGraph(PolicyState)` with typed state.
- Linear edges: `ocr → embed_store → web_research → analyst → company_profile → compiler → END`.
- Each node returns a status string that the UI maps to a progress %.

#### 🔵 Privacy & Security
- 100 % local inference — no cloud LLM calls.
- ChromaDB lives in-process (no disk persistence between sessions).
- PDF never leaves the host.
- Tavily receives only the insurer name + canned query templates, never policy text.

---

## TAB 2 — AGENT FLOW

### End-to-End Pipeline · 6 Sequential Nodes

```
 ①  USER
     │  upload PDF
     ▼
 ②  OCR AGENT  ── load llava → page-by-page JSON → offload
     │
     ▼
 ③  RAG AGENT  ── clause chunk + embed → ChromaDB (per-session)
     │
     ▼
 ④  WEB RESEARCH  ── Tavily light lookups (insurer, regulatory)
     │
     ▼
 ⑤  ANALYST AGENT  ── load deepseek-r1 → 18 section JSONs → offload
     │
     ▼
 ⑥  COMPANY PROFILE  ── Tavily ×7 facets → LLM synthesis → trust score
     │
     ▼
 ⑦  REPORT COMPILER  ── markdown report + structured JSON
     │
     ▼
 ⑧  STREAMLIT UI  ── tabs + downloads
```

---

### Step-by-Step Detail

---

**`01` — USER**
> Upload Insurance PDF

Streamlit file uploader accepts PDF. A UUID `session_id` is generated and the file is written to a temp path.

`OUTPUT: pdf_path, session_id`

---

**`02` — OCR AGENT**
> Load LLaVA → Extract Text → Offload

`PyMuPDF` rasterizes each page at 2× zoom → base64 PNG → sent to `llava` with a strict "return JSON only" prompt → parsed into `{section, text, tables, clause_numbers}` per page → model offloaded via `ollama.generate(..., keep_alive=0)`.

`OUTPUT: ocr_text = [{page, section, text, tables, clause_numbers}]`

---

**`03` — RAG AGENT**
> Chunk → Embed → Store

Paragraph-boundary chunking with `chunk_size=400`. `nomic-embed-text` generates embeddings. ChromaDB stores per-session collection with `{page, section, chunk_index}` metadata.

`OUTPUT: chunks[], ChromaDB collection populated`

---

**`04` — WEB RESEARCH AGENT**
> Heuristic insurer detect + light Tavily lookups

Scans first 3 OCR pages for a line containing "insurance / insurer / ltd / limited". Runs two Tavily queries (claim settlement, regulatory) used as side context for the analyst.

`OUTPUT: insurer_name, external_research{}`

If `TAVILY_API_KEY` is missing this node is a clean no-op.

---

**`05` — ANALYST AGENT**
> Load DeepSeek R1 → 18 grounded section JSONs → Offload

For each of the 18 sections defined in `SECTION_SPECS`:
1. Retrieve top-k chunks using that section's tailored query.
2. Build a prompt with chunks + the section's strict JSON schema + section-specific guidance.
3. Call `deepseek-r1` (temperature 0.2), strip `<think>` blocks, parse JSON safely.
4. Collect any chunk IDs cited into a global `citations[]`.

After all sections complete the model is offloaded.

`OUTPUT: section_analyses{...18 keys...}, citations[]`

---

**`06` — COMPANY PROFILE AGENT** ★ new
> Tavily ×7 facets → LLM synthesis → trust score

Issues 7 advanced-search Tavily queries (templated on `insurer_name`):

| Facet | Query template |
|---|---|
| `company_overview` | `{insurer} insurance company overview history headquarters founded parent group` |
| `claim_settlement_ratio` | `{insurer} claim settlement ratio latest IRDAI annual report percent` |
| `recent_disputes` | `{insurer} insurance complaints lawsuits regulatory action penalty disputes recent` |
| `customer_reviews` | `{insurer} customer reviews complaints feedback service experience` |
| `ratings` | `{insurer} insurance rating CRISIL ICRA AM Best Moody's financial strength rating` |
| `market_share` | `{insurer} market share India insurance sector premium income ranking` |
| `credibility` | `{insurer} solvency ratio IRDAI registration license status credibility trustworthiness` |

Trimmed snippets are passed back through `deepseek-r1` for synthesis into a clean JSON:

```jsonc
{
  "company_overview": "...",
  "claim_settlement_ratio": "...",
  "recent_disputes": ["..."],
  "customer_reviews_summary": "...",
  "ratings": "...",
  "market_share": "...",
  "credibility": "...",
  "overall_assessment": "...",
  "trust_score": 0          // 0-100
}
```

Falls back to concatenated raw snippets if the LLM call fails. Source URLs are preserved per facet for inline citation.

`OUTPUT: company_profile{ insurer, available, summary{}, facets{} }`

---

**`07` — REPORT COMPILER**
> Stitch markdown report + structured JSON

Computes the overall rating (LLM-provided `overall_rating`, else avg per-section score ×10), maps to verdict label, then builds the full consumer markdown report (see TAB 3). Renders the Company Profile section at the end of the markdown with Tavily source URLs.

`OUTPUT: final_report{ headline, sections, company_profile, all_citations, markdown }, report_markdown`

---

**`08` — STREAMLIT UI**
> Render tabs + downloads

Three tabs: rendered Markdown report, structured Company Profile view (with collapsible Tavily sources), Raw JSON. Two download buttons: `.md` and `.json`.

`OUTPUT: Interactive report + downloads`

---

## TAB 3 — REPORT SCHEMA (v2)

The compiler emits **one** markdown document with **18 top-level sections** plus a final **Company Profile** section. The Streamlit UI renders this verbatim.

```
# Your Insurance Policy Review

## Quick Summary
   ### At a Glance
     Policy Name · Insurance Company · Policy Type ·
     Overall Rating · Risk Level · Recommendation
   ### Key Takeaways
     Best Things · Biggest Concerns · Hidden Conditions ·
     Most Important Things Not Covered · Claim Ease

# Basic Policy Information
   ## Policy Details
     Policy Name · Number · Start · End · Renewal · Term
   ## Contact Information
     Customer Support · Claims Support · Emergency Assistance

# What Is Covered?
   Main Benefits · Additional Benefits · Optional Add-Ons ·
   Where and When You Are Covered

# How Much Protection Do You Actually Get?
   Total Coverage Amount · Treatment/Event Limits ·
   Annual Limits · Lifetime Limits · Coverage Caps

# What Is NOT Covered?
   Rejection Situations · Conditions Not Covered ·
   Temporary Restrictions · Important Exclusions ·
   Vague/Ambiguous Clauses

# When Does Coverage Start?
   Initial Waiting Period · Condition-specific Waiting ·
   Pre-existing Disease Waiting · Maternity Waiting

# What Costs Will You Still Pay Yourself?
   Deductibles · Co-Payments · Cost Sharing ·
   Expected Out-of-Pocket Expenses

# Is the Premium Worth It?
   Current Premium · Future Increase Risk ·
   Extra Charges and Fees · Overall Cost Assessment

# How Does the Claim Process Work?
   Steps · Documents · Deadlines · Approval Timeline ·
   Common Rejection Reasons

# Can the Policy Be Renewed or Cancelled?
   Renewal Rules · Grace Period · Company Cancel ·
   Customer Cancel · Lapse Situations

# Important Legal Terms You Should Know
   Your Rights · Insurer Rights · Dispute Resolution ·
   Fraud Rules · Beneficiary/Nominee Rules

# Important Definitions That Could Affect Claims
   Key Terms · Unusual Definitions · Claim-Impacting Definitions

# Potential Risks and Concerns
   High-Risk Clauses · Customer-Unfriendly Terms ·
   Hidden Restrictions · Areas Needing Attention

# How Likely Is a Claim to Be Rejected?
   Overall Claim Risk Rating · Top Denial Reasons ·
   Real-Life Rejection Scenarios

# Real-Life Examples
   Hospitalization · Accident · Critical Illness ·
   Major Surgery · Worst-Case Financial Scenario

# Hidden Surprises We Found
   Unexpected Restrictions · Hidden Costs ·
   Clauses Most People Miss · Pre-Buy Checklist

# How This Policy Compares to Others
   Coverage Quality · Claim Friendliness · Value for Money ·
   Transparency · Overall Competitiveness

# Plain English Summary
   What We Like · What We Don't Like ·
   Best For · Should Consider Other Options

# Final Verdict
   ## Detailed Scores  (each /10)
     Coverage · Coverage Limits · Exclusions ·
     Waiting Periods · Costs · Claims Process ·
     Renewal Protection · Transparency
   ## Overall Rating  /100
   ## Confidence Level  LOW | MEDIUM | HIGH
   ## Final Recommendation
     GOOD_VALUE | BUY_WITH_CAUTION | REVIEW_NEEDED | AVOID

# Company Profile  _(External — Tavily Research)_
   Company Overview · Claim Settlement Ratio ·
   Recent Disputes · Customer Reviews ·
   Ratings · Market Share · Credibility ·
   Overall Company Assessment · Trust Score /100 ·
   Sources Consulted (linked)
```

---

### Citation Convention

Inline, anywhere in the analyst output:

```
[Source: chunk_0042, Page 12, General Exclusions]
[EXTERNAL: CRISIL AAA financial strength rating]
```

Aggregated to the top of the JSON report as `all_citations[]`:

```jsonc
{
  "citation_id": "CIT-007",
  "section": "exclusions",
  "chunk_ref": "chunk_0042",
  "source_type": "POLICY_DOCUMENT"   // or "EXTERNAL"
}
```

---

### Verdict & Risk Bands

| Overall Rating | Recommendation | Risk Level |
|---|---|---|
| 75–100 | `GOOD_VALUE` | `LOW` |
| 55–74  | `BUY_WITH_CAUTION` | `MEDIUM` |
| 35–54  | `REVIEW_NEEDED` | `HIGH` |
| 0–34   | `AVOID` | `HIGH` |

The compiler prefers the LLM-provided `overall_rating` (final_verdict section). If absent or out of range, it averages the per-section scores (×10).

---

## PROJECT LAYOUT

```
ai_insurance/
├── agents/
│   ├── state.py                    # PolicyState (TypedDict)
│   ├── ocr_agent.py                # LLaVA → JSON per page → offload
│   ├── rag_agent.py                # Chunk + ChromaDB + retrieval helper
│   ├── web_research_agent.py       # Heuristic insurer + light Tavily
│   ├── analyst_agent.py            # 18-section grounded prompts
│   ├── company_profile_agent.py    # Tavily ×7 facets + LLM synthesis  ★ new
│   └── compiler_agent.py           # Markdown report + structured JSON
├── utils/
│   └── model_config.py             # VRAM-aware model selection
├── graph.py                        # LangGraph wiring (6 nodes)
├── app.py                          # Streamlit UI (3 tabs + downloads)
├── requirements.txt
├── setup.sh                        # Ollama + models + venv + deps
├── run.sh                          # Launch Streamlit
├── insureiq_colab.ipynb            # Colab + Cloudflare Tunnel deployment
├── insureiq_colab_deployment.md    # Colab deployment guide
├── insureiq_architecture_diagram.md (this file)
└── README.md
```

---

*INSUREIQ · MULTI-AGENT RAG ARCHITECTURE · v2.0*
*OLLAMA · LANGGRAPH · CHROMADB · TAVILY · STREAMLIT*
