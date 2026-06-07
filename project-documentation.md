# InsureIQ — Project Documentation

> A complete reference for the InsureIQ codebase. Start here if you want to understand how the system works, why it's built this way, or how to extend it.

**Version:** 3.0
**Repo:** <https://github.com/aksri648/INSURE-IQ>
**Stack:** LangGraph · Ollama · ChromaDB · Tavily · Streamlit · Tectonic (LaTeX → PDF)

---

## Table of Contents

1. [What InsureIQ Is](#1-what-insureiq-is)
2. [Design Principles](#2-design-principles)
3. [Pipeline Overview (7 Nodes)](#3-pipeline-overview-7-nodes)
4. [Repository Layout](#4-repository-layout)
5. [Pipeline State (`PolicyState`)](#5-pipeline-state-policystate)
6. [Agent-by-Agent Reference](#6-agent-by-agent-reference)
    - [6.1 OCR Agent](#61-ocr-agent)
    - [6.2 RAG Indexer](#62-rag-indexer)
    - [6.3 Web Research Agent](#63-web-research-agent)
    - [6.4 Analyst Agent](#64-analyst-agent)
    - [6.5 Company Profile Agent (Tavily ×7)](#65-company-profile-agent-tavily-7)
    - [6.6 Validator Agent (Deterministic)](#66-validator-agent-deterministic)
    - [6.7 Report Compiler (LaTeX → PDF)](#67-report-compiler-latex--pdf)
7. [Hallucination Defense](#7-hallucination-defense)
8. [Report Structure (PDF)](#8-report-structure-pdf)
9. [Citation & Trust Tags](#9-citation--trust-tags)
10. [Streamlit UI](#10-streamlit-ui)
11. [Models & VRAM Strategy](#11-models--vram-strategy)
12. [Configuration](#12-configuration)
13. [Local Setup](#13-local-setup)
14. [Google Colab + Cloudflare Tunnel](#14-google-colab--cloudflare-tunnel)
15. [Privacy & Security](#15-privacy--security)
16. [Testing & Smoke Checks](#16-testing--smoke-checks)
17. [Extending InsureIQ](#17-extending-insureiq)
18. [Troubleshooting](#18-troubleshooting)
19. [Glossary](#19-glossary)
20. [Changelog](#20-changelog)

---

## 1. What InsureIQ Is

InsureIQ ingests an insurance policy PDF and produces a **fully-cited consumer report** rendered as a typeset PDF. Every finding in the report carries:

- The exact policy clause it relies on (verbatim quote).
- A page reference.
- A trust tag — **TRUSTED** if a deterministic validator confirmed the quote is in the policy, **NEEDS HUMAN REVIEW** if the LLM paraphrased or got the page wrong.

A separate **Company Profile** agent uses Tavily to research the insurer (claim settlement ratio, recent disputes, customer reviews, ratings, market share, credibility) and appends a profile section to the report with a 0–100 trust score and clickable source URLs.

**Why it exists.** Insurance policies are long, dense, and routinely contain exclusions or caps that determine whether a consumer's claim will succeed years later. A generic LLM summary is dangerous: a polite-sounding paraphrase that omits or fabricates a clause is worse than no summary at all. InsureIQ is built so that *nothing the LLM says can reach the report unless a Python function has independently verified that the policy actually contains those words.*

---

## 2. Design Principles

| Principle | How it shows up in the code |
|---|---|
| **Local-first inference** | All LLM calls go to a local Ollama daemon. No OpenAI/Anthropic/Gemini SDKs. |
| **Hallucination-proof citations** | Analyst is constrained to emit only structured findings with a `verbatim_quote` field. The validator (pure Python) verifies the quote is a substring of the cited chunk. |
| **Determinism where it counts** | The verifier is not an LLM. It cannot hallucinate. It either confirms a substring match or it does not. |
| **Sequential VRAM management** | Models load on demand and are explicitly evicted (`keep_alive=0`). Fits on a 15 GB T4. |
| **Strict per-section schemas** | The analyst's section labels are a closed vocabulary. Off-vocab labels are dropped before the report is built. |
| **Graceful degradation** | Missing Tavily key → Web Research + Company Profile skip cleanly. Missing tectonic → fall back to `pdflatex` → reportlab notice PDF. Missing chunks → finding is dropped or tagged for review. |
| **Single deliverable** | The UI exposes exactly one download: the compiled PDF. No JSON, no markdown, no temptation to use an unverified intermediate. |

---

## 3. Pipeline Overview (7 Nodes)

```
PDF
 ▼
[1] OCR Agent              llava → per-page JSON → offload
 ▼
[2] RAG Indexer            chunk + embed → ChromaDB + chunk_index
 ▼
[3] Web Research           heuristic insurer detect + light Tavily
 ▼
[4] Analyst Agent          deepseek-r1 → verifiable findings only
                           each ships {verbatim_quote, chunk_id, page}
 ▼
[5] Company Profile        Tavily ×7 facets → LLM synthesis → trust score
 ▼
[6] Validator Agent  ★     deterministic substring check
                           tag TRUSTED / NEEDS_HUMAN_REVIEW / drop
 ▼
[7] Report Compiler        LaTeX source → tectonic → PDF bytes
 ▼
Streamlit UI               live flowchart + Download Report PDF
```

LangGraph wires these as a linear `StateGraph(PolicyState)`. Each node returns `active_node: <its_name>` so the UI can highlight which one is currently running.

The pipeline is run in a background thread by the Streamlit app; the UI polls a `queue.Queue` of node-completion events on each rerun.

---

## 4. Repository Layout

```
INSURE-IQ/
├── agents/
│   ├── state.py                    # PolicyState TypedDict
│   ├── ocr_agent.py                # LLaVA → per-page JSON
│   ├── rag_agent.py                # Chunker + ChromaDB + chunk_index
│   ├── web_research_agent.py       # Insurer detect + light Tavily
│   ├── analyst_agent.py            # 14-section findings-only schema
│   ├── company_profile_agent.py    # Tavily ×7 + LLM synthesis
│   ├── validator_agent.py          # ★ Deterministic verification
│   └── compiler_agent.py           # LaTeX builder + tectonic call
├── utils/
│   ├── model_config.py             # VRAM-aware model picker
│   └── pdf_builder.py              # tectonic → pdflatex → reportlab
├── graph.py                        # LangGraph wiring + NODE_ORDER/LABELS
├── app.py                          # Streamlit UI (3 columns + flowchart)
├── requirements.txt
├── setup.sh                        # Ollama + tectonic + venv + deps
├── run.sh                          # Launches Streamlit
├── insureiq_colab.ipynb            # Colab + Cloudflare Tunnel notebook
├── insureiq_colab_deployment.md    # Colab guide
├── insureiq_architecture_diagram.md# Architecture diagram doc
├── project-documentation.md        # (this file)
├── README.md                       # Repo landing page
├── .env.example
└── .gitignore
```

---

## 5. Pipeline State (`PolicyState`)

Defined in `agents/state.py`. A single typed `TypedDict` threaded through every node.

| Field | Type | Written by | Used by |
|---|---|---|---|
| `session_id` | `str` | App | All nodes (per-session Chroma collection name) |
| `pdf_path` | `str` | App | OCR |
| `ocr_text` | `list` | OCR | RAG, Web Research |
| `chunks` | `list` | RAG | (informational) |
| `chunk_index` | `dict` | RAG | **Validator** — verbatim `chunk_id → {text, page, section}` |
| `insurer_name` | `str` | Web Research | Company Profile, Compiler |
| `external_research` | `dict` | Web Research | Analyst (context only) |
| `company_profile` | `dict` | Company Profile | Compiler |
| `section_analyses` | `dict` | Analyst | Validator |
| `validated_sections` | `dict` | Validator | Compiler |
| `validation_report` | `dict` | Validator | Compiler |
| `citations` | `list` | Validator | Compiler |
| `final_report` | `dict` | Compiler | UI |
| `latex_source` | `str` | Compiler | UI (right panel viewer) |
| `pdf_bytes` | `bytes` | Compiler | UI (download button) |
| `status` | `str` | Every node | UI (progress text) |
| `active_node` | `str` | Every node | UI (flowchart highlight) |
| `error` | `Optional[str]` | (reserved) | — |

---

## 6. Agent-by-Agent Reference

### 6.1 OCR Agent

**File:** `agents/ocr_agent.py`
**Model:** `llava:7b` or `llava:13b` (auto-selected by VRAM)

PyMuPDF rasterises each page at 2× zoom into a base64 PNG. The image is sent to LLaVA with a strict "return JSON only" prompt asking for:

```json
{
  "section": "<heading or empty>",
  "text": "<verbatim text>",
  "tables": ["<markdown table>"],
  "clause_numbers": ["4.1", "4.2"]
}
```

Output is parsed page by page into `ocr_text: list[{page, section, text, tables, clause_numbers}]`. Malformed responses become `{"section": "", "text": "OCR Error page N: <err>", ...}` so the pipeline never aborts on a single bad page.

Critically, after all pages are processed:

```python
ollama.generate(model=ocr_model, prompt="", keep_alive=0)
```

This evicts LLaVA from VRAM so the analyst model can load cleanly. Without this, two large vision/reasoning models would compete for VRAM on a T4.

---

### 6.2 RAG Indexer

**File:** `agents/rag_agent.py`
**Model:** `nomic-embed-text` (always resident)

Three responsibilities:

1. **Clause-aware chunking.** Splits each page on paragraph boundaries with `chunk_size=400` characters. Each chunk gets a stable id `chunk_NNNN` and metadata `{page, section, chunk_index}`.
2. **Embed + store.** Embeddings are computed via `ollama.embeddings(model="nomic-embed-text", prompt=chunk_text)` and stored in an in-memory ChromaDB collection named `policy_<session_id[:8]>`.
3. **Build the verbatim `chunk_index`.** A `dict[chunk_id, {text, page, section}]` of *raw* chunk text. This is the validator's source of truth — independent of the vector store, so embedding noise can't affect verification.

Exposes a helper `retrieve_chunks(session_id, query, n_results=8)` that returns `list[(text, metadata)]` for the analyst.

---

### 6.3 Web Research Agent

**File:** `agents/web_research_agent.py`

Lightweight Tavily lookups that provide *side context* — never report citations. Two queries on insurer name (`claim settlement ratio IRDAI`, `IRDAI registration solvency ratio`) get stuffed into the analyst's prompt under an `[EXTERNAL]` banner.

Heuristic insurer-name extraction scans the first 3 OCR pages for a line containing `insurance / insurer / ltd / limited`. Used downstream by the Company Profile agent.

If `TAVILY_API_KEY` is missing the node is a no-op and the analyst sees no external research.

---

### 6.4 Analyst Agent

**File:** `agents/analyst_agent.py`
**Model:** `deepseek-r1:7b` or `deepseek-r1:14b`

The most important constraint in the whole system: **the analyst is restricted to emitting verifiable findings only.** No prose summaries, no "overall assessments", no narrative. Anything the validator can't check is not requested in the first place.

#### Sections

14 sections, each with:

- A retrieval query (used against ChromaDB to fetch top-k chunks).
- A fixed list of allowed `label` strings (the closed vocabulary the report renders).
- An optional per-section consumer-friendliness score (0–10).

```
policy_basics · coverage · coverage_limits · exclusions ·
waiting_periods · out_of_pocket · premium_analysis · claims_process ·
renewal_cancellation · legal_terms · definitions · risks_concerns ·
claim_rejection_risk · hidden_surprises
```

#### Finding schema

```json
{
  "findings": [
    {
      "label": "Important Exclusions",
      "claim": "Pre-existing diseases are excluded for 48 months from inception.",
      "chunk_id": "chunk_0042",
      "verbatim_quote": "pre-existing diseases excluded for 48 months from inception",
      "page": 12,
      "section": "General Exclusions"
    }
  ],
  "score": 4
}
```

Every finding **must** include `verbatim_quote` — an exact substring of the cited chunk. Findings missing required fields or referencing labels outside the closed vocabulary are dropped silently inside the analyst, before they ever reach validation.

#### System prompt

The system prompt sets the seven absolute rules: only cite from chunks; quotes must be exact; omit rather than invent; cite the actual chunk id; plain English; JSON only; no narrative fields. Temperature is set to `0.1` to keep paraphrasing pressure low.

#### Offload

After all 14 sections complete, the model is offloaded with `keep_alive=0` so the company-profile synthesis step can reuse the same VRAM cleanly.

---

### 6.5 Company Profile Agent (Tavily ×7)

**File:** `agents/company_profile_agent.py`

Issues 7 advanced-search Tavily queries against the heuristically extracted `insurer_name`:

| Facet | Query template |
|---|---|
| `company_overview` | `{insurer} insurance company overview history headquarters founded parent group` |
| `claim_settlement_ratio` | `{insurer} claim settlement ratio latest IRDAI annual report percent` |
| `recent_disputes` | `{insurer} insurance complaints lawsuits regulatory action penalty disputes recent` |
| `customer_reviews` | `{insurer} customer reviews complaints feedback service experience` |
| `ratings` | `{insurer} insurance rating CRISIL ICRA AM Best Moody's financial strength rating` |
| `market_share` | `{insurer} market share India insurance sector premium income ranking` |
| `credibility` | `{insurer} solvency ratio IRDAI registration license status credibility trustworthiness` |

Top snippets per facet (title + 400-char body + URL) are passed back to `deepseek-r1` for synthesis into:

```json
{
  "company_overview": "...",
  "claim_settlement_ratio": "...",
  "recent_disputes": ["..."],
  "customer_reviews_summary": "...",
  "ratings": "...",
  "market_share": "...",
  "credibility": "...",
  "overall_assessment": "...",
  "trust_score": 78
}
```

Falls back to concatenated raw snippets if the LLM call fails. Source URLs are kept per-facet so the report can render them as clickable footnotes.

All claims here are marked `[EXTERNAL]` — never confused with policy-document citations.

---

### 6.6 Validator Agent (Deterministic)

**File:** `agents/validator_agent.py`
**Model:** *none — pure Python.*

The validator is the load-bearing piece of the hallucination defense. It uses no LLM, so it cannot hallucinate itself.

#### Algorithm

```python
def _validate_finding(finding, chunk_index):
    chunk = chunk_index.get(finding["chunk_id"])
    if not chunk_id or not quote:
        → NEEDS_HUMAN_REVIEW (missing fields)
    if not chunk:
        → NEEDS_HUMAN_REVIEW (chunk_id not in policy index)

    if normalize(quote) in normalize(chunk.text):
        if page matches chunk.page:
            → TRUSTED (exact substring, correct page)
        else:
            → NEEDS_HUMAN_REVIEW (exact text, wrong page)

    if word_overlap_ratio(quote, chunk.text) >= 0.70:
        → NEEDS_HUMAN_REVIEW (paraphrased, possibly OK)

    → DROP (quote is not in the cited chunk at all)
```

`normalize` = lowercase + collapse all whitespace.
`word_overlap_ratio` = fraction of significant words (length > 2) from the quote that appear in the chunk.

#### Outputs

- `validated_sections`: same shape as `section_analyses` but each finding now has `trust_tag`, `validation_notes`, `validation_match`, and `citation_id`.
- `validation_report`: `{counts: {trusted, review, dropped, total}, trusted_ratio}`.
- `citations`: flat list with `{citation_id, section, label, chunk_id, page, section_title, trust_tag}`.

Findings that fail every check are dropped entirely. They never reach the report.

---

### 6.7 Report Compiler (LaTeX → PDF)

**File:** `agents/compiler_agent.py`
**Builder:** `utils/pdf_builder.py`

Walks validated findings into a LaTeX document. Heading hierarchy mirrors the markdown spec exactly so the rendered PDF looks like a polished version of what a markdown viewer would show.

#### Key LaTeX building blocks

- `titlesec` + `xcolor` — colored section headings.
- `tcolorbox` — the headline summary card and the inline trust tags.
- `hyperref` — clickable Tavily source URLs.
- `enumitem` + `parskip` — tight bullet lists between findings.
- `fancyhdr` — footer with "InsureIQ Policy Review · Page N".

Two macros provide the colored inline tags:

```latex
\newcommand{\tagTrusted}{...}   % green pill
\newcommand{\tagReview}{...}    % amber pill
```

#### How a finding renders

```latex
\item Pre-existing diseases are excluded for 48 months. \tagTrusted
      \textcolor{MutedGrey}{\scriptsize [CIT-007 · chunk_0042 · Page 12 · General Exclusions]}
      \\\textcolor{MutedGrey}{\scriptsize\itshape ``pre-existing diseases excluded for 48 months''}
```

For NEEDS_HUMAN_REVIEW findings the same structure with `\tagReview` and an additional reviewer note line (e.g. *"Quote paraphrased (62% word overlap)"*).

#### Overall scoring

The compiler computes the headline rating:

```python
base = mean(section_scores) * 10        # each section 0-10 → 0-100
penalty = (review_findings / total) * 15  # up to -15 if many review tags
overall = clamp(base - penalty, 0, 100)
```

| Overall | Recommendation | Risk Level |
|---|---|---|
| 75–100 | `GOOD VALUE` | `LOW` |
| 55–74 | `BUY WITH CAUTION` | `MEDIUM` |
| 35–54 | `REVIEW NEEDED` | `HIGH` |
| 0–34 | `AVOID` | `HIGH` |

Confidence is derived from the validator:

| trusted / total | Confidence |
|---|---|
| ≥ 0.8 | `HIGH` |
| ≥ 0.5 | `MEDIUM` |
| < 0.5 | `LOW` |

#### PDF builder cascade

`utils/pdf_builder.py::compile_latex_to_pdf(tex)` tries in order:

1. **`tectonic`** — self-contained engine, fetches packages on demand. Preferred.
2. **`pdflatex`** — full TeX Live. Runs twice for accurate refs.
3. **`reportlab`** notice — produces a clearly-labeled "LaTeX engine unavailable" PDF that embeds the LaTeX source so the Download button always returns *something* the user can read.

Returns `bytes`. Errors from each engine are logged but never raised.

---

## 7. Hallucination Defense

A summary of where guarantees come from, end to end.

| Layer | Guarantee |
|---|---|
| **Prompt** | The analyst is asked only for findings that include `verbatim_quote` + `chunk_id`. No "summary" or "overall assessment" fields exist, so the LLM has no slot to invent prose in. |
| **Parser** | Findings missing `chunk_id`, `verbatim_quote`, or with an out-of-vocab `label` are dropped inside the analyst before validation. |
| **Validator** | Pure Python substring match against the verbatim `chunk_index`. The validator itself has no LLM, so it cannot be fooled by fluency. |
| **Tag** | Each surviving finding wears one of two visible tags in the PDF. The reader is told upfront, on the first page, what each tag means. |
| **Drop** | Findings that don't appear in the cited chunk at all are removed completely. The reader never sees them. |
| **Compiler** | Only consumes `validated_sections`. Never reads `section_analyses` directly. |
| **Aggregations** | Overall rating and confidence are computed from validated data + validator stats, not from a final-verdict LLM call. |

The result is a deliberate trade-off: the report may **omit** facts the LLM hinted at but couldn't quote verbatim, but it will never **invent** a fact that isn't in the policy.

---

## 8. Report Structure (PDF)

```
Your Insurance Policy Review                       (centred title)
[Summary card: Policy Name · Insurer · Type ·
 Overall Rating · Risk · Recommendation]

How to Read This Report                            (legend for tags)

Quick Summary
  At a Glance                                      (refers to the card)
  Key Takeaways
    Best Things About This Policy
    Biggest Concerns
    Hidden Conditions to Know About
    Most Important Things Not Covered
    How Easy It May Be to Make a Claim

Basic Policy Information
  Policy Name · Policy Number · Insurer Name · Policy Type ·
  Start Date · End Date · Renewal Date · Policy Term ·
  Customer Support · Claims Support · Emergency Assistance

What Is Covered?
  Main Benefits · Additional Benefits · Optional Add-Ons ·
  Where and When You Are Covered

How Much Protection Do You Actually Get?
  Total Coverage Amount · Treatment/Event Limits ·
  Annual Limits · Lifetime Limits · Coverage Caps

What Is NOT Covered?
  Rejection Situations · Conditions Not Covered ·
  Temporary Restrictions · Important Exclusions ·
  Vague or Ambiguous Clauses

When Does Coverage Start?
  Initial Waiting Period · Condition-Specific Waiting ·
  Pre-Existing Disease Waiting · Maternity Waiting

What Costs Will You Still Pay Yourself?
  Deductibles · Co-Payments · Cost Sharing ·
  Expected Out-of-Pocket Expenses

Is the Premium Worth It?
  Current Premium · Future Increase Risk ·
  Extra Charges & Fees · Overall Cost Assessment

How Does the Claim Process Work?
  Steps · Documents · Deadlines · Approval Timeline ·
  Common Rejection Reasons

Can the Policy Be Renewed or Cancelled?
  Renewal Rules · Grace Period · Company Cancel ·
  Customer Cancel · Lapse Situations

Important Legal Terms You Should Know
  Your Rights · Insurer Rights · Dispute Resolution ·
  Fraud Rules · Beneficiary/Nominee Rules

Important Definitions That Could Affect Claims
  Key Terms · Unusual Definitions · Claim-Impacting Definitions

Potential Risks and Concerns
  High-Risk Clauses · Customer-Unfriendly Terms ·
  Hidden Restrictions · Areas Needing Attention

How Likely Is a Claim to Be Rejected?
  Top Denial Reasons · Real-Life Rejection Scenarios

Hidden Surprises We Found
  Unexpected Restrictions · Hidden Costs ·
  Clauses Most People Miss · Pre-Buy Checklist

Final Verdict
  Detailed Scores (8 axes /10) · Overall Rating /100 ·
  Risk Level · Confidence Level · Final Recommendation

Company Profile  (External — Tavily Research)
  Insurer · Company Overview · Claim Settlement Ratio ·
  Recent Disputes · Customer Reviews · Ratings ·
  Market Share · Credibility · Overall Assessment ·
  Trust Score /100 · Sources Consulted (linked)
```

Sections with no validated findings render with a muted italic note: *"Not specified in the policy."*

---

## 9. Citation & Trust Tags

Every finding shows one of two coloured tags inline:

```
[TRUSTED]            green pill — verbatim quote found in the cited chunk on the cited page
[NEEDS HUMAN REVIEW] amber pill — fuzzy match (≥70% word overlap) or page mismatch
```

A citation badge follows each tag:

```
CIT-007 · chunk_0042 · Page 12 · General Exclusions
```

The verbatim quote from the policy is then shown in italics on the next line:

```
"pre-existing diseases excluded for 48 months from inception"
```

The first page of the PDF includes a "How to Read This Report" legend with both badges and an explanation, so a non-technical consumer immediately understands what they mean.

The structured `citations[]` array (also returned in `final_report["citations"]`) carries:

```json
{
  "citation_id": "CIT-007",
  "section": "exclusions",
  "label": "Important Exclusions",
  "chunk_id": "chunk_0042",
  "page": 12,
  "section_title": "General Exclusions",
  "trust_tag": "TRUSTED"
}
```

---

## 10. Streamlit UI

Three columns (`app.py`):

| Column | Contents |
|---|---|
| **Left** | PDF uploader. *Analyze Policy* button. Post-run summary: insurer / rating / risk / verdict + validator counts (`X TRUSTED · Y NEEDS REVIEW · Z dropped`). |
| **Middle** | Vertical agent flowchart — one rounded box per node, downward arrows between them. The **active** node has a glowing animated cyan border (`@keyframes pulseBorder`); **completed** nodes turn green; **pending** stay grey. |
| **Right** | Read-only LaTeX source viewer + a single **⬇ Download Report PDF** button below it. The PDF is the only deliverable surfaced in the UI. |

#### How the live flowchart works

When the user clicks *Analyze Policy*:

1. The Streamlit thread starts a background worker that runs `pipeline.stream(initial_state)`.
2. For each event the worker pushes `{type: "node_done", node: <name>}` to a `queue.Queue`.
3. When the pipeline completes, the worker pushes `{type: "complete", latex, pdf, report}`.
4. The Streamlit thread `time.sleep(1.0)` + `st.rerun()` while the pipeline is running.
5. On each rerun it drains the queue into `st.session_state.completed_nodes` and `active_node`, then re-renders the flowchart with updated CSS classes.

Result: an interactive UI without long-blocking Streamlit callbacks.

---

## 11. Models & VRAM Strategy

`utils/model_config.py` detects total VRAM via `nvidia-smi` and picks model sizes:

| VRAM | OCR | Analyst / Synthesis | Embedding |
|---|---|---|---|
| > 35 GiB | `llava:13b` | `deepseek-r1:14b` | `nomic-embed-text` |
| else / CPU | `llava:7b` | `deepseek-r1:7b` | `nomic-embed-text` |

Resolved model names are written to `INSUREIQ_MODEL_CONFIG` (default `/tmp/model_config.env`) and read by every node. Override any of the three via env vars `OCR_MODEL`, `ANALYST_MODEL`, `EMBED_MODEL`.

Sequential loading:

```
[OCR Agent]      load llava ─ run ─ offload (keep_alive=0)
[RAG Indexer]    use nomic-embed-text (lightweight, stays resident)
[Web Research]   no model
[Analyst Agent]  load deepseek-r1 ─ run 14 sections ─ keep resident
[Company Profile] reuse same deepseek-r1 ─ synth ─ offload
[Validator]      pure Python, no model
[Compiler]       no model
```

Net VRAM footprint at any moment: one large model + the embedding model. Fits on a single 15 GB T4.

---

## 12. Configuration

| Env var | Purpose | Default |
|---|---|---|
| `TAVILY_API_KEY` | Enables Web Research + Company Profile | unset → both skip cleanly |
| `OCR_MODEL` | Override OCR model | auto from VRAM |
| `ANALYST_MODEL` | Override analyst model | auto from VRAM |
| `EMBED_MODEL` | Override embedding model | `nomic-embed-text` |
| `INSUREIQ_MODEL_CONFIG` | Path to resolved-model config file | `/tmp/model_config.env` |

`app.py` calls `python-dotenv`'s `load_dotenv()` at startup so a `.env` next to `app.py` works in any environment. On Colab the notebook reads `TAVILY_API_KEY` from Colab Secrets (`google.colab.userdata.get`) and writes it to `.env`.

---

## 13. Local Setup

Requires: Linux, Python 3.10+, an NVIDIA GPU with a recent driver, `curl`, `git`.

### One-shot

```bash
git clone https://github.com/aksri648/INSURE-IQ.git
cd INSURE-IQ
./setup.sh
echo "TAVILY_API_KEY=tvly-..." > .env    # optional
./run.sh
```

`setup.sh` performs:

1. Install **tectonic** (LaTeX engine).
2. Install **Ollama** and start it.
3. Detect VRAM, pull `nomic-embed-text`, `llava`, `deepseek-r1` (sizes auto-picked).
4. Write `/tmp/model_config.env`.
5. Create `.venv`, install `requirements.txt`.
6. Scaffold `.env` from `.env.example`.

`run.sh` activates the venv (if present), ensures Ollama is up, then runs `streamlit run app.py` headless on port 8501.

### Manual fallback

```bash
curl -fsSL https://drop-sh.fullyjustified.net | sh
sudo mv tectonic /usr/local/bin/

curl -fsSL https://ollama.com/install.sh | sh
nohup ollama serve >/tmp/ollama.log 2>&1 &
ollama pull nomic-embed-text
ollama pull llava:7b
ollama pull deepseek-r1:7b

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

echo "TAVILY_API_KEY=tvly-..." > .env
streamlit run app.py
```

---

## 14. Google Colab + Cloudflare Tunnel

The notebook `insureiq_colab.ipynb` runs the full project on a free Colab T4 and exposes the Streamlit app on a public `https://*.trycloudflare.com` URL — no account, no domain.

### Pre-flight: add the Tavily secret

1. Open the notebook in Colab.
2. Click the **🔑 key icon** in the left sidebar to open the *Secrets* panel.
3. **+ Add new secret** → Name: `TAVILY_API_KEY`, Value: your `tvly-…` key.
4. Toggle **Notebook access** for this notebook.

Cell 7 fetches the key via:

```python
from google.colab import userdata
TAVILY_API_KEY = userdata.get("TAVILY_API_KEY")
```

with explicit handling for `SecretNotFoundError` and `NotebookAccessError`. The key is written to `/content/INSURE-IQ/.env`. Only a masked preview (`tvly-x…ab`) is printed in the cell output.

### Cells

| Cell | Action |
|---|---|
| 1 | GPU + RAM check |
| 2 | `git clone --depth 1 https://github.com/aksri648/INSURE-IQ` |
| 3 | apt libs + `pip install -r requirements.txt` |
| 4 | Install **tectonic** |
| 5 | Install + start **Ollama** |
| 6 | Pull models sized to detected VRAM |
| 7 | **Load `TAVILY_API_KEY` from Colab Secrets** |
| 8 | Install **cloudflared** |
| 9 | Launch Streamlit + open the Cloudflare tunnel; print the public HTTPS URL |
| (opt) | JS keep-alive snippet for the free tier |

See `insureiq_colab_deployment.md` for the full guide with troubleshooting tables.

---

## 15. Privacy & Security

- **Policy PDFs never leave the host.** OCR happens locally via Ollama. Embeddings are local. The validator is local.
- **ChromaDB is in-memory.** Nothing is persisted between sessions.
- **Tavily only sees public information.** It receives the heuristically extracted insurer name + canned query templates. It never receives policy text.
- **API key handling.** On Colab the key is pulled from the user-scoped Secrets store, not pasted into the notebook source. Only a masked preview is printed.
- **No cloud LLM calls.** No OpenAI / Anthropic / Gemini SDKs are imported anywhere.
- **Tunnel URLs are ephemeral.** `trycloudflare.com` issues a fresh subdomain per `cloudflared` invocation.

If you want a stable subdomain across restarts you can switch to a named Cloudflare tunnel (instructions in `insureiq_colab_deployment.md`).

---

## 16. Testing & Smoke Checks

The repo doesn't ship a formal test suite (yet). The simplest sanity check is to run the validator against a synthetic chunk index:

```python
from agents.validator_agent import _validate_finding

idx = {"chunk_0001": {
    "text": "The pre-existing diseases are excluded for 48 months from policy inception.",
    "page": 12, "section": "Exclusions"
}}

trusted = {"label": "Important Exclusions",
           "claim": "PED waiting period is 48 months.",
           "chunk_id": "chunk_0001",
           "verbatim_quote": "pre-existing diseases are excluded for 48 months",
           "page": 12, "section": "Exclusions"}

fabricated = {**trusted, "verbatim_quote": "totally invented text"}

print(_validate_finding(trusted, idx)["trust_tag"])      # TRUSTED
print(_validate_finding(fabricated, idx))                # None  → dropped
```

End-to-end smoke check (no GPU required, no Ollama):

```python
from agents.compiler_agent import report_compiler_node

state = {
    "session_id": "test1234",
    "pdf_path": "/tmp/fake.pdf",
    "insurer_name": "Acme Insurance",
    "validated_sections": {
        "exclusions": {"findings": [{
            "label": "Important Exclusions",
            "claim": "PED waiting 48 months.",
            "chunk_id": "chunk_0011",
            "verbatim_quote": "pre-existing diseases excluded for 48 months",
            "page": 12, "section": "Exclusions",
            "trust_tag": "TRUSTED",
            "citation_id": "CIT-001",
        }], "score": 4}
    },
    "validation_report": {"counts": {"trusted": 1, "review": 0, "dropped": 0, "total": 1},
                         "trusted_ratio": 1.0},
    "company_profile": {"available": False, "reason": "no key set"},
    "citations": [],
}

out = report_compiler_node(state)
print(len(out["latex_source"]), "chars LaTeX")
print(len(out["pdf_bytes"]), "bytes PDF")
```

If neither `tectonic` nor `pdflatex` is available the compiler emits a reportlab notice PDF — the test still produces a non-empty bytes object.

---

## 17. Extending InsureIQ

### Add a new analyst section

1. Add an entry to `SECTION_SPECS` in `agents/analyst_agent.py` with:
   - `query`: retrieval prompt for ChromaDB.
   - `labels`: closed vocabulary of allowed `label` strings.
   - `score`: `True` if the section contributes to the overall score.
2. Add the same `labels` list under your new section key in `SECTION_LABELS` in `agents/compiler_agent.py`.
3. Add a `parts.append(_render_section("Your Section Title", "your_key", validated))` call inside `report_compiler_node`, in the order you want it to appear.

That's it — the validator and PDF builder pick up the rest.

### Add a new agent node

1. Create `agents/<your_agent>.py` with a `node(state: PolicyState) -> PolicyState` function. Set `active_node` and a unique `status` string in the returned dict.
2. Register the node in `graph.py`:
   ```python
   g.add_node("your_name", your_agent_node)
   g.add_edge("previous_node", "your_name")
   g.add_edge("your_name", "next_node")
   ```
3. Add it to `NODE_ORDER` + `NODE_LABELS` in `graph.py` so the UI flowchart picks it up.

### Swap models

Set the `OCR_MODEL` / `ANALYST_MODEL` / `EMBED_MODEL` env vars before launching. As long as the new model is pulled (`ollama pull …`) and supports the same API, it'll be used.

### Replace the validator

The validator interface is one function: `validator_agent_node(state) -> state`, expected to add `validated_sections`, `validation_report`, and `citations`. You can plug in a different strategy (e.g. embedding-cosine similarity) without touching anything else — just keep the trust tag set to `TRUSTED | NEEDS_HUMAN_REVIEW`.

---

## 18. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `nvidia-smi` not found | CPU runtime | Local: ensure NVIDIA driver. Colab: `Runtime → Change runtime type → T4 GPU`. |
| Cell 5 (Ollama) hangs in Colab | install script failed | `tail /tmp/ollama.log`; re-run the cell. |
| Cell 6 model pull stalls | network throttle | Re-run; pulls resume from the partial blob. |
| Streamlit 502 via tunnel | app still booting | Wait ~15 s; refresh. `tail /tmp/streamlit.log` to inspect. |
| Empty Company Profile section | Tavily key missing | Local: add to `.env`. Colab: 🔑 Secrets → `TAVILY_API_KEY` + notebook access → re-run Cell 7 + Cell 9. |
| OOM mid-analysis | analyst model too big | Set `ANALYST_MODEL=deepseek-r1:7b` before launching. |
| PDF says "LaTeX compiler unavailable" | tectonic + pdflatex missing | Install one; on Colab re-run Cell 4. |
| Lots of NEEDS HUMAN REVIEW tags | analyst paraphrased | Normal on noisy PDFs. Validator is conservative by design. |
| All findings dropped | analyst quotes don't match chunks (e.g. heavy OCR noise) | Check OCR output quality; you may need a sharper OCR model or a clean text-layer PDF. |

---

## 19. Glossary

- **Chunk** — a paragraph-sized slice of OCR'd policy text with a stable `chunk_id`, page number, and section title.
- **`chunk_index`** — `dict[chunk_id, {text, page, section}]` of the verbatim chunk text, used by the validator for substring matching. Distinct from the embedding store.
- **Finding** — an atomic, verifiable claim the analyst extracts. Always includes `label`, `claim`, `chunk_id`, `verbatim_quote`, `page`, `section`.
- **Label** — the report slot a finding fills (e.g. *"Important Exclusions"*). A closed vocabulary per section.
- **Trust tag** — `TRUSTED` or `NEEDS_HUMAN_REVIEW`, assigned by the validator.
- **Verbatim quote** — an exact substring of a chunk, copied by the analyst. The validator confirms this is real.
- **Trust score** — separate 0–100 score on the *insurer* (Company Profile agent), not on the policy.
- **Overall rating** — 0–100 score on the policy, derived from validated section scores minus a review-tag penalty.

---

## 20. Changelog

### v3.0 (current)
- ★ Added **Validator Agent** — deterministic substring verification.
- ★ Added **trust tags** TRUSTED / NEEDS_HUMAN_REVIEW on every finding.
- ★ Switched the report from Markdown to **LaTeX → PDF** (tectonic) as the sole deliverable.
- ★ Rebuilt the Streamlit UI with three columns: upload · live agent flowchart · LaTeX viewer + Download Report PDF.
- ★ Colab notebook now reads `TAVILY_API_KEY` from **Colab Secrets** (`google.colab.userdata`).
- Analyst rewritten to emit findings-only (no free-form prose).
- RAG indexer now builds a `chunk_index` for the validator.
- Compiler computes overall rating with a review-tag penalty; confidence derived from validator stats.
- `utils/pdf_builder.py` introduces the tectonic → pdflatex → reportlab fallback cascade.

### v2.0
- Restructured analyst into 18 sections matching a consumer-friendly markdown template.
- Added **Company Profile Agent** (Tavily ×7 facets + LLM synthesis + trust score).
- Compiler emitted full markdown report; UI showed Markdown / Company Profile / JSON tabs.

### v1.0
- Initial OCR + RAG + Analyst + Web Research + Compiler pipeline.
- JSON-only report with inline `[Source: ...]` citations.
- Streamlit UI with score / verdict / red flags / per-section tabs.

---

*InsureIQ · Project Documentation · v3.0*
*Repo: <https://github.com/aksri648/INSURE-IQ>*
