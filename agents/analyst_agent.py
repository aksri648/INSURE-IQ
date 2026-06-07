"""Analyst agent — emits ONLY verifiable findings.

Every finding ships with:
    - label: which report slot it fills (e.g. "Total Coverage Amount")
    - claim: a consumer-friendly sentence
    - chunk_id: the source chunk
    - verbatim_quote: an EXACT substring from that chunk
    - page: page number
    - section: section title from OCR

The validator agent does a deterministic substring check on `verbatim_quote`
against the chunk store, so no LLM hallucination can slip into the final report.

Free-form summaries are intentionally NOT requested — they cannot be verified.
"""

import json

import ollama

from agents.rag_agent import retrieve_chunks
from agents.state import PolicyState
from utils.model_config import get as get_model


ANALYST_SYSTEM_PROMPT = """You are a senior insurance policy analyst.

ABSOLUTE RULES (violating any rule makes the finding worthless):

1. You may ONLY emit findings backed by the policy text chunks provided.
2. Every finding MUST include a `verbatim_quote` that is an EXACT substring
   (character-for-character) of ONE of the provided chunks. No paraphrasing.
   No summarising. No combining sentences. Quote a real run of characters.
3. If a label cannot be backed by an exact substring from a chunk, DO NOT
   emit a finding for that label. Omitting is correct. Inventing is wrong.
4. Every finding MUST cite the exact chunk_id you copied the quote from.
5. Use plain English in the `claim` field — the consumer reads this.
6. Output ONLY valid JSON matching the requested schema. No prose. No
   markdown fences. No <think> blocks in the output.
7. Do not output an `overall_assessment` or `summary` or any free-form
   narrative field — only the structured findings + the numeric score.
"""


# Per-section: what to retrieve, what labels the report needs, optional score.
SECTION_SPECS: dict = {
    "policy_basics": {
        "query": "policy name policy number insurer start date end date renewal term contact phone email",
        "labels": [
            "Policy Name", "Policy Number", "Insurer Name", "Policy Type",
            "Start Date", "End Date", "Renewal Date", "Policy Term",
            "Customer Support", "Claims Support", "Emergency Assistance",
        ],
        "score": False,
    },
    "coverage": {
        "query": "covered benefits hospitalization treatment day care in-patient out-patient add-on rider geographic scope",
        "labels": [
            "Main Benefits", "Additional Benefits", "Optional Add-Ons",
            "Where and When You Are Covered",
        ],
        "score": True,
    },
    "coverage_limits": {
        "query": "sum insured coverage amount sub limits annual limit lifetime limit cap maximum payable room rent",
        "labels": [
            "Total Coverage Amount", "Treatment/Event Limits",
            "Annual Limits", "Lifetime Limits", "Coverage Caps",
        ],
        "score": True,
    },
    "exclusions": {
        "query": "exclusions not covered excluded conditions limitations rejection denial scenarios",
        "labels": [
            "Situations Where Claims Will Be Rejected",
            "Conditions and Treatments Not Covered",
            "Temporary Restrictions",
            "Important Exclusions",
            "Vague or Ambiguous Clauses",
        ],
        "score": True,
    },
    "waiting_periods": {
        "query": "waiting period initial cooling pre-existing disease PED maternity specific illness days months years",
        "labels": [
            "Initial Waiting Period",
            "Condition-Specific Waiting Periods",
            "Pre-Existing Condition Waiting Period",
            "Maternity Waiting Period",
        ],
        "score": True,
    },
    "out_of_pocket": {
        "query": "deductible co-payment copay cost sharing out of pocket excess voluntary mandatory",
        "labels": [
            "Deductibles", "Co-Payments", "Cost Sharing Requirements",
            "Expected Out-of-Pocket Expenses",
        ],
        "score": True,
    },
    "premium_analysis": {
        "query": "premium amount payment frequency loading discount renewal increase fees charges taxes GST",
        "labels": [
            "Current Premium", "Future Premium Increase Risk",
            "Extra Charges and Fees", "Overall Cost Assessment",
        ],
        "score": True,
    },
    "claims_process": {
        "query": "claim filing process documents required intimation cashless reimbursement timeline TAT settlement",
        "labels": [
            "Steps to Make a Claim", "Documents You Will Need",
            "Claim Submission Deadlines", "Approval Timeline",
            "Common Rejection Reasons",
        ],
        "score": True,
    },
    "renewal_cancellation": {
        "query": "renewal grace period lapse cancellation termination free look refund migration portability",
        "labels": [
            "Renewal Rules", "Grace Period",
            "When the Company Can Cancel", "When You Can Cancel",
            "Lapse Situations",
        ],
        "score": True,
    },
    "legal_terms": {
        "query": "rights duties obligations dispute resolution arbitration ombudsman fraud misrepresentation beneficiary nominee assignment",
        "labels": [
            "Your Rights", "Insurance Company Rights",
            "Dispute Resolution Process",
            "Fraud and Misrepresentation Rules",
            "Beneficiary and Nominee Rules",
        ],
        "score": True,
    },
    "definitions": {
        "query": "definitions terms meaning interpretation hospital injury illness AYUSH day care",
        "labels": [
            "Key Insurance Terms Explained",
            "Unusual Definitions",
            "Definitions That Could Impact Claim Approval",
        ],
        "score": False,
    },
    "risks_concerns": {
        "query": "risk void cancellation lapse forfeiture penalty restriction limitation onus burden",
        "labels": [
            "High-Risk Clauses", "Customer-Unfriendly Terms",
            "Hidden Restrictions", "Areas Needing Extra Attention",
        ],
        "score": True,
    },
    "claim_rejection_risk": {
        "query": "claim denial rejection reasons disclosure waiting period excluded pre-existing fraud non-disclosure",
        "labels": [
            "Top Reasons a Claim Could Be Denied",
            "Real-Life Claim Rejection Scenarios",
        ],
        "score": True,
    },
    "hidden_surprises": {
        "query": "fine print sub-limit cap room rent ICU domiciliary AYUSH OPD modern treatment proportionate deduction",
        "labels": [
            "Unexpected Restrictions", "Hidden Costs",
            "Clauses Most People Miss",
            "Things to Double-Check Before Buying",
        ],
        "score": True,
    },
}


# Sections that are pure aggregations or qualitative — derived deterministically
# by the compiler from validated findings, not asked of the LLM.
# (real_life_examples, comparison, plain_english_summary, final_verdict)

ALLOWED_FINDING_KEYS = {"label", "claim", "chunk_id", "verbatim_quote", "page", "section"}


# ── HELPERS ───────────────────────────────────────────────────────────

def _format_chunks(chunks: list) -> str:
    parts = []
    for doc, meta in chunks:
        cid = meta.get("chunk_id") or f"chunk_{meta.get('chunk_index', '?')}"
        page = meta.get("page", "?")
        section = meta.get("section", "")
        parts.append(f"[{cid} | Page {page} | {section}]\n{doc}")
    return "\n\n".join(parts) if parts else "(no chunks retrieved)"


def _build_section_prompt(section: str, spec: dict, chunks: list) -> str:
    labels_block = "\n".join(f"- {lbl}" for lbl in spec["labels"])
    score_block = (
        '  "score": <integer 1-10 reflecting consumer-friendliness>,\n'
        if spec.get("score") else ""
    )
    return f"""SECTION: {section.upper().replace('_', ' ')}

POLICY TEXT CHUNKS (the only source of truth — quote from these verbatim):
{_format_chunks(chunks)}

REPORT SLOTS YOU MAY FILL (each finding's "label" must be one of these strings):
{labels_block}

Emit one finding per fact you can support with an EXACT substring from a chunk.
A single label may have multiple findings if multiple facts apply (e.g. several
"Important Exclusions"). Skip any label you cannot back with a verbatim quote.

Return ONLY this JSON object — no markdown, no commentary:

{{
  "findings": [
    {{
      "label": "<one of the slot names above>",
      "claim": "<plain-English sentence stating the fact>",
      "chunk_id": "<chunk id you copied from, e.g. chunk_0042>",
      "verbatim_quote": "<EXACT substring of the chunk's text — no edits>",
      "page": <integer page number>,
      "section": "<section title from the chunk header>"
    }}
  ],
{score_block}}}
"""


def _strip_think(raw: str) -> str:
    if "</think>" in raw:
        raw = raw[raw.rfind("</think>") + len("</think>"):].strip()
    return raw


def _safe_parse(raw: str) -> dict:
    raw = raw.replace("```json", "").replace("```", "").strip()
    raw = _strip_think(raw)
    if not raw.startswith("{"):
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            raw = raw[start:end + 1]
    return json.loads(raw)


def _coerce_findings(parsed: dict, allowed_labels: list) -> list:
    raw_findings = parsed.get("findings", []) if isinstance(parsed, dict) else []
    out = []
    label_set = set(allowed_labels)
    for f in raw_findings:
        if not isinstance(f, dict):
            continue
        label = str(f.get("label", "")).strip()
        if label not in label_set:
            # Try fuzzy: case-insensitive
            match = next((lbl for lbl in allowed_labels if lbl.lower() == label.lower()), None)
            if not match:
                continue
            label = match
        cleaned = {
            "label": label,
            "claim": str(f.get("claim", "")).strip(),
            "chunk_id": str(f.get("chunk_id", "")).strip(),
            "verbatim_quote": str(f.get("verbatim_quote", "")).strip(),
            "page": f.get("page"),
            "section": str(f.get("section", "")).strip(),
        }
        if not cleaned["claim"] or not cleaned["chunk_id"] or not cleaned["verbatim_quote"]:
            continue
        out.append(cleaned)
    return out


# ── NODE ──────────────────────────────────────────────────────────────

def analyst_agent_node(state: PolicyState) -> PolicyState:
    analyst_model = get_model("ANALYST_MODEL", "deepseek-r1:7b")
    print(f"[Analyst] Loading {analyst_model}...")

    analyses: dict = {}

    for section, spec in SECTION_SPECS.items():
        print(f"[Analyst] Analyzing: {section}")
        try:
            chunks = retrieve_chunks(state["session_id"], spec["query"])
        except Exception as e:
            analyses[section] = {"error": f"retrieval failed: {e}",
                                 "findings": [], "score": None}
            continue

        prompt = _build_section_prompt(section, spec, chunks)
        try:
            response = ollama.chat(
                model=analyst_model,
                messages=[
                    {"role": "system", "content": ANALYST_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": 0.1},
            )
            raw = response["message"]["content"]
            parsed = _safe_parse(raw)
            findings = _coerce_findings(parsed, spec["labels"])
            score = parsed.get("score") if spec.get("score") else None
            if isinstance(score, (int, float)):
                score = max(0, min(10, int(score)))
            else:
                score = None
            analyses[section] = {"findings": findings, "score": score}
        except Exception as e:
            print(f"[Analyst]  ! {section} failed: {e}")
            analyses[section] = {"error": str(e), "findings": [], "score": None}

    print(f"[Analyst] Offloading {analyst_model} from VRAM...")
    try:
        ollama.generate(model=analyst_model, prompt="", keep_alive=0)
    except Exception:
        pass

    total = sum(len(a.get("findings", [])) for a in analyses.values())
    print(f"[Analyst] Done. {len(analyses)} sections, {total} findings.")
    return {**state,
            "section_analyses": analyses,
            "status": "analysis_complete",
            "active_node": "analyst"}
