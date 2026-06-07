import json

import ollama

from agents.rag_agent import retrieve_chunks
from agents.state import PolicyState
from utils.model_config import get as get_model


ANALYST_SYSTEM_PROMPT = """You are a senior insurance policy analyst with 20 years experience.
You protect consumers by finding hidden exclusions, risks, and unfair terms.
You explain things in plain English a non-expert can understand.

STRICT RULES:
1. ONLY make claims supported by the provided policy text chunks.
2. Every factual claim MUST cite: [Source: chunk_id, Page X, Section Y]
3. External research claims must be tagged [EXTERNAL: description]
4. If a piece of information is not in the policy text, write exactly: "Not specified in the policy"
5. Think step by step before concluding.
6. Output ONLY valid JSON for the requested schema. No preamble. No markdown fences. No prose outside JSON.
7. Keep language consumer-friendly: short sentences, no jargon unless you define it inline."""


# ── SECTION DEFINITIONS ────────────────────────────────────────────────
# Each entry: retrieval query + required JSON schema fragment + guidance.
# Subkeys map 1-to-1 to the markdown template the compiler renders.

SECTION_SPECS: dict = {
    "policy_basics": {
        "query": "policy name policy number insurer start date end date renewal term contact phone email",
        "schema": {
            "policy_name": "",
            "policy_number": "",
            "insurer_name": "",
            "policy_type": "",
            "start_date": "",
            "end_date": "",
            "renewal_date": "",
            "policy_term": "",
            "customer_support": "",
            "claims_support": "",
            "emergency_assistance": "",
            "citations": [],
        },
        "instructions": "Extract administrative metadata. Use 'Not specified in the policy' for missing values.",
    },
    "coverage": {
        "query": "covered benefits hospitalization treatment day care daycare in-patient out-patient add-on rider geographic scope",
        "schema": {
            "main_benefits": [],          # list of strings
            "additional_benefits": [],
            "optional_addons": [],
            "geographic_temporal_coverage": "",
            "summary": "",
            "score": 0,                    # 1-10 (10 = excellent)
            "citations": [],
        },
        "instructions": "List concrete benefits the policy actually pays for. Distinguish core vs optional riders.",
    },
    "coverage_limits": {
        "query": "sum insured coverage amount sub limits annual limit lifetime limit cap maximum payable room rent",
        "schema": {
            "total_coverage_amount": "",
            "treatment_event_limits": [],
            "annual_limits": [],
            "lifetime_limits": [],
            "coverage_caps_to_know": [],
            "summary": "",
            "score": 0,
            "citations": [],
        },
        "instructions": "Quote exact amounts and the conditions they apply to. Flag caps that meaningfully reduce real-world payout.",
    },
    "exclusions": {
        "query": "exclusions not covered excluded conditions limitations rejection denial scenarios",
        "schema": {
            "rejection_situations": [],
            "conditions_not_covered": [],
            "temporary_restrictions": [],
            "important_exclusions": [],
            "vague_or_ambiguous_clauses": [],
            "summary": "",
            "score": 0,
            "red_flags": [],
            "citations": [],
        },
        "instructions": "Highlight exclusions a non-expert would miss. Quote vague language verbatim in vague_or_ambiguous_clauses.",
    },
    "waiting_periods": {
        "query": "waiting period initial cooling pre-existing disease PED maternity specific illness days months years",
        "schema": {
            "initial_waiting_period": "",
            "condition_specific_waiting_periods": [],
            "pre_existing_condition_waiting_period": "",
            "maternity_waiting_period": "",
            "summary": "",
            "score": 0,
            "citations": [],
        },
        "instructions": "Be precise about durations. Mention whether the clock resets on renewal if mentioned.",
    },
    "out_of_pocket": {
        "query": "deductible co-payment copay cost sharing out of pocket excess voluntary mandatory",
        "schema": {
            "deductibles": [],
            "co_payments": [],
            "cost_sharing_requirements": [],
            "expected_out_of_pocket_expenses": "",
            "summary": "",
            "score": 0,
            "citations": [],
        },
        "instructions": "Translate percentages and amounts into the rupee/dollar amount a policyholder might actually pay.",
    },
    "premium_analysis": {
        "query": "premium amount payment frequency loading discount renewal increase fees charges taxes GST",
        "schema": {
            "current_premium": "",
            "future_premium_increase_risk": "",
            "extra_charges_and_fees": [],
            "overall_cost_assessment": "",
            "summary": "",
            "score": 0,
            "citations": [],
        },
        "instructions": "Flag age-based loading, claim-based loading, and any non-obvious fees.",
    },
    "claims_process": {
        "query": "claim filing process documents required intimation cashless reimbursement timeline TAT settlement",
        "schema": {
            "steps_to_make_a_claim": [],
            "documents_required": [],
            "claim_submission_deadlines": "",
            "approval_timeline": "",
            "common_rejection_reasons": [],
            "summary": "",
            "score": 0,
            "citations": [],
        },
        "instructions": "Make steps actionable. Highlight strict deadlines that could void a claim.",
    },
    "renewal_cancellation": {
        "query": "renewal grace period lapse cancellation termination free look refund migration portability",
        "schema": {
            "renewal_rules": "",
            "grace_period": "",
            "when_company_can_cancel": [],
            "when_you_can_cancel": [],
            "lapse_situations": [],
            "summary": "",
            "score": 0,
            "citations": [],
        },
        "instructions": "Note lifelong vs guaranteed renewal differences if stated.",
    },
    "legal_terms": {
        "query": "rights duties obligations dispute resolution arbitration ombudsman fraud misrepresentation beneficiary nominee assignment",
        "schema": {
            "your_rights": [],
            "insurance_company_rights": [],
            "dispute_resolution_process": "",
            "fraud_and_misrepresentation_rules": "",
            "beneficiary_and_nominee_rules": "",
            "summary": "",
            "score": 0,
            "citations": [],
        },
        "instructions": "Translate legal phrasing into plain English while preserving the obligation.",
    },
    "definitions": {
        "query": "definitions terms meaning interpretation hospital injury illness AYUSH day care",
        "schema": {
            "key_terms_explained": [],            # list of {term, plain_meaning}
            "unusual_definitions": [],            # list of {term, definition, why_unusual}
            "claim_impacting_definitions": [],    # list of {term, impact}
            "summary": "",
            "citations": [],
        },
        "instructions": "Quote the exact definition for unusual or claim-impacting terms before commenting.",
    },
    "risks_concerns": {
        "query": "risk void cancellation lapse forfeiture penalty restriction limitation onus burden",
        "schema": {
            "high_risk_clauses": [],
            "customer_unfriendly_terms": [],
            "hidden_restrictions": [],
            "areas_needing_extra_attention": [],
            "summary": "",
            "score": 0,
            "citations": [],
        },
        "instructions": "Rank by potential consumer harm.",
    },
    "claim_rejection_risk": {
        "query": "claim denial rejection reasons disclosure waiting period excluded pre-existing fraud non-disclosure",
        "schema": {
            "overall_claim_risk_rating": "",    # LOW | MEDIUM | HIGH
            "top_reasons_a_claim_could_be_denied": [],
            "real_life_claim_rejection_scenarios": [],
            "summary": "",
            "citations": [],
        },
        "instructions": "Build scenarios from clauses you actually found, not generic ones.",
    },
    "real_life_examples": {
        "query": "hospitalization accident critical illness surgery treatment claim payout coverage scenario",
        "schema": {
            "if_hospitalized": "",
            "if_accident": "",
            "if_critical_illness": "",
            "if_major_surgery": "",
            "worst_case_financial_scenario": "",
            "citations": [],
        },
        "instructions": "Walk through what the policy would pay and what the policyholder would pay, with rough numbers.",
    },
    "hidden_surprises": {
        "query": "fine print sub-limit cap room rent ICU domiciliary AYUSH OPD modern treatment proportionate deduction",
        "schema": {
            "unexpected_restrictions": [],
            "hidden_costs": [],
            "clauses_most_people_miss": [],
            "things_to_double_check_before_buying": [],
            "summary": "",
            "citations": [],
        },
        "instructions": "Focus on traps. Each item should reference a real clause from the chunks.",
    },
    "comparison": {
        "query": "industry standard market norm typical policy benchmark IRDAI guideline",
        "schema": {
            "coverage_quality": "",
            "claim_friendliness": "",
            "value_for_money": "",
            "transparency": "",
            "overall_competitiveness": "",
            "summary": "",
            "citations": [],
        },
        "instructions": "Compare against industry norms only when you have evidence from external research or the policy itself. Avoid speculation.",
    },
    "plain_english_summary": {
        "query": "policy overview suitability buyer profile",
        "schema": {
            "what_we_like": [],
            "what_we_dont_like": [],
            "best_for": [],
            "should_consider_other_options": [],
            "summary": "",
            "citations": [],
        },
        "instructions": "Write as if explaining to a friend with no insurance background.",
    },
    "final_verdict": {
        "query": "policy summary overall verdict recommendation",
        "schema": {
            "detailed_scores": {
                "coverage": 0,
                "coverage_limits": 0,
                "exclusions": 0,
                "waiting_periods": 0,
                "costs": 0,
                "claims_process": 0,
                "renewal_protection": 0,
                "transparency": 0,
            },
            "overall_rating": 0,           # 0-100
            "risk_level": "",              # LOW | MEDIUM | HIGH
            "confidence_level": "",        # LOW | MEDIUM | HIGH
            "final_recommendation": "",    # GOOD_VALUE | BUY_WITH_CAUTION | REVIEW_NEEDED | AVOID
            "one_line_verdict": "",
            "summary": "",
            "citations": [],
        },
        "instructions": "Each score is 0-10. Overall_rating is 0-100. Be honest — penalize hidden exclusions and weak claim processes.",
    },
}


# ── HELPERS ────────────────────────────────────────────────────────────

def _format_chunks(chunks: list) -> str:
    parts = []
    for doc, meta in chunks:
        cid = meta.get("chunk_id") or f"chunk_{meta.get('chunk_index', '?')}"
        page = meta.get("page", "?")
        section = meta.get("section", "?")
        parts.append(f"[{cid} | Page {page} | {section}]\n{doc}")
    return "\n\n".join(parts) if parts else "(no chunks retrieved)"


def _external_summary(external: dict) -> str:
    if not external:
        return ""
    out = ["", "EXTERNAL RESEARCH (mark any use as [EXTERNAL: ...]):"]
    for _, v in external.items():
        if isinstance(v, dict) and v.get("data"):
            for item in v["data"][:2]:
                title = item.get("title", "")
                content = (item.get("content") or "")[:220]
                out.append(f"- {title}: {content}")
    return "\n".join(out)


def _build_section_prompt(section: str, spec: dict, chunks: list, external: dict) -> str:
    return f"""Analyze the {section.upper().replace('_', ' ')} of this insurance policy.

POLICY TEXT CHUNKS:
{_format_chunks(chunks)}
{_external_summary(external)}

GUIDANCE: {spec['instructions']}

Return a single JSON object matching EXACTLY this schema (preserve all keys; fill arrays with concrete items; use "Not specified in the policy" for missing strings):

{json.dumps(spec['schema'], indent=2)}

Cite every factual claim inline within the relevant string or list item using: [Source: chunk_XXXX, Page N, Section Title].
The top-level "citations" array should list the chunk_ids you relied on most heavily."""


def _strip_think(raw: str) -> str:
    if "</think>" in raw:
        raw = raw[raw.rfind("</think>") + len("</think>"):].strip()
    return raw


def _safe_parse(raw: str) -> dict:
    raw = raw.replace("```json", "").replace("```", "").strip()
    raw = _strip_think(raw)
    # First {...} block fallback
    if not raw.startswith("{"):
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            raw = raw[start:end + 1]
    return json.loads(raw)


def _collect_citations(section: str, parsed: dict, start_counter: int) -> tuple[list, int]:
    citations = []
    counter = start_counter
    cited = parsed.get("citations", []) if isinstance(parsed, dict) else []
    for c in cited:
        citations.append({
            "citation_id": f"CIT-{counter:03d}",
            "section": section,
            "chunk_ref": c,
            "source_type": "POLICY_DOCUMENT",
        })
        counter += 1
    return citations, counter


# ── NODE ───────────────────────────────────────────────────────────────

def analyst_agent_node(state: PolicyState) -> PolicyState:
    analyst_model = get_model("ANALYST_MODEL", "deepseek-r1:7b")
    print(f"[Analyst] Loading {analyst_model}...")

    analyses: dict = {}
    citations: list = []
    cit_counter = 1

    for section, spec in SECTION_SPECS.items():
        print(f"[Analyst] Analyzing: {section}")

        try:
            chunks = retrieve_chunks(state["session_id"], spec["query"])
        except Exception as e:
            analyses[section] = {"error": f"retrieval failed: {e}", **spec["schema"]}
            continue

        prompt = _build_section_prompt(
            section, spec, chunks, state.get("external_research", {})
        )

        try:
            response = ollama.chat(
                model=analyst_model,
                messages=[
                    {"role": "system", "content": ANALYST_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": 0.2},
            )
            raw = response["message"]["content"]
            parsed = _safe_parse(raw)
            analyses[section] = parsed

            new_cites, cit_counter = _collect_citations(section, parsed, cit_counter)
            citations.extend(new_cites)

        except Exception as e:
            print(f"[Analyst]  ! {section} failed: {e}")
            analyses[section] = {"error": str(e), **spec["schema"]}

    # Offload analyst model
    print(f"[Analyst] Offloading {analyst_model} from VRAM...")
    try:
        ollama.generate(model=analyst_model, prompt="", keep_alive=0)
    except Exception:
        pass

    print(f"[Analyst] Done. {len(analyses)} sections, {len(citations)} citations.")
    return {**state,
            "section_analyses": analyses,
            "citations": citations,
            "status": "analysis_complete"}
