import json

import ollama

from agents.rag_agent import retrieve_chunks
from agents.state import PolicyState
from utils.model_config import get as get_model


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


def _format_chunks(chunks: list) -> str:
    parts = []
    for doc, meta in chunks:
        cid = meta.get("chunk_id") or f"chunk_{meta.get('chunk_index','?')}"
        page = meta.get("page", "?")
        section = meta.get("section", "?")
        parts.append(f"[{cid} | Page {page} | {section}]\n{doc}")
    return "\n\n".join(parts)


def build_section_prompt(section: str, chunks: list, external: dict) -> str:
    chunk_text = _format_chunks(chunks)

    ext_summary = ""
    if external:
        ext_summary = "\n\nEXTERNAL RESEARCH (mark all as [EXTERNAL]):\n"
        for _, v in external.items():
            if isinstance(v, dict) and "data" in v:
                for item in v["data"][:2]:
                    title = item.get("title", "")
                    content = item.get("content", "")[:200]
                    ext_summary += f"- {title}: {content}\n"

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


def _strip_think(raw: str) -> str:
    # DeepSeek R1 wraps reasoning in <think>...</think>
    if "</think>" in raw:
        raw = raw[raw.rfind("</think>") + len("</think>"):].strip()
    return raw


def analyst_agent_node(state: PolicyState) -> PolicyState:
    analyst_model = get_model("ANALYST_MODEL", "deepseek-r1:7b")
    print(f"[Analyst] Loading {analyst_model}...")

    analyses: dict = {}
    citations: list = []
    cit_counter = 1

    for section, query in SECTION_QUERIES.items():
        print(f"[Analyst] Analyzing: {section}")
        try:
            chunks = retrieve_chunks(state["session_id"], query)
        except Exception as e:
            analyses[section] = {"error": f"retrieval failed: {e}",
                                 "findings": [],
                                 "summary": "Analysis failed",
                                 "score": 0,
                                 "red_flags": []}
            continue

        prompt = build_section_prompt(
            section, chunks, state.get("external_research", {})
        )

        try:
            response = ollama.chat(
                model=analyst_model,
                messages=[
                    {"role": "system", "content": ANALYST_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            raw = response["message"]["content"]
            raw = raw.replace("```json", "").replace("```", "").strip()
            raw = _strip_think(raw)
            parsed = json.loads(raw)
            analyses[section] = parsed

            for finding in parsed.get("findings", []):
                citation_text = finding.get("citation", "")
                citations.append({
                    "citation_id": f"CIT-{cit_counter:03d}",
                    "section": section,
                    "claim": finding.get("claim", ""),
                    "citation_text": citation_text,
                    "source_type": "EXTERNAL" if "[EXTERNAL]" in citation_text
                                   else "POLICY_DOCUMENT",
                })
                cit_counter += 1

        except Exception as e:
            analyses[section] = {"error": str(e),
                                 "findings": [],
                                 "summary": "Analysis failed",
                                 "score": 0,
                                 "red_flags": []}

    # Offload analyst model
    print(f"[Analyst] Offloading {analyst_model} from VRAM...")
    try:
        ollama.generate(model=analyst_model, prompt="", keep_alive=0)
    except Exception:
        pass

    print(f"[Analyst] Done. {len(citations)} citations extracted.")
    return {**state,
            "section_analyses": analyses,
            "citations": citations,
            "status": "analysis_complete"}
