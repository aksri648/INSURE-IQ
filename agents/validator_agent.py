"""Validator agent — deterministic, hallucination-proof.

Verifies every analyst finding by checking that `verbatim_quote` is an actual
substring of the chunk it cites. No LLM involved — pure string matching with
whitespace normalisation, so the validator itself cannot hallucinate.

Each finding is tagged:

    TRUSTED            — verbatim_quote is found verbatim in the cited chunk
                         AND the chunk_id exists AND the page matches.
    NEEDS_HUMAN_REVIEW — anything weaker: fuzzy match, chunk found but quote
                         not exact, chunk_id missing, etc.

Only TRUSTED and NEEDS_HUMAN_REVIEW findings are passed to the compiler.
Findings that fail every check are dropped entirely (so unsupported LLM
output never reaches the report).
"""

import re

from agents.state import PolicyState


_WS = re.compile(r"\s+")


def _normalize(s: str) -> str:
    return _WS.sub(" ", (s or "").strip().lower())


def _contains_quote(chunk_text: str, quote: str) -> bool:
    return _normalize(quote) in _normalize(chunk_text)


def _word_overlap_ratio(quote: str, chunk_text: str) -> float:
    """Fallback fuzzy similarity: fraction of quote words present in chunk."""
    qw = [w for w in _normalize(quote).split() if len(w) > 2]
    if not qw:
        return 0.0
    cw = set(_normalize(chunk_text).split())
    hits = sum(1 for w in qw if w in cw)
    return hits / len(qw)


def _validate_finding(finding: dict, chunk_index: dict) -> dict:
    chunk_id = finding.get("chunk_id", "")
    quote = finding.get("verbatim_quote", "")
    chunk = chunk_index.get(chunk_id)

    notes = []
    if not chunk_id or not quote:
        return {**finding,
                "trust_tag": "NEEDS_HUMAN_REVIEW",
                "validation_notes": "Missing chunk_id or quote",
                "validation_match": "none"}

    if not chunk:
        return {**finding,
                "trust_tag": "NEEDS_HUMAN_REVIEW",
                "validation_notes": f"chunk_id '{chunk_id}' not found in policy index",
                "validation_match": "none"}

    page_match = (finding.get("page") in (None, "", chunk.get("page")))

    if _contains_quote(chunk["text"], quote):
        if not page_match:
            notes.append(f"Page mismatch (claimed {finding.get('page')}, actual {chunk['page']})")
            return {**finding,
                    "page": chunk["page"],
                    "section": chunk["section"] or finding.get("section", ""),
                    "trust_tag": "NEEDS_HUMAN_REVIEW",
                    "validation_notes": "; ".join(notes),
                    "validation_match": "exact"}
        return {**finding,
                "page": chunk["page"],
                "section": chunk["section"] or finding.get("section", ""),
                "trust_tag": "TRUSTED",
                "validation_notes": "Exact substring match",
                "validation_match": "exact"}

    # Fuzzy fallback — keep but flag
    overlap = _word_overlap_ratio(quote, chunk["text"])
    if overlap >= 0.7:
        return {**finding,
                "page": chunk["page"],
                "section": chunk["section"] or finding.get("section", ""),
                "trust_tag": "NEEDS_HUMAN_REVIEW",
                "validation_notes": f"Quote paraphrased ({int(overlap * 100)}% word overlap)",
                "validation_match": "fuzzy"}

    # Quote does not appear in cited chunk at all — drop entirely
    return None


def validator_agent_node(state: PolicyState) -> PolicyState:
    print("[Validator] Verifying findings against policy chunks...")
    chunk_index = state.get("chunk_index", {})
    analyses = state.get("section_analyses", {})

    validated_sections: dict = {}
    counts = {"trusted": 0, "review": 0, "dropped": 0, "total": 0}
    all_citations: list = []
    cit_counter = 1

    for section, body in analyses.items():
        findings = body.get("findings", []) if isinstance(body, dict) else []
        validated_findings = []
        for f in findings:
            counts["total"] += 1
            verdict = _validate_finding(f, chunk_index)
            if verdict is None:
                counts["dropped"] += 1
                continue
            # Stamp a citation id
            verdict["citation_id"] = f"CIT-{cit_counter:03d}"
            cit_counter += 1
            validated_findings.append(verdict)
            all_citations.append({
                "citation_id": verdict["citation_id"],
                "section": section,
                "label": verdict.get("label", ""),
                "chunk_id": verdict.get("chunk_id", ""),
                "page": verdict.get("page"),
                "section_title": verdict.get("section", ""),
                "trust_tag": verdict["trust_tag"],
            })
            if verdict["trust_tag"] == "TRUSTED":
                counts["trusted"] += 1
            else:
                counts["review"] += 1

        validated_sections[section] = {
            "findings": validated_findings,
            "score": body.get("score") if isinstance(body, dict) else None,
        }

    validation_report = {
        "counts": counts,
        "trusted_ratio": (counts["trusted"] / counts["total"]) if counts["total"] else 0.0,
    }
    print(f"[Validator] {counts['trusted']} trusted, {counts['review']} review, "
          f"{counts['dropped']} dropped (of {counts['total']} total).")

    return {**state,
            "validated_sections": validated_sections,
            "validation_report": validation_report,
            "citations": all_citations,
            "status": "validation_complete",
            "active_node": "validator"}
