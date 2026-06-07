"""Tavily-powered company profile agent.

Researches the insurer's reputation, claim settlement ratio, recent disputes,
customer reviews, ratings, market share, and overall credibility. Output is a
structured dict consumed by the report compiler.
"""

import json
import os

from agents.state import PolicyState


# Each query targets a distinct facet of the company's public reputation.
PROFILE_QUERIES = {
    "company_overview": "{insurer} insurance company overview history headquarters founded parent group",
    "claim_settlement_ratio": "{insurer} claim settlement ratio latest IRDAI annual report percent",
    "recent_disputes": "{insurer} insurance complaints lawsuits regulatory action penalty disputes recent",
    "customer_reviews": "{insurer} customer reviews complaints feedback service experience",
    "ratings": "{insurer} insurance rating CRISIL ICRA AM Best Moody's financial strength rating",
    "market_share": "{insurer} market share India insurance sector premium income ranking",
    "credibility": "{insurer} solvency ratio IRDAI registration license status credibility trustworthiness",
}


def _short_results(results: list, limit: int = 4) -> list:
    """Trim Tavily results to a compact citation-friendly form."""
    out = []
    for r in results[:limit]:
        out.append({
            "title": r.get("title", "")[:160],
            "url": r.get("url", ""),
            "snippet": (r.get("content") or "")[:400],
            "source": "EXTERNAL",
        })
    return out


def _summarize_with_llm(insurer: str, facets: dict) -> dict:
    """Optional LLM pass to consolidate raw Tavily snippets into clean fields.

    Falls back to a plain concatenation if Ollama / the analyst model is not
    available, so the agent never blocks the pipeline.
    """
    try:
        import ollama

        from utils.model_config import get as get_model
    except Exception:
        ollama = None

    flat = {k: [r["snippet"] for r in v.get("sources", [])] for k, v in facets.items()}

    fallback = {
        "company_overview": " ".join(flat.get("company_overview", []))[:600] or "Not found",
        "claim_settlement_ratio": " ".join(flat.get("claim_settlement_ratio", []))[:400] or "Not found",
        "recent_disputes": flat.get("recent_disputes", []) or ["None surfaced in public sources"],
        "customer_reviews_summary": " ".join(flat.get("customer_reviews", []))[:500] or "Not found",
        "ratings": " ".join(flat.get("ratings", []))[:400] or "Not found",
        "market_share": " ".join(flat.get("market_share", []))[:400] or "Not found",
        "credibility": " ".join(flat.get("credibility", []))[:500] or "Not found",
        "overall_assessment": "Synthesis unavailable (LLM not invoked).",
        "trust_score": 0,
    }

    if not ollama:
        return fallback

    model = get_model("ANALYST_MODEL", "deepseek-r1:7b")
    prompt = f"""You are an insurance industry analyst. Summarize public information about the insurer below into a JSON object. Be factual, cite none of these claims back to the policy document — they are all EXTERNAL.

Insurer: {insurer}

RAW SOURCES PER FACET (each list is search-result snippets):
{json.dumps(flat, indent=2)[:6000]}

Return ONLY this JSON shape (no markdown, no prose):
{{
  "company_overview": "<2-3 sentences>",
  "claim_settlement_ratio": "<latest known % with year if available, else 'Not disclosed in public sources'>",
  "recent_disputes": ["<short bullet>", "..."],
  "customer_reviews_summary": "<2-3 sentences>",
  "ratings": "<known credit / financial-strength ratings>",
  "market_share": "<rank or % share if available>",
  "credibility": "<2-3 sentence trust assessment>",
  "overall_assessment": "<one paragraph plain-English verdict on the insurer>",
  "trust_score": <integer 0-100>
}}"""

    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2},
        )
        raw = response["message"]["content"]
        raw = raw.replace("```json", "").replace("```", "").strip()
        if "</think>" in raw:
            raw = raw[raw.rfind("</think>") + len("</think>"):].strip()
        if not raw.startswith("{"):
            s = raw.find("{")
            e = raw.rfind("}")
            if s != -1 and e != -1:
                raw = raw[s:e + 1]
        parsed = json.loads(raw)
        # Try to free VRAM after summarisation
        try:
            ollama.generate(model=model, prompt="", keep_alive=0)
        except Exception:
            pass
        return parsed
    except Exception as e:
        fallback["overall_assessment"] = f"LLM synthesis failed: {e}. Raw snippets retained."
        return fallback


def company_profile_node(state: PolicyState) -> PolicyState:
    insurer = (state.get("insurer_name") or "Unknown Insurer").strip()
    tavily_key = os.getenv("TAVILY_API_KEY", "").strip()

    if not tavily_key:
        print("[Company Profile] No TAVILY_API_KEY set. Skipping.")
        return {**state,
                "company_profile": {
                    "insurer": insurer,
                    "available": False,
                    "reason": "TAVILY_API_KEY not configured",
                },
                "status": "company_profile_complete",
                "active_node": "company_profile"}

    try:
        from tavily import TavilyClient
    except ImportError:
        print("[Company Profile] tavily-python not installed. Skipping.")
        return {**state,
                "company_profile": {
                    "insurer": insurer,
                    "available": False,
                    "reason": "tavily-python not installed",
                },
                "status": "company_profile_complete",
                "active_node": "company_profile"}

    client = TavilyClient(api_key=tavily_key)
    print(f"[Company Profile] Researching insurer: {insurer}")

    facets: dict = {}
    for facet, template in PROFILE_QUERIES.items():
        query = template.format(insurer=insurer)
        try:
            result = client.search(query, max_results=4, search_depth="advanced")
            facets[facet] = {
                "query": query,
                "sources": _short_results(result.get("results", [])),
            }
        except Exception as e:
            facets[facet] = {"query": query, "sources": [], "error": str(e)}

    print("[Company Profile] Synthesizing summary...")
    summary = _summarize_with_llm(insurer, facets)

    profile = {
        "insurer": insurer,
        "available": True,
        "summary": summary,
        "facets": facets,
        "source": "EXTERNAL",
    }

    print("[Company Profile] Done.")
    return {**state, "company_profile": profile, "status": "company_profile_complete"}
