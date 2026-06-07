import os

from agents.state import PolicyState


def extract_insurer_name(ocr_results: list) -> str:
    """Heuristic extraction of insurer name from the first few pages."""
    for page in ocr_results[:3]:
        text = page.get("text", "")
        for line in text.split("\n"):
            lower = line.lower()
            if any(kw in lower for kw in
                   ["insurance", "insurer", "company", "ltd", "limited"]):
                return line.strip()[:80]
    return "Unknown Insurer"


def web_research_node(state: PolicyState) -> PolicyState:
    tavily_key = os.getenv("TAVILY_API_KEY", "").strip()
    insurer_name = extract_insurer_name(state["ocr_text"])

    if not tavily_key:
        print("[Web Research] No TAVILY_API_KEY set. Skipping external research.")
        return {**state,
                "external_research": {},
                "insurer_name": insurer_name,
                "status": "web_research_complete"}

    try:
        from tavily import TavilyClient
    except ImportError:
        print("[Web Research] tavily-python not installed. Skipping.")
        return {**state,
                "external_research": {},
                "insurer_name": insurer_name,
                "status": "web_research_complete"}

    client = TavilyClient(api_key=tavily_key)
    print(f"[Web Research] Researching: {insurer_name}")

    research: dict = {}

    try:
        result = client.search(
            f"{insurer_name} claim settlement ratio IRDAI annual report",
            max_results=3,
        )
        research["claim_settlement"] = {
            "data": result.get("results", []),
            "source": "EXTERNAL",
            "query": f"{insurer_name} claim settlement ratio",
        }
    except Exception as e:
        research["claim_settlement"] = {"error": str(e), "source": "EXTERNAL"}

    try:
        result = client.search(
            f"{insurer_name} IRDAI registration solvency ratio",
            max_results=3,
        )
        research["regulatory"] = {
            "data": result.get("results", []),
            "source": "EXTERNAL",
        }
    except Exception as e:
        research["regulatory"] = {"error": str(e), "source": "EXTERNAL"}

    print("[Web Research] Done.")
    return {**state,
            "external_research": research,
            "insurer_name": insurer_name,
            "status": "web_research_complete"}
