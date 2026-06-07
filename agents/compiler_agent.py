from datetime import datetime

from agents.state import PolicyState


def score_policy(analyses: dict) -> int:
    """Compute overall analyst score 0-100."""
    scores = [v.get("score", 5) for v in analyses.values() if isinstance(v, dict)]
    if not scores:
        return 50
    avg = sum(scores) / len(scores)
    return round(avg * 10)


def get_verdict(score: int) -> str:
    if score >= 75:
        return "GOOD_VALUE"
    if score >= 55:
        return "BUY_WITH_CAUTION"
    if score >= 35:
        return "REVIEW_NEEDED"
    return "AVOID"


def report_compiler_node(state: PolicyState) -> PolicyState:
    print("[Compiler] Building final report...")
    analyses = state.get("section_analyses", {})
    score = score_policy(analyses)

    red_flags: list = []
    for _, data in analyses.items():
        if isinstance(data, dict):
            red_flags.extend(data.get("red_flags", []))

    report = {
        "report_metadata": {
            "session_id": state["session_id"],
            "generated_at": datetime.now().isoformat(),
            "policy_file": state["pdf_path"].split("/")[-1],
            "total_pages_analyzed": len(state.get("ocr_text", [])),
            "total_citations": len(state.get("citations", [])),
        },
        "executive_summary": {
            "insurer_name": state.get("insurer_name", "Unknown"),
            "analyst_score": score,
            "verdict": get_verdict(score),
            "key_red_flags": red_flags[:5],
            "summary": analyses.get("executive_summary", {}).get("summary", ""),
        },
        "sections": analyses,
        "insurer_profile": state.get("external_research", {}),
        "all_citations": state.get("citations", []),
    }

    print(f"[Compiler] Report complete. Score: {score}/100. Verdict: {get_verdict(score)}")
    return {**state, "final_report": report, "status": "complete"}
