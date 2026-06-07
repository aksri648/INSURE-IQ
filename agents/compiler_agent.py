"""Report compiler.

Turns the analyst's per-section JSON + the company profile into a
consumer-friendly markdown report that exactly matches the requested template.
"""

from datetime import datetime

from agents.state import PolicyState


# ── FORMATTING HELPERS ────────────────────────────────────────────────

def _g(d: dict, key: str, default: str = "Not specified in the policy") -> str:
    if not isinstance(d, dict):
        return default
    val = d.get(key)
    if val is None or (isinstance(val, str) and not val.strip()):
        return default
    return str(val)


def _bullets(items, empty: str = "Not specified in the policy") -> str:
    if not items:
        return f"* {empty}"
    if isinstance(items, str):
        items = [items]
    lines = []
    for item in items:
        if isinstance(item, dict):
            # Render dict bullets like {term, plain_meaning}
            label = item.get("term") or item.get("name") or item.get("title") or ""
            rest = {k: v for k, v in item.items() if k not in ("term", "name", "title")}
            inner = " — ".join(f"{k}: {v}" for k, v in rest.items() if v)
            line = f"**{label}** — {inner}" if label else inner
            lines.append(f"* {line}")
        else:
            lines.append(f"* {item}")
    return "\n".join(lines)


def _verdict_label(score: int) -> str:
    if score >= 75:
        return "GOOD_VALUE"
    if score >= 55:
        return "BUY_WITH_CAUTION"
    if score >= 35:
        return "REVIEW_NEEDED"
    return "AVOID"


def _risk_label(score: int) -> str:
    if score >= 75:
        return "LOW"
    if score >= 50:
        return "MEDIUM"
    return "HIGH"


def _compute_overall_score(analyses: dict, verdict: dict) -> int:
    # Prefer the LLM's own overall_rating if present and valid
    rating = verdict.get("overall_rating") if isinstance(verdict, dict) else None
    if isinstance(rating, (int, float)) and 0 < rating <= 100:
        return int(rating)

    # Otherwise average the per-section scores (1-10) and scale to /100
    scores = []
    for v in analyses.values():
        if isinstance(v, dict):
            s = v.get("score")
            if isinstance(s, (int, float)) and 0 < s <= 10:
                scores.append(s)
    if not scores:
        return 50
    return round(sum(scores) / len(scores) * 10)


# ── MARKDOWN BUILDER ──────────────────────────────────────────────────

def _build_markdown(state: PolicyState, overall: int, verdict_label: str, risk_label: str) -> str:
    analyses = state.get("section_analyses", {})
    profile = state.get("company_profile", {})

    basics = analyses.get("policy_basics", {})
    coverage = analyses.get("coverage", {})
    limits = analyses.get("coverage_limits", {})
    exclusions = analyses.get("exclusions", {})
    waiting = analyses.get("waiting_periods", {})
    oop = analyses.get("out_of_pocket", {})
    premium = analyses.get("premium_analysis", {})
    claims = analyses.get("claims_process", {})
    renewal = analyses.get("renewal_cancellation", {})
    legal = analyses.get("legal_terms", {})
    definitions = analyses.get("definitions", {})
    risks = analyses.get("risks_concerns", {})
    rejection = analyses.get("claim_rejection_risk", {})
    examples = analyses.get("real_life_examples", {})
    hidden = analyses.get("hidden_surprises", {})
    comparison = analyses.get("comparison", {})
    plain = analyses.get("plain_english_summary", {})
    verdict = analyses.get("final_verdict", {})

    detailed_scores = verdict.get("detailed_scores", {}) if isinstance(verdict, dict) else {}

    md = []
    push = md.append

    # ── Quick summary ──
    push("# Your Insurance Policy Review\n")
    push("## Quick Summary\n")
    push("### At a Glance\n")
    push(f"* **Policy Name:** {_g(basics, 'policy_name')}")
    push(f"* **Insurance Company:** {_g(basics, 'insurer_name', state.get('insurer_name', 'Not specified in the policy'))}")
    push(f"* **Policy Type:** {_g(basics, 'policy_type')}")
    push(f"* **Overall Rating:** {overall}/100")
    push(f"* **Risk Level:** {risk_label}")
    push(f"* **Recommendation:** {verdict_label.replace('_', ' ')}\n")

    push("### Key Takeaways\n")
    push("**Best Things About This Policy**")
    push(_bullets(plain.get("what_we_like")))
    push("\n**Biggest Concerns**")
    push(_bullets(plain.get("what_we_dont_like") or risks.get("high_risk_clauses")))
    push("\n**Hidden Conditions to Know About**")
    push(_bullets(hidden.get("clauses_most_people_miss")))
    push("\n**Most Important Things Not Covered**")
    push(_bullets(exclusions.get("important_exclusions")))
    push("\n**How Easy It May Be to Make a Claim**")
    push(f"{_g(claims, 'summary')}")
    push("\n---\n")

    # ── Basic info ──
    push("# Basic Policy Information\n")
    push("## Policy Details\n")
    push(f"* **Policy Name:** {_g(basics, 'policy_name')}")
    push(f"* **Policy Number:** {_g(basics, 'policy_number')}")
    push(f"* **Start Date:** {_g(basics, 'start_date')}")
    push(f"* **End Date:** {_g(basics, 'end_date')}")
    push(f"* **Renewal Date:** {_g(basics, 'renewal_date')}")
    push(f"* **Policy Term:** {_g(basics, 'policy_term')}\n")

    push("## Contact Information\n")
    push(f"* **Customer Support:** {_g(basics, 'customer_support')}")
    push(f"* **Claims Support:** {_g(basics, 'claims_support')}")
    push(f"* **Emergency Assistance:** {_g(basics, 'emergency_assistance')}")
    push("\n---\n")

    # ── Coverage ──
    push("# What Is Covered?\n")
    push("## Main Benefits Included")
    push(_bullets(coverage.get("main_benefits")))
    push("\n## Additional Benefits Included")
    push(_bullets(coverage.get("additional_benefits")))
    push("\n## Optional Add-Ons Available")
    push(_bullets(coverage.get("optional_addons")))
    push("\n## Where and When You Are Covered")
    push(f"{_g(coverage, 'geographic_temporal_coverage')}")
    push("\n---\n")

    # ── Limits ──
    push("# How Much Protection Do You Actually Get?\n")
    push(f"## Total Coverage Amount\n\n{_g(limits, 'total_coverage_amount')}\n")
    push("## Limits on Specific Treatments or Events")
    push(_bullets(limits.get("treatment_event_limits")))
    push("\n## Annual Limits")
    push(_bullets(limits.get("annual_limits")))
    push("\n## Lifetime Limits")
    push(_bullets(limits.get("lifetime_limits")))
    push("\n## Coverage Caps You Should Know About")
    push(_bullets(limits.get("coverage_caps_to_know")))
    push("\n---\n")

    # ── Exclusions ──
    push("# What Is NOT Covered?\n")
    push("## Situations Where Claims Will Be Rejected")
    push(_bullets(exclusions.get("rejection_situations")))
    push("\n## Conditions and Treatments Not Covered")
    push(_bullets(exclusions.get("conditions_not_covered")))
    push("\n## Temporary Restrictions")
    push(_bullets(exclusions.get("temporary_restrictions")))
    push("\n## Important Exclusions That Could Affect You")
    push(_bullets(exclusions.get("important_exclusions")))
    push("\n## Clauses That May Be Difficult to Interpret")
    push(_bullets(exclusions.get("vague_or_ambiguous_clauses")))
    push("\n---\n")

    # ── Waiting periods ──
    push("# When Does Coverage Start?\n")
    push(f"## Initial Waiting Period\n\n{_g(waiting, 'initial_waiting_period')}\n")
    push("## Waiting Periods for Specific Conditions")
    push(_bullets(waiting.get("condition_specific_waiting_periods")))
    push(f"\n## Pre-Existing Condition Waiting Periods\n\n{_g(waiting, 'pre_existing_condition_waiting_period')}\n")
    push(f"## Maternity Waiting Periods\n\n{_g(waiting, 'maternity_waiting_period')}")
    push("\n---\n")

    # ── Out of pocket ──
    push("# What Costs Will You Still Pay Yourself?\n")
    push("## Deductibles")
    push(_bullets(oop.get("deductibles")))
    push("\n## Co-Payments")
    push(_bullets(oop.get("co_payments")))
    push("\n## Cost Sharing Requirements")
    push(_bullets(oop.get("cost_sharing_requirements")))
    push(f"\n## Expected Out-of-Pocket Expenses\n\n{_g(oop, 'expected_out_of_pocket_expenses')}")
    push("\n---\n")

    # ── Premium ──
    push("# Is the Premium Worth It?\n")
    push(f"## Current Premium\n\n{_g(premium, 'current_premium')}\n")
    push(f"## Future Premium Increase Risk\n\n{_g(premium, 'future_premium_increase_risk')}\n")
    push("## Extra Charges and Fees")
    push(_bullets(premium.get("extra_charges_and_fees")))
    push(f"\n## Overall Cost Assessment\n\n{_g(premium, 'overall_cost_assessment')}")
    push("\n---\n")

    # ── Claims process ──
    push("# How Does the Claim Process Work?\n")
    push("## Steps to Make a Claim")
    push(_bullets(claims.get("steps_to_make_a_claim")))
    push("\n## Documents You Will Need")
    push(_bullets(claims.get("documents_required")))
    push(f"\n## Claim Submission Deadlines\n\n{_g(claims, 'claim_submission_deadlines')}\n")
    push(f"## How Long Claim Approval May Take\n\n{_g(claims, 'approval_timeline')}\n")
    push("## Common Reasons Claims Get Rejected")
    push(_bullets(claims.get("common_rejection_reasons")))
    push("\n---\n")

    # ── Renewal ──
    push("# Can the Policy Be Renewed or Cancelled?\n")
    push(f"## Renewal Rules\n\n{_g(renewal, 'renewal_rules')}\n")
    push(f"## Grace Period\n\n{_g(renewal, 'grace_period')}\n")
    push("## When the Company Can Cancel Coverage")
    push(_bullets(renewal.get("when_company_can_cancel")))
    push("\n## When You Can Cancel Coverage")
    push(_bullets(renewal.get("when_you_can_cancel")))
    push("\n## Situations That May Cause the Policy to Lapse")
    push(_bullets(renewal.get("lapse_situations")))
    push("\n---\n")

    # ── Legal terms ──
    push("# Important Legal Terms You Should Know\n")
    push("## Your Rights")
    push(_bullets(legal.get("your_rights")))
    push("\n## Insurance Company Rights")
    push(_bullets(legal.get("insurance_company_rights")))
    push(f"\n## Dispute Resolution Process\n\n{_g(legal, 'dispute_resolution_process')}\n")
    push(f"## Fraud and Misrepresentation Rules\n\n{_g(legal, 'fraud_and_misrepresentation_rules')}\n")
    push(f"## Beneficiary and Nominee Rules\n\n{_g(legal, 'beneficiary_and_nominee_rules')}")
    push("\n---\n")

    # ── Definitions ──
    push("# Important Definitions That Could Affect Claims\n")
    push("## Key Insurance Terms Explained")
    push(_bullets(definitions.get("key_terms_explained")))
    push("\n## Unusual Definitions")
    push(_bullets(definitions.get("unusual_definitions")))
    push("\n## Definitions That Could Impact Claim Approval")
    push(_bullets(definitions.get("claim_impacting_definitions")))
    push("\n---\n")

    # ── Risks ──
    push("# Potential Risks and Concerns\n")
    push("## High-Risk Clauses")
    push(_bullets(risks.get("high_risk_clauses")))
    push("\n## Customer-Unfriendly Terms")
    push(_bullets(risks.get("customer_unfriendly_terms")))
    push("\n## Hidden Restrictions")
    push(_bullets(risks.get("hidden_restrictions")))
    push("\n## Areas That Need Extra Attention")
    push(_bullets(risks.get("areas_needing_extra_attention")))
    push("\n---\n")

    # ── Claim rejection risk ──
    push("# How Likely Is a Claim to Be Rejected?\n")
    push(f"## Overall Claim Risk Rating\n\n{_g(rejection, 'overall_claim_risk_rating')}\n")
    push("## Top Reasons a Claim Could Be Denied")
    push(_bullets(rejection.get("top_reasons_a_claim_could_be_denied")))
    push("\n## Real-Life Claim Rejection Scenarios")
    push(_bullets(rejection.get("real_life_claim_rejection_scenarios")))
    push("\n---\n")

    # ── Real-life examples ──
    push("# Real-Life Examples\n")
    push(f"## If You Are Hospitalized\n\n{_g(examples, 'if_hospitalized')}\n")
    push(f"## If You Have an Accident\n\n{_g(examples, 'if_accident')}\n")
    push(f"## If You Develop a Critical Illness\n\n{_g(examples, 'if_critical_illness')}\n")
    push(f"## If You Need Major Surgery\n\n{_g(examples, 'if_major_surgery')}\n")
    push(f"## Worst-Case Financial Scenario\n\n{_g(examples, 'worst_case_financial_scenario')}")
    push("\n---\n")

    # ── Hidden surprises ──
    push("# Hidden Surprises We Found\n")
    push("## Unexpected Restrictions")
    push(_bullets(hidden.get("unexpected_restrictions")))
    push("\n## Hidden Costs")
    push(_bullets(hidden.get("hidden_costs")))
    push("\n## Clauses Most People Miss")
    push(_bullets(hidden.get("clauses_most_people_miss")))
    push("\n## Things to Double-Check Before Buying")
    push(_bullets(hidden.get("things_to_double_check_before_buying")))
    push("\n---\n")

    # ── Comparison ──
    push("# How This Policy Compares to Others\n")
    push(f"## Coverage Quality\n\n{_g(comparison, 'coverage_quality')}\n")
    push(f"## Claim Friendliness\n\n{_g(comparison, 'claim_friendliness')}\n")
    push(f"## Value for Money\n\n{_g(comparison, 'value_for_money')}\n")
    push(f"## Transparency\n\n{_g(comparison, 'transparency')}\n")
    push(f"## Overall Competitiveness\n\n{_g(comparison, 'overall_competitiveness')}")
    push("\n---\n")

    # ── Plain English ──
    push("# Plain English Summary\n")
    push("## What We Like About This Policy")
    push(_bullets(plain.get("what_we_like")))
    push("\n## What We Don't Like About This Policy")
    push(_bullets(plain.get("what_we_dont_like")))
    push("\n## Who This Policy Is Best For")
    push(_bullets(plain.get("best_for")))
    push("\n## Who Should Consider Other Options")
    push(_bullets(plain.get("should_consider_other_options")))
    push("\n---\n")

    # ── Final verdict ──
    push("# Final Verdict\n")
    push("## Detailed Scores\n")
    push(f"* **Coverage:** {detailed_scores.get('coverage', '—')}/10")
    push(f"* **Coverage Limits:** {detailed_scores.get('coverage_limits', '—')}/10")
    push(f"* **Exclusions:** {detailed_scores.get('exclusions', '—')}/10")
    push(f"* **Waiting Periods:** {detailed_scores.get('waiting_periods', '—')}/10")
    push(f"* **Costs:** {detailed_scores.get('costs', '—')}/10")
    push(f"* **Claims Process:** {detailed_scores.get('claims_process', '—')}/10")
    push(f"* **Renewal Protection:** {detailed_scores.get('renewal_protection', '—')}/10")
    push(f"* **Transparency:** {detailed_scores.get('transparency', '—')}/10\n")

    push(f"## Overall Rating\n\n**{overall}/100**\n")
    push(f"## Confidence Level\n\n{_g(verdict, 'confidence_level', 'MEDIUM')}\n")
    push(f"## Final Recommendation\n\n**{verdict_label.replace('_', ' ')}** — {_g(verdict, 'one_line_verdict', '')}")
    push("\n---\n")

    # ── Company profile (Tavily-powered) ──
    push("# Company Profile  _(External — Tavily Research)_\n")
    if not profile or not profile.get("available"):
        reason = profile.get("reason", "Not available") if isinstance(profile, dict) else "Not available"
        push(f"*Company research unavailable: {reason}.*\n")
    else:
        summary = profile.get("summary", {})
        push(f"**Insurer:** {profile.get('insurer', 'Unknown')}\n")
        push(f"## Company Overview\n\n{summary.get('company_overview', 'Not found')}\n")
        push(f"## Claim Settlement Ratio\n\n{summary.get('claim_settlement_ratio', 'Not found')}\n")
        push("## Recent Disputes")
        push(_bullets(summary.get("recent_disputes"), empty="No significant public disputes surfaced"))
        push(f"\n## Customer Reviews & Overall Sentiment\n\n{summary.get('customer_reviews_summary', 'Not found')}\n")
        push(f"## Ratings\n\n{summary.get('ratings', 'Not found')}\n")
        push(f"## Market Share in Insurance Sector\n\n{summary.get('market_share', 'Not found')}\n")
        push(f"## Credibility & Trust\n\n{summary.get('credibility', 'Not found')}\n")
        push(f"## Overall Company Assessment\n\n{summary.get('overall_assessment', 'Not found')}\n")
        push(f"**Trust Score:** {summary.get('trust_score', '—')}/100\n")

        # Sources
        push("### Sources Consulted\n")
        for facet, body in profile.get("facets", {}).items():
            for src in body.get("sources", [])[:3]:
                title = src.get("title", "(untitled)")
                url = src.get("url", "")
                push(f"* _{facet.replace('_', ' ').title()}_ — [{title}]({url})")

    return "\n".join(md)


# ── NODE ──────────────────────────────────────────────────────────────

def report_compiler_node(state: PolicyState) -> PolicyState:
    print("[Compiler] Building final report...")
    analyses = state.get("section_analyses", {})
    verdict = analyses.get("final_verdict", {}) if isinstance(analyses, dict) else {}

    overall = _compute_overall_score(analyses, verdict)
    verdict_label = verdict.get("final_recommendation") if isinstance(verdict, dict) else None
    if verdict_label not in {"GOOD_VALUE", "BUY_WITH_CAUTION", "REVIEW_NEEDED", "AVOID"}:
        verdict_label = _verdict_label(overall)
    risk_label = verdict.get("risk_level") if isinstance(verdict, dict) else None
    if risk_label not in {"LOW", "MEDIUM", "HIGH"}:
        risk_label = _risk_label(overall)

    markdown = _build_markdown(state, overall, verdict_label, risk_label)

    report = {
        "report_metadata": {
            "session_id": state["session_id"],
            "generated_at": datetime.now().isoformat(),
            "policy_file": state["pdf_path"].split("/")[-1],
            "total_pages_analyzed": len(state.get("ocr_text", [])),
            "total_citations": len(state.get("citations", [])),
        },
        "headline": {
            "insurer_name": state.get("insurer_name", "Unknown"),
            "overall_rating": overall,
            "risk_level": risk_label,
            "recommendation": verdict_label,
        },
        "sections": analyses,
        "company_profile": state.get("company_profile", {}),
        "all_citations": state.get("citations", []),
        "markdown": markdown,
    }

    print(f"[Compiler] Report complete. Score: {overall}/100. "
          f"Verdict: {verdict_label}. Risk: {risk_label}.")
    return {**state,
            "final_report": report,
            "report_markdown": markdown,
            "status": "complete"}
