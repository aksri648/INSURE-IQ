"""Report compiler.

Consumes ONLY validated findings (post-validator) and emits LaTeX that, when
rendered, looks like the markdown spec the user provided. Includes inline
citation tags so the reader can see whether each finding is TRUSTED or
NEEDS_HUMAN_REVIEW, plus the source page reference.

The compiled PDF is the only thing exposed in the Streamlit UI.
"""

from datetime import datetime

from agents.state import PolicyState
from utils.pdf_builder import compile_latex_to_pdf


# ── LaTeX escaping ───────────────────────────────────────────────────

_LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _esc(s) -> str:
    if s is None:
        return ""
    s = str(s)
    out = []
    for ch in s:
        out.append(_LATEX_ESCAPES.get(ch, ch))
    return "".join(out)


def _esc_url(s) -> str:
    return str(s or "").replace("%", r"\%").replace("#", r"\#")


# ── Document scaffolding ─────────────────────────────────────────────

LATEX_PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[margin=0.9in]{geometry}
\usepackage{titlesec}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{enumitem}
\usepackage{tcolorbox}
\usepackage{fancyhdr}
\usepackage{parskip}

% Colours
\definecolor{InsureBlue}{HTML}{1E40AF}
\definecolor{TrustGreen}{HTML}{047857}
\definecolor{ReviewAmber}{HTML}{B45309}
\definecolor{DangerRed}{HTML}{B91C1C}
\definecolor{MutedGrey}{HTML}{475569}
\definecolor{CardBg}{HTML}{F1F5F9}

% Headings styled to mimic markdown rendering
\titleformat{\section}{\Large\bfseries\color{InsureBlue}}{}{0pt}{}
\titleformat{\subsection}{\large\bfseries\color{InsureBlue}}{}{0pt}{}
\titleformat{\subsubsection}{\normalsize\bfseries\color{MutedGrey}}{}{0pt}{}
\titlespacing*{\section}{0pt}{14pt}{6pt}
\titlespacing*{\subsection}{0pt}{10pt}{4pt}
\titlespacing*{\subsubsection}{0pt}{8pt}{2pt}

\setlist[itemize]{leftmargin=*,itemsep=2pt,topsep=2pt}

\hypersetup{
  colorlinks=true,
  linkcolor=InsureBlue,
  urlcolor=InsureBlue,
}

% Trust / Review tag macros
\newcommand{\tagTrusted}{%
  \tcbox[on line,colback=TrustGreen!10,colframe=TrustGreen,boxrule=0.4pt,
         arc=2pt,left=2pt,right=2pt,top=0pt,bottom=0pt,nobeforeafter]%
  {\scriptsize\textbf{\textcolor{TrustGreen}{TRUSTED}}}}
\newcommand{\tagReview}{%
  \tcbox[on line,colback=ReviewAmber!10,colframe=ReviewAmber,boxrule=0.4pt,
         arc=2pt,left=2pt,right=2pt,top=0pt,bottom=0pt,nobeforeafter]%
  {\scriptsize\textbf{\textcolor{ReviewAmber}{NEEDS HUMAN REVIEW}}}}

\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0pt}
\fancyfoot[C]{\small\textcolor{MutedGrey}{InsureIQ Policy Review \textbullet{} Page \thepage}}

\setcounter{secnumdepth}{0}
"""


# ── Helpers to build a finding line ──────────────────────────────────

def _tag_macro(trust_tag: str) -> str:
    return r"\tagTrusted" if trust_tag == "TRUSTED" else r"\tagReview"


def _citation_inline(f: dict) -> str:
    cid = _esc(f.get("citation_id", ""))
    chunk = _esc(f.get("chunk_id", ""))
    page = _esc(f.get("page", "?"))
    section = _esc(f.get("section", ""))
    tag = _tag_macro(f.get("trust_tag", "NEEDS_HUMAN_REVIEW"))
    pieces = [f"{tag}",
              fr"\textcolor{{MutedGrey}}{{\scriptsize\,[{cid} \textbullet{{}} {chunk} \textbullet{{}} Page {page}"]
    if section:
        pieces.append(f" \\textbullet{{}} {section}")
    pieces.append("]}")
    return "".join(pieces)


def _finding_bullet(f: dict) -> str:
    claim = _esc(f.get("claim", ""))
    quote = _esc(f.get("verbatim_quote", "")).strip()
    cite = _citation_inline(f)
    notes = f.get("validation_notes", "")
    notes_line = ""
    if notes and f.get("trust_tag") == "NEEDS_HUMAN_REVIEW":
        notes_line = (fr"\\\textcolor{{ReviewAmber}}{{\scriptsize "
                      fr"Reviewer note: {_esc(notes)}}}")
    return (r"\item " + claim + " " + cite +
            fr"\\\textcolor{{MutedGrey}}{{\scriptsize\itshape "
            fr"``{quote}''}}" + notes_line)


def _findings_by_label(section_data: dict, labels: list) -> dict:
    grouped = {lbl: [] for lbl in labels}
    for f in section_data.get("findings", []):
        lbl = f.get("label")
        if lbl in grouped:
            grouped[lbl].append(f)
    return grouped


def _render_label_block(label: str, findings: list, empty_msg: str = "Not specified in the policy.") -> str:
    out = [fr"\subsubsection*{{{_esc(label)}}}"]
    if not findings:
        out.append(fr"\textcolor{{MutedGrey}}{{\itshape {_esc(empty_msg)}}}")
        return "\n".join(out) + "\n"
    out.append(r"\begin{itemize}")
    for f in findings:
        out.append(_finding_bullet(f))
    out.append(r"\end{itemize}")
    return "\n".join(out) + "\n"


# Map of section -> ordered labels (must match analyst SECTION_SPECS labels)
SECTION_LABELS = {
    "policy_basics": [
        "Policy Name", "Policy Number", "Insurer Name", "Policy Type",
        "Start Date", "End Date", "Renewal Date", "Policy Term",
        "Customer Support", "Claims Support", "Emergency Assistance",
    ],
    "coverage": [
        "Main Benefits", "Additional Benefits", "Optional Add-Ons",
        "Where and When You Are Covered",
    ],
    "coverage_limits": [
        "Total Coverage Amount", "Treatment/Event Limits",
        "Annual Limits", "Lifetime Limits", "Coverage Caps",
    ],
    "exclusions": [
        "Situations Where Claims Will Be Rejected",
        "Conditions and Treatments Not Covered",
        "Temporary Restrictions",
        "Important Exclusions",
        "Vague or Ambiguous Clauses",
    ],
    "waiting_periods": [
        "Initial Waiting Period",
        "Condition-Specific Waiting Periods",
        "Pre-Existing Condition Waiting Period",
        "Maternity Waiting Period",
    ],
    "out_of_pocket": [
        "Deductibles", "Co-Payments", "Cost Sharing Requirements",
        "Expected Out-of-Pocket Expenses",
    ],
    "premium_analysis": [
        "Current Premium", "Future Premium Increase Risk",
        "Extra Charges and Fees", "Overall Cost Assessment",
    ],
    "claims_process": [
        "Steps to Make a Claim", "Documents You Will Need",
        "Claim Submission Deadlines", "Approval Timeline",
        "Common Rejection Reasons",
    ],
    "renewal_cancellation": [
        "Renewal Rules", "Grace Period",
        "When the Company Can Cancel", "When You Can Cancel",
        "Lapse Situations",
    ],
    "legal_terms": [
        "Your Rights", "Insurance Company Rights",
        "Dispute Resolution Process",
        "Fraud and Misrepresentation Rules",
        "Beneficiary and Nominee Rules",
    ],
    "definitions": [
        "Key Insurance Terms Explained",
        "Unusual Definitions",
        "Definitions That Could Impact Claim Approval",
    ],
    "risks_concerns": [
        "High-Risk Clauses", "Customer-Unfriendly Terms",
        "Hidden Restrictions", "Areas Needing Extra Attention",
    ],
    "claim_rejection_risk": [
        "Top Reasons a Claim Could Be Denied",
        "Real-Life Claim Rejection Scenarios",
    ],
    "hidden_surprises": [
        "Unexpected Restrictions", "Hidden Costs",
        "Clauses Most People Miss",
        "Things to Double-Check Before Buying",
    ],
}


# ── Aggregate scores / verdict ───────────────────────────────────────

VERDICT_BANDS = [
    (75, "GOOD VALUE",        "TrustGreen"),
    (55, "BUY WITH CAUTION",  "ReviewAmber"),
    (35, "REVIEW NEEDED",     "DangerRed"),
    (0,  "AVOID",             "DangerRed"),
]


def _verdict(overall: int) -> tuple[str, str]:
    for threshold, label, color in VERDICT_BANDS:
        if overall >= threshold:
            return label, color
    return "AVOID", "DangerRed"


def _risk_level(overall: int) -> tuple[str, str]:
    if overall >= 75:
        return "LOW", "TrustGreen"
    if overall >= 50:
        return "MEDIUM", "ReviewAmber"
    return "HIGH", "DangerRed"


def _compute_overall(validated: dict, validation_report: dict) -> int:
    scores = []
    for v in validated.values():
        s = v.get("score") if isinstance(v, dict) else None
        if isinstance(s, (int, float)) and 0 < s <= 10:
            scores.append(s)
    base = (sum(scores) / len(scores) * 10) if scores else 50
    # Penalise if many findings needed review
    counts = validation_report.get("counts", {})
    total = counts.get("total", 0) or 0
    review = counts.get("review", 0) or 0
    if total:
        penalty = (review / total) * 15  # up to -15 if every finding was shaky
        base -= penalty
    return max(0, min(100, round(base)))


# ── Insurer + headline extraction from validated findings ────────────

def _first_finding_claim(validated: dict, section: str, label: str, default: str) -> str:
    findings = validated.get(section, {}).get("findings", [])
    for f in findings:
        if f.get("label") == label and f.get("claim"):
            return f["claim"]
    return default


# ── Document builders ────────────────────────────────────────────────

def _build_header(state: PolicyState, validated: dict, overall: int,
                  verdict_label: str, verdict_color: str,
                  risk_label: str, risk_color: str) -> str:
    insurer = state.get("insurer_name") or _first_finding_claim(
        validated, "policy_basics", "Insurer Name", "Not specified in the policy")
    policy_name = _first_finding_claim(
        validated, "policy_basics", "Policy Name", "Not specified in the policy")
    policy_type = _first_finding_claim(
        validated, "policy_basics", "Policy Type", "Not specified in the policy")

    return rf"""
\begin{{center}}
{{\Huge\bfseries\color{{InsureBlue}} Your Insurance Policy Review}}\\[6pt]
{{\large\textcolor{{MutedGrey}}{{Generated {_esc(datetime.now().strftime('%d %b %Y, %H:%M'))}}}}}
\end{{center}}

\vspace{{6pt}}
\begin{{tcolorbox}}[colback=CardBg,colframe=InsureBlue,boxrule=0.6pt,arc=4pt]
\textbf{{Policy Name:}} {_esc(policy_name)}\\
\textbf{{Insurance Company:}} {_esc(insurer)}\\
\textbf{{Policy Type:}} {_esc(policy_type)}\\
\textbf{{Overall Rating:}} \textcolor{{{verdict_color}}}{{\textbf{{{overall}/100}}}}\\
\textbf{{Risk Level:}} \textcolor{{{risk_color}}}{{\textbf{{{risk_label}}}}}\\
\textbf{{Recommendation:}} \textcolor{{{verdict_color}}}{{\textbf{{{verdict_label}}}}}
\end{{tcolorbox}}
"""


def _key_takeaways(validated: dict) -> str:
    likes = [f for f in validated.get("coverage", {}).get("findings", [])]
    concerns = [f for f in validated.get("risks_concerns", {}).get("findings", [])]
    hidden = [f for f in validated.get("hidden_surprises", {}).get("findings", [])]
    exclusions = [f for f in validated.get("exclusions", {}).get("findings", [])
                  if f.get("label") == "Important Exclusions"]
    claims_proc = [f for f in validated.get("claims_process", {}).get("findings", [])
                   if f.get("label") in ("Steps to Make a Claim", "Common Rejection Reasons")]

    def _short_bullets(items, n=4):
        if not items:
            return r"\textcolor{MutedGrey}{\itshape Not specified in the policy.}"
        out = [r"\begin{itemize}"]
        for f in items[:n]:
            out.append(_finding_bullet(f))
        out.append(r"\end{itemize}")
        return "\n".join(out)

    return rf"""
\subsection*{{At a Glance}}
See the summary card above.

\subsection*{{Key Takeaways}}

\subsubsection*{{Best Things About This Policy}}
{_short_bullets(likes)}

\subsubsection*{{Biggest Concerns}}
{_short_bullets(concerns)}

\subsubsection*{{Hidden Conditions to Know About}}
{_short_bullets(hidden)}

\subsubsection*{{Most Important Things Not Covered}}
{_short_bullets(exclusions)}

\subsubsection*{{How Easy It May Be to Make a Claim}}
{_short_bullets(claims_proc)}
"""


def _render_section(title: str, section_key: str, validated: dict) -> str:
    labels = SECTION_LABELS.get(section_key, [])
    section_data = validated.get(section_key, {})
    grouped = _findings_by_label(section_data, labels)
    parts = [fr"\section*{{{_esc(title)}}}"]
    for lbl in labels:
        parts.append(_render_label_block(lbl, grouped.get(lbl, [])))
    return "\n".join(parts)


def _render_final_verdict(validated: dict, overall: int,
                          verdict_label: str, verdict_color: str,
                          risk_label: str, risk_color: str,
                          validation_report: dict) -> str:
    # Per-axis scores from the analyst (1-10)
    axes = [
        ("Coverage", "coverage"),
        ("Coverage Limits", "coverage_limits"),
        ("Exclusions", "exclusions"),
        ("Waiting Periods", "waiting_periods"),
        ("Costs", "out_of_pocket"),
        ("Claims Process", "claims_process"),
        ("Renewal Protection", "renewal_cancellation"),
        ("Transparency", "hidden_surprises"),
    ]
    items = []
    for label, key in axes:
        score = validated.get(key, {}).get("score")
        if score is None:
            score_str = "--/10"
        else:
            score_str = f"{score}/10"
        items.append(fr"\item \textbf{{{_esc(label)}:}} {score_str}")

    counts = validation_report.get("counts", {})
    trusted_ratio = validation_report.get("trusted_ratio", 0)
    confidence = "HIGH" if trusted_ratio >= 0.8 else ("MEDIUM" if trusted_ratio >= 0.5 else "LOW")
    confidence_color = "TrustGreen" if confidence == "HIGH" else ("ReviewAmber" if confidence == "MEDIUM" else "DangerRed")

    items_block = "\n".join(items)
    return rf"""
\section*{{Final Verdict}}

\subsection*{{Detailed Scores}}
\begin{{itemize}}
{items_block}
\end{{itemize}}

\subsection*{{Overall Rating}}
\textbf{{\textcolor{{{verdict_color}}}{{{overall}/100}}}}

\subsection*{{Risk Level}}
\textbf{{\textcolor{{{risk_color}}}{{{risk_label}}}}}

\subsection*{{Confidence Level}}
\textbf{{\textcolor{{{confidence_color}}}{{{confidence}}}}}
\textcolor{{MutedGrey}}{{\small (based on validator: {counts.get('trusted', 0)} trusted, {counts.get('review', 0)} need review, {counts.get('dropped', 0)} dropped)}}

\subsection*{{Final Recommendation}}
\textbf{{\textcolor{{{verdict_color}}}{{{verdict_label}}}}}
"""


def _render_company_profile(profile: dict) -> str:
    if not profile or not profile.get("available"):
        reason = (profile or {}).get("reason", "Not available")
        return rf"""
\section*{{Company Profile \textnormal{{\small (External --- Tavily Research)}}}}
\textcolor{{MutedGrey}}{{\itshape Company research unavailable: {_esc(reason)}.}}
"""
    summary = profile.get("summary", {})

    def _field(label, key, default="Not found"):
        val = summary.get(key) or default
        return fr"\subsection*{{{_esc(label)}}}" + "\n" + _esc(val) + "\n"

    disputes = summary.get("recent_disputes") or []
    disputes_block = r"\textcolor{MutedGrey}{\itshape No significant public disputes surfaced.}"
    if disputes:
        disputes_block = (r"\begin{itemize}" + "\n" +
                          "\n".join(fr"\item {_esc(d)}" for d in disputes) +
                          "\n" + r"\end{itemize}")

    sources_lines = []
    for facet, body in (profile.get("facets") or {}).items():
        for src in (body or {}).get("sources", [])[:3]:
            title = src.get("title") or "(untitled)"
            url = src.get("url") or ""
            sources_lines.append(
                fr"\item \textit{{{_esc(facet.replace('_', ' ').title())}}} --- "
                fr"\href{{{_esc_url(url)}}}{{{_esc(title)}}}"
            )
    sources_block = ""
    if sources_lines:
        sources_block = (r"\subsection*{Sources Consulted}" + "\n" +
                         r"\begin{itemize}" + "\n" +
                         "\n".join(sources_lines) + "\n" +
                         r"\end{itemize}")

    trust = summary.get("trust_score", "--")
    return rf"""
\section*{{Company Profile \textnormal{{\small (External --- Tavily Research)}}}}
\textbf{{Insurer:}} {_esc(profile.get('insurer', 'Unknown'))}

{_field("Company Overview", "company_overview")}
{_field("Claim Settlement Ratio", "claim_settlement_ratio")}
\subsection*{{Recent Disputes}}
{disputes_block}

{_field("Customer Reviews and Overall Sentiment", "customer_reviews_summary")}
{_field("Ratings", "ratings")}
{_field("Market Share in Insurance Sector", "market_share")}
{_field("Credibility and Trust", "credibility")}
{_field("Overall Company Assessment", "overall_assessment")}

\textbf{{Trust Score:}} {_esc(trust)}/100

{sources_block}
"""


def _render_legend() -> str:
    return r"""
\section*{How to Read This Report}
Every finding in this report is tagged by the InsureIQ validator agent:
\begin{itemize}
\item \tagTrusted{} --- the quoted text was found verbatim in the policy
document at the cited page. Safe to rely on.
\item \tagReview{} --- the quote could not be matched exactly. Please verify
this point manually against the policy PDF before acting on it.
\end{itemize}
Each finding shows the original policy quote in italics with its
\textit{citation id}, \textit{chunk id}, and \textit{page reference}.
"""


# ── Public node ──────────────────────────────────────────────────────

def report_compiler_node(state: PolicyState) -> PolicyState:
    print("[Compiler] Building LaTeX report...")

    validated = state.get("validated_sections") or {}
    validation_report = state.get("validation_report") or {}
    profile = state.get("company_profile") or {}

    overall = _compute_overall(validated, validation_report)
    verdict_label, verdict_color = _verdict(overall)
    risk_label, risk_color = _risk_level(overall)

    parts = [LATEX_PREAMBLE, r"\begin{document}"]
    parts.append(_build_header(state, validated, overall,
                               verdict_label, verdict_color,
                               risk_label, risk_color))
    parts.append(_render_legend())

    parts.append(r"\section*{Quick Summary}")
    parts.append(_key_takeaways(validated))

    parts.append(_render_section("Basic Policy Information", "policy_basics", validated))
    parts.append(_render_section("What Is Covered?", "coverage", validated))
    parts.append(_render_section("How Much Protection Do You Actually Get?", "coverage_limits", validated))
    parts.append(_render_section("What Is NOT Covered?", "exclusions", validated))
    parts.append(_render_section("When Does Coverage Start?", "waiting_periods", validated))
    parts.append(_render_section("What Costs Will You Still Pay Yourself?", "out_of_pocket", validated))
    parts.append(_render_section("Is the Premium Worth It?", "premium_analysis", validated))
    parts.append(_render_section("How Does the Claim Process Work?", "claims_process", validated))
    parts.append(_render_section("Can the Policy Be Renewed or Cancelled?", "renewal_cancellation", validated))
    parts.append(_render_section("Important Legal Terms You Should Know", "legal_terms", validated))
    parts.append(_render_section("Important Definitions That Could Affect Claims", "definitions", validated))
    parts.append(_render_section("Potential Risks and Concerns", "risks_concerns", validated))
    parts.append(_render_section("How Likely Is a Claim to Be Rejected?", "claim_rejection_risk", validated))
    parts.append(_render_section("Hidden Surprises We Found", "hidden_surprises", validated))

    parts.append(_render_final_verdict(validated, overall,
                                       verdict_label, verdict_color,
                                       risk_label, risk_color,
                                       validation_report))
    parts.append(_render_company_profile(profile))

    parts.append(r"\end{document}")
    latex_source = "\n".join(parts)

    print(f"[Compiler] LaTeX source built ({len(latex_source)} chars). Compiling PDF...")
    pdf_bytes = compile_latex_to_pdf(latex_source)
    print(f"[Compiler] PDF ready ({len(pdf_bytes)} bytes).")

    final_report = {
        "session_id": state["session_id"],
        "generated_at": datetime.now().isoformat(),
        "policy_file": state["pdf_path"].split("/")[-1],
        "headline": {
            "insurer_name": state.get("insurer_name", "Unknown"),
            "overall_rating": overall,
            "risk_level": risk_label,
            "recommendation": verdict_label,
        },
        "sections": validated,
        "company_profile": profile,
        "validation_report": validation_report,
        "citations": state.get("citations", []),
    }

    return {**state,
            "final_report": final_report,
            "latex_source": latex_source,
            "pdf_bytes": pdf_bytes,
            "status": "complete",
            "active_node": "compiler"}
