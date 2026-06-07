import json
import os
import sys
import tempfile
import uuid

import streamlit as st
from dotenv import load_dotenv

# Make sure local packages are importable when launched from anywhere
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

load_dotenv()

from utils.model_config import write_model_config  # noqa: E402

# Materialise model config file early so all nodes read consistent values
write_model_config()

from graph import pipeline  # noqa: E402


st.set_page_config(
    page_title="InsureIQ — AI Policy Analyst",
    page_icon="⬡",
    layout="wide",
)

st.markdown(
    """
<style>
  .stApp { background: #0a0e1a; color: #e2e8f0; }
  .verdict-GOOD_VALUE       { color: #10b981; font-weight: 700; }
  .verdict-BUY_WITH_CAUTION { color: #f59e0b; font-weight: 700; }
  .verdict-REVIEW_NEEDED    { color: #ef4444; font-weight: 700; }
  .verdict-AVOID            { color: #dc2626; font-weight: 900; }
  .risk-LOW    { color: #10b981; font-weight: 700; }
  .risk-MEDIUM { color: #f59e0b; font-weight: 700; }
  .risk-HIGH   { color: #ef4444; font-weight: 700; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown("## ⬡ InsureIQ — AI Insurance Policy Analyst")
st.caption(
    "Upload any insurance policy PDF for a full cited analysis · Powered by local AI + Tavily company research"
)

uploaded_file = st.file_uploader(
    "Upload Insurance Policy PDF",
    type=["pdf"],
    help="Your PDF stays on this machine — local Ollama inference only",
)

if uploaded_file:
    col1, _, col3 = st.columns([2, 1, 1])
    with col1:
        st.info(f"📄 {uploaded_file.name} — {uploaded_file.size // 1024} KB")
    with col3:
        if st.button("🔍 Analyze Policy", type="primary", use_container_width=True):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            session_id = str(uuid.uuid4())
            st.session_state["last_session_id"] = session_id

            progress_bar = st.progress(0)
            status_text = st.empty()

            stages = [
                ("ocr_complete",              "👁  OCR Extraction",          20),
                ("rag_complete",              "🗄  Building Vector Store",   40),
                ("web_research_complete",     "🌐  External Research",       55),
                ("analysis_complete",         "🤖  Deep Section Analysis",   80),
                ("company_profile_complete",  "🏢  Company Profile",         92),
                ("complete",                  "📋  Compiling Report",       100),
            ]
            stage_map = {s[0]: (s[1], s[2]) for s in stages}

            initial_state = {
                "session_id": session_id,
                "pdf_path": tmp_path,
                "ocr_text": [],
                "chunks": [],
                "insurer_name": "",
                "external_research": {},
                "company_profile": {},
                "section_analyses": {},
                "final_report": {},
                "report_markdown": "",
                "citations": [],
                "error": None,
                "status": "starting",
            }

            status_text.text("🚀 Starting analysis pipeline...")
            final_state = initial_state

            try:
                for event in pipeline.stream(initial_state):
                    for _, node_state in event.items():
                        current_status = node_state.get("status", "")
                        if current_status in stage_map:
                            label, pct = stage_map[current_status]
                            progress_bar.progress(pct)
                            status_text.text(f"{label}...")
                        final_state = node_state

                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")
                st.session_state["report"] = final_state.get("final_report", {})
                st.session_state["markdown"] = final_state.get("report_markdown", "")
            except Exception as e:
                st.error(f"Pipeline error: {e}")
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

# ── Report Display ──────────────────────────────────────────────────
if st.session_state.get("report"):
    report = st.session_state["report"]
    markdown = st.session_state.get("markdown", "")
    headline = report.get("headline", {})
    session_id = st.session_state.get("last_session_id", "session")

    st.divider()
    st.markdown("### 📊 Analysis Report")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        score = headline.get("overall_rating", 0)
        color = (
            "#10b981" if score >= 75 else "#f59e0b" if score >= 55 else "#ef4444"
        )
        st.markdown(
            f"<h2 style='color:{color};margin:0'>{score}"
            f"<span style='font-size:14px'>/100</span></h2>",
            unsafe_allow_html=True,
        )
        st.caption("Overall Rating")
    with col2:
        verdict = headline.get("recommendation", "UNKNOWN")
        st.markdown(
            f"<p class='verdict-{verdict}'>{verdict.replace('_', ' ')}</p>",
            unsafe_allow_html=True,
        )
        st.caption("Recommendation")
    with col3:
        risk = headline.get("risk_level", "UNKNOWN")
        st.markdown(
            f"<p class='risk-{risk}'>{risk}</p>",
            unsafe_allow_html=True,
        )
        st.caption("Risk Level")
    with col4:
        st.metric("Insurer", (headline.get("insurer_name") or "—")[:25])

    tab_report, tab_company, tab_json = st.tabs(
        ["📄 Full Report", "🏢 Company Profile", "🧾 Raw JSON"]
    )

    with tab_report:
        if markdown:
            st.markdown(markdown)
        else:
            st.info("No report content available.")

    with tab_company:
        profile = report.get("company_profile", {})
        if not profile or not profile.get("available"):
            reason = profile.get("reason") if isinstance(profile, dict) else None
            st.warning(
                "Company profile unavailable. "
                f"{('Reason: ' + reason) if reason else 'Set TAVILY_API_KEY in .env to enable.'}"
            )
        else:
            summary = profile.get("summary", {})
            st.markdown(f"**Insurer:** {profile.get('insurer', 'Unknown')}")
            st.markdown(f"**Trust Score:** {summary.get('trust_score', '—')}/100")

            for k, label in [
                ("company_overview",          "Company Overview"),
                ("claim_settlement_ratio",    "Claim Settlement Ratio"),
                ("customer_reviews_summary",  "Customer Reviews"),
                ("ratings",                   "Ratings"),
                ("market_share",              "Market Share"),
                ("credibility",               "Credibility"),
                ("overall_assessment",        "Overall Assessment"),
            ]:
                st.markdown(f"#### {label}")
                st.markdown(summary.get(k, "Not found"))

            disputes = summary.get("recent_disputes") or []
            if disputes:
                st.markdown("#### Recent Disputes")
                for d in disputes:
                    st.markdown(f"- {d}")

            with st.expander("🔗 Sources Consulted (Tavily)"):
                for facet, body in profile.get("facets", {}).items():
                    for src in body.get("sources", [])[:3]:
                        title = src.get("title", "(untitled)")
                        url = src.get("url", "")
                        st.markdown(
                            f"- _{facet.replace('_', ' ').title()}_ — [{title}]({url})"
                        )

    with tab_json:
        st.json(report)

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button(
            label="⬇️ Download Report (Markdown)",
            data=markdown.encode("utf-8") if markdown else b"",
            file_name=f"insureiq_report_{session_id[:8]}.md",
            mime="text/markdown",
            disabled=not markdown,
        )
    with col_b:
        st.download_button(
            label="⬇️ Download Report (JSON)",
            data=json.dumps(report, indent=2),
            file_name=f"insureiq_report_{session_id[:8]}.json",
            mime="application/json",
        )
