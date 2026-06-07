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
  .citation-badge {
    background: #1e2d45; border: 1px solid #00d4ff33;
    border-radius: 4px; padding: 2px 8px;
    font-size: 11px; color: #00d4ff; font-family: monospace;
  }
  .red-flag {
    background: #ef444410; border-left: 3px solid #ef4444;
    padding: 8px 12px; margin: 4px 0; border-radius: 2px;
  }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown("## ⬡ InsureIQ — AI Insurance Policy Analyst")
st.caption(
    "Upload any insurance policy PDF for a full cited analysis · Powered by local AI"
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
                ("ocr_complete",          "👁  OCR Extraction",        25),
                ("rag_complete",          "🗄  Building Vector Store",  50),
                ("web_research_complete", "🌐  External Research",      65),
                ("analysis_complete",     "🤖  Deep Analysis",          85),
                ("complete",              "📋  Compiling Report",      100),
            ]
            stage_map = {s[0]: (s[1], s[2]) for s in stages}

            initial_state = {
                "session_id": session_id,
                "pdf_path": tmp_path,
                "ocr_text": [],
                "chunks": [],
                "insurer_name": "",
                "external_research": {},
                "section_analyses": {},
                "final_report": {},
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
            except Exception as e:
                st.error(f"Pipeline error: {e}")
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

if "report" in st.session_state and st.session_state["report"]:
    report = st.session_state["report"]
    summary = report.get("executive_summary", {})
    sections = report.get("sections", {})
    session_id = st.session_state.get("last_session_id", "session")

    st.divider()
    st.markdown("### 📊 Analysis Report")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        score = summary.get("analyst_score", 0)
        color = (
            "#10b981" if score >= 75 else "#f59e0b" if score >= 55 else "#ef4444"
        )
        st.markdown(
            f"<h2 style='color:{color};margin:0'>{score}"
            f"<span style='font-size:14px'>/100</span></h2>",
            unsafe_allow_html=True,
        )
        st.caption("Analyst Score")
    with col2:
        verdict = summary.get("verdict", "UNKNOWN")
        st.markdown(
            f"<p class='verdict-{verdict}'>{verdict.replace('_', ' ')}</p>",
            unsafe_allow_html=True,
        )
        st.caption("Verdict")
    with col3:
        st.metric("Insurer", summary.get("insurer_name", "—")[:25])
    with col4:
        st.metric("Citations", len(report.get("all_citations", [])))

    flags = summary.get("key_red_flags", [])
    if flags:
        st.markdown("#### ⚠️ Key Red Flags")
        for flag in flags:
            st.markdown(
                f"<div class='red-flag'>⚠️ {flag}</div>", unsafe_allow_html=True
            )

    section_labels = {
        "executive_summary": "📋 Summary",
        "coverage_benefits": "✅ Benefits",
        "exclusions":        "❌ Exclusions",
        "waiting_periods":   "⏳ Waiting",
        "premium_analysis":  "💰 Premium",
        "claims_process":    "📝 Claims",
        "risk_assessment":   "⚠️ Risks",
        "policy_conditions": "📜 Conditions",
    }

    if sections:
        tabs = st.tabs([section_labels.get(k, k) for k in sections.keys()])
        for tab, (_, section_data) in zip(tabs, sections.items()):
            with tab:
                if isinstance(section_data, dict):
                    if section_data.get("summary"):
                        st.markdown(section_data["summary"])
                    for finding in section_data.get("findings", []):
                        with st.expander(
                            f"📌 {finding.get('claim', 'Finding')[:80]}"
                        ):
                            st.write(finding.get("claim", ""))
                            if "citation" in finding:
                                st.markdown(
                                    f"<span class='citation-badge'>"
                                    f"{finding['citation']}</span>",
                                    unsafe_allow_html=True,
                                )

    st.divider()
    st.download_button(
        label="⬇️ Download Full Report (JSON)",
        data=json.dumps(report, indent=2),
        file_name=f"insureiq_report_{session_id[:8]}.json",
        mime="application/json",
    )
