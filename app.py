"""InsureIQ Streamlit UI.

Layout (three columns):
    LEFT    — PDF upload + Analyze button.
    MIDDLE  — vertical flowchart of all agents. The active node's border
              animates while it executes. Completed nodes get a green border.
    RIGHT   — LaTeX source viewer (read-only) + a single "Download Report PDF"
              button below the viewer.

The pipeline is run in a background thread so the UI can poll a shared state
slot and re-render the flowchart as each node completes.
"""

import os
import queue
import sys
import tempfile
import threading
import time
import uuid

import streamlit as st
from dotenv import load_dotenv


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

load_dotenv()

from utils.model_config import write_model_config  # noqa: E402

write_model_config()

from graph import NODE_LABELS, NODE_ORDER, pipeline  # noqa: E402


st.set_page_config(
    page_title="InsureIQ — AI Policy Analyst",
    page_icon="⬡",
    layout="wide",
)


# ── Styles ────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
  .stApp { background: #0a0e1a; color: #e2e8f0; }
  .block-container { padding-top: 1.2rem; }
  h1, h2, h3 { color: #e2e8f0 !important; }

  /* Flowchart node card */
  .node-card {
    border: 2px solid #1e293b;
    background: #0f172a;
    border-radius: 10px;
    padding: 14px 16px;
    margin: 6px 0;
    color: #cbd5e1;
    font-weight: 600;
    text-align: center;
    transition: all 0.25s ease;
  }
  .node-pending   { border-color: #1e293b; color: #64748b; }
  .node-done      { border-color: #047857; color: #d1fae5;
                    background: linear-gradient(180deg, #022c22, #0f172a); }
  .node-active    {
    border-color: #00d4ff;
    color: #e0f2fe;
    background: linear-gradient(180deg, #082f49, #0f172a);
    box-shadow: 0 0 0 0 rgba(0,212,255,0.7);
    animation: pulseBorder 1.4s ease-in-out infinite;
  }
  @keyframes pulseBorder {
    0%   { box-shadow: 0 0 0 0 rgba(0,212,255,0.55); border-color: #00d4ff; }
    50%  { box-shadow: 0 0 0 10px rgba(0,212,255,0); border-color: #38bdf8; }
    100% { box-shadow: 0 0 0 0 rgba(0,212,255,0); border-color: #00d4ff; }
  }
  .node-arrow {
    text-align: center;
    color: #475569;
    font-size: 16px;
    margin: -2px 0;
  }

  /* LaTeX viewer */
  .latex-viewer textarea {
    background: #0f172a !important;
    color: #cbd5e1 !important;
    font-family: 'JetBrains Mono', 'SF Mono', Menlo, monospace !important;
    font-size: 12px !important;
  }

  /* Big primary download */
  div[data-testid="stDownloadButton"] button {
    background: #00d4ff !important;
    color: #0a0e1a !important;
    border: 0 !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    padding: 0.6rem 1rem !important;
  }
  div[data-testid="stDownloadButton"] button:disabled {
    background: #1e293b !important;
    color: #475569 !important;
  }
</style>
""",
    unsafe_allow_html=True,
)


# ── Header ────────────────────────────────────────────────────────────

st.markdown("## ⬡ InsureIQ — AI Insurance Policy Analyst")
st.caption(
    "Local multi-agent pipeline with deterministic validator · LaTeX-rendered cited report"
)


# ── Session-state init ────────────────────────────────────────────────

def _init_state():
    defaults = {
        "completed_nodes": [],
        "active_node": None,
        "pipeline_running": False,
        "pipeline_done": False,
        "latex_source": "",
        "pdf_bytes": b"",
        "final_report": None,
        "error_msg": "",
        "events_queue": None,
        "worker_thread": None,
        "tmp_pdf_path": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ── Pipeline worker ───────────────────────────────────────────────────

def _run_pipeline(initial_state: dict, events: "queue.Queue"):
    """Run the LangGraph pipeline and push events to the queue."""
    try:
        last_state = initial_state
        for event in pipeline.stream(initial_state):
            for node_name, node_state in event.items():
                events.put({"type": "node_done", "node": node_name})
                last_state = node_state
        events.put({"type": "complete",
                    "latex": last_state.get("latex_source", ""),
                    "pdf": last_state.get("pdf_bytes", b""),
                    "report": last_state.get("final_report", {})})
    except Exception as e:
        events.put({"type": "error", "message": str(e)})


def _drain_events():
    """Pull any queued events into st.session_state."""
    q = st.session_state.events_queue
    if q is None:
        return False
    changed = False
    while True:
        try:
            ev = q.get_nowait()
        except queue.Empty:
            break
        changed = True
        if ev["type"] == "node_done":
            node = ev["node"]
            if node not in st.session_state.completed_nodes:
                st.session_state.completed_nodes.append(node)
            # Active node = next one in order, or None if done
            try:
                idx = NODE_ORDER.index(node)
                st.session_state.active_node = (
                    NODE_ORDER[idx + 1] if idx + 1 < len(NODE_ORDER) else None
                )
            except ValueError:
                pass
        elif ev["type"] == "complete":
            st.session_state.latex_source = ev["latex"]
            st.session_state.pdf_bytes = ev["pdf"]
            st.session_state.final_report = ev["report"]
            st.session_state.pipeline_done = True
            st.session_state.pipeline_running = False
            st.session_state.active_node = None
        elif ev["type"] == "error":
            st.session_state.error_msg = ev["message"]
            st.session_state.pipeline_running = False
            st.session_state.active_node = None
    return changed


# ── Three-column layout ───────────────────────────────────────────────

col_left, col_mid, col_right = st.columns([1, 1, 2], gap="large")


# === LEFT: upload ====================================================
with col_left:
    st.markdown("### 1 · Upload")
    uploaded_file = st.file_uploader(
        "Insurance Policy PDF",
        type=["pdf"],
        help="Stays local — no external LLM calls",
    )

    if uploaded_file:
        st.caption(f"📄 {uploaded_file.name} — {uploaded_file.size // 1024} KB")

    analyze_disabled = (
        uploaded_file is None or st.session_state.pipeline_running
    )

    if st.button("🔍 Analyze Policy",
                 type="primary",
                 use_container_width=True,
                 disabled=analyze_disabled):
        # Reset run state
        st.session_state.completed_nodes = []
        st.session_state.active_node = NODE_ORDER[0]
        st.session_state.pipeline_running = True
        st.session_state.pipeline_done = False
        st.session_state.latex_source = ""
        st.session_state.pdf_bytes = b""
        st.session_state.final_report = None
        st.session_state.error_msg = ""

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            st.session_state.tmp_pdf_path = tmp.name

        initial_state = {
            "session_id": str(uuid.uuid4()),
            "pdf_path": st.session_state.tmp_pdf_path,
            "ocr_text": [],
            "chunks": [],
            "chunk_index": {},
            "insurer_name": "",
            "external_research": {},
            "company_profile": {},
            "section_analyses": {},
            "validated_sections": {},
            "validation_report": {},
            "final_report": {},
            "latex_source": "",
            "pdf_bytes": b"",
            "citations": [],
            "error": None,
            "status": "starting",
            "active_node": "ocr",
        }

        st.session_state.events_queue = queue.Queue()
        st.session_state.worker_thread = threading.Thread(
            target=_run_pipeline,
            args=(initial_state, st.session_state.events_queue),
            daemon=True,
        )
        st.session_state.worker_thread.start()

    if st.session_state.error_msg:
        st.error(st.session_state.error_msg)

    if st.session_state.pipeline_done:
        st.success("Analysis complete.")
        report = st.session_state.final_report or {}
        headline = report.get("headline", {}) if isinstance(report, dict) else {}
        if headline:
            st.markdown(f"**Insurer:** {headline.get('insurer_name', '—')}")
            st.markdown(f"**Rating:** {headline.get('overall_rating', '—')}/100")
            st.markdown(f"**Risk:** {headline.get('risk_level', '—')}")
            st.markdown(f"**Verdict:** {headline.get('recommendation', '—')}")
        vr = (report or {}).get("validation_report", {})
        counts = vr.get("counts", {})
        if counts:
            st.caption(
                f"Validator: {counts.get('trusted', 0)} TRUSTED · "
                f"{counts.get('review', 0)} NEEDS REVIEW · "
                f"{counts.get('dropped', 0)} dropped"
            )


# === MIDDLE: flowchart ===============================================
def _node_classes(node: str) -> str:
    if node == st.session_state.active_node and st.session_state.pipeline_running:
        return "node-card node-active"
    if node in st.session_state.completed_nodes:
        return "node-card node-done"
    return "node-card node-pending"


with col_mid:
    st.markdown("### 2 · Agent Pipeline")
    chart_html = []
    for i, node in enumerate(NODE_ORDER):
        cls = _node_classes(node)
        label = NODE_LABELS[node]
        chart_html.append(f'<div class="{cls}">{label}</div>')
        if i < len(NODE_ORDER) - 1:
            chart_html.append('<div class="node-arrow">▼</div>')
    st.markdown("\n".join(chart_html), unsafe_allow_html=True)

    if st.session_state.pipeline_running:
        st.caption("Running… the active agent has a glowing border.")
    elif st.session_state.pipeline_done:
        st.caption("All agents complete.")
    else:
        st.caption("Upload a PDF and press Analyze to start.")


# === RIGHT: LaTeX viewer + PDF download ==============================
with col_right:
    st.markdown("### 3 · Compiled LaTeX Report")
    latex_src = st.session_state.latex_source or (
        "% The LaTeX source for your validated report will appear here\n"
        "% after the pipeline finishes."
    )
    st.markdown('<div class="latex-viewer">', unsafe_allow_html=True)
    st.text_area(
        label="LaTeX source",
        value=latex_src,
        height=560,
        disabled=True,
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.download_button(
        label="⬇️ Download Report PDF",
        data=st.session_state.pdf_bytes or b"",
        file_name="insureiq_report.pdf",
        mime="application/pdf",
        disabled=not st.session_state.pdf_bytes,
        use_container_width=True,
    )


# ── Live refresh while pipeline is running ────────────────────────────

if st.session_state.pipeline_running:
    _drain_events()
    time.sleep(1.0)
    # Cleanup temp pdf if pipeline finished
    if st.session_state.pipeline_done and st.session_state.tmp_pdf_path:
        try:
            os.unlink(st.session_state.tmp_pdf_path)
        except OSError:
            pass
        st.session_state.tmp_pdf_path = None
    st.rerun()
else:
    # One last drain to catch the final 'complete' event if the rerun beat the queue
    if _drain_events():
        st.rerun()
