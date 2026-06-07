# ⬡ InsureIQ — AI Insurance Policy Analyst

**Upload an insurance policy PDF. Get a fully cited, hallucination-proof consumer report as a typeset PDF.**

100 % local LLM inference. A deterministic validator agent confirms every claim against the policy text, so nothing the LLM says reaches the report unless a real run of characters in the policy backs it up.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aksri648/INSURE-IQ/blob/main/insureiq_colab.ipynb)

📘 **Deep dive:** [`project-documentation.md`](./project-documentation.md) — architecture, every agent, prompts, validator algorithm, LaTeX schema, ops.
🏗 **Architecture diagram:** [`insureiq_architecture_diagram.md`](./insureiq_architecture_diagram.md)
☁️ **Colab + Cloudflare deployment:** [`insureiq_colab_deployment.md`](./insureiq_colab_deployment.md)

---

## What it does

- Runs **7 LangGraph agents** sequentially on your machine:
  `OCR → RAG → Web Research → Analyst → Company Profile → Validator → Compiler`
- **Validator agent** does a deterministic substring check on every analyst claim against the policy text. Tags each finding **TRUSTED** ✅ or **NEEDS HUMAN REVIEW** ⚠️. Drops anything fabricated.
- Compiles the report as **LaTeX → PDF** with `tectonic`. Inline coloured trust tags + verbatim quotes + page references on every finding.
- A **Tavily-powered Company Profile** agent appends an insurer dossier: claim settlement ratio, recent disputes, customer reviews, ratings, market share, credibility, trust score, source URLs.
- **Streamlit UI**: left = upload, middle = live agent flowchart (active node glows cyan), right = LaTeX viewer + single **⬇ Download Report PDF** button.

---

## Quick start — Google Colab (recommended)

1. Click **Open in Colab** above.
2. `Runtime → Change runtime type → T4 GPU`.
3. Open the **🔑 Secrets** panel in the left sidebar and add `TAVILY_API_KEY` with notebook access (get a free key at <https://tavily.com>).
4. `Runtime → Run all`.

After ~15–25 min of setup, Cell 9 prints a public `https://*.trycloudflare.com` URL. Open it, drop a PDF, watch the flowchart light up, download the report.

Full guide: [`insureiq_colab_deployment.md`](./insureiq_colab_deployment.md).

---

## Quick start — local

Requires Linux, Python 3.10+, NVIDIA GPU, `curl`, `git`.

```bash
git clone https://github.com/aksri648/INSURE-IQ.git
cd INSURE-IQ
./setup.sh                              # tectonic + Ollama + models + venv
echo "TAVILY_API_KEY=tvly-..." > .env   # optional (else profile is skipped)
./run.sh                                # http://localhost:8501
```

---

## How the validator stops hallucinations

The analyst is constrained to emit **only structured findings**, each carrying a `verbatim_quote` field — an exact substring of the cited chunk.

The validator is **pure Python, no LLM**:

```python
if normalize(quote) in normalize(chunk.text):
    → TRUSTED   if page matches
    → NEEDS_HUMAN_REVIEW  if page mismatch
elif word_overlap_ratio(quote, chunk.text) >= 0.70:
    → NEEDS_HUMAN_REVIEW  (paraphrased)
else:
    → DROP  (not in the policy)
```

A pure-string verifier cannot be fluent-talked into approving an invented clause. The PDF includes a **"How to Read This Report"** legend on the first page explaining the tags to the consumer.

Details + the full per-section schema are in [`project-documentation.md`](./project-documentation.md).

---

## Repository layout

```
INSURE-IQ/
├── agents/                       # 7 LangGraph nodes
│   ├── state.py                  #   PolicyState TypedDict
│   ├── ocr_agent.py              #   LLaVA → JSON
│   ├── rag_agent.py              #   chunk + embed + chunk_index
│   ├── web_research_agent.py     #   light Tavily side context
│   ├── analyst_agent.py          #   verifiable findings only
│   ├── company_profile_agent.py  #   Tavily ×7 facets + LLM synth
│   ├── validator_agent.py        # ★ deterministic verification
│   └── compiler_agent.py         #   LaTeX builder
├── utils/
│   ├── model_config.py           # VRAM-aware model picker
│   └── pdf_builder.py            # tectonic → pdflatex → reportlab
├── graph.py                      # LangGraph wiring
├── app.py                        # Streamlit (3 columns + flowchart)
├── insureiq_colab.ipynb          # Colab + Cloudflare Tunnel notebook
├── setup.sh / run.sh             # local setup + launch
├── requirements.txt
├── project-documentation.md      # ← full reference
├── insureiq_architecture_diagram.md
├── insureiq_colab_deployment.md
└── README.md                     # ← you are here
```

---

## Configuration

| Env var | Purpose | Default |
|---|---|---|
| `TAVILY_API_KEY` | Enables Web Research + Company Profile (on Colab, read from Secrets) | unset → both skip |
| `OCR_MODEL` / `ANALYST_MODEL` / `EMBED_MODEL` | Override auto-picked models | auto from VRAM |
| `INSUREIQ_MODEL_CONFIG` | Path to resolved-model config | `/tmp/model_config.env` |

---

## Privacy

- PDFs never leave the host. All inference is local Ollama.
- ChromaDB is in-memory and per-session.
- Tavily receives only the insurer name + canned query templates, never policy text.
- On Colab, `TAVILY_API_KEY` is read from Colab Secrets — not hard-coded in the notebook.

---

## License

See repo for licensing details.

## Links

- **Repo:** <https://github.com/aksri648/INSURE-IQ>
- **Project documentation:** [`project-documentation.md`](./project-documentation.md)
- **Architecture diagram:** [`insureiq_architecture_diagram.md`](./insureiq_architecture_diagram.md)
- **Colab deployment guide:** [`insureiq_colab_deployment.md`](./insureiq_colab_deployment.md)
- **Open in Colab:** <https://colab.research.google.com/github/aksri648/INSURE-IQ/blob/main/insureiq_colab.ipynb>
