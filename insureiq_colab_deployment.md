# InsureIQ — Google Colab + Cloudflare Tunnel Deployment Guide
### v3.0 · Repo-cloned · 7-Node Pipeline · Deterministic Validator · LaTeX → PDF

> Run the full hallucination-proof InsureIQ pipeline on Colab's free/Pro GPU and expose the Streamlit app publicly via Cloudflare Tunnel — no server, no domain, no cost.
>
> Paired notebook: [`insureiq_colab.ipynb`](./insureiq_colab.ipynb).
> Open in Colab: <https://colab.research.google.com/github/aksri648/INSURE-IQ/blob/main/insureiq_colab.ipynb>

---

## ARCHITECTURE ON COLAB

```
┌────────────────────────────────────────────────────────────────────────┐
│              GOOGLE COLAB  (T4 / A100 GPU)                             │
│                                                                        │
│   git clone github.com/aksri648/INSURE-IQ  →  /content/INSURE-IQ      │
│                                                                        │
│  ┌─────────────┐ ┌──────────────┐ ┌──────────────┐ ┌───────────────┐  │
│  │   Ollama    │ │   ChromaDB   │ │   Tectonic   │ │   Streamlit   │  │
│  │   :11434    │ │  (in-mem)    │ │  LaTeX→PDF   │ │   :8501       │  │
│  │ llava +     │ │  per-session │ │  engine      │ │  3-column UI  │  │
│  │ deepseek-r1 │ │              │ │              │ │  + flowchart  │  │
│  │ nomic-embed │ │ chunk_index  │ │              │ │  + PDF dl     │  │
│  └─────────────┘ └──────────────┘ └──────────────┘ └───────┬───────┘  │
│                                                            │           │
└────────────────────────────────────────────────────────────┼───────────┘
                                                             │
                                                  ┌──────────▼──────────┐
                                                  │  cloudflared tunnel │
                                                  │  (trycloudflare.com)│
                                                  └──────────┬──────────┘
                                                             │ Public HTTPS
                                                             ▼
                                                     Anyone on internet
```

---

## COLAB RUNTIME RECOMMENDATIONS

| Plan  | GPU  | VRAM   | Suitability |
|-------|------|--------|-------------|
| Free  | T4   | 15 GB  | ✅ Works — uses 7B models |
| Pro   | A100 | 40 GB  | ✅ Best — uses 14B models |
| Pro+  | A100 | 80 GB  | ✅ Overkill but smooth |

`utils/model_config.py` auto-picks model sizes from detected VRAM. Override with env vars `OCR_MODEL`, `ANALYST_MODEL`, `EMBED_MODEL` if needed.

---

## NOTEBOOK CELL SUMMARY

The notebook (`insureiq_colab.ipynb`) is 9 cells. Run top to bottom; the last one prints the public URL.

| Cell | What it does |
|------|--------------|
| 1 | GPU + RAM check (`nvidia-smi`, `free -h`). |
| 2 | `git clone --depth 1 https://github.com/aksri648/INSURE-IQ` into `/content/INSURE-IQ`. |
| 3 | `apt` libs (`libgl1`, `libglib2.0-0`) + `pip install -r requirements.txt`. |
| 4 | **Install `tectonic`** (self-contained LaTeX engine) — required to render the final PDF. |
| 5 | Install Ollama via `curl https://ollama.com/install.sh \| sh`. Start `ollama serve` in the background. Wait until `:11434` responds. |
| 6 | Detect VRAM. Pull `nomic-embed-text` + (`llava:13b`/`llava:7b`) + (`deepseek-r1:14b`/`deepseek-r1:7b`). Write resolved names to `/tmp/model_config.env`. |
| 7 | Set `TAVILY_API_KEY` in `os.environ` and persist to `/content/INSURE-IQ/.env`. Required for the Company Profile agent. |
| 8 | Install `cloudflared` binary from GitHub releases. |
| 9 | Start Streamlit in background → start `cloudflared tunnel --url http://localhost:8501` → parse the printed `https://*.trycloudflare.com` URL → keep the tunnel cell running. |

Optional keep-alive JS cell can be inserted before Cell 9 to fight Colab's idle timer on free tier.

---

## QUICK START (3 steps)

1. Open <https://colab.research.google.com/github/aksri648/INSURE-IQ/blob/main/insureiq_colab.ipynb>
2. `Runtime → Change runtime type → T4 GPU` (or A100).
3. `Runtime → Run all`.

Wait for Cell 9 to print:

```
================================================================
  🌐 PUBLIC URL: https://<random>.trycloudflare.com
================================================================
```

Open the URL → drop a policy PDF on the left → click **Analyze Policy** → watch the middle flowchart light up → press **⬇️ Download Report PDF** on the right.

Total cold start ≈ 15–25 min (mostly model pulls). Each policy analysis runs ~5–15 min depending on PDF length and GPU.

---

## WHAT GETS RUN ON COLAB

Pipeline (7 nodes):

```
ocr → embed_store → web_research → analyst → company_profile → validator → compiler → END
```

- **OCR** offloads (`keep_alive=0`) before the analyst loads.
- **Analyst** runs 14 grounded section prompts. Each finding **must** include a `verbatim_quote` that is an exact substring of the cited policy chunk.
- **Company Profile** issues 7 Tavily advanced searches about the insurer.
- **Validator** does a deterministic substring check on every finding against the verbatim `chunk_index` (no LLM). Tags TRUSTED / NEEDS_HUMAN_REVIEW; drops anything whose quote isn't in the cited chunk.
- **Compiler** emits LaTeX → tectonic → PDF. The Streamlit UI shows the LaTeX source on the right and the **Download Report PDF** button.

---

## TAVILY KEY HANDLING

Both Tavily-dependent agents check `os.getenv("TAVILY_API_KEY")` at runtime.

| Key present? | Effect |
|---|---|
| ✅ | Web Research runs 2 queries · Company Profile runs 7 queries + LLM synthesis. |
| ❌ | Both agents short-circuit cleanly. Pipeline still completes; Company Profile section in the PDF shows a "not available" note. |

Get a free key at <https://tavily.com>. Paste it into **Cell 7** of the notebook.

```python
# Cell 7
import os
os.environ["TAVILY_API_KEY"] = "tvly-..."
with open("/content/INSURE-IQ/.env", "w") as f:
    f.write(f"TAVILY_API_KEY={os.environ['TAVILY_API_KEY']}\n")
```

`/content/INSURE-IQ/app.py` calls `load_dotenv()` so the Streamlit subprocess picks up the same key.

---

## CLOUDFLARE TUNNEL — HOW IT WORKS

```
Colab Machine                Cloudflare Edge          User's Browser
     │                             │                        │
     │  cloudflared tunnel --url   │                        │
     │ ────────────────────────────▶                        │
     │  registers random subdomain │                        │
     │                             │◀── HTTPS request ──────│
     │◀── forwards to :8501 ───────│                        │
     │                             │                        │
  Streamlit                        │──── response ─────────▶│
```

- **No account required** — uses `trycloudflare.com` free tier.
- **HTTPS by default** — Cloudflare terminates SSL.
- **No port forwarding** — works behind Colab's NAT.
- **URL is ephemeral** — a new random URL is issued each session.

### Optional: Named Tunnel (stable URL)

1. Create a free Cloudflare account.
2. Run `cloudflared login` in a Colab cell.
3. `cloudflared tunnel create insureiq` → `cloudflared tunnel run --url http://localhost:8501 insureiq`.

---

## COLAB LIMITATIONS & WORKAROUNDS

| Limitation | Impact | Workaround |
|------------|--------|------------|
| Session timeout (free: ~90 min idle) | Pipeline interrupted | Use Colab Pro · enable the keep-alive JS cell |
| VRAM shared with system | OOM on big PDFs / large models | Use 7B models on T4 (default); sequential offload between OCR and analyst |
| No persistent storage | ChromaDB lost on restart | Re-upload PDF — rebuild is fast |
| Tunnel URL changes on restart | Users lose link | Restart Cell 9 and reshare URL · use a named tunnel for stability |
| Network egress limits | Slow model pull | Pre-pull models at session start (Cell 6) |
| Tavily rate limits / no key | Company Profile silently skips | Provide a key in Cell 7 |
| tectonic first-run downloads | First PDF compile fetches fonts | Subsequent compiles are fast; cached in `~/.cache/Tectonic/` |

### Keep-Alive Cell (insert before Cell 9 if needed)

```python
import IPython

IPython.display.display(IPython.display.Javascript("""
function ClickConnect(){
    console.log("Keeping Colab alive...");
    const btn = document.querySelector("#top-toolbar > colab-connect-button");
    if (btn && btn.shadowRoot) {
        const inner = btn.shadowRoot.querySelector("#connect");
        if (inner) inner.click();
    }
}
setInterval(ClickConnect, 60000);
"""))
print("✅ Keep-alive activated")
```

---

## TROUBLESHOOTING

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `nvidia-smi` not found | CPU runtime | `Runtime → Change runtime type → T4 GPU` |
| Cell 5 hangs on "Waiting for Ollama..." | install script failed | `tail /tmp/ollama.log` and re-run cell |
| Cell 6 model pull stalls | network egress throttled | re-run cell · Colab will resume the partial pull |
| Streamlit returns 502 via tunnel | app still booting | wait 10–15 s and refresh; `tail /tmp/streamlit.log` to inspect |
| Empty Company Profile section in PDF | `TAVILY_API_KEY` not set | re-run Cell 7 with a valid key, then restart Cell 9 |
| OOM mid-analysis | analyst model too big | force smaller: `os.environ["ANALYST_MODEL"]="deepseek-r1:7b"` before Cell 9 |
| Download PDF says "LaTeX compiler unavailable" | tectonic install failed | re-run Cell 4 · then restart Cell 9 |
| Lots of NEEDS HUMAN REVIEW tags | analyst paraphrased instead of quoting | normal on long noisy PDFs; the validator is conservative by design |

---

## RUNNING LOCALLY INSTEAD

The repo also ships `setup.sh` / `run.sh` for local Linux + GPU:

```bash
git clone https://github.com/aksri648/INSURE-IQ.git
cd INSURE-IQ
./setup.sh              # installs tectonic + Ollama + venv + pulls models
echo "TAVILY_API_KEY=tvly-..." > .env
./run.sh                # http://localhost:8501
```

See [`README.md`](./README.md) for full local instructions.

---

## QUICK REFERENCE — ALL COLAB CELLS IN ORDER

```
Cell 1  →  GPU + RAM check
Cell 2  →  git clone https://github.com/aksri648/INSURE-IQ
Cell 3  →  apt + pip install -r requirements.txt
Cell 4  →  Install tectonic (LaTeX → PDF)
Cell 5  →  Install + start Ollama server
Cell 6  →  Pull models (auto-sized to VRAM)
Cell 7  →  Set TAVILY_API_KEY (recommended)
Cell 8  →  Install cloudflared
Cell 9  →  Launch Streamlit + Cloudflare tunnel → public URL
```

Total setup time: ~15–25 min (mostly model download)
Analysis time per policy: ~5–15 min depending on PDF length and GPU

---

*InsureIQ · Colab + Cloudflare Deployment Guide · v3.0*
*Repo: https://github.com/aksri648/INSURE-IQ*
