# InsureIQ — Google Colab + Cloudflare Tunnel Deployment Guide
### v2.0 · Repo-cloned · 6-Node Pipeline · 18-Section Report · Tavily Company Profile

> Run the full multi-agent InsureIQ pipeline on Colab's free/Pro GPU and expose Streamlit publicly via Cloudflare Tunnel — no server, no domain, no cost.
>
> This guide is paired with the **ready-to-run notebook** in the repo: [`insureiq_colab.ipynb`](./insureiq_colab.ipynb).
> Open it directly in Colab: <https://colab.research.google.com/github/aksri648/INSURE-IQ/blob/main/insureiq_colab.ipynb>

---

## ARCHITECTURE ON COLAB

```
┌──────────────────────────────────────────────────────────────────┐
│              GOOGLE COLAB  (T4 / A100 GPU)                       │
│                                                                  │
│   git clone github.com/aksri648/INSURE-IQ  →  /content/INSURE-IQ │
│                                                                  │
│  ┌─────────────┐   ┌──────────────┐   ┌────────────────────┐    │
│  │   Ollama    │   │   ChromaDB   │   │   Streamlit  :8501 │    │
│  │   :11434    │   │  (in-mem)    │   │  app.py + graph.py │    │
│  │ llava +     │   │  per-session │   │  6 LangGraph nodes │    │
│  │ deepseek-r1 │   │  collection  │   │                    │    │
│  │ nomic-embed │   │              │   │                    │    │
│  └─────────────┘   └──────────────┘   └─────────┬──────────┘    │
│                                                 │                │
└─────────────────────────────────────────────────┼────────────────┘
                                                  │
                                       ┌──────────▼──────────┐
                                       │  cloudflared tunnel │
                                       │  (trycloudflare.com)│
                                       └──────────┬──────────┘
                                                  │ Public HTTPS
                                                  ▼
                                          Anyone on internet
```

The Colab runtime hosts everything; cloudflared exposes the local Streamlit port as a public `https://<random>.trycloudflare.com` URL.

---

## COLAB RUNTIME RECOMMENDATIONS

| Plan  | GPU  | VRAM   | Suitability |
|-------|------|--------|-------------|
| Free  | T4   | 15 GB  | ✅ Works — uses 7B models |
| Pro   | A100 | 40 GB  | ✅ Best — uses 14B+ models |
| Pro+  | A100 | 80 GB  | ✅ Overkill but smooth |

`utils/model_config.py` auto-picks model sizes from detected VRAM:

| VRAM   | OCR Model    | Analyst / Synthesis Model | Embedding |
|--------|--------------|---------------------------|-----------|
| ≤ 35 GB | `llava:7b`   | `deepseek-r1:7b`         | `nomic-embed-text` |
| > 35 GB | `llava:13b`  | `deepseek-r1:14b`        | `nomic-embed-text` |

Override with env vars `OCR_MODEL`, `ANALYST_MODEL`, `EMBED_MODEL` if needed.

---

## NOTEBOOK CELL SUMMARY

The notebook (`insureiq_colab.ipynb`) is 8 cells. Run top-to-bottom; the last one prints the public URL.

| Cell | What it does |
|------|--------------|
| 1 | GPU + RAM check (`nvidia-smi`, `free -h`). |
| 2 | `git clone --depth 1 https://github.com/aksri648/INSURE-IQ` into `/content/INSURE-IQ`. |
| 3 | `apt` libs (`libgl1`, `libglib2.0-0`) + `pip install -r /content/INSURE-IQ/requirements.txt`. |
| 4 | Install Ollama via `curl -fsSL https://ollama.com/install.sh \| sh`. Start `ollama serve` in the background. Wait until `:11434` responds. |
| 5 | Detect VRAM. Pull `nomic-embed-text` + (`llava:13b`/`llava:7b`) + (`deepseek-r1:14b`/`deepseek-r1:7b`). Write the resolved names to `/tmp/model_config.env`. |
| 6 | Set `TAVILY_API_KEY` in `os.environ` and persist to `/content/INSURE-IQ/.env`. **Leave blank to skip Web Research + Company Profile agents.** |
| 7 | Install `cloudflared` binary from GitHub releases. |
| 8 | Start Streamlit (background subprocess) → start `cloudflared tunnel --url http://localhost:8501` → parse the printed `https://*.trycloudflare.com` URL → keep the tunnel cell running. |

Optional keep-alive JS cell can be inserted before Cell 8 to fight Colab's idle timer on free tier.

---

## QUICK START (3 steps)

1. Open <https://colab.research.google.com/github/aksri648/INSURE-IQ/blob/main/insureiq_colab.ipynb>
2. `Runtime → Change runtime type → T4 GPU` (or A100).
3. `Runtime → Run all`.

Wait for Cell 8 to print:

```
================================================================
  🌐 PUBLIC URL: https://<random>.trycloudflare.com
================================================================
```

Open the URL → upload a policy PDF → click **Analyze Policy**.

Total cold-start ≈ 15–25 min (mostly model pulls). Each policy analysis runs ~5–15 min depending on PDF length and GPU.

---

## WHAT GETS RUN ON COLAB

The notebook clones the repo and runs the project unchanged. Pipeline:

```
ocr  →  embed_store  →  web_research  →  analyst  →  company_profile  →  compiler  →  END
```

- **OCR** uses `llava` with `keep_alive=0` to free VRAM before the analyst loads.
- **Analyst** runs **18 grounded section prompts** with strict JSON schemas, producing the consumer-friendly report template.
- **Company Profile** issues **7 Tavily `search_depth="advanced"` queries** about the insurer (overview, claim settlement ratio, recent disputes, customer reviews, ratings, market share, credibility) and synthesizes them through DeepSeek R1 into a structured profile + trust score.
- **Compiler** emits Markdown + structured JSON. Streamlit renders both with download buttons.

---

## TAVILY KEY HANDLING

Both Tavily-dependent agents (`web_research_agent.py`, `company_profile_agent.py`) check `os.getenv("TAVILY_API_KEY")` at runtime.

| Key present? | Effect |
|---|---|
| ✅ | Web Research runs 2 queries · Company Profile runs 7 queries + LLM synthesis. |
| ❌ | Both agents short-circuit cleanly with `{"available": false, "reason": "TAVILY_API_KEY not configured"}`. Pipeline still completes. |

Get a free key at <https://tavily.com>. Paste it into **Cell 6** of the notebook.

```python
# Cell 6
import os
os.environ["TAVILY_API_KEY"] = "tvly-..."
with open("/content/INSURE-IQ/.env", "w") as f:
    f.write(f"TAVILY_API_KEY={os.environ['TAVILY_API_KEY']}\n")
```

`/content/INSURE-IQ/app.py` calls `load_dotenv()` so the subprocess Streamlit child picks up the same key.

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

If you want a stable subdomain across restarts:

1. Create a free Cloudflare account at cloudflare.com.
2. Run `cloudflared login` in a Colab cell.
3. Create a named tunnel: `cloudflared tunnel create insureiq`.
4. Run: `cloudflared tunnel run --url http://localhost:8501 insureiq`.

---

## COLAB LIMITATIONS & WORKAROUNDS

| Limitation | Impact | Workaround |
|------------|--------|------------|
| Session timeout (free: ~90 min idle) | Pipeline interrupted | Use Colab Pro · enable the keep-alive JS cell |
| VRAM shared with system | OOM on big PDFs / large models | Use 7B models on T4 (default); sequential offload between OCR and analyst |
| No persistent storage | ChromaDB lost on restart | Re-upload PDF — embeddings rebuild quickly |
| Tunnel URL changes on restart | Users lose link | Restart Cell 8 and reshare URL · use a named tunnel for stability |
| Network egress limits | Slow model pull | Pre-pull models at session start (Cell 5) |
| Tavily rate limits / no key | Web Research + Company Profile silently skip | Provide a key in Cell 6 |

### Keep-Alive Cell (insert before Cell 8 if needed)

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
| Cell 4 hangs on "Waiting for Ollama..." | install script failed | `tail /tmp/ollama.log` and re-run cell |
| Cell 5 model pull stalls | network egress throttled | re-run cell · Colab will resume the partial pull |
| Streamlit returns 502 via tunnel | app still booting | wait 10–15 s and refresh; `tail /tmp/streamlit.log` to inspect |
| Empty Company Profile tab | `TAVILY_API_KEY` not set | re-run Cell 6 with a valid key, then restart Cell 8 |
| OOM mid-analysis | analyst model too big | force smaller: `os.environ["ANALYST_MODEL"]="deepseek-r1:7b"` before Cell 8 |

---

## RUNNING LOCALLY INSTEAD

The repo also ships `setup.sh` / `run.sh` for local Linux + GPU:

```bash
git clone https://github.com/aksri648/INSURE-IQ.git
cd INSURE-IQ
./setup.sh              # installs Ollama + venv + pulls models
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
Cell 4  →  Install + start Ollama server
Cell 5  →  Pull models (auto-sized to VRAM)
Cell 6  →  Set TAVILY_API_KEY (optional but recommended)
Cell 7  →  Install cloudflared
Cell 8  →  Launch Streamlit + Cloudflare tunnel → public URL
```

Total setup time: ~15–25 min (mostly model download)
Analysis time per policy: ~5–15 min depending on PDF length and GPU

---

*InsureIQ · Colab + Cloudflare Deployment Guide · v2.0*
*Repo: https://github.com/aksri648/INSURE-IQ*
