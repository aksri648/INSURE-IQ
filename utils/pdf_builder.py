"""Compile LaTeX source to PDF bytes.

Strategy:
  1. Prefer `tectonic` (self-contained, no system TeX install needed).
  2. Fall back to `pdflatex` if available.
  3. Last-resort fallback: render a plain-text "this is not the LaTeX PDF,
     install tectonic" notice via reportlab so the Download button never
     produces an empty file.
"""

import os
import shutil
import subprocess
import tempfile


def _try_tectonic(tex_path: str, workdir: str) -> str | None:
    if not shutil.which("tectonic"):
        return None
    try:
        subprocess.run(
            ["tectonic", "--keep-logs", "--outdir", workdir, tex_path],
            cwd=workdir,
            check=True,
            capture_output=True,
            timeout=180,
        )
    except subprocess.CalledProcessError as e:
        print("[pdf_builder] tectonic failed:\n", e.stderr.decode("utf-8", "ignore")[-2000:])
        return None
    except Exception as e:
        print(f"[pdf_builder] tectonic error: {e}")
        return None
    pdf = os.path.splitext(tex_path)[0] + ".pdf"
    return pdf if os.path.exists(pdf) else None


def _try_pdflatex(tex_path: str, workdir: str) -> str | None:
    if not shutil.which("pdflatex"):
        return None
    try:
        # Run twice for accurate page refs / TOC if present
        for _ in range(2):
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode",
                 "-halt-on-error", "-output-directory", workdir, tex_path],
                cwd=workdir,
                check=True,
                capture_output=True,
                timeout=180,
            )
    except subprocess.CalledProcessError as e:
        print("[pdf_builder] pdflatex failed:\n", e.stdout.decode("utf-8", "ignore")[-2000:])
        return None
    except Exception as e:
        print(f"[pdf_builder] pdflatex error: {e}")
        return None
    pdf = os.path.splitext(tex_path)[0] + ".pdf"
    return pdf if os.path.exists(pdf) else None


def _fallback_reportlab_pdf(tex_source: str) -> bytes:
    """Last-resort: produce a notice PDF so the UI button still works."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
    except Exception:
        return b""

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        path = f.name
    c = canvas.Canvas(path, pagesize=A4)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, 800, "InsureIQ Report — LaTeX compiler unavailable")
    c.setFont("Helvetica", 10)
    msg = [
        "tectonic and pdflatex were both missing on this host.",
        "Install one of them to render the formatted report:",
        "  curl -fsSL https://drop-sh.fullyjustified.net | sh",
        "  # or:  apt install texlive-latex-base texlive-latex-extra",
        "",
        "The LaTeX source for this report follows on the next pages.",
    ]
    y = 770
    for line in msg:
        c.drawString(72, y, line)
        y -= 14
    c.showPage()

    # Dump the source verbatim (truncated per page)
    c.setFont("Courier", 8)
    for chunk in [tex_source[i:i + 4000] for i in range(0, len(tex_source), 4000)]:
        y = 820
        for line in chunk.splitlines():
            c.drawString(36, y, line[:160])
            y -= 9
            if y < 36:
                c.showPage()
                c.setFont("Courier", 8)
                y = 820
        c.showPage()
        c.setFont("Courier", 8)
    c.save()
    with open(path, "rb") as f:
        data = f.read()
    try:
        os.unlink(path)
    except OSError:
        pass
    return data


def compile_latex_to_pdf(tex_source: str) -> bytes:
    """Return PDF bytes (never None / empty for any reasonable input)."""
    workdir = tempfile.mkdtemp(prefix="insureiq_tex_")
    tex_path = os.path.join(workdir, "report.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex_source)

    pdf_path = _try_tectonic(tex_path, workdir) or _try_pdflatex(tex_path, workdir)
    if pdf_path:
        with open(pdf_path, "rb") as f:
            return f.read()

    print("[pdf_builder] No LaTeX engine available — falling back to reportlab notice.")
    return _fallback_reportlab_pdf(tex_source)
