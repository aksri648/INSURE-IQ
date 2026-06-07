import base64
import json

import fitz  # PyMuPDF
import ollama

from agents.state import PolicyState
from utils.model_config import get as get_model


OCR_SYSTEM_PROMPT = """You are a precise insurance document digitizer.
Extract ALL text from this policy page. Preserve:
- Section headings and hierarchy
- Clause numbers (e.g. 4.2.1)
- Table structures as markdown tables
- Bold/emphasized terms

Return ONLY valid JSON:
{
  "section": "<section title or empty string>",
  "text": "<full verbatim text>",
  "tables": ["<table as markdown>"],
  "clause_numbers": ["4.1", "4.2"]
}
Do NOT summarize. Do NOT paraphrase."""


def page_to_base64(doc, page_num: int) -> str:
    page = doc[page_num]
    mat = fitz.Matrix(2, 2)  # 2x zoom for better OCR
    pix = page.get_pixmap(matrix=mat)
    return base64.b64encode(pix.tobytes("png")).decode()


def ocr_agent_node(state: PolicyState) -> PolicyState:
    ocr_model = get_model("OCR_MODEL", "llava:7b")
    print(f"[OCR Agent] Loading {ocr_model}...")

    doc = fitz.open(state["pdf_path"])
    total_pages = len(doc)
    ocr_results = []

    for i in range(total_pages):
        print(f"[OCR Agent] Processing page {i + 1}/{total_pages}")
        img_b64 = page_to_base64(doc, i)

        try:
            response = ollama.chat(
                model=ocr_model,
                messages=[{
                    "role": "user",
                    "content": OCR_SYSTEM_PROMPT,
                    "images": [img_b64],
                }],
            )
            raw = response["message"]["content"]
            raw = raw.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(raw)
        except Exception as e:
            parsed = {"section": "",
                      "text": f"OCR Error page {i + 1}: {e}",
                      "tables": [],
                      "clause_numbers": []}

        ocr_results.append({"page": i + 1, **parsed})

    doc.close()

    # Offload OCR model to free VRAM for the analyst model
    print(f"[OCR Agent] Offloading {ocr_model} from VRAM...")
    try:
        ollama.generate(model=ocr_model, prompt="", keep_alive=0)
    except Exception:
        pass

    print(f"[OCR Agent] Done. Extracted {total_pages} pages.")
    return {**state, "ocr_text": ocr_results, "status": "ocr_complete"}
