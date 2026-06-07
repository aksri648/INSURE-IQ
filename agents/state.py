from typing import TypedDict, Optional


class PolicyState(TypedDict):
    session_id: str
    pdf_path: str
    ocr_text: list           # [{page, text, tables}]
    chunks: list             # [{chunk_id, text, metadata}]
    chunk_index: dict        # chunk_id -> {text, page, section} (verbatim, for validator)
    insurer_name: str
    external_research: dict
    company_profile: dict
    section_analyses: dict   # raw analyst output
    validated_sections: dict # post-validation, with trust tags on each finding
    validation_report: dict  # counts + per-finding verdicts
    final_report: dict       # structured json wrapper
    latex_source: str        # full LaTeX document
    pdf_bytes: bytes         # compiled PDF bytes
    citations: list
    error: Optional[str]
    status: str
    active_node: str         # for UI flowchart highlighting
