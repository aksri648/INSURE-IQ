from typing import TypedDict, Optional


class PolicyState(TypedDict):
    session_id: str
    pdf_path: str
    ocr_text: list          # [{page, text, tables}]
    chunks: list            # [{chunk_id, text, metadata}]
    insurer_name: str
    external_research: dict
    company_profile: dict   # Tavily-powered insurer profile
    section_analyses: dict
    final_report: dict
    report_markdown: str
    citations: list
    error: Optional[str]
    status: str             # current pipeline stage
