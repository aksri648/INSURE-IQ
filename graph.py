from langgraph.graph import StateGraph, END

from agents.analyst_agent import analyst_agent_node
from agents.company_profile_agent import company_profile_node
from agents.compiler_agent import report_compiler_node
from agents.ocr_agent import ocr_agent_node
from agents.rag_agent import embed_and_store_node
from agents.state import PolicyState
from agents.web_research_agent import web_research_node


def build_graph():
    g = StateGraph(PolicyState)

    g.add_node("ocr",             ocr_agent_node)
    g.add_node("embed_store",     embed_and_store_node)
    g.add_node("web_research",    web_research_node)
    g.add_node("analyst",         analyst_agent_node)
    g.add_node("company_profile", company_profile_node)
    g.add_node("compiler",        report_compiler_node)

    g.set_entry_point("ocr")
    g.add_edge("ocr",             "embed_store")
    g.add_edge("embed_store",     "web_research")
    g.add_edge("web_research",    "analyst")
    g.add_edge("analyst",         "company_profile")
    g.add_edge("company_profile", "compiler")
    g.add_edge("compiler",        END)

    return g.compile()


pipeline = build_graph()
