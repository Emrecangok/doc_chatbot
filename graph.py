from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from langchain_chroma import Chroma

from state import DocChatState
from nodes import DocChatNodes, route_start, route_after_grade , route_after_tools


def build_graph(vector_store: Chroma):
    nodes = DocChatNodes(vector_store)
    graph = StateGraph(DocChatState)

    graph.add_node("ingest", nodes.ingest_node)
    graph.add_node("agent", nodes.agent_node)
    graph.add_node("run_tools", ToolNode(nodes.tools))          # eski adı: retrieve_docs
    graph.add_node("grade_docs", nodes.grade_docs_node)
    graph.add_node("rewrite", nodes.rewrite_query_node)
    graph.add_node("generate", nodes.generate_node)
    graph.add_node("fallback", nodes.fallback_generate_node)
    graph.add_node("trim", nodes.trim_messages_node)

    graph.add_conditional_edges(START, route_start, {"ingest": "ingest", "agent": "agent"})
    graph.add_edge("ingest", "agent")

    graph.add_conditional_edges("agent", tools_condition, {
        "tools": "run_tools",
        END: "trim",                     # netleştirme sorusu da buradan geçer
    })

    # YENİ: hangi tool çalıştığına göre
    graph.add_conditional_edges("run_tools", route_after_tools, {
        "grade_docs": "grade_docs",
        "agent": "agent",
    })

    graph.add_conditional_edges("grade_docs", route_after_grade, {
        "generate": "generate",
        "rewrite": "rewrite",
        "fallback": "fallback",
    })
    graph.add_edge("rewrite", "run_tools")

    graph.add_edge("generate", "trim")
    graph.add_edge("fallback", "trim")
    graph.add_edge("trim", END)

    return graph.compile(checkpointer=MemorySaver())


if __name__ == "__main__":
    from rag import get_vector_store          

    vector_store = get_vector_store()
    app = build_graph(vector_store)

