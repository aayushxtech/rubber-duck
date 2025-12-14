from langgraph.graph import StateGraph, END
from app.graph.state import GraphState
from app.graph.nodes.ananlyze import analyze_node


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("analyze", analyze_node)
    graph.set_entry_point("analyze")
    graph.add_edge("analyze", END)
    return graph.compile()
