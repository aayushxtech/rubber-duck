from langgraph.graph import END, StateGraph

from app.graph.nodes.ananlyze import analyze_node
from app.graph.nodes.final_respond_node import final_response_node
from app.graph.nodes.route import route_node
from app.graph.nodes.search_node import search_node
from app.graph.nodes.socratic import socratic_node
from app.graph.state import GraphState


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("analyze", analyze_node)
    graph.add_node("route", route_node)
    graph.add_node("socratic", socratic_node)
    graph.add_node("search", search_node)
    graph.add_node("final", final_response_node)

    graph.set_entry_point("analyze")

    graph.add_edge("analyze", "route")

    graph.add_conditional_edges(
        "route",
        lambda state: state.route,
        {
            "socratic": "socratic",
            "search": "search",
            "final": "final",
        },
    )

    graph.add_edge("socratic", END)

    graph.add_edge("search", "analyze")

    graph.add_edge("final", END)

    return graph.compile()
