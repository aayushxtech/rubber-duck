from app.graph.state import GraphState


def route_node(state: GraphState) -> GraphState:
    if state.analysis is None:
        raise ValueError("Analysis missing in route node")

    # 1. Search only if needed AND not already done
    if state.analysis.needs_search and state.evidence is None:
        state.route = "search"
        return state

    # 2. Socratic only if uncertainties remain
    if state.analysis.uncertainties:
        state.route = "socratic"
        return state

    # 3. Otherwise, converge
    state.route = "final"
    return state
