from app.graph.state import GraphState

def route_node(state: GraphState) -> GraphState:
    if state.analysis is None:
        raise ValueError("Analysis data is required to route the node.")
    
    # elif state.analysis.needs_search == True:
    #     state.route = "search_node"
    else:
        state.route = "socratic"
        
    return state