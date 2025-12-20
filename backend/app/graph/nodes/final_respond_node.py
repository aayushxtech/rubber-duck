from app.graph.state import GraphState

def final_response_node(state: GraphState) -> GraphState:
    # For MVP: just pass through the last Socratic question
    state.final_answer = state.socratic_question
    return state
