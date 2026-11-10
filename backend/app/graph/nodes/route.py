from app.graph.state import ExecutionStatus, GraphState


def route_node(state: GraphState) -> GraphState:
    if state.analysis is None:
        raise ValueError("Analysis missing in route node")

    # 1. Enforce deterministic research budget
    # If analysis indicates factual search is required and budget permits, search first
    if (
        state.analysis.needs_search
        and state.internal_research_count < state.max_research_iterations
    ):
        state.route = "search"
        return state

    # 2. Enforce clarification limit
    # If the clarification limit has been reached, route directly to final response
    if state.clarification_count >= state.max_clarification_turns:
        state.execution_status = ExecutionStatus.COMPLETED
        state.route = "final"
        return state

    # 3. If there are still uncertainties and clarification budget permits, request clarification
    if state.analysis.uncertainties:
        state.execution_status = ExecutionStatus.NEEDS_CLARIFICATION
        state.route = "socratic"
        return state

    # 4. Otherwise, we have converged
    state.execution_status = ExecutionStatus.COMPLETED
    state.route = "final"
    return state
