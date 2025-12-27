from app.graph.state import GraphState


def loop_decider_node(state: GraphState) -> GraphState:
    if state.analysis is None:
        raise ValueError("Analysis must exist before loop decision")

    # HARD GUARD: if a socratic question is pending, wait for user
    if state.socratic_question is not None:
        state.should_loop = True
        return state

    # Ensure counters exist
    if state.turn_count is None:
        state.turn_count = 0
    if state.max_turns is None:
        state.max_turns = 25

    # Increment turn count ONLY when not waiting on user
    state.turn_count += 1

    # Stop if max turns reached
    if state.turn_count >= state.max_turns:
        state.should_loop = False
        return state

    # Stop if no uncertainties remain
    uncertainties = getattr(state.analysis, "uncertainties", None)
    if not uncertainties:
        state.should_loop = False
        return state

    # Otherwise, continue looping
    state.should_loop = True
    return state
