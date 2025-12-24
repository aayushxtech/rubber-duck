from app.graph.state import GraphState


def loop_decider_node(state: GraphState) -> GraphState:
    if state.analysis is None:
        raise ValueError("Analysis must exist before loop decision")

    # Ensure turn_count and max_turns are initialized
    if not hasattr(state, "turn_count") or state.turn_count is None:
        state.turn_count = 0
    if not hasattr(state, "max_turns") or state.max_turns is None:
        state.max_turns = 25  # Default safety limit

    # Increment turn count
    state.turn_count += 1

    # Exit conditions
    if state.turn_count >= state.max_turns:
        state.should_loop = False
        return state

    # Ensure uncertainties is a list
    uncertainties = getattr(state.analysis, "uncertainties", None)
    if not uncertainties:
        state.should_loop = False
        return state

    state.should_loop = True

    return state
