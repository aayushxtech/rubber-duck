from app.graph.state import GraphState
from app.services.llm_helper import call_text_llm


def final_response_node(state: GraphState) -> GraphState:
    initial = state.request.req
    if state.analysis is not None:
        analysis = state.analysis.analysis
        assumptions = state.analysis.assumptions
        uncertainties = state.analysis.uncertainties
    else:
        analysis = None
        assumptions = None
        uncertainties = None

    # For MVP: just pass through the last Socratic question
    prompt = f"""
    TASK:
    You are concluding an iterative reasoning session.
    
    INPUTS:
    initial_question = {initial}
    analysis = {analysis}
    assumptions = {assumptions}
    uncertainties = {uncertainties}
    
    INSTRUCTIONS:
    - Assume the user is ready for a final answer.
    - Provide a clear, direct answer
    - Do not ask questions
    - Do not introduce new uncertainty
    - Acknowledge limits only if necessary
    """

    state.final_answer = call_text_llm(task_name="finalize", prompt=prompt)
    return state
