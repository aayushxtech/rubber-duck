from app.graph.state import GraphState
from app.services.llm_helper import call_text_llm


def socratic_node(state: GraphState) -> GraphState:
    if state.analysis is None or not hasattr(state.analysis, "analysis"):
        raise ValueError("state.analysis is None or missing 'analysis' attribute")
    current_analysis = state.analysis.analysis
    current_assumptions = state.analysis.assumptions or []
    current_uncertainties = state.analysis.uncertainties or []
    previous_question = state.socratic_question

    prev_q_instruction = (
        f"\nPREVIOUS QUESTION ASKED:\n{previous_question}\n"
        "(Do NOT repeat or rephrase this question; ask something new.)\n"
        if previous_question
        else ""
    )

    prompt = f"""You are a thoughtful critical-thinking partner having a natural conversation with the user.
Your job is to help the user examine their thinking, not demonstrate your own intelligence.

CONTEXT:
Analysis: {current_analysis}
Assumptions: {current_assumptions}
Uncertainties: {current_uncertainties}
{prev_q_instruction}
INSTRUCTIONS:
1. Ask exactly ONE question.
2. Must be ONE sentence in plain, conversational English.
3. Aim for 10–25 words. Keep it clear, simple, and direct.
4. Avoid academic jargon, formal metrics, mathematical methods, or technical buzzwords.
5. Challenge an unexamined assumption, a missing connection, or an unexplored consequence.
6. Do NOT interrogate or demand justification of every detail. Help the user reflect naturally.
7. Return ONLY the question itself. No preamble, no explanation, no quotation marks, no markdown.
8. The question MUST end with a question mark.
""".strip()

    raw_text = call_text_llm(
        task_name="socratic_question",
        prompt=prompt,
    )
    question = raw_text.strip().strip('"').strip("'")
    if question and not question.endswith("?"):
        question += "?"

    state.socratic_question = question
    state.turn_count += 1
    state.clarification_count += 1
    return state
