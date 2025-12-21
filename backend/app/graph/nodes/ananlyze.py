from app.graph.state import GraphState
from app.schemas.analysis import DuckAnalysis
from app.services.llm_helper import call_structured_llm


def analyze_node(state: GraphState) -> GraphState:
    if state.last_user_reply is None:
        user_message = state.request.req
        no_last_user_reply_result = call_structured_llm(
            task_name="analyze",
            prompt=f"""
            TASK:
            You are a critical thinking assistant that analyzes user messages to surface hidden assumptions, identify knowledge gaps, and determine when external research is necessary. Your goal is to help users reason more clearly by making implicit beliefs explicit and flagging areas requiring verification.
  

            OUTPUT RULES (MANDATORY):
            - Return ONLY a valid JSON object with no additional text, markdown, code fences, or comments
            - Use double quotes for all keys and string values
            - All required fields (analysis, assumptions, uncertainties, needs_search) must be present
            - Do NOT add fields beyond the specified schema

            SCHEMA:
            {{
            "analysis_id": "string",
            "analysis_time": "ISO8601 datetime string",
            "analysis": "string",
            "assumptions": ["string"],
            "uncertainties": ["string"],
            "needs_search": boolean
            }}

            FIELD GUIDANCE:
            - analysis_id: Unique identifier for this analysis (can be a UUID or any unique string)
            - analysis_time: Current UTC time in ISO8601 format
            - analysis: Brief summary (1-2 sentences) of the user's core issue or question
            - assumptions: Array of implicit beliefs or premises the user is making (empty array if none identified)
            - uncertainties: Array of missing information, ambiguities, or knowledge gaps (empty array if none identified)
            - needs_search: Boolean - true if real-world facts, current data, or external verification would improve the response

            USER MESSAGE:
            {user_message}
            """.strip(),
            schema=DuckAnalysis,
        )
        result = no_last_user_reply_result

    else:
        user_message = state.last_user_reply
        initial_question = state.request.req
        previous_socratic_q = state.socratic_question
        last_user_reply_result = call_structured_llm(
            task_name="analyze",
            prompt=f"""
        TASK:
        You are a critical thinking assistant engaged in an ONGOING reasoning process with a user.

        Your goal is NOT to restart the analysis from scratch.
        Your goal is to REFINE and UPDATE the understanding of the user's original question
        based on new information provided by the user.

        Think of this as progressive clarification, not a fresh problem.
        
        ORIGINAL USER QUESTION:
        {initial_question}
        
        Previous Socratic question:
        {previous_socratic_q}

        USER'S LATEST REPLY (in response to your previous Socratic question):
        {user_message}

        ANALYSIS INSTRUCTIONS:

        - Treat the original user question as the core problem that remains constant.
        - Use the user's latest reply to:
        - Clarify intent
        - Validate or invalidate earlier assumptions
        - Resolve uncertainties where possible
        - REMOVE uncertainties that are clearly answered by the user's reply.
        - ADD new uncertainties ONLY if the reply introduces genuinely new ambiguity.
        - Do NOT rephrase the problem unless the user's reply meaningfully reframes it.
        - Do NOT invent complexity for its own sake.

        Your analysis should show progress toward clarity.

        --------------------------------------------------
        OUTPUT RULES (MANDATORY):
        - Return ONLY a valid JSON object
        - Do NOT include explanations, markdown, or extra text
        - Use double quotes for all keys and string values
        - Do NOT add fields beyond the schema
        - All fields must be present even if empty

        SCHEMA:
        {{
            "analysis_id": "string",
            "analysis_time": "ISO8601 datetime string",
            "analysis": "string",
            "assumptions": ["string"],
            "uncertainties": ["string"],
            "needs_search": boolean
        }}

        FIELD GUIDANCE:
        - analysis: 1–2 sentences summarizing the CURRENT refined understanding
        - assumptions: implicit beliefs still being made (empty array if none)
        - uncertainties: unresolved gaps AFTER considering the user's reply
        - needs_search: true only if external facts or data are required to proceed

        IMPORTANT:
        This is an iterative reasoning process.
        Your output should move closer to clarity than the previous turn.

            """.strip(),
            schema=DuckAnalysis,
        )
        result = last_user_reply_result

    if not result.success:
        raise RuntimeError(
            f"Analyze node failed after {result.attempts} attempts: {result.errors}"
        )

    state.analysis = result.value
    return state
