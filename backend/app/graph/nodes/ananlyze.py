from app.graph.state import GraphState
from app.schemas.analysis import DuckAnalysis
from app.services.llm_helper import call_structured_llm


def analyze_node(state: GraphState) -> GraphState:
    user_message = state.request.req
    # Placeholder for analysis logic
    result = call_structured_llm(
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
    if not result.success:
        raise RuntimeError(
            f"Analyze node failed after {result.attempts} attempts: {result.errors}"
        )

    state.analysis = result.value
    return state
