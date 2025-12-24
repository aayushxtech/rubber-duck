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
            You are a critical thinking assistant that analyzes a user’s message to surface hidden assumptions and identify meaningful uncertainties.

            Your primary role is to help the user think more clearly by making implicit beliefs explicit and by identifying what must be clarified before a high-quality answer can be given.

            You should prefer asking a single, focused Socratic question whenever clarification would materially improve the answer.

            You are NOT trying to maximize rigor or completeness.
            You are NOT trying to answer immediately if clarification is needed.
            You ARE trying to decide whether the next step should be:
            - asking the user a clarifying question
            - performing external research
            - or converging to a final answer

            OUTPUT RULES (MANDATORY):
            - Return ONLY a valid JSON object
            - Use double quotes for all keys and string values
            - Do NOT include markdown, comments, or explanatory text
            - All required fields (analysis, assumptions, uncertainties, needs_search) MUST be present
            - Do NOT add fields beyond the specified schema

            CRITICAL INVARIANTS (MUST FOLLOW):
            - If "uncertainties" is an empty list, then "needs_search" MUST be false
            - External search is permitted ONLY to resolve listed uncertainties
            - Missing user intent or context must NEVER trigger external search
            - If clarification from the user would improve the answer, prefer listing uncertainties over converging

            UNCERTAINTY RULES (IMPORTANT):
            - Uncertainties represent information that must be clarified before giving the best answer
            - Uncertainties may include:
            - missing factual information
            - missing user goals, constraints, time horizon, or context
            - Clarification uncertainties must NOT trigger search
            - Do NOT include:
            - minor preferences that do not affect the substance of the answer
            - stylistic or presentation-related clarifications
            - uncertainties whose resolution would not change the advice

            SOCRATIC PRIORITY RULE (KEY):
            - If the question is abstract, strategic, value-based, or involves trade-offs:
            - assume clarification is needed by default
            - list at least one clarification uncertainty unless the question is already fully specified
            - Do NOT treat the ability to answer under assumptions as sufficient reason to skip clarification

            CONVERGENCE RULE:
            - Converge (return an empty uncertainties list) ONLY if:
            - the question is concrete and well-specified
            - OR clarification would not materially change the answer
            - Do NOT converge merely because an answer is possible under assumptions

            SEARCH GUIDANCE:
            - Set needs_search to true ONLY if:
            - resolving a listed uncertainty requires real-world facts, empirical data, or external verification
            - If uncertainties are empty, needs_search MUST be false

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
            - analysis_id: Unique identifier for this analysis
            - analysis_time: Current UTC time in ISO8601 format
            - analysis: 1–2 sentence summary of the user’s core question
            - assumptions: Implicit beliefs or premises (empty array if none)
            - uncertainties: Information that must be clarified before giving the best answer (empty array only when truly unnecessary)
            - needs_search: True only if resolving listed uncertainties requires external research

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
        search_results = state.evidence
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
        
        SSEARCH RESULTS regarding last socratic question:
        {search_results}

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
        - Only include uncertainties that PREVENT a reasonable, high-quality answer. If the remaining uncertainty would not change the conclusion materially,do NOT include it.
        - Use the SEARCH RESULTS only to reduce the uncer

        Your analysis should show progress toward clarity.

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
