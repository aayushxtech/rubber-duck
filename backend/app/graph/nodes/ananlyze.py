from datetime import UTC, datetime
from uuid import uuid4

from app.graph.state import DuckEvidence, GraphState
from app.schemas.analysis import DuckAnalysis, DuckAnalysisContent
from app.services.llm_helper import call_structured_llm


def analyze_node(state: GraphState) -> GraphState:
    """Perform analysis of the user's problem.

    Three execution paths:
    1️⃣ Initial turn – no evidence, no user reply yet.
    2️⃣ Post‑search turn – evidence has been gathered but the user has not replied.
    3️⃣ Clarification turn – the user replied to a Socratic question.
    The appropriate prompt is built for each case and sent to the structured LLM.
    """

    # Safeguard: reject empty or whitespace-only user question
    if state.request is None or not state.request.req or not state.request.req.strip():
        raise ValueError("User question cannot be empty or whitespace-only")

    # --------------------------------------------------------------------
    # Helper to format evidence for inclusion in prompts
    # --------------------------------------------------------------------
    def _format_evidence(evidence: DuckEvidence | None) -> str:
        if not evidence or not evidence.raw_results:
            return "No external evidence collected."
        lines = []
        for idx, res in enumerate(evidence.raw_results, start=1):
            title = url = snippet = None
            if isinstance(res, dict):
                if (
                    "results" in res
                    and isinstance(res["results"], list)
                    and res["results"]
                ):
                    first = res["results"][0]
                    title = first.get("title")
                    url = first.get("url")
                    snippet = first.get("content")
                else:
                    title = res.get("title")
                    url = res.get("url")
                    snippet = res.get("content")
            # Fallback handling for error dicts
            if not title:
                title = res.get("query", "Unknown source")
            if not url:
                url = "N/A"
            if not snippet:
                snippet = "(no snippet available)"
            lines.append(
                f"[{idx}] Title: {title}\n    URL: {url}\n    Snippet: {snippet}"
            )
        return "\n".join(lines)

    # --------------------------------------------------------------------
    # 1️⃣ Initial turn – first analysis, no prior evidence, no user reply
    # --------------------------------------------------------------------
    if state.last_user_reply is None and (
        state.evidence is None or not state.evidence.raw_results
    ):
        user_message = state.request.req.strip()
        prompt = (
            (
                f"""
            TASK:
            You are a critical thinking assistant that analyzes a user’s message to surface hidden assumptions and identify meaningful uncertainties.

            YOUR PRIMARY ROLE:
            - Produce a JSON object with fields: analysis, assumptions, uncertainties, needs_search.
            - Do NOT perform external search at this stage.

            USER MESSAGE:
            {user_message}
            """
            )

            .strip()
        )

        # Safeguards before calling LLM
        if user_message not in prompt:
            raise ValueError("Constructed analysis prompt missing actual user question")
        if "{user_message}" in prompt:
            raise ValueError(
                "Unresolved {user_message} placeholder detected in analysis prompt"
            )

        result = call_structured_llm(
            task_name="analyze_initial", prompt=prompt, schema=DuckAnalysisContent
        )

    # --------------------------------------------------------------------
    # 2️⃣ Post‑search turn – evidence exists, user has not replied yet
    # --------------------------------------------------------------------
    elif (
        state.last_user_reply is None
        and state.evidence is not None
        and state.analysis is not None
    ):
        evidence_str = _format_evidence(state.evidence)
        original_question = state.request.req.strip()
        prompt = (
            (
                """
            TASK:
            You are revisiting the original question after external research.

            CONTEXT:
            ORIGINAL QUESTION: {original_question}
            PREVIOUS ANALYSIS: {previous_analysis}
            PREVIOUS ASSUMPTIONS: {previous_assumptions}
            PREVIOUS UNCERTAINTIES: {previous_uncertainties}

            EVIDENCE (may contain multiple sources):
            {evidence_str}

            INSTRUCTIONS:
            - For each uncertainty, decide whether the evidence directly resolves it, partially resolves it, or does not resolve it.
            - Remove uncertainties that are directly resolved.
            - Keep uncertainties that are only partially addressed or still ambiguous.
            - If contradictory evidence is present, retain the uncertainty and note the conflict.
            - Do NOT assume an uncertainty is resolved merely because evidence exists.
            - Set `needs_search` to true only if a factual gap remains that the current evidence does not adequately cover.

            Respond with a valid JSON object with fields: analysis, assumptions, uncertainties, needs_search.
            """
            )
            .format(
                original_question=original_question,
                previous_analysis=state.analysis.analysis or "",
                previous_assumptions=state.analysis.assumptions or [],
                previous_uncertainties=state.analysis.uncertainties or [],
                evidence_str=evidence_str,
            )
            .strip()
        )

        # Safeguards before calling LLM
        if original_question not in prompt:
            raise ValueError("Constructed analysis prompt missing actual user question")
        if "{user_message}" in prompt:
            raise ValueError(
                "Unresolved {user_message} placeholder detected in analysis prompt"
            )

        result = call_structured_llm(
            task_name="analyze_post_search", prompt=prompt, schema=DuckAnalysisContent
        )

    # --------------------------------------------------------------------
    # 3️⃣ Clarification turn – user replied to a Socratic question
    # --------------------------------------------------------------------
    else:
        user_message = state.last_user_reply
        initial_question = state.request.req.strip()
        previous_socratic_q = state.socratic_question
        evidence_str = _format_evidence(state.evidence)
        prompt = (
            (
                """
            TASK:
            You are a critical thinking assistant engaged in an ongoing reasoning process with a user.

            ORIGINAL USER QUESTION:
            {initial_question}

            PREVIOUS SOCRATIC QUESTION:
            {previous_socratic_q}

            USER'S LATEST REPLY:
            {user_message}

            EVIDENCE (if any):
            {evidence_str}

            ANALYSIS INSTRUCTIONS:
            - Update the analysis based on the user's reply.
            - REMOVE uncertainties that the reply clearly resolves.
            - ADD new uncertainties only if the reply creates new ambiguity.
            - Use the evidence to answer factual parts where possible, but do NOT fabricate.
            - Set `needs_search` to true ONLY if a factual gap remains that the evidence does not cover.

            Return a JSON object with fields: analysis, assumptions, uncertainties, needs_search.
            """
            )
            .format(
                initial_question=initial_question,
                previous_socratic_q=previous_socratic_q or "(none)",
                user_message=user_message,
                evidence_str=evidence_str,
            )
            .strip()
        )

        # Safeguards before calling LLM
        if initial_question not in prompt:
            raise ValueError("Constructed analysis prompt missing actual user question")
        if "{user_message}" in prompt:
            raise ValueError(
                "Unresolved {user_message} placeholder detected in analysis prompt"
            )

        result = call_structured_llm(
            task_name="analyze_clarification", prompt=prompt, schema=DuckAnalysisContent
        )

    # --------------------------------------------------------------------
    # Common post‑processing – handle LLM failures & assign Python metadata
    # --------------------------------------------------------------------
    if not result.success or result.value is None:
        raise RuntimeError(
            f"Analyze node failed after {result.attempts} attempts: {result.errors}"
        )

    # Generate metadata in Python (NOT the LLM)
    analysis_id = uuid4()
    analysis_time = datetime.now(UTC)

    raw_val = result.value
    # Validate and construct final DuckAnalysis with Python metadata
    if isinstance(raw_val, (DuckAnalysisContent, DuckAnalysis)):
        state.analysis = DuckAnalysis(
            analysis_id=analysis_id,
            analysis_time=analysis_time,
            analysis=raw_val.analysis,
            assumptions=raw_val.assumptions,
            uncertainties=raw_val.uncertainties,
            needs_search=raw_val.needs_search,
        )
    elif isinstance(raw_val, dict):
        state.analysis = DuckAnalysis(
            analysis_id=analysis_id,
            analysis_time=analysis_time,
            analysis=raw_val.get("analysis", ""),
            assumptions=raw_val.get("assumptions"),
            uncertainties=raw_val.get("uncertainties"),
            needs_search=bool(raw_val.get("needs_search", False)),
        )
    else:
        raise ValueError(f"Unexpected analysis response type: {type(raw_val)}")

    return state
