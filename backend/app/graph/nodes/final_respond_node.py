
from app.graph.state import DuckEvidence, GraphState
from app.services.llm_helper import call_text_llm


def final_response_node(state: GraphState) -> GraphState:
    """Generate the final answer, grounded in any collected evidence.

    The prompt receives the original question, latest analysis, assumptions,
    remaining uncertainties, and a formatted view of the evidence. The LLM is
    instructed to cite only sources present in the evidence block and to note
    when evidence is insufficient.
    """

    # Helper to format evidence for inclusion in the final prompt (mirrors analyze_node)
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

    initial = state.request.req
    analysis_text = state.analysis.analysis if state.analysis else ""
    assumptions = state.analysis.assumptions if state.analysis else []
    uncertainties = state.analysis.uncertainties if state.analysis else []
    evidence_str = _format_evidence(state.evidence)

    prompt = (
        (
            f"""
        TASK:
        You are concluding an iterative reasoning session and must provide a final, well‑grounded answer.

        INPUTS:
        - Initial question: {initial}
        - Latest analysis: {analysis_text}
        - Assumptions: {assumptions}
        - Remaining uncertainties: {uncertainties}
        - Retrieved evidence (may contain multiple sources):\n{evidence_str}

        INSTRUCTIONS:
        1. Base your answer on the analysis and any retrieved evidence.
        2. When you cite a factual claim, reference the source title and URL exactly as they appear in the evidence block.
        3. Do NOT fabricate citations or claim that a source supports a statement it does not.
        4. If evidence is insufficient to resolve a remaining uncertainty, explicitly acknowledge the limitation.
        5. Provide a concise, direct answer; do not ask further questions.
        """
        )
        
        .strip()
    )

    state.final_answer = call_text_llm(task_name="finalize", prompt=prompt)
    state.socratic_question = None
    state.turn_count += 1
    return state
