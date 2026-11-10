from unittest.mock import patch

from app.graph.graph import build_graph
from app.graph.state import ExecutionStatus, GraphState
from app.services.parsed_response import ParsedResponse


def test_search_failure_not_treated_as_successful_research(
    make_duck_request, make_duck_analysis
):
    """Verify that when search raises an error, it is recorded as an error and not genuine facts."""
    req = make_duck_request("Fact that fails to fetch")
    state = GraphState(request=req)
    graph = build_graph()

    analysis1 = make_duck_analysis(
        analysis="Needs search.",
        uncertainties=["Remote facts"],
        needs_search=True,
    )
    analysis2 = make_duck_analysis(
        analysis="Search failed, uncertainties persist.",
        uncertainties=["Remote facts still missing"],
        needs_search=False,
    )

    with (
        patch("app.graph.nodes.ananlyze.call_structured_llm") as mock_struct,
        patch("app.graph.nodes.search_node.tavily_client") as mock_tavily,
        patch("app.graph.nodes.socratic.call_text_llm") as mock_text,
    ):
        mock_struct.side_effect = [
            ParsedResponse(
                value=analysis1, raw_text="{}", model="mock", attempts=1, success=True
            ),
            ParsedResponse(
                value=analysis2, raw_text="{}", model="mock", attempts=1, success=True
            ),
        ]
        mock_tavily.search.side_effect = Exception("Tavily 500 error")
        mock_text.return_value = "Can you provide this fact directly?"

        result = graph.invoke(state)

    gs = result if isinstance(result, GraphState) else GraphState(**result)
    assert gs.evidence is not None
    assert len(gs.evidence.raw_results) == 1
    # Result must contain error, not fabricated valid content
    assert "error" in gs.evidence.raw_results[0]
    assert gs.execution_status == ExecutionStatus.NEEDS_CLARIFICATION


def test_invalid_model_response_does_not_fabricate_success(make_duck_request):
    """Verify that completely unparseable model output returns success=False and does not fabricate data."""
    req = make_duck_request("Prompt causing invalid output")
    state = GraphState(request=req)
    graph = build_graph()

    with patch("app.graph.nodes.ananlyze.call_structured_llm") as mock_struct:
        mock_struct.return_value = ParsedResponse(
            value=None,
            raw_text="Random conversational gibberish without JSON",
            model="mock",
            attempts=5,
            success=False,
            errors=["JSON decode error: Expecting value: line 1 column 1"],
        )

        try:
            graph.invoke(state)
            failed = False
        except RuntimeError as e:
            failed = True
            assert "Analyze node failed after 5 attempts" in str(e)

        assert failed, (
            "Expected graph invocation to fail when LLM returns invalid output"
        )


def test_research_budget_strictly_enforced_under_repeated_requests(
    make_duck_request, make_duck_analysis
):
    """Verify research loop cannot exceed max_research_iterations under any condition."""
    req = make_duck_request("Unresolvable question that always asks for search")
    state = GraphState(
        request=req,
        internal_research_count=2,
        max_research_iterations=3,
    )
    graph = build_graph()

    # Model persistently requests search
    analysis_needs_search = make_duck_analysis(
        analysis="Still needs more search.",
        uncertainties=["Never-ending mystery"],
        needs_search=True,
    )
    analysis_final_fallback = make_duck_analysis(
        analysis="Budget reached, ask user clarification.",
        uncertainties=["Never-ending mystery"],
        needs_search=False,
    )

    with (
        patch("app.graph.nodes.ananlyze.call_structured_llm") as mock_struct,
        patch("app.graph.nodes.search_node.tavily_client") as mock_tavily,
        patch("app.graph.nodes.socratic.call_text_llm") as mock_text,
    ):
        mock_struct.side_effect = [
            ParsedResponse(
                value=analysis_needs_search,
                raw_text="{}",
                model="mock",
                attempts=1,
                success=True,
            ),
            ParsedResponse(
                value=analysis_needs_search,
                raw_text="{}",
                model="mock",
                attempts=1,
                success=True,
            ),
        ]
        mock_tavily.search.return_value = {
            "results": [
                {"title": "Source", "url": "https://example.com", "content": "Info"}
            ]
        }
        mock_text.return_value = "What is your perspective on this mystery?"

        result = graph.invoke(state)

    gs = result if isinstance(result, GraphState) else GraphState(**result)
    assert gs.internal_research_count == 3
    assert gs.route == "socratic"
    assert gs.execution_status == ExecutionStatus.NEEDS_CLARIFICATION
