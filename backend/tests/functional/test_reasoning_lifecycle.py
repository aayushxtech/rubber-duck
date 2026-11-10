from unittest.mock import patch

import pytest
from app.graph.graph import build_graph
from app.graph.state import ExecutionStatus, GraphState
from app.services.parsed_response import ParsedResponse


def test_1_request_answered_directly(make_duck_request, make_duck_analysis):
    """Scenario 1: Direct answer when initial analysis has no uncertainties."""
    req = make_duck_request("What is 2 + 2?")
    state = GraphState(request=req)
    graph = build_graph()

    mock_analysis = make_duck_analysis(
        analysis="Arithmetic question with known direct solution.",
        uncertainties=[],
        needs_search=False,
    )

    with (
        patch("app.graph.nodes.ananlyze.call_structured_llm") as mock_struct,
        patch("app.graph.nodes.final_respond_node.call_text_llm") as mock_text,
    ):
        mock_struct.return_value = ParsedResponse(
            value=mock_analysis,
            raw_text="{}",
            model="mock",
            attempts=1,
            success=True,
        )
        mock_text.return_value = "2 + 2 equals 4."

        result = graph.invoke(state)

    gs = result if isinstance(result, GraphState) else GraphState(**result)
    assert gs.execution_status == ExecutionStatus.COMPLETED
    assert gs.final_answer == "2 + 2 equals 4."
    assert gs.route == "final"
    assert gs.socratic_question is None


def test_2_request_requiring_clarification(make_duck_request, make_duck_analysis):
    """Scenario 2: Analysis surfaces uncertainties without search need -> Socratic question."""
    req = make_duck_request("I want to build an app.")
    state = GraphState(request=req)
    graph = build_graph()

    mock_analysis = make_duck_analysis(
        analysis="User goal is ambiguous.",
        uncertainties=["Target platform", "User audience"],
        needs_search=False,
    )

    with (
        patch("app.graph.nodes.ananlyze.call_structured_llm") as mock_struct,
        patch("app.graph.nodes.socratic.call_text_llm") as mock_text,
    ):
        mock_struct.return_value = ParsedResponse(
            value=mock_analysis,
            raw_text="{}",
            model="mock",
            attempts=1,
            success=True,
        )
        mock_text.return_value = (
            "What platform (web, mobile, or desktop) are you targeting?"
        )

        result = graph.invoke(state)

    gs = result if isinstance(result, GraphState) else GraphState(**result)
    assert gs.execution_status == ExecutionStatus.NEEDS_CLARIFICATION
    assert gs.route == "socratic"
    assert (
        gs.socratic_question
        == "What platform (web, mobile, or desktop) are you targeting?"
    )
    assert gs.final_answer is None


def test_3_clarification_response_from_user(make_duck_request, make_duck_analysis):
    """Scenario 3: User provides clarification reply, resolving remaining uncertainties."""
    req = make_duck_request("I want to build an app.")
    initial_analysis = make_duck_analysis(
        analysis="Prior analysis with open questions.",
        uncertainties=["Target platform"],
        needs_search=False,
    )
    state = GraphState(
        request=req,
        last_user_reply="I am building a web app using FastAPI and React.",
        socratic_question="What platform are you targeting?",
        analysis=initial_analysis,
    )
    graph = build_graph()

    resolved_analysis = make_duck_analysis(
        analysis="Platform clarified as web (FastAPI + React).",
        uncertainties=[],
        needs_search=False,
    )

    with (
        patch("app.graph.nodes.ananlyze.call_structured_llm") as mock_struct,
        patch("app.graph.nodes.final_respond_node.call_text_llm") as mock_text,
    ):
        mock_struct.return_value = ParsedResponse(
            value=resolved_analysis,
            raw_text="{}",
            model="mock",
            attempts=1,
            success=True,
        )
        mock_text.return_value = (
            "A recommended architecture is a decoupled REST/SPA setup."
        )

        result = graph.invoke(state)

    gs = result if isinstance(result, GraphState) else GraphState(**result)
    assert gs.execution_status == ExecutionStatus.COMPLETED
    assert (
        gs.final_answer == "A recommended architecture is a decoupled REST/SPA setup."
    )


def test_4_request_requiring_external_research(make_duck_request, make_duck_analysis):
    """Scenario 4: Request with factual uncertainty triggering research route."""
    req = make_duck_request("What is the latest release date of Python 3.13?")
    state = GraphState(request=req)
    graph = build_graph()

    needs_search_analysis = make_duck_analysis(
        analysis="Requires external release date verification.",
        uncertainties=["Python 3.13 release date"],
        needs_search=True,
    )
    post_search_analysis = make_duck_analysis(
        analysis="Python 3.13 released in October 2024.",
        uncertainties=[],
        needs_search=False,
    )

    with (
        patch("app.graph.nodes.ananlyze.call_structured_llm") as mock_struct,
        patch("app.graph.nodes.search_node.tavily_client") as mock_tavily,
        patch("app.graph.nodes.final_respond_node.call_text_llm") as mock_text,
    ):
        mock_struct.side_effect = [
            ParsedResponse(
                value=needs_search_analysis,
                raw_text="{}",
                model="mock",
                attempts=1,
                success=True,
            ),
            ParsedResponse(
                value=post_search_analysis,
                raw_text="{}",
                model="mock",
                attempts=1,
                success=True,
            ),
        ]
        mock_tavily.search.return_value = {
            "results": [
                {
                    "title": "Python Release",
                    "url": "https://python.org",
                    "content": "Released Oct 7, 2024",
                }
            ]
        }
        mock_text.return_value = (
            "Python 3.13 was officially released on October 7, 2024."
        )

        result = graph.invoke(state)

    gs = result if isinstance(result, GraphState) else GraphState(**result)
    assert gs.execution_status == ExecutionStatus.COMPLETED
    assert gs.internal_research_count == 1
    assert gs.evidence is not None
    assert len(gs.evidence.raw_results) == 1


def test_5_research_followed_by_successful_analysis(
    make_duck_request, make_duck_analysis
):
    """Scenario 5: Search succeeds, post-search analysis ingests evidence and converges."""
    req = make_duck_request("What is Project Valhalla in Java?")
    state = GraphState(request=req)
    graph = build_graph()

    analysis1 = make_duck_analysis(
        analysis="Needs search for Valhalla status.",
        uncertainties=["Java Project Valhalla purpose"],
        needs_search=True,
    )
    analysis2 = make_duck_analysis(
        analysis="Valhalla introduces value objects to Java.",
        uncertainties=[],
        needs_search=False,
    )

    with (
        patch("app.graph.nodes.ananlyze.call_structured_llm") as mock_struct,
        patch("app.graph.nodes.search_node.tavily_client") as mock_tavily,
        patch("app.graph.nodes.final_respond_node.call_text_llm") as mock_text,
    ):
        mock_struct.side_effect = [
            ParsedResponse(
                value=analysis1, raw_text="{}", model="mock", attempts=1, success=True
            ),
            ParsedResponse(
                value=analysis2, raw_text="{}", model="mock", attempts=1, success=True
            ),
        ]
        mock_tavily.search.return_value = {
            "results": [
                {
                    "title": "OpenJDK",
                    "url": "https://openjdk.org/projects/valhalla",
                    "content": "Value types",
                }
            ]
        }
        mock_text.return_value = (
            "Project Valhalla aims to augment Java with value objects."
        )

        result = graph.invoke(state)

    gs = result if isinstance(result, GraphState) else GraphState(**result)
    assert gs.execution_status == ExecutionStatus.COMPLETED
    assert gs.final_answer is not None


def test_6_research_with_insufficient_evidence(make_duck_request, make_duck_analysis):
    """Scenario 6: Research returns insufficient evidence, uncertainties remain -> Socratic clarification."""
    req = make_duck_request("Is quantum teleportation practical for humans?")
    state = GraphState(request=req)
    graph = build_graph()

    analysis1 = make_duck_analysis(
        analysis="Initial query needs search.",
        uncertainties=["Quantum macroscopic limits"],
        needs_search=True,
    )
    # Post-search analysis determines evidence does not resolve human scale practicality
    analysis2 = make_duck_analysis(
        analysis="Evidence only covers subatomic teleportation; human scale remains unresolved.",
        uncertainties=["Human scale macroscopic applicability"],
        needs_search=False,  # no further search needed, ask user clarification
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
        mock_tavily.search.return_value = {
            "results": [
                {
                    "title": "Quantum Physics",
                    "url": "https://example.com/quantum",
                    "content": "Teleportation observed for photons.",
                }
            ]
        }
        mock_text.return_value = (
            "Are you asking about physical matter transfer or information transmission?"
        )

        result = graph.invoke(state)

    gs = result if isinstance(result, GraphState) else GraphState(**result)
    assert gs.execution_status == ExecutionStatus.NEEDS_CLARIFICATION
    assert gs.socratic_question is not None


def test_7_research_with_conflicting_evidence(make_duck_request, make_duck_analysis):
    """Scenario 7: Conflicting sources retained as uncertainties -> route to Socratic."""
    req = make_duck_request("Does caffeine improve long-term memory?")
    state = GraphState(request=req)
    graph = build_graph()

    analysis1 = make_duck_analysis(
        analysis="Check memory effects of caffeine.",
        uncertainties=["Caffeine memory effects"],
        needs_search=True,
    )
    analysis2 = make_duck_analysis(
        analysis="Conflicting studies found: some indicate enhancement, others show no long-term effect.",
        uncertainties=[
            "Conflicting empirical evidence on duration of memory enhancement"
        ],
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
        mock_tavily.search.return_value = {
            "results": [
                {
                    "title": "Study A",
                    "url": "https://a.com",
                    "content": "Caffeine improves consolidation.",
                },
                {
                    "title": "Study B",
                    "url": "https://b.com",
                    "content": "Caffeine has no significant long term memory benefit.",
                },
            ]
        }
        mock_text.return_value = (
            "Which specific memory types or dosage levels are you focusing on?"
        )

        result = graph.invoke(state)

    gs = result if isinstance(result, GraphState) else GraphState(**result)
    assert gs.execution_status == ExecutionStatus.NEEDS_CLARIFICATION
    assert gs.route == "socratic"


def test_8_exhausted_research_budget(make_duck_request, make_duck_analysis):
    """Scenario 8: Exhausted research budget terminates search and proceeds deterministically."""
    req = make_duck_request("Hard problem requiring multiple searches.")
    # Initialize with research count already at max budget
    state = GraphState(
        request=req,
        internal_research_count=3,
        max_research_iterations=3,
    )
    graph = build_graph()

    # Analysis still claims needs_search=True and uncertainties exist
    analysis = make_duck_analysis(
        analysis="Still uncertain.",
        uncertainties=["Remaining gap"],
        needs_search=True,
    )

    with (
        patch("app.graph.nodes.ananlyze.call_structured_llm") as mock_struct,
        patch("app.graph.nodes.socratic.call_text_llm") as mock_text,
        patch("app.graph.nodes.search_node.tavily_client") as mock_tavily,
    ):
        mock_struct.return_value = ParsedResponse(
            value=analysis, raw_text="{}", model="mock", attempts=1, success=True
        )
        mock_text.return_value = "How would you define the remaining gap?"

        result = graph.invoke(state)

        # Tavily search must not be called because budget was already exhausted
        mock_tavily.search.assert_not_called()

    gs = result if isinstance(result, GraphState) else GraphState(**result)
    # Because budget exhausted and uncertainties exist, route to socratic clarification
    assert gs.execution_status == ExecutionStatus.NEEDS_CLARIFICATION
    assert gs.route == "socratic"


def test_9_failed_llm_execution(make_duck_request):
    """Scenario 9: Failed LLM execution raises controlled exception in analyze node."""
    req = make_duck_request("Trigger failure.")
    state = GraphState(request=req)
    graph = build_graph()

    with patch("app.graph.nodes.ananlyze.call_structured_llm") as mock_struct:
        mock_struct.return_value = ParsedResponse(
            value=None,
            raw_text=None,
            model="mock",
            attempts=5,
            success=False,
            errors=["LLM call error: ConnectionError"],
        )

        with pytest.raises(RuntimeError) as exc_info:
            graph.invoke(state)

        assert "Analyze node failed" in str(exc_info.value)


def test_10_failed_search_execution(make_duck_request, make_duck_analysis):
    """Scenario 10: Failed Tavily search captures error without crashing and increments budget."""
    req = make_duck_request("Search with failing provider.")
    state = GraphState(request=req)
    graph = build_graph()

    analysis1 = make_duck_analysis(
        analysis="Need search.",
        uncertainties=["External fact"],
        needs_search=True,
    )
    analysis2 = make_duck_analysis(
        analysis="Search failed, proceed with clarification.",
        uncertainties=["External fact still unverified"],
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
        # Search raises an exception
        mock_tavily.search.side_effect = Exception("Tavily service 503 unavailable")
        mock_text.return_value = "Can you provide more detail on this fact?"

        result = graph.invoke(state)

    gs = result if isinstance(result, GraphState) else GraphState(**result)
    assert gs.internal_research_count == 1
    assert gs.evidence is not None
    # Error should be captured in raw_results
    assert any("error" in res for res in gs.evidence.raw_results)
    assert gs.execution_status == ExecutionStatus.NEEDS_CLARIFICATION
