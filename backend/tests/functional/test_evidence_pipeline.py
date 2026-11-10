from datetime import UTC, datetime
from unittest.mock import patch
from uuid import uuid4

from app.graph.graph import build_graph
from app.graph.state import GraphState
from app.schemas.evidence import DuckEvidence
from app.services.parsed_response import ParsedResponse


def test_evidence_survives_graph_transitions(make_duck_request, make_duck_analysis):
    """Verify evidence collected during search remains available in final state."""
    req = make_duck_request("Who discovered penicillin?")
    state = GraphState(request=req)
    graph = build_graph()

    analysis1 = make_duck_analysis(
        analysis="Needs search.",
        uncertainties=["Penicillin discoverer"],
        needs_search=True,
    )
    analysis2 = make_duck_analysis(
        analysis="Discovered by Alexander Fleming.",
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
                    "title": "History",
                    "url": "https://history.org/penicillin",
                    "content": "Fleming in 1928",
                }
            ]
        }
        mock_text.return_value = "Alexander Fleming discovered penicillin in 1928."

        result = graph.invoke(state)

    gs = result if isinstance(result, GraphState) else GraphState(**result)
    assert gs.evidence is not None
    assert len(gs.evidence.raw_results) == 1
    assert gs.evidence.raw_results[0]["results"][0]["title"] == "History"


def test_multiple_snippets_same_source_preserved(make_duck_request, make_duck_analysis):
    """Verify distinct snippets from the same URL are both preserved after deduplication."""
    req = make_duck_request("Details about Python GIL")
    prior_evidence = DuckEvidence(
        evidence_id=uuid4(),
        search_id=uuid4(),
        raw_results=[
            {"url": "https://docs.python.org/gil", "content": "The GIL is a mutex."}
        ],
        collected_at=datetime.now(UTC),
    )
    state = GraphState(
        request=req,
        evidence=prior_evidence,
    )
    graph = build_graph()

    analysis1 = make_duck_analysis(
        analysis="Needs more GIL details.",
        uncertainties=["GIL removal in 3.13"],
        needs_search=True,
    )
    analysis2 = make_duck_analysis(
        analysis="Sufficient info.",
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
        # Return a different snippet from the SAME URL
        mock_tavily.search.return_value = {
            "results": [
                {
                    "url": "https://docs.python.org/gil",
                    "content": "PEP 703 makes the GIL optional.",
                }
            ]
        }
        mock_text.return_value = (
            "The GIL is a mutex, but PEP 703 makes it optional in 3.13."
        )

        result = graph.invoke(state)

    gs = result if isinstance(result, GraphState) else GraphState(**result)
    assert len(gs.evidence.raw_results) == 2


def test_duplicate_evidence_removed(make_duck_request, make_duck_analysis):
    """Verify genuinely identical evidence (same URL + same content) is deduplicated."""
    req = make_duck_request("Query with duplicates")
    prior_evidence = DuckEvidence(
        evidence_id=uuid4(),
        search_id=uuid4(),
        raw_results=[
            {"url": "https://example.com/item", "content": "Exact duplicate text."}
        ],
        collected_at=datetime.now(UTC),
    )
    state = GraphState(
        request=req,
        evidence=prior_evidence,
    )
    graph = build_graph()

    analysis1 = make_duck_analysis(
        analysis="Need search.",
        uncertainties=["Item check"],
        needs_search=True,
    )
    analysis2 = make_duck_analysis(
        analysis="Done.",
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
        # Search returns exact same URL and content
        mock_tavily.search.return_value = {
            "results": [
                {"url": "https://example.com/item", "content": "Exact duplicate text."}
            ]
        }
        mock_text.return_value = "Done."

        result = graph.invoke(state)

    gs = result if isinstance(result, GraphState) else GraphState(**result)
    assert len(gs.evidence.raw_results) == 1


def test_search_queries_not_repeated(make_duck_request, make_duck_analysis):
    """Verify previously attempted normalized queries are skipped and tracked."""
    req = make_duck_request("Query tracking test")
    # Pre-populate attempted_queries with the normalized query for uncertainty "FastAPI"
    state = GraphState(
        request=req,
        attempted_queries=["evidence about fastapi"],
    )
    graph = build_graph()

    analysis1 = make_duck_analysis(
        analysis="Needs search.",
        uncertainties=["FastAPI"],
        needs_search=True,
    )
    analysis2 = make_duck_analysis(
        analysis="Clarification required.",
        uncertainties=["FastAPI"],
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
        mock_text.return_value = "What aspect of FastAPI?"

        result = graph.invoke(state)

        # Tavily search should NOT have been invoked because "evidence about fastapi" was already attempted
        mock_tavily.search.assert_not_called()

    gs = result if isinstance(result, GraphState) else GraphState(**result)
    assert "evidence about fastapi" in gs.attempted_queries


def test_evidence_size_limit_enforced(make_duck_request, make_duck_analysis):
    """Verify max_evidence_results limit is enforced deterministically."""
    req = make_duck_request("Query with many snippets")
    prior_evidence = DuckEvidence(
        evidence_id=uuid4(),
        search_id=uuid4(),
        raw_results=[
            {"url": f"https://example.com/{i}", "content": f"Snippet {i}"}
            for i in range(4)
        ],
        collected_at=datetime.now(UTC),
    )
    state = GraphState(
        request=req,
        evidence=prior_evidence,
        max_evidence_results=3,
    )
    graph = build_graph()

    analysis1 = make_duck_analysis(
        analysis="Needs search.",
        uncertainties=["Many items"],
        needs_search=True,
    )
    analysis2 = make_duck_analysis(
        analysis="Finished.",
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
        # Return 1 new distinct snippet
        mock_tavily.search.return_value = {
            "results": [{"url": "https://example.com/new", "content": "Snippet new"}]
        }
        mock_text.return_value = "Summary."

        result = graph.invoke(state)

    gs = result if isinstance(result, GraphState) else GraphState(**result)
    assert len(gs.evidence.raw_results) == 3


def test_final_response_receives_collected_evidence(
    make_duck_request, make_duck_analysis
):
    """Verify final response prompt explicitly formats and receives collected evidence."""
    req = make_duck_request("Explain Raft consensus")
    state = GraphState(request=req)
    graph = build_graph()

    analysis1 = make_duck_analysis(
        analysis="Need search.",
        uncertainties=["Raft election"],
        needs_search=True,
    )
    analysis2 = make_duck_analysis(
        analysis="Evidence gathered.",
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
                    "title": "Raft Paper",
                    "url": "https://raft.github.io",
                    "content": "Leader election uses randomized timers.",
                }
            ]
        }
        mock_text.return_value = "Raft achieves consensus via leader election."

        result = graph.invoke(state)

        # Inspect prompt passed to final_response_node call_text_llm
        call_kwargs = mock_text.call_args.kwargs
        assert "Raft Paper" in call_kwargs["prompt"]
        assert "https://raft.github.io" in call_kwargs["prompt"]
        assert "Leader election uses randomized timers." in call_kwargs["prompt"]
        assert "Do NOT fabricate citations" in call_kwargs["prompt"]
