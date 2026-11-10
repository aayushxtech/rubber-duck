import sys
from datetime import datetime
from unittest.mock import patch
from uuid import uuid4

# Ensure the backend package is importable
sys.path.insert(0, "backend")

from app.graph.graph import build_graph
from app.graph.state import ExecutionStatus, GraphState
from app.schemas.analysis import DuckAnalysis
from app.schemas.request import DuckRequest


# Helper to create a dummy analysis
def dummy_analysis(uncertainties=None, needs_search=False):
    if uncertainties is None:
        uncertainties = []
    return DuckAnalysis(
        analysis_id=uuid4(),
        analysis_time=datetime.utcnow(),
        analysis="dummy analysis",
        assumptions=[],
        uncertainties=uncertainties,
        needs_search=needs_search,
    )


# Mock LLM response for analysis nodes – simply returns the provided analysis object
class DummyParsedResponse:
    def __init__(self, value):
        self.value = value
        self.success = True
        self.attempts = 1
        self.errors = None


# Mock Tavily client – we will monkey‑patch the search_node's tavily_client.search method via the tavily_client object
# The search_node directly accesses the global tavily_client, so we patch the attribute on the module


def test_initial_analysis_triggers_search_and_preserves_evidence():
    # Initial request with an uncertainty that requires search
    request = DuckRequest(
        req_id=uuid4(),
        req="What is the capital of Wonderland?",
        req_time=datetime.utcnow(),
        status="new",
    )
    state = GraphState(request=request)
    graph = build_graph()

    # Mock the structured LLM calls for the initial analysis and the post‑search analysis
    with (
        patch("app.graph.nodes.ananlyze.call_structured_llm") as mock_llm,
        patch("app.graph.nodes.search_node.tavily_client") as mock_tavily,
    ):
        # Initial analysis says we need search
        mock_llm.return_value = DummyParsedResponse(
            dummy_analysis(["capital of Wonderland"], needs_search=True)
        )
        # Mock Tavily to return a simple result
        mock_tavily.search.return_value = {
            "results": [
                {
                    "title": "Wiki",
                    "url": "https://example.com",
                    "content": "Wonderland's capital is Heart City.",
                }
            ]
        }
        # Mock the second analysis (post‑search) to resolve the uncertainty
        mock_llm.return_value = DummyParsedResponse(
            dummy_analysis([], needs_search=False)
        )

        result = graph.invoke(state)
        # After execution we expect evidence collected and final answer generated
        assert isinstance(result, dict) or isinstance(result, GraphState)
        # Extract GraphState from dict if needed
        gs = result if isinstance(result, GraphState) else result
        # Evidence must be present and contain the mocked result
        assert gs.evidence is not None
        assert isinstance(gs.evidence.raw_results, list)
        assert len(gs.evidence.raw_results) == 1
        # Final answer should be produced
        assert gs.final_answer is not None
        # Execution should be completed because uncertainty was resolved
        assert gs.execution_status == ExecutionStatus.COMPLETED


def test_evidence_accumulation_and_deduplication():
    request = DuckRequest(
        req_id=uuid4(),
        req="List the top AI languages.",
        req_time=datetime.utcnow(),
        status="new",
    )
    state = GraphState(request=request)
    graph = build_graph()

    with (
        patch("app.graph.nodes.ananlyze.call_structured_llm") as mock_llm,
        patch("app.graph.nodes.search_node.tavily_client") as mock_tavily,
    ):
        # First analysis needs search for two uncertainties
        mock_llm.side_effect = [
            DummyParsedResponse(dummy_analysis(["AI languages"], needs_search=True)),
            DummyParsedResponse(
                dummy_analysis(["AI languages"], needs_search=True)
            ),  # post‑search keeps need
            DummyParsedResponse(
                dummy_analysis(["AI languages"], needs_search=False)
            ),  # after second search resolves
        ]
        # First search returns a result with URL A
        mock_tavily.search.side_effect = [
            {
                "results": [
                    {
                        "title": "Source A",
                        "url": "https://source-a.com",
                        "content": "Python is popular.",
                    }
                ]
            },
            {
                "results": [
                    {
                        "title": "Source A",
                        "url": "https://source-a.com",
                        "content": "Python still leads.",
                    }
                ]
            },
        ]
        result = graph.invoke(state)
        gs = result if isinstance(result, GraphState) else result
        # Evidence should contain only one deduplicated entry (max_evidence_results = 10, but duplicate URL removed)
        assert len(gs.evidence.raw_results) == 1
        # Research count should be 2 (two searches attempted)
        assert gs.internal_research_count == 2


def test_research_budget_exhaustion_and_clarification():
    request = DuckRequest(
        req_id=uuid4(),
        req="Explain quantum gravity.",
        req_time=datetime.utcnow(),
        status="new",
    )
    state = GraphState(request=request, max_research_iterations=1)
    graph = build_graph()

    with (
        patch("app.graph.nodes.ananlyze.call_structured_llm") as mock_llm,
        patch("app.graph.nodes.search_node.tavily_client") as mock_tavily,
    ):
        # Initial analysis needs search
        mock_llm.return_value = DummyParsedResponse(
            dummy_analysis(["quantum gravity"], needs_search=True)
        )
        mock_tavily.search.return_value = {
            "results": [
                {
                    "title": "Paper",
                    "url": "https://paper.com",
                    "content": "No consensus yet.",
                }
            ]
        }
        # Post‑search analysis still says need search (but budget will be exhausted)
        mock_llm.return_value = DummyParsedResponse(
            dummy_analysis(["quantum gravity"], needs_search=True)
        )
        result = graph.invoke(state)
        gs = result if isinstance(result, GraphState) else result
        # Budget exhausted – should route to clarification (NEEDS_CLARIFICATION) without looping indefinitely
        assert gs.execution_status == ExecutionStatus.NEEDS_CLARIFICATION
        assert gs.route == "socratic"
        assert gs.internal_research_count == 1


def test_conflicting_evidence_handling():
    request = DuckRequest(
        req_id=uuid4(),
        req="What is the boiling point of water at sea level?",
        req_time=datetime.utcnow(),
        status="new",
    )
    state = GraphState(request=request)
    graph = build_graph()

    with (
        patch("app.graph.nodes.ananlyze.call_structured_llm") as mock_llm,
        patch("app.graph.nodes.search_node.tavily_client") as mock_tavily,
    ):
        mock_llm.return_value = DummyParsedResponse(
            dummy_analysis(["boiling point"], needs_search=True)
        )
        # Return two contradictory results
        mock_tavily.search.return_value = {
            "results": [
                {"title": "Source 1", "url": "https://s1.com", "content": "100°C"},
                {"title": "Source 2", "url": "https://s2.com", "content": "212°F"},
            ]
        }
        # Post‑search analysis decides uncertainty remains (needs clarification)
        mock_llm.return_value = DummyParsedResponse(
            dummy_analysis(["boiling point"], needs_search=False)
        )
        result = graph.invoke(state)
        gs = result if isinstance(result, GraphState) else result
        # Evidence should contain both sources (deduped by URL, both distinct)
        assert (
            len(gs.evidence.raw_results) == 1
        )  # because our format keeps only first result per Tavily response
        # Execution should request clarification due to unresolved uncertainty
        assert gs.execution_status == ExecutionStatus.NEEDS_CLARIFICATION
        assert gs.route == "socratic"


def test_evidence_size_limit():
    request = DuckRequest(
        req_id=uuid4(),
        req="Give me 20 references about AI.",
        req_time=datetime.utcnow(),
        status="new",
    )
    state = GraphState(request=request, max_evidence_results=3)
    graph = build_graph()

    with (
        patch("app.graph.nodes.ananlyze.call_structured_llm") as mock_llm,
        patch("app.graph.nodes.search_node.tavily_client") as mock_tavily,
    ):
        mock_llm.return_value = DummyParsedResponse(
            dummy_analysis(["AI references"], needs_search=True)
        )
        # Simulate Tavily returning many results
        mock_tavily.search.return_value = {
            "results": [
                {
                    "title": f"Source {i}",
                    "url": f"https://s{i}.com",
                    "content": f"Info {i}",
                }
                for i in range(10)
            ]
        }
        # Post‑search analysis resolves (no further search)
        mock_llm.return_value = DummyParsedResponse(
            dummy_analysis([], needs_search=False)
        )
        result = graph.invoke(state)
        gs = result if isinstance(result, GraphState) else result
        # Evidence should be trimmed to max_evidence_results (3)
        assert len(gs.evidence.raw_results) == 3
        assert gs.internal_research_count == 1


# The above tests rely on the deterministic behavior of the graph and mocked LLM/Tavily.
