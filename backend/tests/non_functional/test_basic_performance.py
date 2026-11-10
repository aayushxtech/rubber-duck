import time
from unittest.mock import patch

from app.services.parsed_response import ParsedResponse


def test_mocked_request_completes_within_reasonable_time(client, make_duck_analysis):
    """Verify that deterministic execution of a direct request executes in < 500ms when mocked."""
    analysis = make_duck_analysis(
        analysis="Fast deterministic path.",
        uncertainties=[],
        needs_search=False,
    )

    with (
        patch("app.graph.nodes.ananlyze.call_structured_llm") as mock_struct,
        patch("app.graph.nodes.final_respond_node.call_text_llm") as mock_text,
    ):
        mock_struct.return_value = ParsedResponse(
            value=analysis, raw_text="{}", model="mock", attempts=1, success=True
        )
        mock_text.return_value = "Instant answer."

        start = time.perf_counter()
        resp = client.post(
            "/api/ducks/start", json={"initial_question": "Quick question"}
        )
        elapsed = time.perf_counter() - start

    assert resp.status_code == 200
    # Mocked request overhead should easily complete in under 500ms
    assert elapsed < 0.5, f"Execution took unexpectedly long: {elapsed:.3f}s"


def test_research_loop_bounded_execution(client, make_duck_analysis):
    """Verify research loop terminates within bounded time and calls when budget exhausted."""
    analysis = make_duck_analysis(
        analysis="Needs search.",
        uncertainties=["Loop test uncertainty"],
        needs_search=True,
    )

    with (
        patch("app.graph.nodes.ananlyze.call_structured_llm") as mock_struct,
        patch("app.graph.nodes.search_node.tavily_client") as mock_tavily,
        patch("app.graph.nodes.socratic.call_text_llm") as mock_text,
    ):
        mock_struct.return_value = ParsedResponse(
            value=analysis, raw_text="{}", model="mock", attempts=1, success=True
        )
        mock_tavily.search.return_value = {
            "results": [
                {
                    "title": "Loop Source",
                    "url": "https://example.com/loop",
                    "content": "Loop info",
                }
            ]
        }
        mock_text.return_value = "Clarification after budget limit."

        start = time.perf_counter()
        resp = client.post(
            "/api/ducks/start", json={"initial_question": "Search topic"}
        )
        elapsed = time.perf_counter() - start

    assert resp.status_code == 200
    assert elapsed < 1.0, f"Research loop took unexpectedly long: {elapsed:.3f}s"
