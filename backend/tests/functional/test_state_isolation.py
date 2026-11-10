from datetime import UTC, datetime
from unittest.mock import patch
from uuid import uuid4

from app.services.parsed_response import ParsedResponse


def test_independent_requests_state_isolation(client, make_duck_analysis):
    """Verify concurrent or sequential requests have isolated graph states."""
    analysis1 = make_duck_analysis(
        analysis="Analysis for request 1",
        uncertainties=["Question 1 uncertainty"],
        needs_search=False,
    )
    analysis2 = make_duck_analysis(
        analysis="Analysis for request 2",
        uncertainties=["Question 2 uncertainty"],
        needs_search=False,
    )

    with (
        patch("app.graph.nodes.ananlyze.call_structured_llm") as mock_struct,
        patch("app.graph.nodes.socratic.call_text_llm") as mock_text,
    ):
        # First request
        mock_struct.return_value = ParsedResponse(
            value=analysis1, raw_text="{}", model="mock", attempts=1, success=True
        )
        mock_text.return_value = "Socratic Q1?"
        resp1 = client.post(
            "/api/ducks/start", json={"initial_question": "First unique topic"}
        )

        # Second request
        mock_struct.return_value = ParsedResponse(
            value=analysis2, raw_text="{}", model="mock", attempts=1, success=True
        )
        mock_text.return_value = "Socratic Q2?"
        resp2 = client.post(
            "/api/ducks/start", json={"initial_question": "Second unique topic"}
        )

    data1 = resp1.json()
    data2 = resp2.json()

    assert data1["socratic_question"] == "Socratic Q1?"
    assert data2["socratic_question"] == "Socratic Q2?"
    assert data1["analysis"]["analysis"] != data2["analysis"]["analysis"]


def test_evidence_not_leaked_across_requests(client, make_duck_analysis):
    """Verify evidence from a request that searched does not bleed into subsequent request."""
    analysis_with_search = make_duck_analysis(
        analysis="Needs research",
        uncertainties=["Fact Alpha"],
        needs_search=True,
    )
    analysis_post_search = make_duck_analysis(
        analysis="Researched Alpha",
        uncertainties=[],
        needs_search=False,
    )
    analysis_direct = make_duck_analysis(
        analysis="Direct answer without search",
        uncertainties=[],
        needs_search=False,
    )

    with (
        patch("app.graph.nodes.ananlyze.call_structured_llm") as mock_struct,
        patch("app.graph.nodes.search_node.tavily_client") as mock_tavily,
        patch("app.graph.nodes.final_respond_node.call_text_llm") as mock_text,
    ):
        # Request 1 triggers search
        mock_struct.side_effect = [
            ParsedResponse(
                value=analysis_with_search,
                raw_text="{}",
                model="mock",
                attempts=1,
                success=True,
            ),
            ParsedResponse(
                value=analysis_post_search,
                raw_text="{}",
                model="mock",
                attempts=1,
                success=True,
            ),
        ]
        mock_tavily.search.return_value = {
            "results": [
                {
                    "title": "Alpha Source",
                    "url": "https://alpha.com",
                    "content": "Alpha info",
                }
            ]
        }
        mock_text.return_value = "Final Answer 1."

        payload1 = {
            "initial_question": "Search topic",
            "prev_socratic_q": "prev",
            "user_reply": "reply",
            "analysis": {
                "analysis_id": str(uuid4()),
                "analysis_time": datetime.now(UTC).isoformat(),
                "analysis": "Initial",
                "assumptions": [],
                "uncertainties": ["Fact Alpha"],
                "needs_search": True,
            },
            "turn_count": 1,
            "evidence": None,
        }
        resp1 = client.post("/api/ducks/reply", json=payload1)

        # Request 2 does NOT search
        mock_struct.side_effect = [
            ParsedResponse(
                value=analysis_direct,
                raw_text="{}",
                model="mock",
                attempts=1,
                success=True,
            ),
        ]
        mock_text.return_value = "Final Answer 2."

        payload2 = {
            "initial_question": "Non search topic",
            "prev_socratic_q": "prev",
            "user_reply": "reply",
            "analysis": {
                "analysis_id": str(uuid4()),
                "analysis_time": datetime.now(UTC).isoformat(),
                "analysis": "Initial",
                "assumptions": [],
                "uncertainties": [],
                "needs_search": False,
            },
            "turn_count": 1,
            "evidence": None,
        }
        resp2 = client.post("/api/ducks/reply", json=payload2)

    data1 = resp1.json()
    data2 = resp2.json()

    assert data1["evidence"] is not None
    assert len(data1["evidence"]["raw_results"]) == 1
    # Second request must have no evidence leaked from the first
    assert data2["evidence"] is None


def test_independent_research_counter_initialization(make_duck_request):
    """Verify GraphState initializes research counters and query history independently."""
    from app.graph.state import GraphState

    req1 = make_duck_request("Topic 1")
    req2 = make_duck_request("Topic 2")

    state1 = GraphState(request=req1)
    state2 = GraphState(request=req2)

    state1.internal_research_count += 2
    state1.turn_count += 1
    state1.attempted_queries.append("evidence about topic 1")

    # state2 must remain untouched
    assert state2.internal_research_count == 0
    assert state2.turn_count == 0
    assert state2.attempted_queries == []
