"""Tests for graceful reasoning termination at maximum clarification limit."""

from datetime import UTC, datetime
from unittest.mock import patch
from uuid import uuid4

from app.graph.graph import build_graph
from app.graph.state import GraphState
from app.schemas.analysis import DuckAnalysis, DuckAnalysisContent
from app.schemas.request import DuckRequest
from app.services.parsed_response import ParsedResponse


def _mock_analysis_content(
    analysis="Analysis statement", uncertainties=None, needs_search=False
):
    return DuckAnalysisContent(
        analysis=analysis,
        assumptions=["Standard assumption"],
        uncertainties=uncertainties if uncertainties is not None else ["Uncertainty A"],
        needs_search=needs_search,
    )


# ---------------------------------------------------------------------------
# Focused Offline Tests
# ---------------------------------------------------------------------------


def test_1_normal_clarification_flow_below_limit(client):
    """1. Normal clarification flow below the limit generates a Socratic question."""
    mock_content = _mock_analysis_content(
        uncertainties=["What is the deployment environment?"]
    )

    with (
        patch("app.graph.nodes.ananlyze.call_structured_llm") as mock_struct,
        patch("app.graph.nodes.socratic.call_text_llm") as mock_text,
    ):
        mock_struct.return_value = ParsedResponse(
            value=mock_content, raw_text="{}", model="mock", attempts=1, success=True
        )
        mock_text.return_value = "What is the deployment environment?"

        resp = client.post(
            "/api/ducks/start", json={"initial_question": "How do I deploy this app?"}
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["socratic_question"] == "What is the deployment environment?"
    assert data["final_answer"] is None
    assert data["turn_count"] == 1


def test_2_final_permitted_socratic_question_generated(client):
    """2. The final permitted Socratic question (turn 3) is generated when turn_count is 2."""
    mock_content = _mock_analysis_content(uncertainties=["Remaining uncertainty 3"])

    with (
        patch("app.graph.nodes.ananlyze.call_structured_llm") as mock_struct,
        patch("app.graph.nodes.socratic.call_text_llm") as mock_text,
    ):
        mock_struct.return_value = ParsedResponse(
            value=mock_content, raw_text="{}", model="mock", attempts=1, success=True
        )
        mock_text.return_value = "What is your fallback strategy?"

        resp = client.post(
            "/api/ducks/reply",
            json={
                "initial_question": "How do I deploy?",
                "prev_socratic_q": "What is the cloud provider?",
                "user_reply": "We are deploying to AWS.",
                "analysis": {
                    "analysis_id": str(uuid4()),
                    "analysis_time": datetime.now(UTC).isoformat(),
                    "analysis": "Deploying to AWS.",
                    "assumptions": [],
                    "uncertainties": ["Need fallback strategy."],
                    "needs_search": False,
                },
                "turn_count": 2,
                "evidence": None,
            },
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["socratic_question"] == "What is your fallback strategy?"
    assert data["final_answer"] is None
    assert data["turn_count"] == 3


def test_3_and_4_user_reply_to_final_question_gracefully_terminates(client):
    """3 & 4. User reply to question 3 is processed, and routes directly to final answer without another Socratic question."""
    mock_content = _mock_analysis_content(
        analysis="AWS multi-region with Route53 DNS failover.",
        uncertainties=[
            "Still some minor cost uncertainties"
        ],  # Uncertainties remain, but limit reached!
        needs_search=False,
    )

    with (
        patch("app.graph.nodes.ananlyze.call_structured_llm") as mock_struct,
        patch("app.graph.nodes.socratic.call_text_llm") as mock_socratic,
        patch("app.graph.nodes.final_respond_node.call_text_llm") as mock_final,
    ):
        mock_struct.return_value = ParsedResponse(
            value=mock_content, raw_text="{}", model="mock", attempts=1, success=True
        )
        mock_final.return_value = "Recommended deployment plan: use AWS ECS with multi-region Route53 failover."

        resp = client.post(
            "/api/ducks/reply",
            json={
                "initial_question": "How do I deploy?",
                "prev_socratic_q": "What is your fallback strategy?",
                "user_reply": "Our fallback is multi-region DNS failover.",
                "analysis": {
                    "analysis_id": str(uuid4()),
                    "analysis_time": datetime.now(UTC).isoformat(),
                    "analysis": "Prior analysis text",
                    "assumptions": [],
                    "uncertainties": ["Fallback strategy"],
                    "needs_search": False,
                },
                "turn_count": 3,
                "evidence": None,
            },
        )

    assert resp.status_code == 200
    data = resp.json()
    # Socratic node must NOT have been called
    mock_socratic.assert_not_called()
    # Final answer must be returned
    assert (
        data["final_answer"]
        == "Recommended deployment plan: use AWS ECS with multi-region Route53 failover."
    )
    assert data["socratic_question"] is None
    assert data["turn_count"] == 4


def test_5_final_response_incorporates_last_clarification():
    """5. Final response prompt receives the latest analysis incorporating the user's last clarification."""
    graph = build_graph()
    mock_content = _mock_analysis_content(
        analysis="Incorporated user latest clarification: zero-downtime rolling deploys required.",
        uncertainties=["Minor edge case"],
        needs_search=False,
    )
    captured_final_prompts = []

    def mock_final_call(*, task_name, prompt):
        captured_final_prompts.append(prompt)
        return "Final comprehensive conclusion."

    state = GraphState(
        request=DuckRequest(
            req_id=uuid4(),
            req="How do I deploy with zero downtime?",
            req_time=datetime.now(UTC),
            status="socratic",
        ),
        last_user_reply="We require blue-green deployments with health check gating.",
        analysis=DuckAnalysis(
            analysis_id=uuid4(),
            analysis_time=datetime.now(UTC),
            analysis="Prior analysis",
            assumptions=[],
            uncertainties=["Deployment strategy"],
            needs_search=False,
        ),
        turn_count=3,
        clarification_count=3,
        max_clarification_turns=3,
        socratic_question="What deployment strategy do you need?",
    )

    with (
        patch("app.graph.nodes.ananlyze.call_structured_llm") as mock_struct,
        patch(
            "app.graph.nodes.final_respond_node.call_text_llm",
            side_effect=mock_final_call,
        ),
    ):
        mock_struct.return_value = ParsedResponse(
            value=mock_content, raw_text="{}", model="mock", attempts=1, success=True
        )
        result = graph.invoke(state)

    assert result["final_answer"] == "Final comprehensive conclusion."
    assert result["socratic_question"] is None
    assert len(captured_final_prompts) == 1
    final_prompt = captured_final_prompts[0]
    assert "zero-downtime rolling deploys required" in final_prompt


def test_6_evidence_preserved_during_graceful_termination(client):
    """6. Evidence collected is preserved during graceful termination at the turn limit."""
    mock_content = _mock_analysis_content(
        analysis="Evidence-supported analysis.",
        uncertainties=["Residual ambiguity"],
        needs_search=False,
    )
    evidence_payload = {
        "evidence_id": str(uuid4()),
        "search_id": str(uuid4()),
        "raw_results": [
            {
                "title": "AWS Architecture Guide",
                "url": "https://aws.amazon.com",
                "content": "Best practices",
            }
        ],
        "meta": {},
        "collected_at": datetime.now(UTC).isoformat(),
    }

    with (
        patch("app.graph.nodes.ananlyze.call_structured_llm") as mock_struct,
        patch("app.graph.nodes.final_respond_node.call_text_llm") as mock_final,
    ):
        mock_struct.return_value = ParsedResponse(
            value=mock_content, raw_text="{}", model="mock", attempts=1, success=True
        )
        mock_final.return_value = "Final answer citing AWS Architecture Guide."

        resp = client.post(
            "/api/ducks/reply",
            json={
                "initial_question": "How do I deploy?",
                "prev_socratic_q": "What is the cloud provider?",
                "user_reply": "AWS ECS.",
                "analysis": {
                    "analysis_id": str(uuid4()),
                    "analysis_time": datetime.now(UTC).isoformat(),
                    "analysis": "Analysis",
                    "assumptions": [],
                    "uncertainties": ["Ambiguity"],
                    "needs_search": False,
                },
                "turn_count": 3,
                "evidence": evidence_payload,
            },
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["final_answer"] is not None
    assert data["evidence"] is not None
    assert len(data["evidence"]["raw_results"]) == 1
    assert data["evidence"]["raw_results"][0]["title"] == "AWS Architecture Guide"


def test_7_internal_research_budget_remains_independent():
    """7. Reaching the clarification limit does not discard useful research if research budget permits."""
    graph = build_graph()
    # Analysis indicates research is needed
    content_needs_search = _mock_analysis_content(
        analysis="Factual research required on ECS Fargate limits.",
        uncertainties=["ECS Fargate memory limits"],
        needs_search=True,
    )
    content_post_search = _mock_analysis_content(
        analysis="Resolved memory limits via AWS documentation.",
        uncertainties=["Memory limits resolved"],
        needs_search=False,
    )

    state = GraphState(
        request=DuckRequest(
            req_id=uuid4(),
            req="What are ECS Fargate limits?",
            req_time=datetime.now(UTC),
            status="socratic",
        ),
        last_user_reply="We want serverless containers.",
        analysis=DuckAnalysis(
            analysis_id=uuid4(),
            analysis_time=datetime.now(UTC),
            analysis="Initial",
            assumptions=[],
            uncertainties=["Fargate limits"],
            needs_search=False,
        ),
        turn_count=3,
        clarification_count=3,
        max_clarification_turns=3,
        internal_research_count=0,
        max_research_iterations=3,
        socratic_question="Do you prefer EC2 or Fargate?",
    )

    mock_struct_responses = [
        ParsedResponse(
            value=content_needs_search,
            raw_text="{}",
            model="mock",
            attempts=1,
            success=True,
        ),
        ParsedResponse(
            value=content_post_search,
            raw_text="{}",
            model="mock",
            attempts=1,
            success=True,
        ),
    ]

    with (
        patch(
            "app.graph.nodes.ananlyze.call_structured_llm",
            side_effect=mock_struct_responses,
        ),
        patch("app.graph.nodes.search_node.tavily_client") as mock_tavily,
        patch("app.graph.nodes.final_respond_node.call_text_llm") as mock_final,
    ):
        mock_tavily.search.return_value = {
            "results": [
                {
                    "title": "Fargate Limits",
                    "url": "http://aws.com",
                    "content": "Max 30GB",
                }
            ]
        }
        mock_final.return_value = "Fargate supports up to 30GB RAM."

        result = graph.invoke(state)

    # Search occurred because budget permitted
    assert result["internal_research_count"] == 1
    # Terminated at final answer because clarification limit was reached
    assert result["final_answer"] == "Fargate supports up to 30GB RAM."
    assert result["socratic_question"] is None


def test_8_completed_session_cannot_continue_and_handles_stale_replies(client):
    """8. A completed session cannot accidentally continue, and replies without pending clarification return 400."""
    # Case A: Reply sent with turn_count > MAX_CLARIFICATION_TURNS (session already completed)
    resp1 = client.post(
        "/api/ducks/reply",
        json={
            "initial_question": "Question",
            "prev_socratic_q": "Old Q?",
            "user_reply": "Reply after completion",
            "analysis": {
                "analysis_id": str(uuid4()),
                "analysis_time": datetime.now(UTC).isoformat(),
                "analysis": "Analysis",
                "assumptions": [],
                "uncertainties": [],
                "needs_search": False,
            },
            "turn_count": 4,  # > MAX_CLARIFICATION_TURNS (3)
            "evidence": None,
        },
    )
    assert resp1.status_code == 400
    assert "Session has already concluded" in resp1.json()["detail"]

    # Case B: Reply sent with no prev_socratic_q (no clarification pending)
    resp2 = client.post(
        "/api/ducks/reply",
        json={
            "initial_question": "Question",
            "prev_socratic_q": "   ",
            "user_reply": "Reply without pending question",
            "analysis": {
                "analysis_id": str(uuid4()),
                "analysis_time": datetime.now(UTC).isoformat(),
                "analysis": "Analysis",
                "assumptions": [],
                "uncertainties": [],
                "needs_search": False,
            },
            "turn_count": 1,
            "evidence": None,
        },
    )
    assert resp2.status_code == 400
    assert "No clarification is currently pending" in resp2.json()["detail"]


def test_9_direct_answer_requests_still_work_correctly(client):
    """9. Direct-answer requests still work correctly and return final_answer immediately."""
    mock_content = _mock_analysis_content(
        analysis="Direct arithmetic calculation.",
        uncertainties=[],  # No uncertainties!
        needs_search=False,
    )

    with (
        patch("app.graph.nodes.ananlyze.call_structured_llm") as mock_struct,
        patch("app.graph.nodes.final_respond_node.call_text_llm") as mock_final,
    ):
        mock_struct.return_value = ParsedResponse(
            value=mock_content, raw_text="{}", model="mock", attempts=1, success=True
        )
        mock_final.return_value = "2 + 2 equals 4."

        resp = client.post(
            "/api/ducks/start", json={"initial_question": "What is 2 + 2?"}
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["final_answer"] == "2 + 2 equals 4."
    assert data["socratic_question"] is None
    assert data["turn_count"] == 1
