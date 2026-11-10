from datetime import UTC, datetime
from unittest.mock import patch
from uuid import uuid4

from app.services.parsed_response import ParsedResponse


def test_health_check(client):
    """Verify health check endpoint returns 200 and expected schema."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_start_endpoint_valid_request(client, make_duck_analysis):
    """Verify POST /api/ducks/start accepts valid payload and returns StartResponse."""
    mock_analysis = make_duck_analysis(
        analysis="Initial breakdown",
        uncertainties=["Clarify scope"],
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
        mock_text.return_value = "What is the primary constraint?"

        response = client.post(
            "/api/ducks/start",
            json={"initial_question": "How should I structure my database?"},
        )

    assert response.status_code == 200
    data = response.json()
    assert "socratic_question" in data
    assert "final_answer" in data
    assert "analysis" in data
    assert "turn_count" in data
    assert "evidence" in data
    assert data["socratic_question"] == "What is the primary constraint?"
    assert data["final_answer"] is None
    assert data["analysis"]["analysis"] == "Initial breakdown"
    assert data["turn_count"] == 1


def test_start_endpoint_direct_answer_returns_null_socratic_question(
    client, make_duck_analysis
):
    """Verify POST /api/ducks/start returns non-empty final_answer and socratic_question: null when answered directly."""
    direct_analysis = make_duck_analysis(
        analysis="Direct convergence analysis",
        uncertainties=[],
        needs_search=False,
    )

    with (
        patch("app.graph.nodes.ananlyze.call_structured_llm") as mock_struct,
        patch("app.graph.nodes.final_respond_node.call_text_llm") as mock_text,
    ):
        mock_struct.return_value = ParsedResponse(
            value=direct_analysis,
            raw_text="{}",
            model="mock",
            attempts=1,
            success=True,
        )
        mock_text.return_value = "Direct final answer."

        response = client.post(
            "/api/ducks/start",
            json={"initial_question": "What is Python?"},
        )

    assert response.status_code == 200
    data = response.json()
    assert "final_answer" in data
    assert data["final_answer"] == "Direct final answer."
    assert data["socratic_question"] is None
    assert data["turn_count"] == 1


def test_start_endpoint_missing_required_fields(client):
    """Verify POST /api/ducks/start rejects missing initial_question."""
    response = client.post("/api/ducks/start", json={})
    assert response.status_code == 422
    errors = response.json().get("detail", [])
    assert any("initial_question" in str(err) for err in errors)


def test_start_endpoint_empty_initial_question(client):
    """Verify POST /api/ducks/start rejects empty string due to min_length=1."""
    response = client.post("/api/ducks/start", json={"initial_question": ""})
    assert response.status_code == 422


def test_start_endpoint_invalid_type(client):
    """Verify POST /api/ducks/start rejects non-string types for initial_question."""
    response = client.post("/api/ducks/start", json={"initial_question": 12345})
    assert response.status_code == 422


def test_reply_endpoint_valid_request(client, make_duck_analysis):
    """Verify POST /api/ducks/reply accepts valid payload and returns ReplyResponse."""
    analysis = make_duck_analysis(
        analysis="Post-reply analysis",
        uncertainties=[],
        needs_search=False,
    )

    with (
        patch("app.graph.nodes.ananlyze.call_structured_llm") as mock_struct,
        patch("app.graph.nodes.final_respond_node.call_text_llm") as mock_text,
    ):
        mock_struct.return_value = ParsedResponse(
            value=analysis,
            raw_text="{}",
            model="mock",
            attempts=1,
            success=True,
        )
        mock_text.return_value = "Here is the recommended database structure."

        payload = {
            "initial_question": "How should I structure my database?",
            "prev_socratic_q": "What is the primary constraint?",
            "user_reply": "Read-heavy workload with low write frequency.",
            "analysis": {
                "analysis_id": str(uuid4()),
                "analysis_time": datetime.now(UTC).isoformat(),
                "analysis": "Initial analysis",
                "assumptions": ["PostgreSQL is used"],
                "uncertainties": ["Workload pattern"],
                "needs_search": False,
            },
            "turn_count": 1,
            "evidence": None,
        }

        response = client.post("/api/ducks/reply", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "final_answer" in data
    assert data["final_answer"] == "Here is the recommended database structure."
    assert "analysis" in data
    assert "turn_count" in data
    assert data["turn_count"] == 2


def test_reply_endpoint_missing_fields(client):
    """Verify POST /api/ducks/reply rejects missing fields."""
    response = client.post(
        "/api/ducks/reply",
        json={"initial_question": "How do I scale?"},
    )
    assert response.status_code == 422


def test_reply_endpoint_turn_limit_exceeded(client):
    """Verify POST /api/ducks/reply returns HTTP 400 when turn_count exceeds max clarification limit."""
    payload = {
        "initial_question": "How should I scale?",
        "prev_socratic_q": "What is the budget?",
        "user_reply": "Under $1000/mo",
        "analysis": {
            "analysis_id": str(uuid4()),
            "analysis_time": datetime.now(UTC).isoformat(),
            "analysis": "Analysis",
            "assumptions": [],
            "uncertainties": [],
            "needs_search": False,
        },
        "turn_count": 5,
        "evidence": None,
    }
    response = client.post("/api/ducks/reply", json=payload)
    assert response.status_code == 400
    assert "Session has already concluded" in response.json().get("detail", "")


def test_unexpected_error_does_not_expose_traceback(client):
    """Verify server exceptions do not leak stack trace in response body."""
    with patch(
        "app.graph.nodes.ananlyze.call_structured_llm",
        side_effect=Exception("Database connection secret_pw=123 failed!"),
    ):
        response = client.post(
            "/api/ducks/start",
            json={"initial_question": "Trigger error"},
        )
    assert response.status_code == 500
    body = response.text
    # Traceback should not be returned in API response body
    assert "Traceback (most recent call last)" not in body
    assert "secret_pw=123" not in body
