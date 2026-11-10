import re
from unittest.mock import patch

from app.services.parsed_response import ParsedResponse


def test_api_does_not_leak_environment_secrets(client, make_duck_analysis):
    """Verify responses never include pattern of API keys (Groq / Tavily tokens)."""
    analysis = make_duck_analysis(
        analysis="Standard analysis.",
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
        mock_text.return_value = "Answer with no secrets."

        response = client.post(
            "/api/ducks/start", json={"initial_question": "Check security"}
        )

    assert response.status_code == 200
    body = response.text

    # Groq API keys start with gsk_
    # Tavily API keys start with tvly-
    assert not re.search(r"gsk_[a-zA-Z0-9]{20,}", body)
    assert not re.search(r"tvly-[a-zA-Z0-9]{20,}", body)


def test_error_response_does_not_leak_internal_tracebacks(client):
    """Verify that internal unexpected exceptions do not leak stack traces to client."""
    with patch("app.graph.nodes.ananlyze.call_structured_llm") as mock_struct:
        mock_struct.side_effect = RuntimeError(
            "Internal memory fault in service secret_token=abcdef"
        )

        response = client.post(
            "/api/ducks/start",
            json={"initial_question": "Trigger 500 error"},
        )

    assert response.status_code == 500
    body = response.text
    assert "Traceback" not in body
    assert 'File "' not in body
    assert "secret_token" not in body


def test_health_endpoint_does_not_reveal_sensitive_configuration(client):
    """Verify health endpoint returns only status and does not dump os.environ or config."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) == {"status"}
    assert data["status"] == "ok"
