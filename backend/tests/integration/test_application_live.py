import os

import pytest


@pytest.mark.live
def test_live_application_start_endpoint(client):
    """Live integration test: exercise actual FastAPI /api/ducks/start with real providers."""
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY is not configured in local environment")

    # A short, unambiguous arithmetic or basic question to keep token usage small
    response = client.post(
        "/api/ducks/start",
        json={"initial_question": "Is Python an interpreted language?"},
    )

    assert response.status_code == 200
    data = response.json()
    assert "analysis" in data
    assert "turn_count" in data
    # Should produce either a socratic question or direct analysis
    assert data["turn_count"] >= 1
    assert "analysis" in data["analysis"]
