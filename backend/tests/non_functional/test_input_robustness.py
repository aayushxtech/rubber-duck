from unittest.mock import patch

from app.services.parsed_response import ParsedResponse


def test_empty_string_input_rejected(client):
    """Verify empty string is rejected by request validation schema."""
    resp = client.post("/api/ducks/start", json={"initial_question": ""})
    assert resp.status_code == 422


def test_whitespace_only_input(client, make_duck_analysis):
    """Verify whitespace-only input is accepted by schema or handled cleanly without crashing."""
    analysis = make_duck_analysis(
        analysis="Empty content detected",
        uncertainties=["Clarify question"],
        needs_search=False,
    )

    with (
        patch("app.graph.nodes.ananlyze.call_structured_llm") as mock_struct,
        patch("app.graph.nodes.socratic.call_text_llm") as mock_text,
    ):
        mock_struct.return_value = ParsedResponse(
            value=analysis, raw_text="{}", model="mock", attempts=1, success=True
        )
        mock_text.return_value = "Can you state your question?"

        resp = client.post("/api/ducks/start", json={"initial_question": "   \n\t  "})

    # Should either validate (if min_length counts whitespace) or cleanly return 200/422 without 500 crash
    assert resp.status_code in (200, 422)


def test_very_long_input_handled_without_crash(client, make_duck_analysis):
    """Verify long prompt (e.g. 10,000 chars) does not crash the server or raise 500."""
    long_question = "Explain this topic: " + ("word " * 2000)
    analysis = make_duck_analysis(
        analysis="Long input processed",
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
        mock_text.return_value = "Summary of long input."

        resp = client.post("/api/ducks/start", json={"initial_question": long_question})

    assert resp.status_code == 200
    assert resp.json()["analysis"]["analysis"] == "Long input processed"


def test_malformed_json_body_returns_422(client):
    """Verify syntactically invalid JSON payload returns 422 Unprocessable Entity."""
    resp = client.post(
        "/api/ducks/start",
        content='{"initial_question": invalid_json',
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422


def test_unexpected_input_types_rejected(client):
    """Verify passing array or boolean for initial_question is rejected with 422."""
    resp1 = client.post(
        "/api/ducks/start", json={"initial_question": ["item1", "item2"]}
    )
    assert resp1.status_code == 422

    resp2 = client.post("/api/ducks/start", json={"initial_question": True})
    assert resp2.status_code == 422


def test_unicode_and_emojis_accepted(client, make_duck_analysis):
    """Verify unusual but valid Unicode (multilingual, emojis, special characters) is handled cleanly."""
    unicode_question = "🦆 Rubber Duck: こんにちは世界! هل يعمل الذكاء الاصطناعي؟ 🚀✨"
    analysis = make_duck_analysis(
        analysis="Multilingual query processed",
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
        mock_text.return_value = "Unicode answer: 鴨 🦆"

        resp = client.post(
            "/api/ducks/start", json={"initial_question": unicode_question}
        )

    assert resp.status_code == 200
    assert resp.json()["socratic_question"] is None
    assert resp.json()["final_answer"] == "Unicode answer: 鴨 🦆"
    assert resp.json()["turn_count"] == 1
