import json
from unittest.mock import MagicMock, patch

import groq
import pytest
from app.services.llm_helper import (
    MissingAPIKeyError,
    call_structured_llm,
    call_text_llm,
    get_groq_client,
)
from pydantic import BaseModel, Field


class DummySchema(BaseModel):
    name: str = Field(..., min_length=1)
    score: int


def test_lazy_groq_client_init(monkeypatch):
    """Verify Groq client is lazily created and raises MissingAPIKeyError when key is unset."""
    # Reset singleton client
    import app.services.llm_helper as helper

    monkeypatch.setattr(helper, "_groq_client", None)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with patch("app.services.llm_helper.load_dotenv"):
        with pytest.raises(MissingAPIKeyError) as exc_info:
            get_groq_client()
        assert "GROQ_API_KEY is not set" in str(exc_info.value)


def test_structured_llm_successful_parse():
    """Verify successful JSON parsing and Pydantic validation on attempt 1."""
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({"name": "RubberDuck", "score": 100})
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_completion

    with patch("app.services.llm_helper.get_groq_client", return_value=mock_client):
        res = call_structured_llm(
            task_name="test_success",
            prompt="Return dummy schema",
            schema=DummySchema,
            max_retries=3,
        )

    assert res.success is True
    assert res.attempts == 1
    assert res.value.name == "RubberDuck"
    assert res.value.score == 100
    assert res.errors is None


def test_structured_llm_json_repair_mechanism():
    """Verify malformed JSON triggers repair prompt and succeeds on attempt 2."""
    mock_client = MagicMock()

    # Attempt 1: invalid JSON; Attempt 2: valid JSON
    choice_bad = MagicMock()
    choice_bad.message.content = "Not a json string"
    choice_good = MagicMock()
    choice_good.message.content = json.dumps({"name": "FixedDuck", "score": 95})

    comp1 = MagicMock()
    comp1.choices = [choice_bad]
    comp2 = MagicMock()
    comp2.choices = [choice_good]

    mock_client.chat.completions.create.side_effect = [comp1, comp2]

    with patch("app.services.llm_helper.get_groq_client", return_value=mock_client):
        res = call_structured_llm(
            task_name="test_repair",
            prompt="Original prompt",
            schema=DummySchema,
            max_retries=3,
        )

    assert res.success is True
    assert res.attempts == 2
    assert res.value.name == "FixedDuck"
    # Verify second prompt was a repair prompt
    second_call_prompt = mock_client.chat.completions.create.call_args_list[1][1][
        "messages"
    ][0]["content"]
    assert "ATTEMPT 1 OF 3" in second_call_prompt
    assert "JSON decode error" in second_call_prompt


def test_structured_llm_validation_error_repair():
    """Verify schema validation error triggers repair prompt."""
    mock_client = MagicMock()

    # Attempt 1: schema missing required field 'score'; Attempt 2: valid
    choice_bad = MagicMock()
    choice_bad.message.content = json.dumps({"name": "Incomplete"})
    choice_good = MagicMock()
    choice_good.message.content = json.dumps({"name": "Complete", "score": 50})

    comp1 = MagicMock()
    comp1.choices = [choice_bad]
    comp2 = MagicMock()
    comp2.choices = [choice_good]

    mock_client.chat.completions.create.side_effect = [comp1, comp2]

    with patch("app.services.llm_helper.get_groq_client", return_value=mock_client):
        res = call_structured_llm(
            task_name="test_validation_repair",
            prompt="Original prompt",
            schema=DummySchema,
            max_retries=3,
        )

    assert res.success is True
    assert res.attempts == 2
    assert res.value.score == 50


def test_structured_llm_permanent_provider_error_stops_immediately():
    """Verify permanent errors (e.g. 401 AuthenticationError) stop retries on attempt 1."""
    mock_client = MagicMock()
    # Mock a 401 AuthenticationError
    req = MagicMock()
    resp = MagicMock()
    resp.status_code = 401
    auth_error = groq.AuthenticationError("Invalid API key", response=resp, body=None)

    mock_client.chat.completions.create.side_effect = auth_error

    with patch("app.services.llm_helper.get_groq_client", return_value=mock_client):
        res = call_structured_llm(
            task_name="test_auth_fail",
            prompt="prompt",
            schema=DummySchema,
            max_retries=5,
        )

    assert res.success is False
    assert res.attempts == 1  # Must not retry permanent error
    assert any(
        "Permanent LLM call error: AuthenticationError" in err for err in res.errors
    )


def test_structured_llm_transient_provider_error_retries_with_backoff():
    """Verify transient error (e.g. APITimeoutError) retries and succeeds."""
    mock_client = MagicMock()
    req = MagicMock()
    timeout_err = groq.APITimeoutError(request=req)

    choice_good = MagicMock()
    choice_good.message.content = json.dumps({"name": "Recovered", "score": 88})
    comp_good = MagicMock()
    comp_good.choices = [choice_good]

    mock_client.chat.completions.create.side_effect = [timeout_err, comp_good]

    with (
        patch("app.services.llm_helper.get_groq_client", return_value=mock_client),
        patch("time.sleep") as mock_sleep,
    ):
        res = call_structured_llm(
            task_name="test_transient_retry",
            prompt="prompt",
            schema=DummySchema,
            max_retries=3,
        )

    assert res.success is True
    assert res.attempts == 2
    assert res.value.name == "Recovered"
    mock_sleep.assert_called_once()


def test_structured_llm_empty_response_handling():
    """Verify empty model response is rejected."""
    mock_client = MagicMock()
    choice_empty = MagicMock()
    choice_empty.message.content = None
    comp = MagicMock()
    comp.choices = [choice_empty]

    mock_client.chat.completions.create.return_value = comp

    with patch("app.services.llm_helper.get_groq_client", return_value=mock_client):
        res = call_structured_llm(
            task_name="test_empty",
            prompt="prompt",
            schema=DummySchema,
            max_retries=2,
        )

    assert res.success is False
    assert any("ValueError" in err for err in res.errors)


def test_text_llm_success():
    """Verify call_text_llm returns stripped string response."""
    mock_client = MagicMock()
    choice = MagicMock()
    choice.message.content = "   This is the final answer.   "
    comp = MagicMock()
    comp.choices = [choice]
    mock_client.chat.completions.create.return_value = comp

    with patch("app.services.llm_helper.get_groq_client", return_value=mock_client):
        ans = call_text_llm(task_name="test_text", prompt="prompt")

    assert ans == "This is the final answer."


def test_text_llm_permanent_error_raises_immediately():
    """Verify call_text_llm raises on permanent error without retrying."""
    mock_client = MagicMock()
    resp = MagicMock()
    resp.status_code = 404
    not_found = groq.NotFoundError("Model not found", response=resp, body=None)
    mock_client.chat.completions.create.side_effect = not_found

    with patch("app.services.llm_helper.get_groq_client", return_value=mock_client):
        with pytest.raises(RuntimeError) as exc_info:
            call_text_llm(task_name="test_text_perm", prompt="prompt")

        assert "LLM call failed: NotFoundError" in str(exc_info.value)
        assert mock_client.chat.completions.create.call_count == 1


def test_text_llm_transient_retry():
    """Verify call_text_llm retries on transient connection error."""
    mock_client = MagicMock()
    req = MagicMock()
    conn_err = groq.APIConnectionError(request=req)

    choice_good = MagicMock()
    choice_good.message.content = "Success text"
    comp_good = MagicMock()
    comp_good.choices = [choice_good]

    mock_client.chat.completions.create.side_effect = [conn_err, comp_good]

    with (
        patch("app.services.llm_helper.get_groq_client", return_value=mock_client),
        patch("time.sleep"),
    ):
        text = call_text_llm(task_name="test_text_retry", prompt="prompt")

    assert text == "Success text"
    assert mock_client.chat.completions.create.call_count == 2
