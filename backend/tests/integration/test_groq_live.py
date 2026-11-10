import os

import pytest
from app.services.llm_helper import call_structured_llm, call_text_llm
from app.services.model_registery import LIGHT_WEIGHT_MODEL, REASONING_MODEL
from pydantic import BaseModel, Field


class LiveCheckSchema(BaseModel):
    message: str = Field(..., description="A short greeting or confirmation message.")
    is_working: bool


@pytest.mark.live
def test_live_structured_llm_call():
    """Live integration test: verifies call_structured_llm against real Groq endpoint."""
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY is not configured in local environment")

    prompt = (
        "Return a JSON object conforming to LiveCheckSchema with message='RubberDuck live check' and is_working=true. "
        "Return ONLY valid JSON."
    )
    result = call_structured_llm(
        task_name="live_structured_check",
        prompt=prompt,
        schema=LiveCheckSchema,
        model=REASONING_MODEL,
        max_tokens=150,
        max_retries=2,
    )

    assert result.success is True
    assert result.value is not None
    assert result.value.is_working is True
    assert isinstance(result.value.message, str)


@pytest.mark.live
def test_live_text_llm_call():
    """Live integration test: verifies call_text_llm against real Groq endpoint."""
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY is not configured in local environment")

    prompt = "Respond with exactly the single word: PONG"
    result = call_text_llm(
        task_name="live_text_check",
        prompt=prompt,
        model=LIGHT_WEIGHT_MODEL,
        max_tokens=20,
    )

    assert isinstance(result, str)
    assert len(result) > 0
    assert "PONG" in result.upper()
