"""Tests for Socratic question generation prompt, guidelines, and state handling."""

from datetime import UTC, datetime
from unittest.mock import patch
from uuid import uuid4

from app.graph.nodes.socratic import socratic_node
from app.graph.state import GraphState
from app.schemas.analysis import DuckAnalysis
from app.schemas.request import DuckRequest


def _make_state(
    analysis_text="The user assumes increasing test coverage directly reduces production defects.",
    assumptions=None,
    uncertainties=None,
    previous_socratic_q=None,
    turn_count=0,
):
    req = DuckRequest(
        req_id=uuid4(),
        req="Does higher test coverage guarantee better software?",
        req_time=datetime.now(UTC),
        status="socratic",
    )
    analysis = DuckAnalysis(
        analysis_id=uuid4(),
        analysis_time=datetime.now(UTC),
        analysis=analysis_text,
        assumptions=assumptions or ["Test suite tests meaningful behaviour."],
        uncertainties=uncertainties or ["Quality of assertions versus line count."],
        needs_search=False,
    )
    return GraphState(
        request=req,
        analysis=analysis,
        socratic_question=previous_socratic_q,
        turn_count=turn_count,
    )


def test_socratic_node_single_question_generated():
    """1. Exactly one question is generated and ends with a question mark."""
    state = _make_state()
    mock_q = "What happens if tests only assert trivial conditions?"

    with patch(
        "app.graph.nodes.socratic.call_text_llm", return_value=mock_q
    ) as mock_llm:
        res = socratic_node(state)

    assert res.socratic_question == mock_q
    assert res.socratic_question.count("?") == 1
    assert res.socratic_question.endswith("?")


def test_socratic_node_concise_and_understandable():
    """2. The question is concise, conversational, and within targeted length."""
    state = _make_state()
    mock_q = "Could a suite have 100% coverage and still miss critical bugs?"

    with patch("app.graph.nodes.socratic.call_text_llm", return_value=mock_q):
        res = socratic_node(state)

    words = res.socratic_question.split()
    assert 5 <= len(words) <= 30
    assert not any(
        jargon in res.socratic_question.lower()
        for jargon in ["epistemic", "operationalize", "mathematical fusion"]
    )


def test_socratic_node_prompt_grounded_in_analysis():
    """3. The prompt constructed by socratic_node embeds analysis, assumptions, and uncertainties."""
    captured_prompts = []

    def mock_call(*, task_name, prompt):
        captured_prompts.append(prompt)
        return "What makes you confident in this assumption?"

    state = _make_state(
        analysis_text="Critical analysis on microservice boundaries.",
        assumptions=["Network latency is negligible."],
        uncertainties=["Service discovery overhead."],
    )

    with patch("app.graph.nodes.socratic.call_text_llm", side_effect=mock_call):
        socratic_node(state)

    assert len(captured_prompts) == 1
    prompt = captured_prompts[0]
    assert "Critical analysis on microservice boundaries." in prompt
    assert "Network latency is negligible." in prompt
    assert "Service discovery overhead." in prompt


def test_socratic_node_prompt_avoids_irrelevant_technical_jargon():
    """4. The prompt explicitly instructs plain English and forbids dense jargon/interrogation."""
    captured_prompts = []

    with patch(
        "app.graph.nodes.socratic.call_text_llm",
        side_effect=lambda task_name, prompt: captured_prompts.append(prompt) or "Why?",
    ):
        socratic_node(_make_state())

    prompt = captured_prompts[0]
    assert "plain, conversational English" in prompt
    assert "avoid academic jargon" in prompt.lower()
    assert "do not interrogate" in prompt.lower()
    assert "exactly one question" in prompt.lower()


def test_socratic_node_does_not_repeat_previous_question():
    """5. When a prior Socratic question exists, the prompt instructs not to repeat or rephrase it."""
    captured_prompts = []
    prev_q = "What is the primary constraint holding back this design?"
    state = _make_state(previous_socratic_q=prev_q)

    with patch(
        "app.graph.nodes.socratic.call_text_llm",
        side_effect=lambda task_name, prompt: (
            captured_prompts.append(prompt) or "New question?"
        ),
    ):
        socratic_node(state)

    prompt = captured_prompts[0]
    assert "PREVIOUS QUESTION ASKED:" in prompt
    assert prev_q in prompt
    assert "Do NOT repeat or rephrase this question" in prompt


def test_socratic_node_state_handling_and_turn_increment():
    """6. The node strips extra quotes, ensures question mark, and increments turn count."""
    state = _make_state(turn_count=2)
    # LLM returns quotes and misses question mark
    mock_raw = '  "What could go wrong if the primary server fails"  '

    with patch("app.graph.nodes.socratic.call_text_llm", return_value=mock_raw):
        res = socratic_node(state)

    assert res.socratic_question == "What could go wrong if the primary server fails?"
    assert res.turn_count == 3
