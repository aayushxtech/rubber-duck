import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from textual.widgets import Input

# Ensure cli/ is on sys.path for importing cli modules
sys.path.insert(0, os.path.abspath("cli"))

from client import (
    BackendAPIError,
    BackendConnectionError,
    RubberDuckClient,
    RubberDuckError,
)
from main import RubberDuckApp
from session import Session

# ---------------------------------------------------------------------------
# 1. Session Unit Tests
# ---------------------------------------------------------------------------


def test_session_initialization():
    session = Session()
    assert session.is_active is False
    assert session.is_completed is False
    assert session.is_awaiting_clarification is False
    assert session.turn_count == 0


def test_session_start_new():
    session = Session()
    session.start_new("What is quantum computing?")
    assert session.is_active is True
    assert session.is_completed is False
    assert session.is_awaiting_clarification is False
    assert session.initial_question == "What is quantum computing?"
    assert session.turn_count == 0


def test_session_update_socratic_question():
    session = Session()
    session.start_new("What is quantum computing?")
    data = {
        "socratic_question": "What specifically do you mean by computing?",
        "final_answer": None,
        "analysis": {
            "analysis": "User asks abstract question",
            "assumptions": [],
            "uncertainties": [],
        },
        "turn_count": 1,
        "evidence": None,
    }
    session.update_from_response(data)
    assert session.is_active is True
    assert session.is_completed is False
    assert session.is_awaiting_clarification is True
    assert (
        session.latest_socratic_question
        == "What specifically do you mean by computing?"
    )
    assert session.turn_count == 1


def test_session_update_final_answer():
    session = Session()
    session.start_new("What is 2+2?")
    data = {
        "socratic_question": None,
        "final_answer": "4",
        "analysis": {"analysis": "Simple arithmetic"},
        "turn_count": 1,
        "evidence": None,
    }
    session.update_from_response(data)
    assert session.is_active is True
    assert session.is_completed is True
    assert session.is_awaiting_clarification is False
    assert session.latest_final_answer == "4"


def test_session_reset():
    session = Session()
    session.start_new("Question")
    session.reset()
    assert session.is_active is False
    assert session.initial_question is None
    assert session.turn_count == 0


# ---------------------------------------------------------------------------
# 2. RubberDuckClient HTTP Tests (Mocked)
# ---------------------------------------------------------------------------


def test_client_check_health_success():
    async def _run():
        client = RubberDuckClient("http://mock-backend:8000")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch(
            "httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp
        ):
            healthy = await client.check_health()
            assert healthy is True

    asyncio.run(_run())


def test_client_check_health_failure():
    async def _run():
        client = RubberDuckClient("http://mock-backend:8000")
        with patch(
            "httpx.AsyncClient.get", side_effect=Exception("Connection refused")
        ):
            healthy = await client.check_health()
            assert healthy is False

    asyncio.run(_run())


def test_client_start_reasoning_success():
    async def _run():
        client = RubberDuckClient("http://mock-backend:8000")
        expected_response = {
            "socratic_question": "Why?",
            "final_answer": None,
            "analysis": {"analysis": "Test analysis"},
            "turn_count": 1,
            "evidence": None,
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = expected_response

        with patch(
            "httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp
        ) as mock_post:
            res = await client.start_reasoning("Test question")
            assert res == expected_response
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args.kwargs
            assert call_kwargs["json"] == {"initial_question": "Test question"}

    asyncio.run(_run())


def test_client_start_reasoning_connection_error():
    async def _run():
        client = RubberDuckClient("http://mock-backend:8000")
        import httpx

        with patch(
            "httpx.AsyncClient.post",
            side_effect=httpx.ConnectError("Connection refused"),
        ):
            with pytest.raises(BackendConnectionError) as exc_info:
                await client.start_reasoning("Test question")
            assert "Cannot connect to RubberDuck backend" in str(exc_info.value)

    asyncio.run(_run())


def test_client_start_reasoning_api_error():
    async def _run():
        client = RubberDuckClient("http://mock-backend:8000")
        mock_resp = MagicMock()
        mock_resp.status_code = 422
        mock_resp.json.return_value = {"detail": "Validation error"}

        with patch(
            "httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp
        ):
            with pytest.raises(BackendAPIError) as exc_info:
                await client.start_reasoning("   ")
            assert exc_info.value.status_code == 422
            assert "Validation error" in exc_info.value.detail

    asyncio.run(_run())


def test_client_reply_reasoning_success():
    async def _run():
        client = RubberDuckClient("http://mock-backend:8000")
        session = Session()
        session.start_new("Original question")
        session.update_from_response(
            {
                "socratic_question": "Clarify X?",
                "final_answer": None,
                "analysis": {"analysis": "Prior analysis"},
                "turn_count": 1,
                "evidence": None,
            }
        )

        expected_reply_response = {
            "socratic_question": None,
            "final_answer": "Final conclusion reached",
            "analysis": {"analysis": "Updated analysis"},
            "turn_count": 2,
            "evidence": None,
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = expected_reply_response

        with patch(
            "httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp
        ) as mock_post:
            res = await client.reply_reasoning(session, "Here is my clarification on X")
            assert res == expected_reply_response
            call_kwargs = mock_post.call_args.kwargs
            payload = call_kwargs["json"]
            assert payload["initial_question"] == "Original question"
            assert payload["prev_socratic_q"] == "Clarify X?"
            assert payload["user_reply"] == "Here is my clarification on X"
            assert payload["analysis"] == {"analysis": "Prior analysis"}
            assert payload["turn_count"] == 1

    asyncio.run(_run())


def test_client_reply_reasoning_validation():
    async def _run():
        client = RubberDuckClient("http://mock-backend:8000")
        session = Session()
        with pytest.raises(
            RubberDuckError, match="Cannot reply without an active session"
        ):
            await client.reply_reasoning(session, "reply")

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 3. RubberDuckApp Textual UI Headless Tests
# ---------------------------------------------------------------------------


def test_textual_app_mount_and_commands():
    async def _run():
        app = RubberDuckApp(base_url="http://mock-backend:8000")
        with patch.object(
            app.client, "check_health", new_callable=AsyncMock, return_value=True
        ):
            async with app.run_test() as pilot:
                # Check widgets exist
                assert app.query_one("#header_bar") is not None
                assert app.query_one("#conversation") is not None
                assert app.query_one("#user_input") is not None

                inp = app.query_one("#user_input", Input)

                # Test /help
                app.post_message(Input.Submitted(inp, value="/help"))
                await pilot.pause()

                # Test /status
                app.post_message(Input.Submitted(inp, value="/status"))
                await pilot.pause()

                # Test /clear
                app.post_message(Input.Submitted(inp, value="/clear"))
                await pilot.pause()

    asyncio.run(_run())


def test_textual_app_ask_and_reply_flow():
    async def _run():
        app = RubberDuckApp(base_url="http://mock-backend:8000")
        start_payload = {
            "socratic_question": "What is the premise?",
            "final_answer": None,
            "analysis": {
                "analysis": "Initial analysis text",
                "assumptions": ["Assumption A"],
                "uncertainties": ["Uncertainty B"],
                "needs_search": False,
            },
            "turn_count": 1,
            "evidence": None,
        }
        reply_payload = {
            "socratic_question": None,
            "final_answer": "Final resolved answer",
            "analysis": {
                "analysis": "Completed analysis",
                "assumptions": ["Assumption A"],
                "uncertainties": [],
                "needs_search": False,
            },
            "turn_count": 2,
            "evidence": {
                "raw_results": [
                    {
                        "title": "Source 1",
                        "url": "https://example.com",
                        "content": "Snippet 1",
                    }
                ]
            },
        }

        with (
            patch.object(
                app.client, "check_health", new_callable=AsyncMock, return_value=True
            ),
            patch.object(
                app.client,
                "start_reasoning",
                new_callable=AsyncMock,
                return_value=start_payload,
            ),
            patch.object(
                app.client,
                "reply_reasoning",
                new_callable=AsyncMock,
                return_value=reply_payload,
            ),
        ):
            async with app.run_test() as pilot:
                inp = app.query_one("#user_input", Input)

                # Ask question
                app.post_message(
                    Input.Submitted(inp, value="/ask How does gravity work?")
                )
                await pilot.pause()
                await app.workers.wait_for_complete()
                await pilot.pause()

                assert app.session.is_awaiting_clarification is True
                assert app.session.latest_socratic_question == "What is the premise?"

                # Test /analysis command
                app.post_message(Input.Submitted(inp, value="/analysis"))
                await pilot.pause()

                # Reply to Socratic inquiry via plain text input
                app.post_message(
                    Input.Submitted(
                        inp,
                        value="The premise is general relativity and spacetime curvature.",
                    )
                )
                await pilot.pause()
                await app.workers.wait_for_complete()
                await pilot.pause()

                assert app.session.is_completed is True
                assert app.session.latest_final_answer == "Final resolved answer"

                # Test /evidence command
                app.post_message(Input.Submitted(inp, value="/evidence"))
                await pilot.pause()

    asyncio.run(_run())
