import os
from typing import Any

import httpx

try:
    from session import Session
except ImportError:
    from cli.session import Session


class RubberDuckError(Exception):
    """Base exception for CLI client errors."""



class BackendConnectionError(RubberDuckError):
    """Raised when unable to reach the FastAPI backend."""



class BackendAPIError(RubberDuckError):
    """Raised when backend returns a non-2xx status code."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"API Error {status_code}: {detail}")


class RubberDuckClient:
    """HTTP client communicating with the RubberDuck FastAPI backend."""

    def __init__(self, base_url: str | None = None, timeout: float = 60.0):
        self.base_url = (
            base_url or os.getenv("RUBBERDUCK_BACKEND_URL", "http://localhost:8000")
        ).rstrip("/")
        self.timeout = timeout

    async def check_health(self) -> bool:
        """Check if backend is reachable and healthy."""
        url = f"{self.base_url}/health"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url)
                return resp.status_code == 200
        except Exception:
            return False

    async def start_reasoning(self, question: str) -> dict[str, Any]:
        """Send initial question to POST /api/ducks/start."""
        url = f"{self.base_url}/api/ducks/start"
        payload = {"initial_question": question.strip()}

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(url, json=payload)
        except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError) as e:
            raise BackendConnectionError(
                f"Cannot connect to RubberDuck backend at {self.base_url}. Is it running?"
            ) from e
        except Exception as e:
            raise BackendConnectionError(f"Unexpected connection failure: {e}") from e

        if resp.status_code != 200:
            detail = self._extract_error_detail(resp)
            raise BackendAPIError(resp.status_code, detail)

        return resp.json()

    async def reply_reasoning(
        self, session: Session, user_reply: str
    ) -> dict[str, Any]:
        """Send user clarification to POST /api/ducks/reply."""
        if not session.initial_question:
            raise RubberDuckError(
                "Cannot reply without an active session initial question."
            )
        if not session.latest_socratic_question:
            raise RubberDuckError("No pending Socratic question to reply to.")
        if not session.latest_analysis:
            raise RubberDuckError(
                "Session has no analysis state to forward to backend."
            )

        url = f"{self.base_url}/api/ducks/reply"
        payload = {
            "initial_question": session.initial_question,
            "prev_socratic_q": session.latest_socratic_question,
            "user_reply": user_reply.strip(),
            "analysis": session.latest_analysis,
            "turn_count": session.turn_count,
            "evidence": session.latest_evidence,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(url, json=payload)
        except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError) as e:
            raise BackendConnectionError(
                f"Cannot connect to RubberDuck backend at {self.base_url}. Is it running?"
            ) from e
        except Exception as e:
            raise BackendConnectionError(f"Unexpected connection failure: {e}") from e

        if resp.status_code != 200:
            detail = self._extract_error_detail(resp)
            raise BackendAPIError(resp.status_code, detail)

        return resp.json()

    @staticmethod
    def _extract_error_detail(resp: httpx.Response) -> str:
        """Safely extract error message from response body."""
        try:
            data = resp.json()
            if isinstance(data, dict):
                return data.get("detail", str(data))
            return str(data)
        except Exception:
            return resp.text or f"HTTP {resp.status_code}"
