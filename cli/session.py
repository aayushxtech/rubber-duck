from dataclasses import dataclass
from typing import Any


@dataclass
class Session:
    """In-memory session state for RubberDuck CLI."""

    initial_question: str | None = None
    latest_socratic_question: str | None = None
    latest_final_answer: str | None = None
    latest_analysis: dict[str, Any] | None = None
    latest_evidence: dict[str, Any] | None = None
    turn_count: int = 0
    is_active: bool = False
    is_completed: bool = False

    @property
    def is_awaiting_clarification(self) -> bool:
        """True if the graph paused on a Socratic question and expects user reply."""
        return (
            self.is_active
            and not self.is_completed
            and bool(self.latest_socratic_question)
        )

    def start_new(self, question: str) -> None:
        """Reset and start a fresh reasoning session."""
        self.initial_question = question.strip()
        self.latest_socratic_question = None
        self.latest_final_answer = None
        self.latest_analysis = None
        self.latest_evidence = None
        self.turn_count = 0
        self.is_active = True
        self.is_completed = False

    def update_from_response(self, data: dict[str, Any]) -> None:
        """Update session state from a start or reply API response."""
        self.latest_socratic_question = data.get("socratic_question")
        self.latest_final_answer = data.get("final_answer")
        self.latest_analysis = data.get("analysis")
        self.latest_evidence = data.get("evidence")
        self.turn_count = data.get("turn_count", self.turn_count)

        if self.latest_final_answer or not self.latest_socratic_question:
            self.is_completed = True

    def reset(self) -> None:
        """Clear all session data."""
        self.initial_question = None
        self.latest_socratic_question = None
        self.latest_final_answer = None
        self.latest_analysis = None
        self.latest_evidence = None
        self.turn_count = 0
        self.is_active = False
        self.is_completed = False
