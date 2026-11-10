from enum import Enum

from pydantic import BaseModel

from app.schemas.analysis import DuckAnalysis
from app.schemas.evidence import DuckEvidence
from app.schemas.request import DuckRequest
from app.schemas.search import DuckSearch


class ExecutionStatus(str, Enum):
    IN_PROGRESS = "IN_PROGRESS"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class GraphState(BaseModel):
    request: DuckRequest
    analysis: DuckAnalysis | None = None
    execution_status: ExecutionStatus = ExecutionStatus.IN_PROGRESS
    route: str | None = None
    turn_count: int = 0
    clarification_count: int = 0
    max_clarification_turns: int = 3
    # Deterministic research budget
    internal_research_count: int = 0
    max_research_iterations: int = 3
    # Evidence handling limits
    max_evidence_results: int = 10  # Upper bound for stored evidence snippets
    # Track queries that have already been attempted (normalized)
    attempted_queries: list[str] = []
    search: DuckSearch | None = None
    evidence: DuckEvidence | None = None
    socratic_question: str | None = None
    final_answer: str | None = None
    last_user_reply: str | None = None
