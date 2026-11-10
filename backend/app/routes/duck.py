import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from app.graph.graph import build_graph
from app.graph.state import GraphState
from app.schemas.analysis import DuckAnalysis
from app.schemas.evidence import DuckEvidence
from app.schemas.request import DuckRequest

router = APIRouter()
graph = build_graph()
MAX_CLARIFICATION_TURNS = 3
RECURSION_LIMIT = 30


# --- Request/Response Schemas ---


class StartRequest(BaseModel):
    initial_question: str = Field(..., min_length=1)

    @field_validator("initial_question")
    @classmethod
    def validate_initial_question(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("initial_question cannot be empty or whitespace-only")
        return v.strip()


class StartResponse(BaseModel):
    socratic_question: str | None = None
    final_answer: str | None = None
    analysis: DuckAnalysis
    turn_count: int
    evidence: DuckEvidence | None = None


class ReplyRequest(BaseModel):
    initial_question: str = Field(..., min_length=1)
    prev_socratic_q: str | None = None
    user_reply: str = Field(..., min_length=1)
    analysis: DuckAnalysis
    turn_count: int
    evidence: DuckEvidence | None = None

    @field_validator("initial_question", "user_reply")
    @classmethod
    def validate_non_empty_strings(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty or whitespace-only")
        return v.strip()


class ReplyResponse(BaseModel):
    socratic_question: str | None = None
    final_answer: str | None = None
    analysis: DuckAnalysis
    turn_count: int
    evidence: DuckEvidence | None = None


# --- Endpoints ---


@router.post("/ducks/start", response_model=StartResponse)
async def start_duck_reasoning(req: StartRequest):
    duck_request = DuckRequest(
        req_id=uuid.uuid4(),
        req=req.initial_question,
        req_time=datetime.now(UTC),
        status="socratic",
    )
    state = GraphState(
        request=duck_request,
        last_user_reply=None,
        turn_count=0,
        clarification_count=0,
        max_clarification_turns=MAX_CLARIFICATION_TURNS,
    )

    try:
        result = graph.invoke(state, config={"recursion_limit": RECURSION_LIMIT})
    except HTTPException:
        raise
    except Exception:
        # Do not expose internal exceptions, API keys, or stack traces
        raise HTTPException(
            status_code=500, detail="An error occurred while processing the request."
        )

    return StartResponse(
        socratic_question=result.get("socratic_question"),
        final_answer=result.get("final_answer"),
        analysis=result.get("analysis", {}),
        turn_count=result.get("turn_count", 0),
        evidence=result.get("evidence"),
    )


@router.post("/ducks/reply", response_model=ReplyResponse)
async def reply_duck_reasoning(req: ReplyRequest):
    # Differentiate between no clarification pending and session already concluded
    if not req.prev_socratic_q or not req.prev_socratic_q.strip():
        raise HTTPException(
            status_code=400, detail="No clarification is currently pending."
        )

    if req.turn_count > MAX_CLARIFICATION_TURNS:
        raise HTTPException(
            status_code=400,
            detail="Session has already concluded. No clarification is pending.",
        )

    duck_request = DuckRequest(
        req_id=uuid.uuid4(),
        req=req.initial_question,
        req_time=datetime.now(UTC),
        status="socratic",
    )

    state = GraphState(
        request=duck_request,
        last_user_reply=req.user_reply,
        analysis=req.analysis
        if isinstance(req.analysis, DuckAnalysis)
        else DuckAnalysis(**req.analysis.dict()),
        turn_count=req.turn_count,
        clarification_count=req.turn_count,
        max_clarification_turns=MAX_CLARIFICATION_TURNS,
        socratic_question=req.prev_socratic_q,
        evidence=req.evidence,
    )

    try:
        result = graph.invoke(state, config={"recursion_limit": RECURSION_LIMIT})
    except HTTPException:
        raise
    except Exception:
        # Do not expose internal exceptions, API keys, or stack traces
        raise HTTPException(
            status_code=500, detail="An error occurred while processing the request."
        )

    return ReplyResponse(
        socratic_question=result.get("socratic_question"),
        final_answer=result.get("final_answer"),
        analysis=result.get("analysis", {}),
        turn_count=result.get("turn_count", req.turn_count),
        evidence=result.get("evidence"),
    )
