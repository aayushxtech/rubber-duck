from datetime import datetime, timezone
import uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.graph.graph import build_graph
from app.graph.state import GraphState
from app.schemas.request import DuckRequest
from app.schemas.analysis import DuckAnalysis  # Ensure DuckAnalysis is imported

router = APIRouter()
graph = build_graph()
MAX_TURNS = 5
RECURSION_LIMIT = 30


# --- Request/Response Schemas ---


class StartRequest(BaseModel):
    initial_question: str = Field(..., min_length=1)


class StartResponse(BaseModel):
    socratic_question: str
    analysis: DuckAnalysis
    turn_count: int


class ReplyRequest(BaseModel):
    initial_question: str = Field(..., min_length=1)
    prev_socratic_q: str = Field(..., min_length=1)
    user_reply: str = Field(..., min_length=1)
    analysis: DuckAnalysis
    turn_count: int


class ReplyResponse(BaseModel):
    socratic_question: str | None = None
    final_answer: str | None = None
    analysis: DuckAnalysis
    turn_count: int

# --- Endpoints ---


@router.post("/ducks/start", response_model=StartResponse)
async def start_duck_reasoning(req: StartRequest):

    duck_request = DuckRequest(
        req_id=uuid.uuid4(),
        req=req.initial_question,
        req_time=datetime.now(timezone.utc),
        status="socratic"
    )
    state = GraphState(
        request=duck_request,
        last_user_reply=None,
        turn_count=0
    )
    result = graph.invoke(state, config={"recursion_limit": RECURSION_LIMIT})

    return StartResponse(
        socratic_question=result.get("socratic_question") or "",
        analysis=result.get("analysis", {}),
        turn_count=result.get("turn_count", 0)
    )


@router.post("/ducks/reply", response_model=ReplyResponse)
async def reply_duck_reasoning(req: ReplyRequest):
    if req.turn_count >= MAX_TURNS:
        raise HTTPException(
            status_code=400, detail="Maximum number of turns reached.")

    duck_request = DuckRequest(
        req_id=uuid.uuid4(),
        req=req.initial_question,
        req_time=datetime.now(timezone.utc),
        status="socratic"
    )

    previous_socratic_q = getattr(req.analysis, "socratic_question", None) or getattr(
        req, "socratic_question", None)

    state = GraphState(
        request=duck_request,
        last_user_reply=req.user_reply,
        analysis=req.analysis if isinstance(
            req.analysis, DuckAnalysis) else DuckAnalysis(**req.analysis.dict()),
        turn_count=req.turn_count,
        # This is critical for the analyze node logic
        socratic_question=previous_socratic_q
    )
    result = graph.invoke(state, config={"recursion_limit": RECURSION_LIMIT})

    return ReplyResponse(
        socratic_question=result.get("socratic_question"),
        final_answer=result.get("final_answer"),
        analysis=result.get("analysis", {}),
        turn_count=result.get("turn_count", req.turn_count)
    )
