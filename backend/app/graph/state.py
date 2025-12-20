from typing import Optional
from pydantic import BaseModel
from app.schemas.request import DuckRequest
from app.schemas.analysis import DuckAnalysis
from app.schemas.search import DuckSearch
from app.schemas.evidence import DuckEvidence


class GraphState(BaseModel):
    request: DuckRequest
    analysis: Optional[DuckAnalysis] = None
    route: Optional[str] = None
    should_loop: Optional[bool] = False
    search: Optional[DuckSearch] = None
    evidence: Optional[DuckEvidence] = None
    socratic_question: Optional[str] = None
    user_reply: Optional[str] = None
    turn_count: int = 0
    max_turns: int = 3
    final_answer: Optional[str] = None
