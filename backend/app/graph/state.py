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
    search: Optional[DuckSearch] = None
    evidence: Optional[DuckEvidence] = None
    socratic_question: Optional[str] = None
    final_answer: Optional[str] = None
