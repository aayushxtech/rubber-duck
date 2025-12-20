from pydantic import BaseModel
from uuid import UUID
from typing import Optional, List
from datetime import datetime


class DuckAnalysis(BaseModel):
    analysis_id: UUID
    analysis_time: datetime
    analysis: str
    assumptions: Optional[List[str]]
    uncertainties: Optional[List[str]]
    needs_search: bool