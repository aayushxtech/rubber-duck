from pydantic import BaseModel
from uuid import UUID
from typing import Optional, Dict
from datetime import datetime


class DuckAnalysis(BaseModel):
    analysis_id: UUID
    analysis: str
    meta: Optional[Dict] = {}
    analysis_time: datetime
