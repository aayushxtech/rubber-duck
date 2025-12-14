from pydantic import BaseModel
from uuid import UUID
from typing import Optional, Dict
from datetime import datetime


class DuckEvidence(BaseModel):
    evidence_id: UUID
    evidence: str
    meta: Optional[Dict] = {}
    evidence_time: datetime
