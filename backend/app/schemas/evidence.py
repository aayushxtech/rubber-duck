from pydantic import BaseModel
from uuid import UUID
from typing import List, Optional, Dict
from datetime import datetime


class DuckEvidence(BaseModel):
    evidence_id: UUID
    search_id: UUID
    raw_results: List[Dict]
    meta: Optional[Dict] = None
    collected_at: datetime
