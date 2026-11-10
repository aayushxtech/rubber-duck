from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class DuckEvidence(BaseModel):
    evidence_id: UUID
    search_id: UUID
    raw_results: list[dict]
    meta: dict | None = None
    collected_at: datetime
