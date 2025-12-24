from typing import List, Optional, Dict
from pydantic import BaseModel
from uuid import UUID
from datetime import datetime


class DuckSearch(BaseModel):
    search_id: UUID
    queries: List[str]
    filters: Optional[Dict] = None
    limit: int = 10
    provider: str = "tavily"
    created_at: datetime
