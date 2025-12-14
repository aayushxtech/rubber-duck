from typing import List, Optional, Dict
from pydantic import BaseModel
from uuid import UUID
from datetime import datetime


class DuckSearch(BaseModel):
    search_id: UUID
    query: str
    filters: Optional[Dict] = {}
    limit: Optional[int] = 10
    type: Optional[str] = "socratic"
    raw: Dict = {}
    results: Optional[List[Dict]] = []
    time: datetime
