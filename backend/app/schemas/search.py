from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class DuckSearch(BaseModel):
    search_id: UUID
    queries: list[str]
    filters: dict | None = None
    limit: int = 10
    provider: str = "tavily"
    created_at: datetime
