from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class DuckRequest(BaseModel):
    req_id: UUID
    req: str
    meta: dict | None = {}
    req_time: datetime
    status: str
