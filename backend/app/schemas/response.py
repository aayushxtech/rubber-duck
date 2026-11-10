from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class DuckResponse(BaseModel):
    res_id: UUID
    res: str
    meta: dict | None = {}
    res_time: datetime
