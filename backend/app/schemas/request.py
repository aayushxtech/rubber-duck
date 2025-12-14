from pydantic import BaseModel
from uuid import UUID
from typing import Optional, Dict
from datetime import datetime


class DuckRequest(BaseModel):
    req_id: UUID
    req: str
    meta: Optional[Dict] = {}
    req_time: datetime
    status: str
