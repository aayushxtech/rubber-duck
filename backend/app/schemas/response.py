from pydantic import BaseModel
from uuid import UUID
from typing import Optional, Dict
from datetime import datetime


class DuckResponse(BaseModel):
    res_id: UUID
    res: str
    meta: Optional[Dict] = {}
    res_time: datetime
