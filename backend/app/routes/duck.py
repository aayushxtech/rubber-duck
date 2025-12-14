from datetime import datetime
import uuid
from fastapi import APIRouter

router = APIRouter()


@router.post("/ducks")
async def create_duck(req: str):
    # Placeholder for actual logic
    req_id = uuid.uuid4()
    req = req
    req_time = datetime.now()
    status = "created"

    return {"req_id": req_id,
            "req": req,
            "req_time": req_time,
            "status": status}
