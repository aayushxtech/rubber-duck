from datetime import datetime
import uuid
from fastapi import APIRouter
from app.graph.graph import StateGraph, build_graph
from app.graph.state import GraphState
from app.schemas.request import DuckRequest

router = APIRouter()
graph = build_graph()


@router.post("/ducks")
async def duck(_req: str) -> dict:
    req_id = uuid.uuid8()
    req = _req
    req_time = datetime.now()
    status = "socratic"

    duck_request = DuckRequest(
        req_id=req_id,
        req=req,
        req_time=req_time,
        status=status
    )

    initial_graph_state = GraphState(
        request=duck_request
    )
    final_state = graph.invoke(
        initial_graph_state,
        config={"recursion_limit": 100}
    )

    return {
        "soc_q": final_state["socratic_question"],
        "turns_used": final_state["turn_count"]
    }
