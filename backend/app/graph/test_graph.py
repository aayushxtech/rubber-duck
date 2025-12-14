from uuid import uuid4
from datetime import datetime
from app.graph.state import GraphState
from app.schemas.request import DuckRequest
from app.graph.graph import build_graph


def test_langraph():
    state = GraphState(
        request=DuckRequest(
            req_id=uuid4(),
            req="I think Fedora is bad for development but I'm not sure why.",
            meta={},
            req_time=datetime.now(),
            status="new",
        )
    )
    graph = build_graph()
    final_state = graph.invoke(state)
    print(final_state["analysis"])


if __name__ == "__main__":
    test_langraph()
