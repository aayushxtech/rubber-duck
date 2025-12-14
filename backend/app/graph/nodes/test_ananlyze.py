from uuid import uuid4
from datetime import datetime
from app.graph.state import GraphState
from app.schemas.request import DuckRequest
from app.graph.nodes.ananlyze import analyze_node


def test_analyze_node():
    state = GraphState(
        request=DuckRequest(
            req_id=uuid4(),
            req="I think Fedora is bad for development but I'm not sure why.",
            meta={},
            req_time=datetime.now(),
            status="new",
        )
    )

    new_state = analyze_node(state)
    print(new_state.analysis)
    assert new_state.analysis is not None
    assert hasattr(new_state.analysis, "analysis")
    assert isinstance(new_state.analysis.analysis, str)


if __name__ == "__main__":
    test_analyze_node()
