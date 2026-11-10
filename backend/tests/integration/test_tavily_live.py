import os
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from app.graph.nodes.search_node import search_node, tavily_client
from app.graph.state import GraphState
from app.schemas.analysis import DuckAnalysis
from app.schemas.request import DuckRequest


@pytest.mark.live
def test_live_tavily_search_node():
    """Live integration test: verifies Tavily search node with real Tavily client."""
    if not os.getenv("TAVILY_API_KEY") or tavily_client is None:
        pytest.skip("TAVILY_API_KEY is not configured in local environment")

    req = DuckRequest(
        req_id=uuid4(),
        req="Python 3.12 release year",
        meta={},
        req_time=datetime.now(UTC),
        status="live_test",
    )
    analysis = DuckAnalysis(
        analysis_id=uuid4(),
        analysis_time=datetime.now(UTC),
        analysis="Needs year confirmation",
        assumptions=[],
        uncertainties=["Python 3.12 release year"],
        needs_search=True,
    )
    state = GraphState(
        request=req,
        analysis=analysis,
        max_evidence_results=3,
    )

    updated_state = search_node(state)

    assert updated_state.internal_research_count == 1
    assert updated_state.evidence is not None
    assert len(updated_state.evidence.raw_results) > 0
    assert updated_state.analysis.needs_search is False
