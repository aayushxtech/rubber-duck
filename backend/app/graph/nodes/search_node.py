from datetime import datetime
import os
from uuid import uuid4
from dotenv import load_dotenv
from app.graph.state import GraphState
from app.graph.state import DuckSearch
from app.graph.state import DuckEvidence

from tavily import TavilyClient

load_dotenv()

tavily_key = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=tavily_key) if tavily_key else None


def uncertainty_to_query(u: str) -> str:
    return f"evidence about {u}"


def search_node(state: GraphState) -> GraphState:
    if (
        state.analysis is None
        or not state.analysis.needs_search
        or not state.analysis.uncertainties
    ):
        return state

    if not tavily_client:
        raise RuntimeError("Tavily client not initialized")

    search_id = uuid4()
    queries = []
    raw_results = []

    for uncertainty in state.analysis.uncertainties:
        query = uncertainty_to_query(uncertainty)
        queries.append(query)

        try:
            result = tavily_client.search(query=query)
        except Exception as e:
            result = {"error": str(e), "query": query}

        raw_results.append(result)

    state.search = DuckSearch(
        search_id=search_id,
        queries=queries,
        provider="tavily",
        created_at=datetime.now(),
    )

    state.evidence = DuckEvidence(
        evidence_id=uuid4(),
        search_id=search_id,
        raw_results=raw_results,
        collected_at=datetime.now(),
    )
    state.analysis.needs_search = False

    return state
