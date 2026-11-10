import os
from datetime import datetime
from uuid import uuid4

from app.graph.state import DuckEvidence, DuckSearch, GraphState
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

tavily_key = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=tavily_key) if tavily_key else None


def uncertainty_to_query(u: str) -> str:
    return f"evidence about {u}"


def search_node(state: GraphState) -> GraphState:
    # Exit early if no search is required
    if (
        state.analysis is None
        or not state.analysis.needs_search
        or not state.analysis.uncertainties
    ):
        return state

    # Increment deterministic research budget counter
    state.internal_research_count += 1

    if not tavily_client:
        raise RuntimeError("Tavily client not initialized")

    search_id = uuid4()
    queries = []
    new_results = []

    # Prepare to collect new results for uncertainties that have not been attempted yet
    new_results = []
    queries = []
    for uncertainty in state.analysis.uncertainties:
        query = uncertainty_to_query(uncertainty)
        # Normalise query for tracking (strip and lower)
        norm_query = query.strip().lower()
        if norm_query in getattr(state, "attempted_queries", []):
            # Skip this uncertainty – it has already been searched
            continue
        queries.append(query)
        try:
            result = tavily_client.search(query=query)
        except Exception as e:
            result = {"error": str(e), "query": query}
        new_results.append(result)
        # Record the attempted query
        state.attempted_queries.append(norm_query)

    state.search = DuckSearch(
        search_id=search_id,
        queries=queries,
        provider="tavily",
        created_at=datetime.now(),
    )

    # Combine previous evidence (if any) with new results
    combined = []
    if state.evidence is not None and isinstance(state.evidence.raw_results, list):
        combined.extend(state.evidence.raw_results)
    combined.extend(new_results)

    # Deduplicate by normalized URL + content when possible, preserving distinct snippets
    seen = set()
    deduped = []
    for res in combined:
        # Extract normalized identifier components
        url = None
        content = None
        if isinstance(res, dict):
            # Try to get URL from first result entry or top‑level key
            if "results" in res and isinstance(res["results"], list) and res["results"]:
                first = res["results"][0]
                url = first.get("url")
                # Prefer content from the same first result if present
                content = first.get("content")
            else:
                url = res.get("url")
                content = res.get("content")
        # Normalise missing values safely
        norm_url = url.strip().lower() if isinstance(url, str) else ""
        norm_content = content.strip().lower() if isinstance(content, str) else ""
        # Composite key – empty strings are acceptable; they simply produce fewer duplicates
        key = f"{norm_url}||{norm_content}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(res)

    # Enforce deterministic maximum evidence size defined in GraphState
    max_evidence = getattr(state, "max_evidence_results", 10)
    if len(deduped) > max_evidence:
        deduped = deduped[:max_evidence]

    state.evidence = DuckEvidence(
        evidence_id=uuid4(),
        search_id=search_id,
        raw_results=deduped,
        collected_at=datetime.now(),
    )
    # After a successful search, we assume the current uncertainties are not automatically resolved.
    # The analyzer will decide whether further research is needed.
    state.analysis.needs_search = False

    return state
