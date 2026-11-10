import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.main import app
from app.schemas.analysis import DuckAnalysis
from app.schemas.evidence import DuckEvidence
from app.schemas.request import DuckRequest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "live: mark test as requiring real external API calls (opt-in via RUN_LIVE_TESTS=1)",
    )


def pytest_runtest_setup(item):
    """Skip live tests unless RUN_LIVE_TESTS=1 is explicitly set."""
    if "live" in item.keywords:
        if os.getenv("RUN_LIVE_TESTS") != "1":
            pytest.skip("Skipping live integration test (set RUN_LIVE_TESTS=1 to run)")


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def make_duck_analysis():
    """Factory fixture for DuckAnalysis."""

    def _create(
        analysis="Test analysis statement.",
        assumptions=None,
        uncertainties=None,
        needs_search=False,
    ):
        return DuckAnalysis(
            analysis_id=uuid4(),
            analysis_time=datetime.now(UTC),
            analysis=analysis,
            assumptions=assumptions or ["Assumption 1"],
            uncertainties=uncertainties or [],
            needs_search=needs_search,
        )

    return _create


@pytest.fixture
def make_duck_request():
    """Factory fixture for DuckRequest."""

    def _create(req="How does photosysnthesis work?", status="socratic"):
        return DuckRequest(
            req_id=uuid4(),
            req=req,
            meta={},
            req_time=datetime.now(UTC),
            status=status,
        )

    return _create


@pytest.fixture
def make_duck_evidence():
    """Factory fixture for DuckEvidence."""

    def _create(raw_results=None):
        return DuckEvidence(
            evidence_id=uuid4(),
            search_id=uuid4(),
            raw_results=raw_results
            or [
                {
                    "title": "Photosynthesis Overview",
                    "url": "https://example.com/photosynthesis",
                    "content": "Photosynthesis converts light into chemical energy.",
                }
            ],
            collected_at=datetime.now(UTC),
        )

    return _create
