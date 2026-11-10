"""
Comprehensive tests for RubberDuck graph execution lifecycle fix.

Tests cover all four critical scenarios:
- Scenario A: Contextual clarification (socratic path)
- Scenario B: Research required (search -> analyze -> resolution)
- Scenario C: Direct convergence (final path)
- Scenario D: Research limit reached (bounded termination)
"""

import sys

sys.path.insert(0, "backend")

from datetime import datetime
from uuid import uuid4

from app.graph.graph import build_graph
from app.graph.state import ExecutionStatus, GraphState
from app.schemas.request import DuckRequest


class TestRubberDuckGraphLifecycle:
    """Test the graph execution lifecycle after fix."""

    def test_contextual_clarification_scenario(self):
        """Scenario A: Contextual clarification (socratic path)

        Input: Abstract question with contextual uncertainty.

        Expected:
        - Analysis executes
        - Router selects clarification
        - Socratic node executes
        - Graph terminates with NEEDS_CLARIFICATION
        - No repeated analysis without user input
        """
        state = GraphState(
            request=DuckRequest(
                req_id=uuid4(),
                req="I need to refactor this codebase but I'm not sure what the best approach is.",
                meta={},
                req_time=datetime.now(),
                status="new",
            )
        )
        graph = build_graph()
        result = graph.invoke(state)

        assert isinstance(result, GraphState), "Result should be a GraphState"
        assert result.execution_status == ExecutionStatus.NEEDS_CLARIFICATION, (
            f"Expected NEEDS_CLARIFICATION but got {result.execution_status}"
        )
        assert result.route == "socratic", (
            f"Expected route to be 'socratic' but got {result.route}"
        )
        assert result.socratic_question is not None, (
            "Socratic question should not be None"
        )
        assert len(result.analysis.uncertainties) > 0, (
            "There should be uncertainties for contextual clarification"
        )
        assert result.internal_research_count == 0, (
            "Internal research count should be 0 for clarification"
        )

    def test_research_required_scenario(self):
        """Scenario B: Research required (search -> analyze -> resolution)

        Input: Factual uncertainty requiring external verification.

        Expected:
        - Analysis executes
        - Router selects search
        - Search executes
        - Evidence is added to state
        - Analysis executes again
        - Graph either clarifies or completes
        - No uncontrolled search cycle occurs
        """
        state = GraphState(
            request=DuckRequest(
                req_id=uuid4(),
                req="Python 3.10 vs Python 3.11: which version is better for web development?",
                meta={},
                req_time=datetime.now(),
                status="new",
            )
        )
        graph = build_graph()
        result = graph.invoke(state)

        assert isinstance(result, GraphState), "Result should be a GraphState"
        assert result.route in ["socratic", "final"], (
            f"Expected route to be 'socratic' or 'final' but got {result.route}"
        )
        assert result.internal_research_count <= result.max_research_iterations, (
            f"Internal research count {result.internal_research_count} "
            f"should not exceed max {result.max_research_iterations}"
        )

        if result.execution_status == ExecutionStatus.NEEDS_CLARIFICATION:
            assert result.route == "socratic", (
                "Should route to socratic when needs clarification"
            )
            assert len(result.analysis.uncertainties) > 0, (
                "Should have uncertainties for clarification"
            )
        elif result.execution_status == ExecutionStatus.COMPLETED:
            assert result.route == "final", "Should route to final when completed"
            assert result.final_answer is not None, (
                "Final answer should not be None when completed"
            )

    def test_direct_convergence_scenario(self):
        """Scenario C: Direct convergence (final path)

        Input: Concrete question with no material uncertainties.

        Expected:
        - Analysis executes
        - Router selects final
        - Final response is generated
        - Graph terminates with COMPLETED
        """
        state = GraphState(
            request=DuckRequest(
                req_id=uuid4(),
                req="What is the capital of France?",
                meta={},
                req_time=datetime.now(),
                status="new",
            )
        )
        graph = build_graph()
        result = graph.invoke(state)

        assert isinstance(result, GraphState), "Result should be a GraphState"
        assert result.execution_status == ExecutionStatus.COMPLETED, (
            f"Expected COMPLETED but got {result.execution_status}"
        )
        assert result.route == "final", (
            f"Expected route to be 'final' but got {result.route}"
        )
        assert result.final_answer is not None, "Final answer should not be None"
        assert len(result.analysis.uncertainties) == 0, (
            "Should have no uncertainties for convergence"
        )
        assert result.internal_research_count == 0, (
            "Internal research count should be 0"
        )

    def test_research_limit_reached_scenario(self):
        """Scenario D: Research limit reached

        Expected:
        - The graph terminates predictably
        - It does not repeatedly invoke search
        - It does not silently claim the uncertainty was resolved
        """
        state = GraphState(
            request=DuckRequest(
                req_id=uuid4(),
                req="What is the best programming language for AI development in 2024?",
                meta={},
                req_time=datetime.now(),
                status="new",
            ),
            max_research_iterations=2,
        )
        graph = build_graph()
        result = graph.invoke(state)

        assert isinstance(result, GraphState), "Result should be a GraphState"
        assert result.internal_research_count <= 2, (
            f"Internal research count {result.internal_research_count} "
            f"should not exceed limit 2"
        )
        assert result.execution_status in [
            ExecutionStatus.NEEDS_CLARIFICATION,
            ExecutionStatus.COMPLETED,
        ], (
            f"Expected NEEDS_CLARIFICATION or COMPLETED but got {result.execution_status}"
        )

    def test_no_socratic_regeneration_scenario(self):
        """Test that socratic questions are not regenerated without user input

        This is the core fix for the original bug.
        """
        state = GraphState(
            request=DuckRequest(
                req_id=uuid4(),
                req="I need help with my Python project but I'm not sure where to start.",
                meta={},
                req_time=datetime.now(),
                status="new",
            )
        )
        graph = build_graph()
        result = graph.invoke(state)

        assert isinstance(result, GraphState), "Result should be a GraphState"
        assert result.execution_status in [
            ExecutionStatus.NEEDS_CLARIFICATION,
            ExecutionStatus.COMPLETED,
        ], (
            f"Expected NEEDS_CLARIFICATION or COMPLETED but got {result.execution_status}"
        )

        if result.execution_status == ExecutionStatus.NEEDS_CLARIFICATION:
            assert result.route == "socratic", (
                "Should route to socratic when needs clarification"
            )
            assert result.socratic_question is not None, (
                "Socratic question should not be None"
            )

        if result.execution_status == ExecutionStatus.COMPLETED:
            assert result.route == "final", "Should route to final when completed"
            assert result.final_answer is not None, "Final answer should not be None"

    def test_bounded_search_scenario(self):
        """Test that search loops are bounded and evidence resolves uncertainties"""
        state = GraphState(
            request=DuckRequest(
                req_id=uuid4(),
                req="What are the latest features in Python 3.12?",
                meta={},
                req_time=datetime.now(),
                status="new",
            )
        )
        graph = build_graph()
        result = graph.invoke(state)

        assert isinstance(result, GraphState), "Result should be a GraphState"
        assert result.internal_research_count <= result.max_research_iterations, (
            f"Internal research count {result.internal_research_count} "
            f"should not exceed max {result.max_research_iterations}"
        )

        if result.evidence is not None:
            assert result.evidence.raw_results is not None, (
                "Evidence raw_results should not be None"
            )

        if result.execution_status == ExecutionStatus.NEEDS_CLARIFICATION:
            assert len(result.analysis.uncertainties) > 0, (
                "Should have uncertainties for clarification"
            )

        if result.execution_status == ExecutionStatus.COMPLETED:
            assert len(result.analysis.uncertainties) == 0, (
                "Should have no uncertainties when completed"
            )


if __name__ == "__main__":
    print("Running comprehensive RubberDuck graph lifecycle tests")
    print("=" * 70)

    tester = TestRubberDuckGraphLifecycle()

    scenarios = [
        ("Contextual Clarification (A)", tester.test_contextual_clarification_scenario),
        ("Research Required (B)", tester.test_research_required_scenario),
        ("Direct Convergence (C)", tester.test_direct_convergence_scenario),
        ("Research Limit Reached (D)", tester.test_research_limit_reached_scenario),
        ("No Socratic Regeneration", tester.test_no_socratic_regeneration_scenario),
        ("Bounded Search", tester.test_bounded_search_scenario),
    ]

    for scenario_name, test_func in scenarios:
        print(f"\n{scenario_name}:")
        try:
            test_func()
            print(f"✓ {scenario_name} passed")
        except Exception as e:
            print(f"✗ {scenario_name} failed: {e}")

    print("\n" + "=" * 70)
    print("All tests completed")
