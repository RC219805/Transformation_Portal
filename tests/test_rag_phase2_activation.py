"""
Tests for Phase 2 RAG System Activation.

This module tests the Knowledge Engine feedback loop activation,
including CI result ingestion, dependency graph construction,
and intelligent test selection.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

import pytest

# Add agents directory to path
agents_path = Path(__file__).parent.parent / '.github' / 'agents'
sys.path.insert(0, str(agents_path))

from rag_system.phase2_activation import (  # noqa: E402
    Phase2Activator,
    TestResultEntry,
    QualityMetricEntry,
    DependencyGraphStats,
    KnowledgeBaseState,
    OBSERVED_CI_RESULTS,
)


@pytest.fixture
def activator():
    """Create a Phase2Activator instance."""
    return Phase2Activator()


@pytest.fixture
def sample_ci_results():
    """Create sample CI results for testing."""
    return {
        "run_id": "test_run_001",
        "timestamp": "2025-11-30T12:00:00Z",
        "summary": {
            "total_tests": 100,
            "passed": 90,
            "skipped": 10,
            "failed": 0,
            "errors": 0,
            "duration_seconds": 5.5
        },
        "codeql_results": {
            "alerts": 0,
            "queries_run": 20,
            "modules_extracted": 50
        },
        "test_categories": {
            "unit_tests": {
                "test_module_a.py": {"passed": 30, "skipped": 5},
                "test_module_b.py": {"passed": 25, "skipped": 0}
            },
            "integration_tests": {
                "test_integration.py": {"passed": 35, "skipped": 5}
            }
        },
        "phase2_files_validated": [
            "file1.py", "file2.py", "file3.py",
            "file4.py", "file5.py", "file6.py"
        ],
        "extraction_warnings": [
            "Warning 1",
            "Warning 2"
        ]
    }


class TestTestResultEntry:
    """Tests for TestResultEntry dataclass."""

    def test_create_entry(self):
        """Test creating a test result entry."""
        entry = TestResultEntry(
            test_id="test_module.py::test_func",
            test_file="test_module.py",
            test_name="test_func",
            status="passed"
        )
        assert entry.test_id == "test_module.py::test_func"
        assert entry.status == "passed"
        assert entry.timestamp  # Should be auto-filled

    def test_entry_with_error(self):
        """Test creating a failed test entry with error message."""
        entry = TestResultEntry(
            test_id="test_module.py::test_fail",
            test_file="test_module.py",
            test_name="test_fail",
            status="failed",
            error_message="AssertionError: Expected 1, got 2"
        )
        assert entry.status == "failed"
        assert "AssertionError" in entry.error_message


class TestQualityMetricEntry:
    """Tests for QualityMetricEntry dataclass."""

    def test_create_metric(self):
        """Test creating a quality metric entry."""
        metric = QualityMetricEntry(
            metric_id="test_pass_rate",
            metric_type="test_health",
            value=90.0,
            unit="percent",
            source="CI Run #1"
        )
        assert metric.metric_id == "test_pass_rate"
        assert metric.value == 90.0
        assert metric.context == {}  # Default empty dict

    def test_metric_with_context(self):
        """Test creating metric with context."""
        metric = QualityMetricEntry(
            metric_id="coverage",
            metric_type="code_quality",
            value=85.5,
            unit="percent",
            source="Coverage Report",
            context={"lines_covered": 1000, "total_lines": 1170}
        )
        assert metric.context["lines_covered"] == 1000


class TestPhase2Activator:
    """Tests for Phase2Activator class."""

    def test_init(self, activator):
        """Test activator initialization."""
        assert activator.test_results == []
        assert activator.metrics == []
        assert activator.patterns == {}
        assert activator.dependency_stats is None

    def test_ingest_ci_results(self, activator, sample_ci_results):
        """Test ingesting CI results."""
        report = activator.ingest_ci_results(sample_ci_results)

        # Check report structure
        assert "timestamp" in report
        assert "source" in report
        assert report["entries_created"] > 0
        assert report["metrics_recorded"] == 3
        assert report["patterns_detected"] >= 2  # At least phase2 and warnings

    def test_ingest_creates_test_entries(self, activator, sample_ci_results):
        """Test that ingestion creates test entries."""
        activator.ingest_ci_results(sample_ci_results)

        # Should have 100 entries (90 passed + 10 skipped)
        assert len(activator.test_results) == 100

        # Check categories are recorded
        categories = {tr.category for tr in activator.test_results}
        assert "unit_tests" in categories
        assert "integration_tests" in categories

    def test_ingest_creates_metrics(self, activator, sample_ci_results):
        """Test that ingestion creates quality metrics."""
        activator.ingest_ci_results(sample_ci_results)

        assert len(activator.metrics) == 3

        metric_ids = {m.metric_id for m in activator.metrics}
        assert "test_pass_rate" in metric_ids
        assert "test_execution_time" in metric_ids
        assert "security_alerts" in metric_ids

    def test_ingest_detects_patterns(self, activator, sample_ci_results):
        """Test pattern detection during ingestion."""
        activator.ingest_ci_results(sample_ci_results)

        assert "phase2_complete_deployment" in activator.patterns
        assert "stub_module_warnings" in activator.patterns

    def test_build_dependency_graph(self, activator):
        """Test dependency graph construction."""
        report = activator.build_dependency_graph()

        assert report["status"] == "graph_constructed"
        assert report["nodes"] > 0
        assert report["edges"] > 0
        assert len(report["critical_paths"]) > 0

        # Check stats were stored
        assert activator.dependency_stats is not None
        assert activator.dependency_stats.total_nodes == report["nodes"]

    def test_generate_test_selection_strategy(self, activator):
        """Test intelligent test selection generation."""
        selection = activator.generate_test_selection_strategy()

        assert "changed_files" in selection
        assert "affected_tests" in selection
        assert "test_reduction_percent" in selection
        assert selection["strategy"] == "dependency_aware_selection"
        assert 0 <= selection["confidence"] <= 1

    def test_test_selection_with_custom_files(self, activator):
        """Test test selection with custom changed files."""
        changed_files = [
            "src/streaming/processor.py",
            "src/streaming/checkpoint.py"
        ]
        selection = activator.generate_test_selection_strategy(changed_files)

        assert selection["changed_files"] == changed_files
        assert len(selection["affected_tests"]) >= 0

    def test_export_knowledge_base(self, activator, sample_ci_results, tmp_path):
        """Test exporting knowledge base to files."""
        # First ingest some data
        activator.ingest_ci_results(sample_ci_results)
        activator.build_dependency_graph()

        # Export to temp directory
        export_report = activator.export_knowledge_base(tmp_path)

        # Check files were created
        assert (tmp_path / "test_results.json").exists()
        assert (tmp_path / "quality_metrics.json").exists()
        assert (tmp_path / "detected_patterns.json").exists()
        assert (tmp_path / "dependency_stats.json").exists()
        assert (tmp_path / "knowledge_state.json").exists()

        # Verify content
        with open(tmp_path / "knowledge_state.json") as f:
            state = json.load(f)
        assert state["test_results_count"] == 100
        assert state["dependency_graph_built"] is True

    def test_generate_activation_report(self, activator, sample_ci_results):
        """Test activation report generation."""
        activator.ingest_ci_results(sample_ci_results)
        activator.build_dependency_graph()

        report = activator.generate_activation_report()

        # Check report contains key sections
        assert "PHASE 2 RAG SYSTEM ACTIVATION REPORT" in report
        assert "KNOWLEDGE ENGINE INGESTION" in report
        assert "DEPENDENCY GRAPH CONSTRUCTION" in report
        assert "INTELLIGENT TEST SELECTION" in report
        assert "STRATEGIC VALUE DELIVERED" in report
        assert "ACTIVATION COMPLETE" in report


class TestObservedCIResults:
    """Tests for the OBSERVED_CI_RESULTS constant."""

    def test_observed_results_structure(self):
        """Test that OBSERVED_CI_RESULTS has correct structure."""
        assert "run_id" in OBSERVED_CI_RESULTS
        assert "summary" in OBSERVED_CI_RESULTS
        assert "test_categories" in OBSERVED_CI_RESULTS
        assert "codeql_results" in OBSERVED_CI_RESULTS

    def test_observed_results_summary(self):
        """Test the summary data in OBSERVED_CI_RESULTS."""
        summary = OBSERVED_CI_RESULTS["summary"]
        assert summary["total_tests"] == 1117
        assert summary["passed"] == 913
        assert summary["failed"] == 0

    def test_ingest_observed_results(self, activator):
        """Test ingesting the actual observed CI results."""
        report = activator.ingest_ci_results(OBSERVED_CI_RESULTS)

        # Calculate expected count dynamically from test categories
        expected_entries = sum(
            results.get("passed", 0) + results.get("skipped", 0)
            for category_tests in OBSERVED_CI_RESULTS.get(
                "test_categories", {}
            ).values()
            for results in category_tests.values()
        )
        assert report["entries_created"] == expected_entries
        assert report["metrics_recorded"] == 3
        assert report["patterns_detected"] == 3


class TestKnowledgeBaseIntegration:
    """Integration tests for the knowledge base."""

    def test_full_activation_workflow(self, activator, tmp_path):
        """Test the complete activation workflow."""
        # Step 1: Ingest CI results
        ingestion = activator.ingest_ci_results(OBSERVED_CI_RESULTS)
        assert ingestion["entries_created"] > 0

        # Step 2: Build dependency graph
        graph = activator.build_dependency_graph()
        assert graph["status"] == "graph_constructed"

        # Step 3: Generate test selection
        selection = activator.generate_test_selection_strategy()
        assert selection["test_reduction_percent"] > 0

        # Step 4: Export knowledge base
        export = activator.export_knowledge_base(tmp_path)
        assert len(export["exported_files"]) >= 4

        # Step 5: Generate report
        report = activator.generate_activation_report()
        assert "ACTIVATION COMPLETE" in report

    def test_knowledge_base_files_are_valid_json(self, activator, tmp_path):
        """Test that all exported files are valid JSON."""
        activator.ingest_ci_results(OBSERVED_CI_RESULTS)
        activator.build_dependency_graph()
        activator.export_knowledge_base(tmp_path)

        json_files = [
            "test_results.json",
            "quality_metrics.json",
            "detected_patterns.json",
            "dependency_stats.json",
            "knowledge_state.json"
        ]

        for filename in json_files:
            filepath = tmp_path / filename
            assert filepath.exists(), f"{filename} should exist"
            with open(filepath) as f:
                data = json.load(f)  # Should not raise
            assert data is not None
