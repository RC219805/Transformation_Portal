"""
Tests for Phase 2 RAG System Integration.

This module tests the four vectors of Phase 2:
- Vector 1: Git Hook Integration
- Vector 2: Consolidated CI/CD (workflow file presence)
- Vector 3: Knowledge Engine Feedback Loop
- Vector 4: Cross-Pipeline Dependency Analysis
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
import yaml

# Get repository root (tests directory is in repo root)
REPO_ROOT = Path(__file__).resolve().parent.parent
RAG_SYSTEM_DIR = REPO_ROOT / ".github/agents/rag_system"

# Add RAG system to path
sys.path.insert(0, str(RAG_SYSTEM_DIR))


class TestPhase2GitHooks:
    """Tests for Vector 1: Git Hook Integration."""

    def test_git_hook_config_defaults(self):
        """Test GitHookConfig has sensible defaults."""
        from git_hooks import GitHookConfig

        config = GitHookConfig()
        assert config.repo_root == "."
        assert config.rag_cache_dir == ".rag_cache"
        assert "post-commit" in config.enabled_hooks
        assert config.incremental_enabled is True
        assert config.background_indexing is True

    def test_git_hook_manager_initialization(self):
        """Test GitHookManager can be initialized."""
        from git_hooks import GitHookManager, GitHookConfig

        config = GitHookConfig(repo_root=".")
        manager = GitHookManager(config)
        assert manager is not None
        assert manager.config.repo_root == "."

    @patch("subprocess.run")
    def test_change_detector_get_current_commit(self, mock_run):
        """Test ChangeDetector can get current commit."""
        from git_hooks import ChangeDetector, GitHookConfig

        mock_run.return_value = MagicMock(
            stdout="abc123def456\n", returncode=0
        )
        config = GitHookConfig()
        detector = ChangeDetector(".", config)
        commit = detector.get_current_commit()
        assert commit == "abc123def456"

    @patch("subprocess.run")
    def test_change_detector_get_current_branch(self, mock_run):
        """Test ChangeDetector can get current branch."""
        from git_hooks import ChangeDetector, GitHookConfig

        mock_run.return_value = MagicMock(stdout="main\n", returncode=0)
        config = GitHookConfig()
        detector = ChangeDetector(".", config)
        branch = detector.get_current_branch()
        assert branch == "main"

    def test_hook_installer_init(self):
        """Test HookInstaller initialization."""
        from git_hooks import HookInstaller, GitHookConfig

        config = GitHookConfig(enabled_hooks=["post-commit"])
        installer = HookInstaller(".", config)
        assert installer is not None


class TestPhase2ConsolidatedCI:
    """Tests for Vector 2: Consolidated CI/CD Workflow."""

    def test_consolidated_workflow_exists(self):
        """Test that ci-consolidated.yml exists."""
        workflow_path = REPO_ROOT / ".github/workflows/ci-consolidated.yml"
        assert workflow_path.exists(), f"Expected {workflow_path} to exist"

    def test_consolidated_workflow_has_stages(self):
        """Test that consolidated workflow has expected stages."""
        workflow_path = REPO_ROOT / ".github/workflows/ci-consolidated.yml"
        with open(workflow_path) as f:
            content = yaml.safe_load(f)

        # Check basic structure
        assert "jobs" in content
        jobs = content["jobs"]

        # Should have setup job
        assert "setup" in jobs

        # Should have lint job
        assert "lint" in jobs

    def test_deprecated_workflows_marked(self):
        """Test that old workflows are deprecated."""
        workflows_dir = REPO_ROOT / ".github/workflows"

        deprecated_files = list(workflows_dir.glob("*.deprecated"))
        # Should have at least one deprecated file
        assert len(deprecated_files) >= 1


class TestPhase2KnowledgeEngine:
    """Tests for Vector 3: Knowledge Engine Feedback Loop."""

    def test_test_status_enum(self):
        """Test TestStatus enum values."""
        from knowledge_feedback import TestStatus

        assert TestStatus.PASSED.value == "passed"
        assert TestStatus.FAILED.value == "failed"
        assert TestStatus.SKIPPED.value == "skipped"

    def test_metric_type_enum(self):
        """Test MetricType enum values."""
        from knowledge_feedback import MetricType

        assert MetricType.COVERAGE_LINE.value == "coverage_line"
        assert MetricType.COVERAGE_BRANCH.value == "coverage_branch"
        assert MetricType.LINT_SCORE.value == "lint_score"

    def test_knowledge_engine_initialization(self):
        """Test KnowledgeEngine can be initialized."""
        from knowledge_feedback import KnowledgeEngine

        engine = KnowledgeEngine()
        assert engine is not None

    def test_knowledge_engine_status(self):
        """Test KnowledgeEngine get_status method."""
        from knowledge_feedback import KnowledgeEngine

        engine = KnowledgeEngine()
        status = engine.get_status()

        assert "knowledge_entries" in status
        assert "patterns_tracked" in status
        assert "metrics_history_size" in status
        assert "storage_path" in status

    def test_failure_analyzer_has_patterns(self):
        """Test FailureAnalyzer has built-in patterns."""
        from knowledge_feedback import FailureAnalyzer, KnowledgeEngineConfig

        config = KnowledgeEngineConfig()
        analyzer = FailureAnalyzer(config)

        # Verify patterns exist
        assert len(analyzer.patterns) > 0, "FailureAnalyzer should have built-in patterns"
        # Verify patterns have required attributes
        for pattern in analyzer.patterns:
            assert hasattr(pattern, 'name'), "Pattern should have a name attribute"
            assert hasattr(pattern, 'error_regex'), "Pattern should have an error_regex attribute"


class TestPhase2DependencyAnalysis:
    """Tests for Vector 4: Cross-Pipeline Dependency Analysis."""

    def test_dependency_node_dataclass(self):
        """Test DependencyNode dataclass."""
        from dependency_analysis import DependencyNode

        node = DependencyNode(
            node_id="src/test.py",
            node_type="module",
            name="test.py"
        )
        assert node.node_id == "src/test.py"
        assert node.node_type == "module"
        assert node.name == "test.py"

    def test_dependency_edge_dataclass(self):
        """Test DependencyEdge dataclass."""
        from dependency_analysis import DependencyEdge

        edge = DependencyEdge(
            source="src/a.py",
            target="src/b.py",
            edge_type="imports"
        )
        assert edge.source == "src/a.py"
        assert edge.target == "src/b.py"
        assert edge.edge_type == "imports"

    def test_dependency_analyzer_initialization(self):
        """Test DependencyAnalyzer can be initialized."""
        from dependency_analysis import DependencyAnalyzer

        analyzer = DependencyAnalyzer()
        assert analyzer is not None

    def test_import_graph_builder_initialization(self):
        """Test ImportGraphBuilder initialization."""
        from dependency_analysis import ImportGraphBuilder, DependencyConfig

        config = DependencyConfig()
        builder = ImportGraphBuilder(config)
        assert builder is not None

    def test_workflow_graph_builder_initialization(self):
        """Test WorkflowGraphBuilder initialization."""
        from dependency_analysis import WorkflowGraphBuilder, DependencyConfig

        config = DependencyConfig()
        builder = WorkflowGraphBuilder(config)
        assert builder is not None

    def test_test_graph_builder_initialization(self):
        """Test TestGraphBuilder initialization."""
        from dependency_analysis import TestGraphBuilder, DependencyConfig

        config = DependencyConfig()
        builder = TestGraphBuilder(config)
        assert builder is not None

    def test_impact_report_dataclass(self):
        """Test ImpactReport dataclass."""
        from dependency_analysis import ImpactReport

        report = ImpactReport(
            changed_files=["src/a.py"],
            direct_dependents=["src/b.py"],
            direct_dependencies=["src/c.py"],
            all_affected=["src/a.py", "src/b.py"],
            affected_tests=["tests/test_a.py"],
            affected_workflows=["CI Pipeline"],
            impact_score=0.5,
            affected_loc=100,
            recommended_tests=["tests/test_a.py"]
        )
        assert report.impact_score == 0.5
        assert len(report.affected_tests) == 1


class TestPhase2Integration:
    """Integration tests for Phase 2 components working together."""

    def test_all_phase2_components_importable(self):
        """Test that all Phase 2 components can be imported."""
        from git_hooks import GitHookManager
        from knowledge_feedback import KnowledgeEngine
        from dependency_analysis import DependencyAnalyzer

        assert GitHookManager is not None
        assert KnowledgeEngine is not None
        assert DependencyAnalyzer is not None

    def test_phase2_files_exist(self):
        """Test that all Phase 2 implementation files exist."""
        expected_files = [
            "git_hooks.py",
            "knowledge_feedback.py",
            "dependency_analysis.py",
        ]

        for filename in expected_files:
            filepath = RAG_SYSTEM_DIR / filename
            assert filepath.exists(), f"Expected {filepath} to exist"

    def test_phase2_documentation_exists(self):
        """Test that Phase 2 documentation exists."""
        status_file = RAG_SYSTEM_DIR / "PHASE2_IMPLEMENTATION_STATUS.md"
        assert status_file.exists(), f"Expected {status_file} to exist"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
