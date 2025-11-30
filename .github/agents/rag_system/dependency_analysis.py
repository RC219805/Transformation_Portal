#!/usr/bin/env python3
"""
Transformation Portal RAG System - Cross-Pipeline Dependency Analysis
======================================================================
Phase 2 Vector 4: Intelligent dependency tracking and impact assessment
across CI/CD pipelines and codebase components.

This module provides:
- Dependency graph construction from imports and workflows
- Change impact analysis for PRs
- Test selection optimization
- Pipeline ordering recommendations
- Circular dependency detection

Architecture:
    DependencyAnalyzer
    ├── ImportGraphBuilder (Python import analysis)
    ├── WorkflowGraphBuilder (CI/CD workflow dependencies)
    ├── ImpactCalculator (change propagation analysis)
    ├── TestSelector (intelligent test selection)
    └── RecommendationEngine (optimization suggestions)

Graph Types:
    - import_graph: Python module dependencies
    - workflow_graph: CI/CD job dependencies
    - test_graph: Test-to-code relationships
    - combined_graph: Unified dependency view

Analysis Capabilities:
    - Forward impact: What depends on changed files?
    - Backward trace: What does this file depend on?
    - Critical path: Longest dependency chain
    - Affected tests: Tests for changed code
    - Pipeline impact: Workflows affected by changes

Usage:
    # Build dependency graph
    python dependency_analysis.py build

    # Analyze impact of changes
    python dependency_analysis.py impact --files src/module.py

    # Get test recommendations
    python dependency_analysis.py tests --files src/module.py

    # Visualize dependencies
    python dependency_analysis.py visualize --module depth_pipeline

Author: Transformation Portal
Version: 2.1.0 (Phase 2)
"""

from __future__ import annotations

import ast
import fnmatch
import json
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Configure module logger
logger = logging.getLogger("rag_system.dependency_analysis")


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class DependencyNode:
    """A node in the dependency graph."""

    node_id: str  # Unique identifier (file path or workflow name)
    node_type: str  # "module", "workflow", "test", "config"
    name: str

    # Location
    file_path: Optional[str] = None

    # Metadata
    lines_of_code: int = 0
    complexity: int = 0  # Cyclomatic complexity estimate
    last_modified: Optional[str] = None

    # Tags
    tags: List[str] = field(default_factory=list)


@dataclass
class DependencyEdge:
    """An edge in the dependency graph."""

    source: str  # Node ID
    target: str  # Node ID
    edge_type: str  # "imports", "calls", "triggers", "tests"

    # Metadata
    weight: float = 1.0  # Strength of dependency
    line_numbers: List[int] = field(default_factory=list)


@dataclass
class ImpactReport:
    """Report of change impact analysis."""

    changed_files: List[str]

    # Direct dependencies
    direct_dependents: List[str] = field(default_factory=list)
    direct_dependencies: List[str] = field(default_factory=list)

    # Transitive impact
    all_affected: List[str] = field(default_factory=list)
    affected_tests: List[str] = field(default_factory=list)
    affected_workflows: List[str] = field(default_factory=list)

    # Metrics
    impact_score: float = 0.0  # 0-1 scale of change risk
    affected_loc: int = 0

    # Recommendations
    recommended_tests: List[str] = field(default_factory=list)
    recommended_reviewers: List[str] = field(default_factory=list)


@dataclass
class DependencyGraph:
    """Complete dependency graph."""

    nodes: Dict[str, DependencyNode] = field(default_factory=dict)
    edges: List[DependencyEdge] = field(default_factory=list)

    # Adjacency lists for efficient traversal
    forward_edges: Dict[str, List[str]] = field(
        default_factory=lambda: defaultdict(list)
    )
    backward_edges: Dict[str, List[str]] = field(
        default_factory=lambda: defaultdict(list)
    )

    # Metadata
    built_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    root_path: str = "."


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class DependencyConfig:
    """Configuration for dependency analysis."""

    # Paths
    repo_root: str = "."
    cache_dir: str = ".rag_cache/dependencies"

    # Analysis scope
    include_patterns: List[str] = field(default_factory=lambda: [
        "*.py",
        ".github/workflows/*.yml",
        ".github/workflows/*.yaml",
    ])

    exclude_patterns: List[str] = field(default_factory=lambda: [
        "deprecated/*",
        ".venv/*",
        "__pycache__/*",
        ".rag_cache/*",
        "*.pyc",
    ])

    # Analysis options
    analyze_imports: bool = True
    analyze_workflows: bool = True
    analyze_tests: bool = True
    max_depth: int = 10  # Maximum traversal depth

    # Test mapping
    test_directories: List[str] = field(default_factory=lambda: ["tests/"])
    test_patterns: List[str] = field(
        default_factory=lambda: ["test_*.py", "*_test.py"]
    )


# =============================================================================
# Import Graph Builder
# =============================================================================


class ImportGraphBuilder:
    """
    Builds dependency graph from Python imports.

    Analyzes:
    - Import statements (import x, from x import y)
    - Relative imports
    - Dynamic imports (importlib)
    """

    def __init__(self, config: DependencyConfig):
        self.config = config
        self.repo_root = Path(config.repo_root)

    def build(self, graph: DependencyGraph) -> None:
        """Build import graph for all Python files."""
        python_files = self._find_python_files()

        logger.info(f"Analyzing imports in {len(python_files)} Python files")

        for file_path in python_files:
            self._analyze_file(file_path, graph)

    def _find_python_files(self) -> List[Path]:
        """Find all Python files to analyze."""
        files = []

        for pattern in self.config.include_patterns:
            if pattern.endswith(".py"):
                for path in self.repo_root.rglob(pattern):
                    # Check exclusions
                    rel_path = str(path.relative_to(self.repo_root))
                    excluded = any(
                        fnmatch.fnmatch(rel_path, excl)
                        for excl in self.config.exclude_patterns
                    )

                    if not excluded and path.is_file():
                        files.append(path)

        return files

    def _analyze_file(self, file_path: Path, graph: DependencyGraph) -> None:
        """Analyze a single Python file for imports."""
        rel_path = str(file_path.relative_to(self.repo_root))

        # Add node
        node = DependencyNode(
            node_id=rel_path,
            node_type="module",
            name=file_path.stem,
            file_path=rel_path,
        )

        try:
            content = file_path.read_text(encoding="utf-8")
            node.lines_of_code = len(content.splitlines())

            # Parse AST
            tree = ast.parse(content, filename=str(file_path))

            # Extract imports
            imports = self._extract_imports(tree, rel_path)

            # Estimate complexity
            node.complexity = self._estimate_complexity(tree)

            # Determine tags
            if "test" in rel_path.lower():
                node.tags.append("test")
            if "__init__" in rel_path:
                node.tags.append("package")

        except (SyntaxError, UnicodeDecodeError) as e:
            logger.warning(f"Failed to parse {rel_path}: {e}")
            imports = []

        graph.nodes[rel_path] = node

        # Add edges
        for imported_module, line_num in imports:
            resolved = self._resolve_import(imported_module, rel_path)

            if resolved and resolved in graph.nodes:
                edge = DependencyEdge(
                    source=rel_path,
                    target=resolved,
                    edge_type="imports",
                    line_numbers=[line_num],
                )
                graph.edges.append(edge)
                graph.forward_edges[rel_path].append(resolved)
                graph.backward_edges[resolved].append(rel_path)

    def _extract_imports(
        self,
        tree: ast.AST,
        file_path: str,
    ) -> List[tuple]:
        """Extract import statements from AST."""
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append((alias.name, node.lineno))

            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""

                # Handle relative imports
                if node.level > 0:
                    # Relative import
                    package_path = Path(file_path).parent
                    for _ in range(node.level - 1):
                        package_path = package_path.parent

                    if module:
                        module = f"{package_path}.{module}".replace("/", ".")
                    else:
                        module = str(package_path).replace("/", ".")

                imports.append((module, node.lineno))

        return imports

    def _resolve_import(self, module: str, from_file: str) -> Optional[str]:
        """Resolve import string to file path."""
        # Convert module path to file path
        parts = module.split(".")

        # Try as package
        package_path = "/".join(parts) + "/__init__.py"
        if (self.repo_root / package_path).exists():
            return package_path

        # Try as module
        module_path = "/".join(parts) + ".py"
        if (self.repo_root / module_path).exists():
            return module_path

        # Try in src directory
        src_module_path = "src/" + module_path
        if (self.repo_root / src_module_path).exists():
            return src_module_path

        # External module, not in graph
        return None

    def _estimate_complexity(self, tree: ast.AST) -> int:
        """Estimate cyclomatic complexity."""
        complexity = 1  # Base complexity

        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, (ast.And, ast.Or)):
                complexity += 1
            elif isinstance(node, ast.comprehension):
                complexity += 1

        return complexity


# =============================================================================
# Workflow Graph Builder
# =============================================================================


class WorkflowGraphBuilder:
    """
    Builds dependency graph from CI/CD workflows.

    Analyzes:
    - Job dependencies (needs)
    - Workflow triggers (workflow_run)
    - Path triggers
    """

    def __init__(self, config: DependencyConfig):
        self.config = config
        self.repo_root = Path(config.repo_root)

    def build(self, graph: DependencyGraph) -> None:
        """Build workflow dependency graph."""
        try:
            import yaml  # noqa: F401 - used in _analyze_workflow
        except ImportError:
            logger.warning("PyYAML not installed, skipping workflow analysis")
            return

        workflows_dir = self.repo_root / ".github" / "workflows"

        if not workflows_dir.exists():
            logger.warning("No workflows directory found")
            return

        workflow_files = (
            list(workflows_dir.glob("*.yml")) +
            list(workflows_dir.glob("*.yaml"))
        )

        logger.info(f"Analyzing {len(workflow_files)} workflow files")

        for workflow_path in workflow_files:
            self._analyze_workflow(workflow_path, graph)

    def _analyze_workflow(
        self,
        workflow_path: Path,
        graph: DependencyGraph
    ) -> None:
        """Analyze a single workflow file."""
        try:
            import yaml  # noqa: F401 - imported here to check availability
        except ImportError:
            return

        rel_path = str(workflow_path.relative_to(self.repo_root))

        try:
            content = workflow_path.read_text(encoding="utf-8")
            workflow = yaml.safe_load(content)

            if not workflow:
                return

            workflow_name = workflow.get("name", workflow_path.stem)

            # Add workflow node
            node = DependencyNode(
                node_id=f"workflow:{workflow_name}",
                node_type="workflow",
                name=workflow_name,
                file_path=rel_path,
                lines_of_code=len(content.splitlines()),
                tags=["ci"],
            )
            graph.nodes[node.node_id] = node

            # Analyze jobs
            jobs = workflow.get("jobs", {})
            for job_name, job_config in jobs.items():
                job_node_id = f"job:{workflow_name}:{job_name}"

                # Add job node
                job_node = DependencyNode(
                    node_id=job_node_id,
                    node_type="job",
                    name=job_name,
                    file_path=rel_path,
                    tags=["ci", "job"],
                )
                graph.nodes[job_node_id] = job_node

                # Job belongs to workflow
                edge = DependencyEdge(
                    source=node.node_id,
                    target=job_node_id,
                    edge_type="contains",
                )
                graph.edges.append(edge)

                # Job dependencies (needs)
                needs = job_config.get("needs", [])
                if isinstance(needs, str):
                    needs = [needs]

                for needed_job in needs:
                    needed_node_id = f"job:{workflow_name}:{needed_job}"
                    edge = DependencyEdge(
                        source=job_node_id,
                        target=needed_node_id,
                        edge_type="needs",
                    )
                    graph.edges.append(edge)
                    graph.forward_edges[job_node_id].append(needed_node_id)
                    graph.backward_edges[needed_node_id].append(job_node_id)

            # Analyze triggers
            on_config = workflow.get("on", {})
            if isinstance(on_config, dict):
                # Path triggers
                push_paths = on_config.get("push", {}).get("paths", [])
                pr_paths = on_config.get("pull_request", {}).get("paths", [])

                all_paths = push_paths + pr_paths
                for path_pattern in all_paths:
                    # Create edge from path to workflow
                    edge = DependencyEdge(
                        source=path_pattern,
                        target=node.node_id,
                        edge_type="triggers",
                    )
                    graph.edges.append(edge)

                # Workflow triggers
                workflow_run = on_config.get("workflow_run", {})
                if workflow_run:
                    triggering_workflows = workflow_run.get("workflows", [])
                    for trigger_name in triggering_workflows:
                        edge = DependencyEdge(
                            source=f"workflow:{trigger_name}",
                            target=node.node_id,
                            edge_type="triggers",
                        )
                        graph.edges.append(edge)

        except Exception as e:
            logger.warning(f"Failed to analyze workflow {rel_path}: {e}")


# =============================================================================
# Test Graph Builder
# =============================================================================


class TestGraphBuilder:
    """
    Builds test-to-code dependency mappings.

    Analyzes:
    - Test file naming conventions
    - Import relationships in tests
    - Pytest fixtures
    """

    def __init__(self, config: DependencyConfig):
        self.config = config
        self.repo_root = Path(config.repo_root)

    def build(self, graph: DependencyGraph) -> None:
        """Build test dependency mappings."""
        test_files = self._find_test_files()

        logger.info(f"Analyzing {len(test_files)} test files")

        for test_path in test_files:
            self._map_test_to_code(test_path, graph)

    def _find_test_files(self) -> List[Path]:
        """Find all test files."""
        test_files = []

        for test_dir in self.config.test_directories:
            test_path = self.repo_root / test_dir
            if test_path.exists():
                for pattern in self.config.test_patterns:
                    test_files.extend(test_path.rglob(pattern))

        return test_files

    def _map_test_to_code(
        self,
        test_path: Path,
        graph: DependencyGraph
    ) -> None:
        """Map a test file to the code it tests."""
        rel_path = str(test_path.relative_to(self.repo_root))
        test_name = test_path.stem

        # Common patterns: test_module.py -> module.py
        if test_name.startswith("test_"):
            target_name = test_name[5:]  # Remove 'test_' prefix
        elif test_name.endswith("_test"):
            target_name = test_name[:-5]  # Remove '_test' suffix
        else:
            target_name = test_name

        # Look for matching source files
        for node_id, node in graph.nodes.items():
            if node.node_type == "module" and node.name == target_name:
                # Found matching module
                edge = DependencyEdge(
                    source=rel_path,
                    target=node_id,
                    edge_type="tests",
                    weight=1.0,
                )
                graph.edges.append(edge)
                graph.forward_edges[rel_path].append(node_id)
                graph.backward_edges[node_id].append(rel_path)

                # Tag the test node
                if rel_path in graph.nodes:
                    graph.nodes[rel_path].tags.append(f"tests:{node_id}")


# =============================================================================
# Impact Calculator
# =============================================================================


class ImpactCalculator:
    """
    Calculates change impact through dependency graph.
    """

    def __init__(self, config: DependencyConfig, graph: DependencyGraph):
        self.config = config
        self.graph = graph

    def analyze_impact(self, changed_files: List[str]) -> ImpactReport:
        """
        Analyze impact of changes to specified files.

        Args:
            changed_files: List of file paths that have changed

        Returns:
            ImpactReport with analysis results
        """
        report = ImpactReport(changed_files=changed_files)

        # Find direct dependents (who imports these files)
        for file_path in changed_files:
            if file_path in self.graph.backward_edges:
                report.direct_dependents.extend(
                    self.graph.backward_edges[file_path]
                )

            # Direct dependencies (what this file imports)
            if file_path in self.graph.forward_edges:
                report.direct_dependencies.extend(
                    self.graph.forward_edges[file_path]
                )

        # Deduplicate
        report.direct_dependents = list(set(report.direct_dependents))
        report.direct_dependencies = list(set(report.direct_dependencies))

        # Calculate transitive impact (BFS)
        all_affected: Set[str] = set(changed_files)
        queue = deque(changed_files)
        depth = 0

        while queue and depth < self.config.max_depth:
            level_size = len(queue)

            for _ in range(level_size):
                current = queue.popleft()

                # Add dependents
                for dependent in self.graph.backward_edges.get(current, []):
                    if dependent not in all_affected:
                        all_affected.add(dependent)
                        queue.append(dependent)

            depth += 1

        report.all_affected = list(all_affected)

        # Identify affected tests
        for affected in all_affected:
            node = self.graph.nodes.get(affected)
            if node and "test" in node.tags:
                report.affected_tests.append(affected)

            # Also find tests that test this module
            for dependent in self.graph.backward_edges.get(affected, []):
                dep_node = self.graph.nodes.get(dependent)
                if dep_node and "test" in dep_node.tags:
                    if dependent not in report.affected_tests:
                        report.affected_tests.append(dependent)

        # Identify affected workflows
        for affected in all_affected:
            node = self.graph.nodes.get(affected)
            if node and node.node_type == "workflow":
                report.affected_workflows.append(affected)

        # Calculate impact score
        report.impact_score = self._calculate_impact_score(report)

        # Calculate affected LOC
        for affected in all_affected:
            node = self.graph.nodes.get(affected)
            if node:
                report.affected_loc += node.lines_of_code

        # Generate test recommendations
        report.recommended_tests = self._recommend_tests(report)

        return report

    def _calculate_impact_score(self, report: ImpactReport) -> float:
        """Calculate impact score (0-1)."""
        total_nodes = len(self.graph.nodes)
        if total_nodes == 0:
            return 0.0

        # Base score from affected percentage
        affected_ratio = len(report.all_affected) / total_nodes

        # Boost for affected tests
        test_factor = min(len(report.affected_tests) * 0.1, 0.3)

        # Boost for affected workflows
        workflow_factor = min(len(report.affected_workflows) * 0.15, 0.2)

        score = min(affected_ratio + test_factor + workflow_factor, 1.0)
        return round(score, 3)

    def _recommend_tests(self, report: ImpactReport) -> List[str]:
        """Generate test recommendations."""
        recommendations = []

        # Direct tests for changed files
        for changed in report.changed_files:
            # Look for test_<module> pattern
            changed_name = Path(changed).stem
            test_pattern = f"test_{changed_name}"

            for node_id, node in self.graph.nodes.items():
                if node.node_type == "module" and test_pattern in node.name:
                    if node_id not in recommendations:
                        recommendations.append(node_id)

        # Add affected tests
        recommendations.extend(
            t for t in report.affected_tests
            if t not in recommendations
        )

        return recommendations[:20]  # Limit recommendations

    def get_critical_path(self, start_node: str) -> List[str]:
        """Find the longest dependency chain from a node."""
        if start_node not in self.graph.nodes:
            return []

        # DFS to find longest path
        visited: Set[str] = set()
        longest_path: List[str] = []

        def dfs(node: str, current_path: List[str]) -> None:
            nonlocal longest_path

            if node in visited:
                return

            visited.add(node)
            current_path.append(node)

            if len(current_path) > len(longest_path):
                longest_path = list(current_path)

            for dependent in self.graph.backward_edges.get(node, []):
                dfs(dependent, current_path)

            current_path.pop()
            visited.remove(node)

        dfs(start_node, [])
        return longest_path

    def find_circular_dependencies(self) -> List[List[str]]:
        """Detect circular dependencies in the graph."""
        cycles = []
        visited: Set[str] = set()
        rec_stack: Set[str] = set()
        path: List[str] = []

        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in self.graph.forward_edges.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    cycles.append(path[cycle_start:] + [neighbor])
                    return False

            path.pop()
            rec_stack.remove(node)
            return False

        for node in self.graph.nodes:
            if node not in visited:
                dfs(node)

        return cycles


# =============================================================================
# Test Selector
# =============================================================================


class TestSelector:
    """
    Intelligent test selection based on change impact.
    """

    def __init__(self, config: DependencyConfig, graph: DependencyGraph):
        self.config = config
        self.graph = graph
        self.impact_calc = ImpactCalculator(config, graph)

    def select_tests(
        self,
        changed_files: List[str],
        max_tests: int = 50,
    ) -> Dict[str, Any]:
        """
        Select optimal set of tests for changed files.

        Returns:
            Dictionary with selected tests and metadata
        """
        impact = self.impact_calc.analyze_impact(changed_files)

        # Prioritize tests
        prioritized = self._prioritize_tests(
            impact.recommended_tests, changed_files
        )

        # Select top tests
        selected = prioritized[:max_tests]

        # Calculate coverage estimate
        covered_modules: Set[str] = set()
        for test in selected:
            for target in self.graph.forward_edges.get(test, []):
                covered_modules.add(target)

        return {
            "selected_tests": selected,
            "total_recommended": len(impact.recommended_tests),
            "coverage_estimate": (
                len(covered_modules) / max(len(changed_files), 1)
            ),
            "impact_score": impact.impact_score,
            "test_command": self._generate_pytest_command(selected),
        }

    def _prioritize_tests(
        self,
        tests: List[str],
        changed_files: List[str],
    ) -> List[str]:
        """Prioritize tests by relevance to changes."""
        scored_tests = []

        for test in tests:
            score = 0

            # Direct test for changed file
            for changed in changed_files:
                changed_name = Path(changed).stem
                if changed_name in test:
                    score += 10

            # Test imports changed file
            for target in self.graph.forward_edges.get(test, []):
                if target in changed_files:
                    score += 5

            # Prefer smaller tests (faster)
            node = self.graph.nodes.get(test)
            if node and node.lines_of_code < 100:
                score += 2

            scored_tests.append((test, score))

        # Sort by score descending
        scored_tests.sort(key=lambda x: x[1], reverse=True)

        return [test for test, _ in scored_tests]

    def _generate_pytest_command(self, tests: List[str]) -> str:
        """Generate pytest command for selected tests."""
        if not tests:
            return "pytest tests/"

        # Convert paths to pytest format
        test_args = " ".join(tests)
        return f"pytest {test_args} -v"


# =============================================================================
# Dependency Analyzer (Unified Interface)
# =============================================================================


class DependencyAnalyzer:
    """
    Unified interface for cross-pipeline dependency analysis.
    """

    def __init__(self, config: Optional[DependencyConfig] = None):
        self.config = config or DependencyConfig()
        self.graph = DependencyGraph(root_path=self.config.repo_root)

        # Initialize builders
        self.import_builder = ImportGraphBuilder(self.config)
        self.workflow_builder = WorkflowGraphBuilder(self.config)
        self.test_builder = TestGraphBuilder(self.config)

        # Storage path
        self.storage_path = Path(self.config.cache_dir) / "graph.json"

    def build_graph(self, force: bool = False) -> Dict[str, Any]:
        """
        Build complete dependency graph.

        Args:
            force: Force rebuild even if cached

        Returns:
            Build statistics
        """
        # Check cache
        if not force and self._load_cached_graph():
            return {
                "status": "loaded_from_cache",
                "nodes": len(self.graph.nodes),
                "edges": len(self.graph.edges),
            }

        # Reset graph
        self.graph = DependencyGraph(root_path=self.config.repo_root)

        # Build components
        if self.config.analyze_imports:
            self.import_builder.build(self.graph)

        if self.config.analyze_workflows:
            self.workflow_builder.build(self.graph)

        if self.config.analyze_tests:
            self.test_builder.build(self.graph)

        # Save to cache
        self._save_graph()

        return {
            "status": "built",
            "nodes": len(self.graph.nodes),
            "edges": len(self.graph.edges),
            "modules": sum(
                1 for n in self.graph.nodes.values()
                if n.node_type == "module"
            ),
            "workflows": sum(
                1 for n in self.graph.nodes.values()
                if n.node_type == "workflow"
            ),
            "tests": sum(
                1 for n in self.graph.nodes.values()
                if "test" in n.tags
            ),
        }

    def _load_cached_graph(self) -> bool:
        """Load graph from cache."""
        if not self.storage_path.exists():
            return False

        try:
            with open(self.storage_path, "r") as f:
                data = json.load(f)

            # Reconstruct graph
            self.graph = DependencyGraph(
                root_path=data.get("root_path", "."),
                built_at=data.get("built_at", ""),
            )

            for node_data in data.get("nodes", []):
                node = DependencyNode(**node_data)
                self.graph.nodes[node.node_id] = node

            for edge_data in data.get("edges", []):
                edge = DependencyEdge(**edge_data)
                self.graph.edges.append(edge)
                self.graph.forward_edges[edge.source].append(edge.target)
                self.graph.backward_edges[edge.target].append(edge.source)

            logger.info(
                f"Loaded graph from cache: {len(self.graph.nodes)} nodes"
            )
            return True

        except (json.JSONDecodeError, IOError, TypeError) as e:
            logger.warning(f"Failed to load cached graph: {e}")
            return False

    def _save_graph(self) -> None:
        """Save graph to cache."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "root_path": self.graph.root_path,
            "built_at": self.graph.built_at,
            "nodes": [asdict(n) for n in self.graph.nodes.values()],
            "edges": [asdict(e) for e in self.graph.edges],
        }

        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved graph to {self.storage_path}")

    def analyze_impact(self, changed_files: List[str]) -> ImpactReport:
        """Analyze impact of file changes."""
        if not self.graph.nodes:
            self.build_graph()

        calc = ImpactCalculator(self.config, self.graph)
        return calc.analyze_impact(changed_files)

    def select_tests(
        self,
        changed_files: List[str],
        max_tests: int = 50,
    ) -> Dict[str, Any]:
        """Select tests for changed files."""
        if not self.graph.nodes:
            self.build_graph()

        selector = TestSelector(self.config, self.graph)
        return selector.select_tests(changed_files, max_tests)

    def find_cycles(self) -> List[List[str]]:
        """Find circular dependencies."""
        if not self.graph.nodes:
            self.build_graph()

        calc = ImpactCalculator(self.config, self.graph)
        return calc.find_circular_dependencies()

    def get_dependents(self, node_id: str) -> List[str]:
        """Get all nodes that depend on given node."""
        return self.graph.backward_edges.get(node_id, [])

    def get_dependencies(self, node_id: str) -> List[str]:
        """Get all nodes that given node depends on."""
        return self.graph.forward_edges.get(node_id, [])

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        if not self.graph.nodes:
            return {"status": "not_built"}

        # Calculate stats
        module_count = sum(
            1 for n in self.graph.nodes.values()
            if n.node_type == "module"
        )
        test_count = sum(
            1 for n in self.graph.nodes.values()
            if "test" in n.tags
        )
        workflow_count = sum(
            1 for n in self.graph.nodes.values()
            if n.node_type == "workflow"
        )

        total_loc = sum(n.lines_of_code for n in self.graph.nodes.values())
        avg_complexity = 0
        if module_count > 0:
            avg_complexity = sum(
                n.complexity for n in self.graph.nodes.values()
                if n.node_type == "module"
            ) / module_count

        return {
            "total_nodes": len(self.graph.nodes),
            "total_edges": len(self.graph.edges),
            "modules": module_count,
            "tests": test_count,
            "workflows": workflow_count,
            "total_loc": total_loc,
            "avg_complexity": round(avg_complexity, 2),
            "built_at": self.graph.built_at,
        }

    def visualize_module(
        self,
        module: str,
        depth: int = 2,
    ) -> str:
        """Generate ASCII visualization of module dependencies."""
        if module not in self.graph.nodes:
            return f"Module '{module}' not found in graph"

        lines = [f"Dependencies for: {module}", ""]

        # Show dependencies (what it imports)
        deps = self.graph.forward_edges.get(module, [])
        if deps:
            lines.append("Imports:")
            for dep in deps[:10]:
                lines.append(f"  └── {dep}")

        # Show dependents (who imports it)
        dependents = self.graph.backward_edges.get(module, [])
        if dependents:
            lines.append("\nImported by:")
            for dep in dependents[:10]:
                lines.append(f"  └── {dep}")

        return "\n".join(lines)

    def export_for_rag(self) -> List[Dict[str, Any]]:
        """Export graph as chunks for RAG indexing."""
        chunks = []

        for node_id, node in self.graph.nodes.items():
            deps = self.graph.forward_edges.get(node_id, [])
            dependents = self.graph.backward_edges.get(node_id, [])

            content_lines = [
                f"Module: {node.name}",
                f"Type: {node.node_type}",
                f"Path: {node.file_path or 'N/A'}",
                f"Lines: {node.lines_of_code}",
                f"Complexity: {node.complexity}",
            ]

            if deps:
                content_lines.append(f"Imports: {', '.join(deps[:5])}")

            if dependents:
                content_lines.append(f"Imported by: {', '.join(dependents[:5])}")

            chunks.append({
                "chunk_id": f"dep:{node_id}",
                "content": "\n".join(content_lines),
                "file_path": f"dependencies/{node_id}",
                "chunk_type": "dependency",
                "metadata": {
                    "node_type": node.node_type,
                    "dependencies": deps,
                    "dependents": dependents,
                },
            })

        return chunks


# =============================================================================
# CLI Interface
# =============================================================================


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Transformation Portal - Cross-Pipeline Dependency Analysis"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Build command
    build_parser = subparsers.add_parser("build", help="Build dependency graph")
    build_parser.add_argument("--force", action="store_true", help="Force rebuild")

    # Impact command
    impact_parser = subparsers.add_parser("impact", help="Analyze change impact")
    impact_parser.add_argument(
        "--files", nargs="+", required=True, help="Changed files"
    )

    # Tests command
    tests_parser = subparsers.add_parser("tests", help="Select tests for changes")
    tests_parser.add_argument(
        "--files", nargs="+", required=True, help="Changed files"
    )
    tests_parser.add_argument("--max", type=int, default=50, help="Max tests")

    # Cycles command
    subparsers.add_parser("cycles", help="Find circular dependencies")

    # Visualize command
    viz_parser = subparsers.add_parser("visualize", help="Visualize dependencies")
    viz_parser.add_argument("--module", required=True, help="Module to visualize")

    # Stats command
    subparsers.add_parser("stats", help="Show graph statistics")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    analyzer = DependencyAnalyzer()

    if args.command == "build":
        print("Building dependency graph...")
        result = analyzer.build_graph(force=args.force)
        print(f"\n✓ Graph built: {result['nodes']} nodes, {result['edges']} edges")
        if "modules" in result:
            print(f"  Modules: {result['modules']}")
            print(f"  Workflows: {result['workflows']}")
            print(f"  Tests: {result['tests']}")

    elif args.command == "impact":
        print(f"Analyzing impact of {len(args.files)} changed files...\n")
        report = analyzer.analyze_impact(args.files)

        print(f"Impact Score: {report.impact_score:.1%}")
        print(f"Affected LOC: {report.affected_loc}")
        print(f"\nDirect dependents: {len(report.direct_dependents)}")
        for dep in report.direct_dependents[:10]:
            print(f"  - {dep}")

        print(f"\nAffected tests: {len(report.affected_tests)}")
        for test in report.affected_tests[:10]:
            print(f"  - {test}")

        print(f"\nAffected workflows: {len(report.affected_workflows)}")
        for wf in report.affected_workflows[:5]:
            print(f"  - {wf}")

    elif args.command == "tests":
        print(f"Selecting tests for {len(args.files)} changed files...\n")
        result = analyzer.select_tests(args.files, args.max)

        print(f"Selected {len(result['selected_tests'])} tests:")
        for test in result['selected_tests'][:20]:
            print(f"  - {test}")

        print(f"\nCoverage estimate: {result['coverage_estimate']:.1%}")
        print(f"Impact score: {result['impact_score']:.1%}")
        print(f"\nCommand: {result['test_command']}")

    elif args.command == "cycles":
        print("Searching for circular dependencies...\n")
        cycles = analyzer.find_cycles()

        if cycles:
            print(f"Found {len(cycles)} circular dependencies:")
            for i, cycle in enumerate(cycles[:10], 1):
                print(f"\n{i}. {' -> '.join(cycle)}")
        else:
            print("✓ No circular dependencies found")

    elif args.command == "visualize":
        print(analyzer.visualize_module(args.module))

    elif args.command == "stats":
        stats = analyzer.get_stats()
        if stats.get("status") == "not_built":
            print("Graph not built. Run 'build' first.")
        else:
            print("\n=== Dependency Graph Statistics ===\n")
            print(f"Total nodes: {stats['total_nodes']}")
            print(f"Total edges: {stats['total_edges']}")
            print(f"Modules: {stats['modules']}")
            print(f"Tests: {stats['tests']}")
            print(f"Workflows: {stats['workflows']}")
            print(f"Total LOC: {stats['total_loc']}")
            print(f"Avg complexity: {stats['avg_complexity']}")
            print(f"Built at: {stats['built_at']}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
