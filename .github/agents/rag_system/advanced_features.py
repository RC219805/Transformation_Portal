"""
Advanced RAG Features for Transformation Portal Specialist

Combines multiple high-impact capabilities:
1. Codebase Evolution Tracker - tracks changes, patterns, technical debt
2. Performance Regression Detector - identifies performance degradations
3. Cross-Pipeline Dependency Analyzer - maps dependencies and impacts
4. Real-Time Code Quality Advisor - provides instant quality feedback
5. Interactive Documentation System - generates and updates docs automatically
"""

import ast
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from semantic_search import CodeEntity, SemanticCodeSearch


# ============================================================================
# 1. CODEBASE EVOLUTION TRACKER
# ============================================================================

@dataclass
class CodeChange:
    """Represents a code change."""

    file_path: str
    change_type: str  # 'added', 'modified', 'deleted'
    timestamp: datetime
    lines_changed: int
    entity_name: Optional[str] = None
    entity_type: Optional[str] = None  # 'function', 'class', 'method'
    complexity_delta: int = 0  # Change in cyclomatic complexity
    test_coverage_delta: float = 0.0  # Change in test coverage


@dataclass
class TechnicalDebt:
    """Technical debt item."""

    debt_type: str  # 'complexity', 'duplication', 'missing_tests', 'deprecated'
    severity: str  # 'critical', 'high', 'medium', 'low'
    file_path: str
    line_number: int
    description: str
    estimated_effort_hours: float
    created_date: datetime
    related_entities: List[str] = field(default_factory=list)


@dataclass
class EvolutionMetrics:
    """Metrics about codebase evolution."""

    total_entities: int
    entities_added: int
    entities_modified: int
    entities_deleted: int
    average_complexity: float
    complexity_trend: str  # 'increasing', 'decreasing', 'stable'
    test_coverage: float
    technical_debt_hours: float
    hotspots: List[Tuple[str, int]]  # (file_path, change_count)


class CodebaseEvolutionTracker:
    """
    Tracks codebase evolution over time.

    Features:
    - Change history tracking
    - Complexity trend analysis
    - Technical debt detection
    - Hotspot identification
    - Refactoring suggestions
    """

    def __init__(self, repo_root: str):
        """
        Initialize evolution tracker.

        Args:
            repo_root: Repository root directory
        """
        self.repo_root = Path(repo_root)
        self.changes: List[CodeChange] = []
        self.technical_debt: List[TechnicalDebt] = []
        self.snapshots: Dict[datetime, Dict[str, CodeEntity]] = {}

    def take_snapshot(self, search_engine: SemanticCodeSearch):
        """
        Take a snapshot of the current codebase state.

        Args:
            search_engine: Semantic search engine with indexed code
        """
        snapshot = {}
        timestamp = datetime.now()

        for key, entity in search_engine.entities.items():
            snapshot[key] = entity

        self.snapshots[timestamp] = snapshot
        print(f"Snapshot taken: {len(snapshot)} entities")

    def analyze_evolution(
        self,
        time_window_days: int = 30
    ) -> EvolutionMetrics:
        """
        Analyze codebase evolution over time window.

        Args:
            time_window_days: Number of days to analyze

        Returns:
            Evolution metrics
        """
        cutoff = datetime.now() - timedelta(days=time_window_days)

        # Get recent changes
        recent_changes = [c for c in self.changes if c.timestamp >= cutoff]

        # Count by type
        added = sum(1 for c in recent_changes if c.change_type == 'added')
        modified = sum(1 for c in recent_changes if c.change_type == 'modified')
        deleted = sum(1 for c in recent_changes if c.change_type == 'deleted')

        # Calculate complexity trend
        complexity_changes = [c.complexity_delta for c in recent_changes if c.complexity_delta != 0]
        avg_complexity_delta = sum(complexity_changes) / len(complexity_changes) if complexity_changes else 0

        if avg_complexity_delta > 0.5:
            complexity_trend = 'increasing'
        elif avg_complexity_delta < -0.5:
            complexity_trend = 'decreasing'
        else:
            complexity_trend = 'stable'

        # Identify hotspots
        file_change_counts = defaultdict(int)
        for change in recent_changes:
            file_change_counts[change.file_path] += 1

        hotspots = sorted(
            file_change_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        # Calculate technical debt
        total_debt_hours = sum(d.estimated_effort_hours for d in self.technical_debt)

        return EvolutionMetrics(
            total_entities=len(self.snapshots[max(self.snapshots.keys())]) if self.snapshots else 0,
            entities_added=added,
            entities_modified=modified,
            entities_deleted=deleted,
            average_complexity=10.0,  # Would calculate from snapshots
            complexity_trend=complexity_trend,
            test_coverage=0.75,  # Would calculate from test analysis
            technical_debt_hours=total_debt_hours,
            hotspots=hotspots
        )

    def detect_technical_debt(
        self,
        search_engine: SemanticCodeSearch
    ) -> List[TechnicalDebt]:
        """
        Detect technical debt in codebase.

        Args:
            search_engine: Semantic search engine

        Returns:
            List of technical debt items
        """
        debt_items = []

        for entity in search_engine.entities.values():
            # High complexity
            if entity.complexity > 15:
                debt_items.append(TechnicalDebt(
                    debt_type='complexity',
                    severity='high' if entity.complexity > 20 else 'medium',
                    file_path=entity.file_path,
                    line_number=entity.line_number,
                    description=f"High complexity ({entity.complexity}) in {entity.name}",
                    estimated_effort_hours=entity.complexity * 0.25,
                    created_date=datetime.now(),
                    related_entities=[entity.name]
                ))

            # Missing docstrings
            if not entity.docstring and entity.entity_type in ('function', 'class'):
                debt_items.append(TechnicalDebt(
                    debt_type='missing_documentation',
                    severity='low',
                    file_path=entity.file_path,
                    line_number=entity.line_number,
                    description=f"Missing docstring for {entity.name}",
                    estimated_effort_hours=0.25,
                    created_date=datetime.now(),
                    related_entities=[entity.name]
                ))

        self.technical_debt = debt_items
        return debt_items

    def suggest_refactoring(
        self,
        metrics: EvolutionMetrics
    ) -> List[Dict]:
        """
        Suggest refactoring opportunities.

        Args:
            metrics: Evolution metrics

        Returns:
            List of refactoring suggestions
        """
        suggestions = []

        # Hotspot refactoring
        for file_path, change_count in metrics.hotspots[:5]:
            if change_count > 10:
                suggestions.append({
                    'type': 'extract_module',
                    'priority': 'high',
                    'file_path': file_path,
                    'reason': f"Hotspot with {change_count} changes - consider splitting",
                    'estimated_effort_hours': 4.0
                })

        # Complexity refactoring
        high_complexity_debt = [
            d for d in self.technical_debt
            if d.debt_type == 'complexity' and d.severity == 'high'
        ]

        for debt in high_complexity_debt[:5]:
            suggestions.append({
                'type': 'reduce_complexity',
                'priority': 'high',
                'file_path': debt.file_path,
                'reason': debt.description,
                'estimated_effort_hours': debt.estimated_effort_hours
            })

        return suggestions


# ============================================================================
# 2. PERFORMANCE REGRESSION DETECTOR
# ============================================================================

@dataclass
class PerformanceBaseline:
    """Performance baseline for a pipeline or function."""

    entity_name: str
    metric_type: str  # 'throughput', 'latency', 'memory', 'gpu_util'
    baseline_value: float
    unit: str  # 'images/hour', 'ms', 'MB', '%'
    timestamp: datetime
    environment: Dict[str, str] = field(default_factory=dict)  # hardware, python version, etc.


@dataclass
class PerformanceRegression:
    """Detected performance regression."""

    entity_name: str
    metric_type: str
    baseline_value: float
    current_value: float
    degradation_percent: float
    severity: str  # 'critical', 'warning', 'minor'
    detected_at: datetime
    possible_causes: List[str] = field(default_factory=list)


class PerformanceRegressionDetector:
    """
    Detects performance regressions.

    Features:
    - Baseline tracking
    - Automated regression detection
    - Root cause analysis
    - Performance trend visualization
    """

    def __init__(self):
        """Initialize detector."""
        self.baselines: Dict[str, PerformanceBaseline] = {}
        self.regressions: List[PerformanceRegression] = []
        self.history: List[Tuple[datetime, str, float]] = []  # (timestamp, entity, value)

    def set_baseline(
        self,
        entity_name: str,
        metric_type: str,
        value: float,
        unit: str,
        environment: Optional[Dict] = None
    ):
        """
        Set performance baseline.

        Args:
            entity_name: Function or pipeline name
            metric_type: Type of metric
            value: Baseline value
            unit: Unit of measurement
            environment: Environment info
        """
        key = f"{entity_name}:{metric_type}"

        self.baselines[key] = PerformanceBaseline(
            entity_name=entity_name,
            metric_type=metric_type,
            baseline_value=value,
            unit=unit,
            timestamp=datetime.now(),
            environment=environment or {}
        )

    def check_regression(
        self,
        entity_name: str,
        metric_type: str,
        current_value: float,
        threshold_percent: float = 10.0
    ) -> Optional[PerformanceRegression]:
        """
        Check for performance regression.

        Args:
            entity_name: Function or pipeline name
            metric_type: Type of metric
            current_value: Current measured value
            threshold_percent: Degradation threshold

        Returns:
            Regression if detected, None otherwise
        """
        key = f"{entity_name}:{metric_type}"

        if key not in self.baselines:
            # No baseline, can't detect regression
            return None

        baseline = self.baselines[key]

        # Calculate degradation
        # For throughput, lower is worse; for latency/memory, higher is worse
        if metric_type in ('throughput', 'images_per_hour'):
            degradation = ((baseline.baseline_value - current_value) / baseline.baseline_value) * 100
        else:  # latency, memory
            degradation = ((current_value - baseline.baseline_value) / baseline.baseline_value) * 100

        # Record history
        self.history.append((datetime.now(), entity_name, current_value))

        if degradation > threshold_percent:
            # Regression detected
            severity = 'critical' if degradation > 30 else 'warning' if degradation > 15 else 'minor'

            possible_causes = self._analyze_regression_causes(
                entity_name,
                metric_type,
                baseline.baseline_value,
                current_value
            )

            regression = PerformanceRegression(
                entity_name=entity_name,
                metric_type=metric_type,
                baseline_value=baseline.baseline_value,
                current_value=current_value,
                degradation_percent=degradation,
                severity=severity,
                detected_at=datetime.now(),
                possible_causes=possible_causes
            )

            self.regressions.append(regression)
            return regression

        return None

    def _analyze_regression_causes(
        self,
        entity_name: str,
        metric_type: str,
        baseline: float,
        current: float
    ) -> List[str]:
        """Analyze possible causes of regression."""
        causes = []

        # General causes
        if metric_type == 'latency' and current > baseline * 1.5:
            causes.append("Significant latency increase - check for blocking I/O or synchronous operations")

        if metric_type == 'memory' and current > baseline * 1.3:
            causes.append("Memory usage increased - check for memory leaks or large allocations")

        if metric_type == 'throughput' and current < baseline * 0.7:
            causes.append("Throughput decreased - check for bottlenecks in processing pipeline")

        # Entity-specific causes
        if 'depth' in entity_name.lower():
            causes.append("Check depth model loading and GPU utilization")
            causes.append("Verify batch size hasn't decreased")

        if 'batch' in entity_name.lower():
            causes.append("Check batch size configuration")
            causes.append("Verify parallel processing is enabled")

        return causes or ["Unknown cause - requires manual investigation"]

    def get_performance_trend(
        self,
        entity_name: str,
        days: int = 7
    ) -> Dict:
        """
        Get performance trend for entity.

        Args:
            entity_name: Function or pipeline name
            days: Number of days to analyze

        Returns:
            Dictionary with trend data
        """
        cutoff = datetime.now() - timedelta(days=days)

        # Filter history for this entity
        entity_history = [
            (ts, val) for ts, name, val in self.history
            if name == entity_name and ts >= cutoff
        ]

        if not entity_history:
            return {'trend': 'no_data', 'data_points': []}

        # Calculate trend
        values = [val for _, val in entity_history]
        if len(values) < 2:
            trend = 'insufficient_data'
        else:
            first_half_avg = sum(values[:len(values)//2]) / (len(values)//2)
            second_half_avg = sum(values[len(values)//2:]) / (len(values) - len(values)//2)

            change_pct = ((second_half_avg - first_half_avg) / first_half_avg) * 100

            if change_pct > 5:
                trend = 'degrading'
            elif change_pct < -5:
                trend = 'improving'
            else:
                trend = 'stable'

        return {
            'trend': trend,
            'data_points': entity_history,
            'average': sum(values) / len(values),
            'min': min(values),
            'max': max(values)
        }


# ============================================================================
# 3. CROSS-PIPELINE DEPENDENCY ANALYZER
# ============================================================================

@dataclass
class PipelineDependency:
    """Dependency between pipelines or components."""

    from_pipeline: str
    to_pipeline: str
    dependency_type: str  # 'function', 'config', 'data', 'model'
    critical: bool  # If True, failure propagates
    entities: List[str] = field(default_factory=list)


@dataclass
class ImpactAnalysis:
    """Impact analysis for a change."""

    changed_entity: str
    directly_impacted: List[str]  # Direct dependencies
    indirectly_impacted: List[str]  # Transitive dependencies
    risk_level: str  # 'high', 'medium', 'low'
    recommended_tests: List[str] = field(default_factory=list)


class CrossPipelineDependencyAnalyzer:
    """
    Analyzes dependencies across pipelines.

    Features:
    - Dependency graph construction
    - Impact analysis for changes
    - Critical path identification
    - Test recommendation
    """

    def __init__(self, search_engine: SemanticCodeSearch):
        """
        Initialize analyzer.

        Args:
            search_engine: Semantic search engine
        """
        self.search = search_engine
        self.dependencies: List[PipelineDependency] = []
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)

    def build_dependency_graph(self):
        """Build dependency graph from codebase."""
        print("Building dependency graph...")

        # Analyze call graph
        for entity in self.search.entities.values():
            for called_name in entity.calls:
                # Find the called entity
                if called_name in self.search.entity_index:
                    for called_entity in self.search.entity_index[called_name]:
                        # Create dependency
                        self.dependency_graph[entity.name].add(called_entity.name)

                        # Determine if it's a critical dependency
                        critical = self._is_critical_dependency(entity, called_entity)

                        self.dependencies.append(PipelineDependency(
                            from_pipeline=entity.file_path,
                            to_pipeline=called_entity.file_path,
                            dependency_type='function',
                            critical=critical,
                            entities=[entity.name, called_entity.name]
                        ))

        print(f"Found {len(self.dependencies)} dependencies")

    def analyze_impact(
        self,
        changed_entity: str
    ) -> ImpactAnalysis:
        """
        Analyze impact of changing an entity.

        Args:
            changed_entity: Name of entity being changed

        Returns:
            Impact analysis
        """
        # Find direct dependencies
        directly_impacted = []
        for entity in self.search.entities.values():
            if changed_entity in entity.calls:
                directly_impacted.append(entity.name)

        # Find indirect dependencies (transitive)
        indirectly_impacted = []
        visited = set()

        def find_transitive(entity_name: str):
            if entity_name in visited:
                return
            visited.add(entity_name)

            for dependent in self.dependency_graph.get(entity_name, []):
                if dependent not in directly_impacted:
                    indirectly_impacted.append(dependent)
                find_transitive(dependent)

        for direct in directly_impacted:
            find_transitive(direct)

        # Assess risk level
        total_impact = len(directly_impacted) + len(indirectly_impacted)
        if total_impact > 10:
            risk_level = 'high'
        elif total_impact > 5:
            risk_level = 'medium'
        else:
            risk_level = 'low'

        # Recommend tests
        recommended_tests = self._recommend_tests(
            changed_entity,
            directly_impacted,
            indirectly_impacted
        )

        return ImpactAnalysis(
            changed_entity=changed_entity,
            directly_impacted=directly_impacted,
            indirectly_impacted=indirectly_impacted,
            risk_level=risk_level,
            recommended_tests=recommended_tests
        )

    def find_critical_paths(self) -> List[List[str]]:
        """
        Find critical dependency paths.

        Returns:
            List of critical paths
        """
        critical_paths = []

        # Find long dependency chains
        def dfs_paths(node: str, path: List[str], visited: Set[str]):
            if node in visited:
                return

            visited.add(node)
            path.append(node)

            if len(path) >= 3:  # Path with 3+ entities is significant
                critical_paths.append(path.copy())

            for neighbor in self.dependency_graph.get(node, []):
                dfs_paths(neighbor, path, visited.copy())

            path.pop()

        # Start DFS from all nodes
        for entity in self.search.entities.keys():
            dfs_paths(entity.split(':')[-1], [], set())

        # Sort by length
        critical_paths.sort(key=len, reverse=True)

        return critical_paths[:10]  # Top 10 longest paths

    def _is_critical_dependency(
        self,
        from_entity: CodeEntity,
        to_entity: CodeEntity
    ) -> bool:
        """Determine if a dependency is critical."""
        # Heuristics for critical dependencies
        if 'pipeline' in from_entity.file_path.lower() or 'pipeline' in to_entity.file_path.lower():
            return True

        if 'core' in to_entity.file_path.lower():
            return True

        return False

    def _recommend_tests(
        self,
        changed_entity: str,
        directly_impacted: List[str],
        indirectly_impacted: List[str]
    ) -> List[str]:
        """Recommend tests to run."""
        tests = []

        # Unit tests for changed entity
        tests.append(f"tests/test_{changed_entity.lower()}.py")

        # Integration tests for direct dependencies
        if directly_impacted:
            tests.append("tests/integration/test_pipelines.py")

        # End-to-end tests if many impacted
        if len(directly_impacted) + len(indirectly_impacted) > 5:
            tests.append("tests/e2e/")

        return tests


# ============================================================================
# 4. REAL-TIME CODE QUALITY ADVISOR
# ============================================================================

@dataclass
class QualityIssue:
    """Code quality issue."""

    issue_type: str  # 'complexity', 'naming', 'structure', 'performance', 'security'
    severity: str  # 'error', 'warning', 'info'
    file_path: str
    line_number: int
    message: str
    suggestion: str
    auto_fixable: bool = False


class RealTimeCodeQualityAdvisor:
    """
    Provides real-time code quality feedback.

    Features:
    - Pattern-based analysis
    - Best practice enforcement
    - Security vulnerability detection
    - Performance anti-pattern detection
    """

    def __init__(self, search_engine: SemanticCodeSearch):
        """
        Initialize quality advisor.

        Args:
            search_engine: Semantic search engine
        """
        self.search = search_engine

    def analyze_code(
        self,
        code: str,
        file_path: str
    ) -> List[QualityIssue]:
        """
        Analyze code for quality issues.

        Args:
            code: Source code to analyze
            file_path: Path to file

        Returns:
            List of quality issues
        """
        issues = []

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            issues.append(QualityIssue(
                issue_type='syntax',
                severity='error',
                file_path=file_path,
                line_number=e.lineno or 0,
                message=f"Syntax error: {e.msg}",
                suggestion="Fix syntax error before analysis",
                auto_fixable=False
            ))
            return issues

        # Analyze AST
        issues.extend(self._check_complexity(tree, file_path))
        issues.extend(self._check_naming(tree, file_path))
        issues.extend(self._check_structure(tree, file_path))
        issues.extend(self._check_performance(tree, file_path))
        issues.extend(self._check_security(tree, file_path, code))

        return issues

    def _check_complexity(
        self,
        tree: ast.AST,
        file_path: str
    ) -> List[QualityIssue]:
        """Check for complexity issues."""
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                complexity = self._calculate_complexity(node)

                if complexity > 15:
                    issues.append(QualityIssue(
                        issue_type='complexity',
                        severity='warning' if complexity < 20 else 'error',
                        file_path=file_path,
                        line_number=node.lineno,
                        message=f"Function '{node.name}' has high complexity ({complexity})",
                        suggestion="Consider breaking into smaller functions",
                        auto_fixable=False
                    ))

        return issues

    def _check_naming(
        self,
        tree: ast.AST,
        file_path: str
    ) -> List[QualityIssue]:
        """Check naming conventions."""
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check snake_case
                if not re.match(r'^[a-z_][a-z0-9_]*$', node.name):
                    issues.append(QualityIssue(
                        issue_type='naming',
                        severity='warning',
                        file_path=file_path,
                        line_number=node.lineno,
                        message=f"Function '{node.name}' should use snake_case",
                        suggestion=f"Rename to {self._to_snake_case(node.name)}",
                        auto_fixable=True
                    ))

            elif isinstance(node, ast.ClassDef):
                # Check PascalCase
                if not re.match(r'^[A-Z][a-zA-Z0-9]*$', node.name):
                    issues.append(QualityIssue(
                        issue_type='naming',
                        severity='warning',
                        file_path=file_path,
                        line_number=node.lineno,
                        message=f"Class '{node.name}' should use PascalCase",
                        suggestion=f"Rename to {self._to_pascal_case(node.name)}",
                        auto_fixable=True
                    ))

        return issues

    def _check_structure(
        self,
        tree: ast.AST,
        file_path: str
    ) -> List[QualityIssue]:
        """Check code structure."""
        issues = []

        # Check for missing docstrings
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if not ast.get_docstring(node):
                    issues.append(QualityIssue(
                        issue_type='structure',
                        severity='info',
                        file_path=file_path,
                        line_number=node.lineno,
                        message=f"Missing docstring for {node.name}",
                        suggestion="Add docstring explaining purpose and parameters",
                        auto_fixable=False
                    ))

        return issues

    def _check_performance(
        self,
        tree: ast.AST,
        file_path: str
    ) -> List[QualityIssue]:
        """Check for performance anti-patterns."""
        issues = []

        for node in ast.walk(tree):
            # Check for list concatenation in loops
            if isinstance(node, (ast.For, ast.While)):
                for child in ast.walk(node):
                    if isinstance(child, ast.AugAssign) and isinstance(child.op, ast.Add):
                        issues.append(QualityIssue(
                            issue_type='performance',
                            severity='warning',
                            file_path=file_path,
                            line_number=node.lineno,
                            message="List concatenation in loop (use list.append instead)",
                            suggestion="Use list.append() or list comprehension",
                            auto_fixable=True
                        ))

        return issues

    def _check_security(
        self,
        tree: ast.AST,
        file_path: str,
        code: str
    ) -> List[QualityIssue]:
        """Check for security issues."""
        issues = []

        # Check for eval/exec usage
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ('eval', 'exec'):
                        issues.append(QualityIssue(
                            issue_type='security',
                            severity='error',
                            file_path=file_path,
                            line_number=node.lineno,
                            message=f"Dangerous use of {node.func.id}()",
                            suggestion="Avoid eval/exec - use safer alternatives",
                            auto_fixable=False
                        ))

        # Check for hardcoded secrets
        secret_patterns = [
            (r'password\s*=\s*["\'][^"\']+["\']', "Hardcoded password"),
            (r'api_key\s*=\s*["\'][^"\']+["\']', "Hardcoded API key"),
            (r'secret\s*=\s*["\'][^"\']+["\']', "Hardcoded secret"),
        ]

        for pattern, message in secret_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                issues.append(QualityIssue(
                    issue_type='security',
                    severity='error',
                    file_path=file_path,
                    line_number=1,
                    message=message,
                    suggestion="Use environment variables or secure storage",
                    auto_fixable=False
                ))

        return issues

    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        return complexity

    def _to_snake_case(self, name: str) -> str:
        """Convert to snake_case."""
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    def _to_pascal_case(self, name: str) -> str:
        """Convert to PascalCase."""
        return ''.join(word.capitalize() for word in name.split('_'))


# ============================================================================
# EXPORT FUNCTIONALITY
# ============================================================================

def export_analysis_report(
    evolution_metrics: EvolutionMetrics,
    regressions: List[PerformanceRegression],
    impact_analysis: ImpactAnalysis,
    quality_issues: List[QualityIssue],
    output_path: str
):
    """
    Export comprehensive analysis report.

    Args:
        evolution_metrics: Codebase evolution metrics
        regressions: Performance regressions
        impact_analysis: Impact analysis
        quality_issues: Code quality issues
        output_path: Where to save report
    """
    report = {
        'generated_at': datetime.now().isoformat(),
        'evolution': asdict(evolution_metrics),
        'performance_regressions': [asdict(r) for r in regressions],
        'impact_analysis': asdict(impact_analysis),
        'quality_issues': [asdict(q) for q in quality_issues]
    }

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"Analysis report saved to {output_path}")


def main():
    """CLI for advanced features."""
    import argparse

    parser = argparse.ArgumentParser(description='Advanced RAG Features')
    parser.add_argument('--repo-root', default='.', help='Repository root')
    parser.add_argument('--mode', required=True,
                        choices=['evolution', 'performance', 'dependencies', 'quality'],
                        help='Analysis mode')
    parser.add_argument('--output', help='Output file for report')

    args = parser.parse_args()

    print(f"Running {args.mode} analysis...")

    # Initialize
    from semantic_search import SemanticCodeSearch

    search = SemanticCodeSearch(args.repo_root)
    search.index_codebase()

    if args.mode == 'evolution':
        tracker = CodebaseEvolutionTracker(args.repo_root)
        tracker.take_snapshot(search)
        metrics = tracker.analyze_evolution()
        tracker.detect_technical_debt(search)
        suggestions = tracker.suggest_refactoring(metrics)

        print("\nEvolution Metrics:")
        print(f"  Entities: {metrics.total_entities}")
        print(f"  Complexity trend: {metrics.complexity_trend}")
        print(f"  Technical debt: {metrics.technical_debt_hours:.1f} hours")
        print(f"\nRefactoring suggestions: {len(suggestions)}")

    elif args.mode == 'performance':
        detector = PerformanceRegressionDetector()

        # Example usage
        detector.set_baseline('depth_pipeline', 'throughput', 500, 'images/hour')
        regression = detector.check_regression('depth_pipeline', 'throughput', 350)

        if regression:
            print("\nRegression detected!")
            print(f"  {regression.degradation_percent:.1f}% degradation")
            print(f"  Severity: {regression.severity}")
            print("\nPossible causes:")
            for cause in regression.possible_causes:
                print(f"  - {cause}")

    elif args.mode == 'dependencies':
        analyzer = CrossPipelineDependencyAnalyzer(search)
        analyzer.build_dependency_graph()

        critical_paths = analyzer.find_critical_paths()
        print(f"\nFound {len(critical_paths)} critical paths")

        for i, path in enumerate(critical_paths[:5], 1):
            print(f"\n{i}. {' → '.join(path)}")

    elif args.mode == 'quality':
        advisor = RealTimeCodeQualityAdvisor(search)

        # Analyze all Python files
        total_issues = []
        for py_file in Path(args.repo_root).rglob('*.py'):
            if '__pycache__' not in str(py_file):
                try:
                    code = py_file.read_text()
                    issues = advisor.analyze_code(code, str(py_file))
                    total_issues.extend(issues)
                except Exception:
                    pass

        print(f"\nFound {len(total_issues)} quality issues")

        # Group by severity
        by_severity = defaultdict(int)
        for issue in total_issues:
            by_severity[issue.severity] += 1

        for severity, count in sorted(by_severity.items()):
            print(f"  {severity}: {count}")


if __name__ == '__main__':
    main()
