#!/usr/bin/env python3
"""
Transformation Portal RAG System - Knowledge Engine Feedback Loop
==================================================================
Phase 2 Vector 3: Automated integration of test results, quality metrics,
and CI/CD outcomes into the RAG knowledge base.

This module provides:
- Test result ingestion and indexing
- Quality metric tracking and trend analysis
- Failure pattern recognition
- Automated knowledge base updates
- PR context enrichment

Architecture:
    KnowledgeEngine
    ├── TestResultIngester (pytest/unittest output parsing)
    ├── QualityMetricsTracker (coverage, lint scores, performance)
    ├── FailureAnalyzer (pattern recognition, root cause hints)
    ├── KnowledgeUpdater (RAG integration)
    └── FeedbackReporter (PR comments, summaries)

Knowledge Types:
    - test_results: Individual test outcomes with context
    - coverage_data: Line/branch coverage by module
    - lint_findings: Code quality issues by category
    - performance_metrics: Benchmark results over time
    - failure_patterns: Recurring issues and resolutions

Integration Points:
    - CI/CD: Automatic ingestion post-pipeline
    - RAG System: Searchable test knowledge
    - PR Reviews: Context-aware suggestions
    - Developer Queries: "Why did test X fail?"

Usage:
    # Ingest test results
    python knowledge_feedback.py ingest --junit results.xml

    # Query failure patterns
    python knowledge_feedback.py query "authentication timeout"

    # Generate PR context
    python knowledge_feedback.py pr-context --pr 123

    # Show quality trends
    python knowledge_feedback.py trends --days 30

Author: Transformation Portal
Version: 2.1.0 (Phase 2)
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import statistics
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Configure module logger
logger = logging.getLogger("rag_system.knowledge_feedback")


# =============================================================================
# Data Models
# =============================================================================


class TestStatus(Enum):
    """Test execution status."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"
    XFAIL = "xfail"  # Expected failure
    XPASS = "xpass"  # Unexpected pass


class MetricType(Enum):
    """Types of quality metrics."""
    COVERAGE_LINE = "coverage_line"
    COVERAGE_BRANCH = "coverage_branch"
    LINT_SCORE = "lint_score"
    COMPLEXITY = "complexity"
    TEST_DURATION = "test_duration"
    MEMORY_USAGE = "memory_usage"


@dataclass
class TestResult:
    """Individual test result with full context."""

    test_id: str  # Unique identifier (file::class::method)
    name: str
    status: TestStatus
    duration_seconds: float

    # Location
    file_path: str
    line_number: Optional[int] = None
    class_name: Optional[str] = None

    # Failure details
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    stack_trace: Optional[str] = None

    # Context
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    commit_sha: Optional[str] = None
    branch: Optional[str] = None
    ci_run_id: Optional[str] = None

    # Tags for categorization
    tags: List[str] = field(default_factory=list)

    def to_chunk_content(self) -> str:
        """Convert to indexable chunk content."""
        lines = [
            f"Test: {self.name}",
            f"Status: {self.status.value}",
            f"File: {self.file_path}",
            f"Duration: {self.duration_seconds:.3f}s",
        ]

        if self.error_message:
            lines.append(f"Error: {self.error_message[:500]}")

        if self.error_type:
            lines.append(f"Error Type: {self.error_type}")

        if self.tags:
            lines.append(f"Tags: {', '.join(self.tags)}")

        return "\n".join(lines)


@dataclass
class QualityMetric:
    """Quality metric data point."""

    metric_type: MetricType
    value: float
    unit: str

    # Scope
    module: Optional[str] = None
    file_path: Optional[str] = None

    # Context
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    commit_sha: Optional[str] = None

    # Thresholds
    threshold_warning: Optional[float] = None
    threshold_error: Optional[float] = None


@dataclass
class FailurePattern:
    """Recognized failure pattern with resolution hints."""

    pattern_id: str
    name: str
    description: str

    # Pattern matching
    error_regex: str
    file_patterns: List[str] = field(default_factory=list)

    # Statistics
    occurrence_count: int = 0
    last_seen: Optional[str] = None
    affected_tests: List[str] = field(default_factory=list)

    # Resolution
    resolution_hints: List[str] = field(default_factory=list)
    related_docs: List[str] = field(default_factory=list)
    auto_fixable: bool = False


@dataclass
class KnowledgeEntry:
    """Entry in the knowledge base."""

    entry_id: str
    entry_type: str  # test_result, metric, pattern, etc.
    content: str

    # Metadata
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    source: str = "ci_pipeline"

    # Relationships
    related_files: List[str] = field(default_factory=list)
    related_entries: List[str] = field(default_factory=list)

    # Search optimization
    keywords: List[str] = field(default_factory=list)
    embedding_id: Optional[str] = None


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class KnowledgeEngineConfig:
    """Configuration for the knowledge engine."""

    # Storage
    knowledge_dir: str = ".rag_cache/knowledge"
    max_entries: int = 10000
    retention_days: int = 90

    # Ingestion
    auto_ingest: bool = True
    ingest_test_results: bool = True
    ingest_coverage: bool = True
    ingest_lint: bool = True

    # Analysis
    pattern_detection: bool = True
    trend_analysis: bool = True
    min_pattern_occurrences: int = 3

    # Integration
    rag_system_path: str = ".github/agents/rag_system"
    update_rag_index: bool = True

    # Reporting
    pr_comments: bool = True
    slack_notifications: bool = False

    @classmethod
    def from_yaml(cls, path: str) -> "KnowledgeEngineConfig":
        """Load configuration from YAML file."""
        import yaml

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        return cls(**data.get("knowledge_engine", {}))


# =============================================================================
# Test Result Ingester
# =============================================================================


class TestResultIngester:
    """
    Parses and ingests test results from various formats.

    Supported formats:
    - JUnit XML (pytest, unittest)
    - pytest JSON
    - Coverage XML/JSON
    """

    def __init__(self, config: KnowledgeEngineConfig):
        self.config = config

    def ingest_junit_xml(self, path: str) -> List[TestResult]:
        """Parse JUnit XML test results."""
        results = []

        try:
            tree = ET.parse(path)
            root = tree.getroot()

            # Handle both single testsuite and testsuites root
            testsuites = root.findall(".//testsuite")
            if not testsuites and root.tag == "testsuite":
                testsuites = [root]

            for testsuite in testsuites:
                suite_name = testsuite.get("name", "")

                for testcase in testsuite.findall("testcase"):
                    result = self._parse_testcase(testcase, suite_name)
                    if result:
                        results.append(result)

            logger.info(f"Ingested {len(results)} test results from {path}")

        except ET.ParseError as e:
            logger.error(f"Failed to parse JUnit XML: {e}")
        except FileNotFoundError:
            logger.error(f"JUnit XML file not found: {path}")

        return results

    def _parse_testcase(
        self,
        testcase: ET.Element,
        suite_name: str,
    ) -> Optional[TestResult]:
        """Parse a single testcase element."""
        name = testcase.get("name", "unknown")
        classname = testcase.get("classname", "")
        time_str = testcase.get("time", "0")

        try:
            duration = float(time_str)
        except ValueError:
            duration = 0.0

        # Determine status
        failure = testcase.find("failure")
        error = testcase.find("error")
        skipped = testcase.find("skipped")

        if failure is not None:
            status = TestStatus.FAILED
            error_message = failure.get("message", "")
            error_type = failure.get("type", "")
            stack_trace = failure.text
        elif error is not None:
            status = TestStatus.ERROR
            error_message = error.get("message", "")
            error_type = error.get("type", "")
            stack_trace = error.text
        elif skipped is not None:
            status = TestStatus.SKIPPED
            error_message = skipped.get("message")
            error_type = None
            stack_trace = None
        else:
            status = TestStatus.PASSED
            error_message = None
            error_type = None
            stack_trace = None

        # Derive file path from classname
        file_path = self._classname_to_filepath(classname)

        # Generate unique ID
        test_id = f"{file_path}::{classname}::{name}"

        return TestResult(
            test_id=test_id,
            name=name,
            status=status,
            duration_seconds=duration,
            file_path=file_path,
            class_name=classname,
            error_message=error_message,
            error_type=error_type,
            stack_trace=stack_trace,
            tags=self._extract_tags(name, classname),
        )

    def _classname_to_filepath(self, classname: str) -> str:
        """Convert Python classname to file path."""
        if not classname:
            return "unknown"

        # Handle patterns like "tests.test_module.TestClass"
        parts = classname.split(".")

        # Remove class name (last part if capitalized)
        if parts and parts[-1][0].isupper():
            parts = parts[:-1]

        if parts:
            return "/".join(parts) + ".py"

        return "unknown.py"

    def _extract_tags(self, name: str, classname: str) -> List[str]:
        """Extract tags from test name and class."""
        tags = []

        # Common pytest markers in name
        markers = ["slow", "integration", "unit", "ml", "gpu", "api"]
        full_name = f"{classname}.{name}".lower()

        for marker in markers:
            if marker in full_name:
                tags.append(marker)

        # Infer category from path
        if "test_" in name:
            tags.append("test")

        return tags

    def ingest_pytest_json(self, path: str) -> List[TestResult]:
        """Parse pytest JSON report."""
        results = []

        try:
            with open(path, "r") as f:
                data = json.load(f)

            for test in data.get("tests", []):
                nodeid = test.get("nodeid", "")
                outcome = test.get("outcome", "passed")
                duration = test.get("duration", 0.0)

                # Parse nodeid (file::class::method)
                parts = nodeid.split("::")
                file_path = parts[0] if parts else "unknown.py"
                class_name = parts[1] if len(parts) > 1 else None
                name = parts[-1] if parts else "unknown"

                # Map outcome to status
                status_map = {
                    "passed": TestStatus.PASSED,
                    "failed": TestStatus.FAILED,
                    "skipped": TestStatus.SKIPPED,
                    "error": TestStatus.ERROR,
                    "xfailed": TestStatus.XFAIL,
                    "xpassed": TestStatus.XPASS,
                }
                status = status_map.get(outcome, TestStatus.ERROR)

                # Extract failure info
                call_info = test.get("call", {})
                error_message = None
                stack_trace = None

                if status in (TestStatus.FAILED, TestStatus.ERROR):
                    longrepr = call_info.get("longrepr", "")
                    if isinstance(longrepr, str):
                        error_message = longrepr[:500]
                        stack_trace = longrepr

                results.append(TestResult(
                    test_id=nodeid,
                    name=name,
                    status=status,
                    duration_seconds=duration,
                    file_path=file_path,
                    class_name=class_name,
                    error_message=error_message,
                    stack_trace=stack_trace,
                ))

            logger.info(f"Ingested {len(results)} test results from pytest JSON")

        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.error(f"Failed to parse pytest JSON: {e}")

        return results

    def ingest_coverage_xml(self, path: str) -> List[QualityMetric]:
        """Parse coverage.xml (Cobertura format)."""
        metrics = []

        try:
            tree = ET.parse(path)
            root = tree.getroot()

            # Overall coverage
            line_rate = float(root.get("line-rate", 0))
            branch_rate = float(root.get("branch-rate", 0))

            metrics.append(QualityMetric(
                metric_type=MetricType.COVERAGE_LINE,
                value=line_rate * 100,
                unit="percent",
                module="overall",
                threshold_warning=70.0,
                threshold_error=50.0,
            ))

            metrics.append(QualityMetric(
                metric_type=MetricType.COVERAGE_BRANCH,
                value=branch_rate * 100,
                unit="percent",
                module="overall",
                threshold_warning=60.0,
                threshold_error=40.0,
            ))

            # Per-package coverage
            for package in root.findall(".//package"):
                pkg_name = package.get("name", "")
                pkg_line_rate = float(package.get("line-rate", 0))

                metrics.append(QualityMetric(
                    metric_type=MetricType.COVERAGE_LINE,
                    value=pkg_line_rate * 100,
                    unit="percent",
                    module=pkg_name,
                ))

            logger.info(f"Ingested {len(metrics)} coverage metrics")

        except (ET.ParseError, FileNotFoundError) as e:
            logger.error(f"Failed to parse coverage XML: {e}")

        return metrics


# =============================================================================
# Quality Metrics Tracker
# =============================================================================


class QualityMetricsTracker:
    """
    Tracks quality metrics over time and provides trend analysis.
    """

    def __init__(self, config: KnowledgeEngineConfig):
        self.config = config
        self.storage_path = Path(config.knowledge_dir) / "metrics.json"
        self.metrics_history: List[Dict[str, Any]] = []
        self._load_history()

    def _load_history(self) -> None:
        """Load metrics history from disk."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r") as f:
                    self.metrics_history = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.metrics_history = []

    def _save_history(self) -> None:
        """Save metrics history to disk."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Enforce retention policy
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.config.retention_days)
        cutoff_str = cutoff.isoformat()

        self.metrics_history = [
            m for m in self.metrics_history
            if m.get("timestamp", "") > cutoff_str
        ]

        with open(self.storage_path, "w") as f:
            json.dump(self.metrics_history, f, indent=2)

    def record_metrics(self, metrics: List[QualityMetric]) -> None:
        """Record new metrics."""
        for metric in metrics:
            metric_dict = asdict(metric)
            # Convert enum to string for JSON serialization
            metric_dict["metric_type"] = metric.metric_type.value
            self.metrics_history.append(metric_dict)

        self._save_history()
        logger.info(f"Recorded {len(metrics)} quality metrics")

    def get_trends(
        self,
        metric_type: MetricType,
        days: int = 30,
        module: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze trends for a specific metric type.

        Returns:
            Dictionary with trend analysis including:
            - current: Most recent value
            - average: Mean over period
            - trend: "improving", "declining", or "stable"
            - change_percent: Percentage change
            - data_points: List of (timestamp, value) pairs
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        cutoff_str = cutoff.isoformat()

        # Filter relevant metrics
        filtered = [
            m for m in self.metrics_history
            if (m.get("metric_type") == metric_type.value and
                m.get("timestamp", "") > cutoff_str and
                (module is None or m.get("module") == module))
        ]

        if not filtered:
            return {"status": "no_data"}

        # Sort by timestamp
        filtered.sort(key=lambda m: m.get("timestamp", ""))

        values = [m.get("value", 0) for m in filtered]
        timestamps = [m.get("timestamp", "") for m in filtered]

        # Calculate statistics
        current = values[-1]
        average = statistics.mean(values)

        # Determine trend
        if len(values) >= 3:
            first_half_avg = statistics.mean(values[:len(values)//2])
            second_half_avg = statistics.mean(values[len(values)//2:])

            change = second_half_avg - first_half_avg
            change_percent = (
                (change / first_half_avg * 100) if first_half_avg != 0 else 0
            )

            if change_percent > 5:
                trend = "improving"
            elif change_percent < -5:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
            change_percent = 0

        return {
            "status": "ok",
            "current": current,
            "average": round(average, 2),
            "minimum": min(values),
            "maximum": max(values),
            "trend": trend,
            "change_percent": round(change_percent, 2),
            "data_points": list(zip(timestamps, values)),
            "sample_count": len(values),
        }

    def check_thresholds(
        self,
        metrics: List[QualityMetric],
    ) -> List[Dict[str, Any]]:
        """Check metrics against thresholds and return violations."""
        violations = []

        for metric in metrics:
            if metric.threshold_error and metric.value < metric.threshold_error:
                violations.append({
                    "metric": metric.metric_type.value,
                    "module": metric.module,
                    "value": metric.value,
                    "threshold": metric.threshold_error,
                    "severity": "error",
                    "message": (
                        f"{metric.metric_type.value} ({metric.value:.1f}%) "
                        f"below error threshold ({metric.threshold_error}%)"
                    ),
                })
            elif metric.threshold_warning and metric.value < metric.threshold_warning:
                violations.append({
                    "metric": metric.metric_type.value,
                    "module": metric.module,
                    "value": metric.value,
                    "threshold": metric.threshold_warning,
                    "severity": "warning",
                    "message": (
                        f"{metric.metric_type.value} ({metric.value:.1f}%) "
                        f"below warning threshold ({metric.threshold_warning}%)"
                    ),
                })

        return violations


# =============================================================================
# Failure Analyzer
# =============================================================================


class FailureAnalyzer:
    """
    Analyzes test failures to detect patterns and provide resolution hints.
    """

    # Built-in failure patterns
    BUILTIN_PATTERNS = [
        FailurePattern(
            pattern_id="import_error",
            name="Import Error",
            description="Module import failure, often due to missing dependencies",
            error_regex=r"ImportError|ModuleNotFoundError",
            resolution_hints=[
                "Check if the module is installed: pip list | grep <module>",
                "Verify requirements.txt includes the dependency",
                "Check for circular imports in the codebase",
            ],
        ),
        FailurePattern(
            pattern_id="assertion_error",
            name="Assertion Failure",
            description="Test assertion did not hold",
            error_regex=r"AssertionError|assert .+ failed",
            resolution_hints=[
                "Review the expected vs actual values in the assertion",
                "Check if test data or fixtures have changed",
                "Verify the code under test matches expectations",
            ],
        ),
        FailurePattern(
            pattern_id="timeout",
            name="Test Timeout",
            description="Test exceeded time limit",
            error_regex=r"timeout|timed out|TimeoutError",
            resolution_hints=[
                "Check for infinite loops or blocking I/O",
                "Consider increasing timeout for slow tests",
                "Mark test as @pytest.mark.slow if intentionally long",
            ],
        ),
        FailurePattern(
            pattern_id="connection_error",
            name="Connection Error",
            description="Network or database connection failure",
            error_regex=r"ConnectionError|ConnectionRefused|ECONNREFUSED",
            resolution_hints=[
                "Ensure required services are running",
                "Check network configuration and firewall rules",
                "Use mocks for external dependencies in unit tests",
            ],
        ),
        FailurePattern(
            pattern_id="type_error",
            name="Type Error",
            description="Type mismatch in operation",
            error_regex=r"TypeError",
            resolution_hints=[
                "Check function signatures and argument types",
                "Verify return types match expectations",
                "Add type hints and run mypy for static analysis",
            ],
        ),
        FailurePattern(
            pattern_id="attribute_error",
            name="Attribute Error",
            description="Missing attribute or method",
            error_regex=r"AttributeError",
            resolution_hints=[
                "Verify object initialization is complete",
                "Check for typos in attribute/method names",
                "Ensure correct object type is being used",
            ],
        ),
        FailurePattern(
            pattern_id="key_error",
            name="Key Error",
            description="Missing dictionary key or index",
            error_regex=r"KeyError|IndexError",
            resolution_hints=[
                "Use .get() with defaults for optional keys",
                "Validate input data structure before access",
                "Check array bounds before indexing",
            ],
        ),
        FailurePattern(
            pattern_id="fixture_error",
            name="Fixture Error",
            description="Pytest fixture setup/teardown failure",
            error_regex=r"fixture .+ not found|ScopeMismatch",
            resolution_hints=[
                "Verify fixture is defined in conftest.py or imported",
                "Check fixture scope compatibility",
                "Ensure fixture dependencies are available",
            ],
        ),
        FailurePattern(
            pattern_id="cuda_error",
            name="CUDA/GPU Error",
            description="GPU-related failure",
            error_regex=r"CUDA|cuda|RuntimeError.*GPU|out of memory",
            resolution_hints=[
                "Ensure GPU drivers are installed and compatible",
                "Reduce batch size or model size",
                "Set CUDA_VISIBLE_DEVICES to control GPU usage",
                "Mark test as @pytest.mark.gpu for GPU-only tests",
            ],
        ),
        FailurePattern(
            pattern_id="permission_error",
            name="Permission Error",
            description="File or resource permission denied",
            error_regex=r"PermissionError|Permission denied",
            resolution_hints=[
                "Check file/directory permissions",
                "Ensure test user has required access",
                "Use temporary directories for test artifacts",
            ],
        ),
    ]

    def __init__(self, config: KnowledgeEngineConfig):
        self.config = config
        self.patterns = list(self.BUILTIN_PATTERNS)
        self.storage_path = Path(config.knowledge_dir) / "patterns.json"
        self._load_custom_patterns()

    def _load_custom_patterns(self) -> None:
        """Load custom patterns from disk."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r") as f:
                    data = json.load(f)

                for pattern_data in data.get("custom_patterns", []):
                    self.patterns.append(FailurePattern(**pattern_data))

            except (json.JSONDecodeError, IOError):
                pass

    def _save_patterns(self) -> None:
        """Save pattern statistics to disk."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Only save statistics, not built-in patterns
        custom = [p for p in self.patterns if p.pattern_id not in
                  [bp.pattern_id for bp in self.BUILTIN_PATTERNS]]

        data = {
            "custom_patterns": [asdict(p) for p in custom],
            "statistics": {
                p.pattern_id: {
                    "occurrence_count": p.occurrence_count,
                    "last_seen": p.last_seen,
                    "affected_tests": p.affected_tests[-100:],  # Keep last 100
                }
                for p in self.patterns
            },
        }

        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)

    def analyze_failure(
        self,
        result: TestResult,
    ) -> List[Dict[str, Any]]:
        """
        Analyze a test failure and match against known patterns.

        Returns:
            List of matched patterns with resolution hints
        """
        if result.status not in (TestStatus.FAILED, TestStatus.ERROR):
            return []

        matches = []
        error_text = (
            f"{result.error_type or ''} "
            f"{result.error_message or ''} "
            f"{result.stack_trace or ''}"
        )

        for pattern in self.patterns:
            if re.search(pattern.error_regex, error_text, re.IGNORECASE):
                # Update pattern statistics
                pattern.occurrence_count += 1
                pattern.last_seen = datetime.now(timezone.utc).isoformat()
                if result.test_id not in pattern.affected_tests:
                    pattern.affected_tests.append(result.test_id)

                matches.append({
                    "pattern_id": pattern.pattern_id,
                    "pattern_name": pattern.name,
                    "description": pattern.description,
                    "resolution_hints": pattern.resolution_hints,
                    "related_docs": pattern.related_docs,
                    "occurrence_count": pattern.occurrence_count,
                    "confidence": self._calculate_confidence(pattern, result),
                })

        if matches:
            self._save_patterns()

        return matches

    def _calculate_confidence(
        self,
        pattern: FailurePattern,
        result: TestResult,
    ) -> float:
        """Calculate confidence score for pattern match."""
        confidence = 0.5  # Base confidence

        # Increase for exact error type match
        if result.error_type and pattern.error_regex in result.error_type:
            confidence += 0.2

        # Increase for file pattern match
        for file_pattern in pattern.file_patterns:
            if re.search(file_pattern, result.file_path):
                confidence += 0.1

        # Increase for recurring pattern
        if pattern.occurrence_count > 5:
            confidence += 0.1

        return min(confidence, 1.0)

    def get_failure_summary(
        self,
        results: List[TestResult],
    ) -> Dict[str, Any]:
        """Generate a summary of failures with pattern analysis."""
        failures = [
            r for r in results
            if r.status in (TestStatus.FAILED, TestStatus.ERROR)
        ]

        if not failures:
            return {"status": "all_passed", "failure_count": 0}

        # Analyze all failures
        pattern_counts: Dict[str, int] = defaultdict(int)
        all_hints: Set[str] = set()

        for failure in failures:
            matches = self.analyze_failure(failure)
            for match in matches:
                pattern_counts[match["pattern_name"]] += 1
                all_hints.update(match["resolution_hints"])

        # Find most common patterns
        sorted_patterns = sorted(
            pattern_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return {
            "status": "failures_detected",
            "failure_count": len(failures),
            "error_count": sum(1 for r in failures if r.status == TestStatus.ERROR),
            "top_patterns": sorted_patterns[:5],
            "resolution_hints": list(all_hints)[:10],
            "affected_files": list(set(f.file_path for f in failures)),
        }


# =============================================================================
# Knowledge Updater
# =============================================================================


class KnowledgeUpdater:
    """
    Updates the RAG knowledge base with new information.
    """

    def __init__(self, config: KnowledgeEngineConfig):
        self.config = config
        self.storage_path = Path(config.knowledge_dir) / "entries.json"
        self.entries: Dict[str, KnowledgeEntry] = {}
        self._load_entries()

    def _load_entries(self) -> None:
        """Load existing entries from disk."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r") as f:
                    data = json.load(f)

                for entry_data in data.get("entries", []):
                    entry = KnowledgeEntry(**entry_data)
                    self.entries[entry.entry_id] = entry

            except (json.JSONDecodeError, IOError):
                pass

    def _save_entries(self) -> None:
        """Save entries to disk."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Enforce max entries
        if len(self.entries) > self.config.max_entries:
            # Remove oldest entries
            sorted_entries = sorted(
                self.entries.values(),
                key=lambda e: e.updated_at,
            )
            for entry in sorted_entries[:len(self.entries) - self.config.max_entries]:
                del self.entries[entry.entry_id]

        data = {
            "entries": [asdict(e) for e in self.entries.values()],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)

    def add_test_results(self, results: List[TestResult]) -> int:
        """Add test results to knowledge base."""
        added = 0

        for result in results:
            entry_id = f"test:{hashlib.md5(result.test_id.encode()).hexdigest()[:12]}"

            # Extract keywords
            keywords = [
                result.name,
                result.status.value,
                result.file_path,
            ]
            if result.error_type:
                keywords.append(result.error_type)
            keywords.extend(result.tags)

            entry = KnowledgeEntry(
                entry_id=entry_id,
                entry_type="test_result",
                content=result.to_chunk_content(),
                source="test_runner",
                related_files=[result.file_path],
                keywords=keywords,
            )

            self.entries[entry_id] = entry
            added += 1

        self._save_entries()
        logger.info(f"Added {added} test result entries to knowledge base")
        return added

    def add_failure_patterns(
        self,
        failures: List[TestResult],
        analyzer: FailureAnalyzer,
    ) -> int:
        """Add failure pattern entries to knowledge base."""
        added = 0

        for failure in failures:
            if failure.status not in (TestStatus.FAILED, TestStatus.ERROR):
                continue

            matches = analyzer.analyze_failure(failure)

            for match in matches:
                entry_id = f"pattern:{match['pattern_id']}:{failure.test_id[:20]}"

                content_lines = [
                    f"Failure Pattern: {match['pattern_name']}",
                    f"Test: {failure.name}",
                    f"File: {failure.file_path}",
                    f"Error: {failure.error_message or 'Unknown'}",
                    "",
                    "Resolution Hints:",
                ]
                content_lines.extend(
                    f"- {hint}" for hint in match["resolution_hints"]
                )

                entry = KnowledgeEntry(
                    entry_id=entry_id,
                    entry_type="failure_pattern",
                    content="\n".join(content_lines),
                    source="failure_analyzer",
                    related_files=[failure.file_path],
                    keywords=[
                        match["pattern_name"],
                        failure.error_type or "",
                        failure.file_path,
                    ],
                )

                self.entries[entry_id] = entry
                added += 1

        self._save_entries()
        return added

    def get_relevant_entries(
        self,
        query: str,
        entry_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[KnowledgeEntry]:
        """Retrieve relevant entries for a query."""
        query_lower = query.lower()
        query_words = set(query_lower.split())

        scored_entries = []

        for entry in self.entries.values():
            if entry_type and entry.entry_type != entry_type:
                continue

            # Simple keyword matching
            score = 0
            entry_words = set(entry.content.lower().split())
            entry_words.update(k.lower() for k in entry.keywords)

            matches = query_words & entry_words
            score = len(matches) / len(query_words) if query_words else 0

            if score > 0:
                scored_entries.append((score, entry))

        # Sort by score descending
        scored_entries.sort(key=lambda x: x[0], reverse=True)

        return [entry for _, entry in scored_entries[:limit]]

    def export_for_rag(self) -> List[Dict[str, Any]]:
        """Export entries in format suitable for RAG indexing."""
        chunks = []

        for entry in self.entries.values():
            chunks.append({
                "chunk_id": entry.entry_id,
                "content": entry.content,
                "file_path": f"knowledge/{entry.entry_type}/{entry.entry_id}",
                "chunk_type": "knowledge",
                "metadata": {
                    "entry_type": entry.entry_type,
                    "keywords": entry.keywords,
                    "source": entry.source,
                    "related_files": entry.related_files,
                },
            })

        return chunks


# =============================================================================
# Feedback Reporter
# =============================================================================


class FeedbackReporter:
    """
    Generates reports and PR comments from knowledge engine data.
    """

    def __init__(self, config: KnowledgeEngineConfig):
        self.config = config

    def generate_test_summary(
        self,
        results: List[TestResult],
        failure_summary: Dict[str, Any],
    ) -> str:
        """Generate a markdown test summary."""
        total = len(results)
        passed = sum(1 for r in results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in results if r.status == TestStatus.FAILED)
        errors = sum(1 for r in results if r.status == TestStatus.ERROR)
        skipped = sum(1 for r in results if r.status == TestStatus.SKIPPED)

        # Calculate pass rate
        pass_rate = (passed / total * 100) if total > 0 else 0

        # Status emoji
        if failed == 0 and errors == 0:
            status_emoji = "✅"
            status_text = "All tests passed!"
        elif errors > 0:
            status_emoji = "❌"
            status_text = "Tests failed with errors"
        else:
            status_emoji = "⚠️"
            status_text = "Some tests failed"

        lines = [
            f"## {status_emoji} Test Results Summary",
            "",
            f"**Status:** {status_text}",
            "",
            "| Metric | Count |",
            "|--------|-------|",
            f"| Total | {total} |",
            f"| ✅ Passed | {passed} |",
            f"| ❌ Failed | {failed} |",
            f"| 💥 Errors | {errors} |",
            f"| ⏭️ Skipped | {skipped} |",
            f"| **Pass Rate** | **{pass_rate:.1f}%** |",
            "",
        ]

        # Add failure patterns if any
        if failure_summary.get("top_patterns"):
            lines.extend([
                "### 🔍 Detected Failure Patterns",
                "",
            ])
            for pattern_name, count in failure_summary["top_patterns"]:
                lines.append(f"- **{pattern_name}**: {count} occurrences")
            lines.append("")

        # Add resolution hints
        if failure_summary.get("resolution_hints"):
            lines.extend([
                "### 💡 Resolution Hints",
                "",
            ])
            for hint in failure_summary["resolution_hints"][:5]:
                lines.append(f"- {hint}")
            lines.append("")

        # Add affected files
        if failure_summary.get("affected_files"):
            lines.extend([
                "### 📁 Affected Files",
                "",
            ])
            for file_path in failure_summary["affected_files"][:10]:
                lines.append(f"- `{file_path}`")

        return "\n".join(lines)

    def generate_quality_report(
        self,
        tracker: QualityMetricsTracker,
        days: int = 30,
    ) -> str:
        """Generate a quality trends report."""
        lines = [
            "## 📊 Quality Metrics Report",
            "",
            f"*Trends over the last {days} days*",
            "",
        ]

        # Coverage trends
        line_trends = tracker.get_trends(MetricType.COVERAGE_LINE, days, "overall")
        branch_trends = tracker.get_trends(MetricType.COVERAGE_BRANCH, days, "overall")

        if line_trends.get("status") == "ok":
            trend_emoji = {"improving": "📈", "declining": "📉", "stable": "➡️"}.get(
                line_trends["trend"], "❓"
            )
            lines.extend([
                "### Line Coverage",
                f"- **Current:** {line_trends['current']:.1f}%",
                f"- **Average:** {line_trends['average']:.1f}%",
                f"- **Trend:** {trend_emoji} {line_trends['trend']} "
                f"({line_trends['change_percent']:+.1f}%)",
                "",
            ])

        if branch_trends.get("status") == "ok":
            trend_emoji = {"improving": "📈", "declining": "📉", "stable": "➡️"}.get(
                branch_trends["trend"], "❓"
            )
            lines.extend([
                "### Branch Coverage",
                f"- **Current:** {branch_trends['current']:.1f}%",
                f"- **Average:** {branch_trends['average']:.1f}%",
                f"- **Trend:** {trend_emoji} {branch_trends['trend']} "
                f"({branch_trends['change_percent']:+.1f}%)",
                "",
            ])

        return "\n".join(lines)

    def generate_pr_context(
        self,
        results: List[TestResult],
        updater: KnowledgeUpdater,
        changed_files: List[str],
    ) -> str:
        """Generate context-aware PR comment."""
        lines = [
            "## 🤖 Knowledge Engine Insights",
            "",
        ]

        # Find relevant historical information
        for file_path in changed_files[:5]:
            relevant = updater.get_relevant_entries(file_path, limit=3)

            if relevant:
                lines.append(f"### Related to `{file_path}`")
                for entry in relevant:
                    if entry.entry_type == "failure_pattern":
                        lines.append("- ⚠️ Historical failure pattern detected")
                    elif entry.entry_type == "test_result":
                        lines.append("- 📝 Previous test context available")
                lines.append("")

        return "\n".join(lines)


# =============================================================================
# Knowledge Engine (Unified Interface)
# =============================================================================


class KnowledgeEngine:
    """
    Unified interface for the knowledge feedback loop.

    Orchestrates:
    - Test result ingestion
    - Quality metrics tracking
    - Failure pattern analysis
    - Knowledge base updates
    - Report generation
    """

    def __init__(self, config: Optional[KnowledgeEngineConfig] = None):
        self.config = config or KnowledgeEngineConfig()

        # Initialize components
        self.ingester = TestResultIngester(self.config)
        self.tracker = QualityMetricsTracker(self.config)
        self.analyzer = FailureAnalyzer(self.config)
        self.updater = KnowledgeUpdater(self.config)
        self.reporter = FeedbackReporter(self.config)

        # Ensure storage directory exists
        Path(self.config.knowledge_dir).mkdir(parents=True, exist_ok=True)

    def ingest_test_run(
        self,
        junit_path: Optional[str] = None,
        pytest_json_path: Optional[str] = None,
        coverage_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Ingest results from a complete test run.

        Returns:
            Summary of ingested data
        """
        results: List[TestResult] = []
        metrics: List[QualityMetric] = []

        # Ingest test results
        if junit_path:
            results.extend(self.ingester.ingest_junit_xml(junit_path))

        if pytest_json_path:
            results.extend(self.ingester.ingest_pytest_json(pytest_json_path))

        # Ingest coverage
        if coverage_path:
            metrics.extend(self.ingester.ingest_coverage_xml(coverage_path))

        # Update knowledge base
        entries_added = self.updater.add_test_results(results)

        # Analyze failures and add patterns
        failures = [
            r for r in results
            if r.status in (TestStatus.FAILED, TestStatus.ERROR)
        ]
        patterns_added = self.updater.add_failure_patterns(failures, self.analyzer)

        # Track metrics
        if metrics:
            self.tracker.record_metrics(metrics)

        # Generate failure summary
        failure_summary = self.analyzer.get_failure_summary(results)

        return {
            "tests_ingested": len(results),
            "metrics_recorded": len(metrics),
            "entries_added": entries_added,
            "patterns_detected": patterns_added,
            "failure_summary": failure_summary,
        }

    def query(
        self,
        query: str,
        entry_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query the knowledge base."""
        entries = self.updater.get_relevant_entries(query, entry_type)

        return [
            {
                "entry_id": e.entry_id,
                "type": e.entry_type,
                "content": e.content,
                "keywords": e.keywords,
            }
            for e in entries
        ]

    def generate_report(
        self,
        report_type: str = "summary",
        **kwargs,
    ) -> str:
        """Generate a report."""
        if report_type == "summary":
            results = kwargs.get("results", [])
            failure_summary = self.analyzer.get_failure_summary(results)
            return self.reporter.generate_test_summary(results, failure_summary)

        elif report_type == "quality":
            days = kwargs.get("days", 30)
            return self.reporter.generate_quality_report(self.tracker, days)

        elif report_type == "pr_context":
            results = kwargs.get("results", [])
            changed_files = kwargs.get("changed_files", [])
            return self.reporter.generate_pr_context(
                results, self.updater, changed_files
            )

        else:
            return f"Unknown report type: {report_type}"

    def get_status(self) -> Dict[str, Any]:
        """Get knowledge engine status."""
        return {
            "knowledge_entries": len(self.updater.entries),
            "patterns_tracked": len(self.analyzer.patterns),
            "metrics_history_size": len(self.tracker.metrics_history),
            "storage_path": str(self.config.knowledge_dir),
        }


# =============================================================================
# CLI Interface
# =============================================================================


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Transformation Portal - Knowledge Engine Feedback Loop"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest test results")
    ingest_parser.add_argument("--junit", help="JUnit XML file path")
    ingest_parser.add_argument("--pytest-json", help="Pytest JSON report path")
    ingest_parser.add_argument("--coverage", help="Coverage XML file path")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query knowledge base")
    query_parser.add_argument("query", help="Search query")
    query_parser.add_argument("--type", help="Entry type filter")

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate report")
    report_parser.add_argument("--type", default="quality", help="Report type")
    report_parser.add_argument("--days", type=int, default=30, help="Days for trends")

    # Status command
    subparsers.add_parser("status", help="Show status")

    # Trends command
    trends_parser = subparsers.add_parser("trends", help="Show quality trends")
    trends_parser.add_argument("--days", type=int, default=30, help="Days to analyze")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    engine = KnowledgeEngine()

    if args.command == "ingest":
        result = engine.ingest_test_run(
            junit_path=args.junit,
            pytest_json_path=args.pytest_json,
            coverage_path=args.coverage,
        )
        print("\n✓ Ingestion complete:")
        print(f"  Tests: {result['tests_ingested']}")
        print(f"  Metrics: {result['metrics_recorded']}")
        print(f"  Knowledge entries: {result['entries_added']}")
        print(f"  Patterns detected: {result['patterns_detected']}")

        if result['failure_summary'].get('failure_count', 0) > 0:
            print(
                f"\n⚠ Failures detected: "
                f"{result['failure_summary']['failure_count']}"
            )

    elif args.command == "query":
        results = engine.query(args.query, getattr(args, 'type', None))
        print(f"\n🔍 Found {len(results)} relevant entries:\n")
        for entry in results:
            print(f"[{entry['type']}] {entry['entry_id']}")
            print(f"  {entry['content'][:200]}...")
            print()

    elif args.command == "report":
        report = engine.generate_report(args.type, days=args.days)
        print(report)

    elif args.command == "status":
        status = engine.get_status()
        print("\n=== Knowledge Engine Status ===\n")
        print(f"Knowledge entries: {status['knowledge_entries']}")
        print(f"Patterns tracked: {status['patterns_tracked']}")
        print(f"Metrics history: {status['metrics_history_size']} records")
        print(f"Storage: {status['storage_path']}")

    elif args.command == "trends":
        report = engine.generate_report("quality", days=args.days)
        print(report)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
