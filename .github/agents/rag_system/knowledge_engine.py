"""
Knowledge Integration Engine for RAG System

Provides pattern analysis, feedback loops, recommendations, and query interface
for continuous improvement of image processing workflows.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import json
import statistics


@dataclass
class PatternAnalysis:
    """Analysis of patterns in processing pipelines."""

    pipeline_name: str
    total_runs: int
    success_rate: float
    avg_processing_time: float
    median_processing_time: float
    p95_processing_time: float

    # Failure analysis
    failure_modes: Dict[str, int] = field(default_factory=dict)  # error_type: count
    error_patterns: List[str] = field(default_factory=list)

    # Performance trends
    time_trend: str = "stable"  # "improving", "degrading", "stable"
    quality_trend: str = "stable"  # "improving", "degrading", "stable"

    # Common parameters
    common_parameters: Dict[str, any] = field(default_factory=dict)
    optimal_parameters: Dict[str, any] = field(default_factory=dict)


@dataclass
class Recommendation:
    """Recommendation for pipeline improvement."""

    recommendation_type: str  # "missing_test", "undocumented_feature", "regression", "optimization"
    severity: str  # "critical", "high", "medium", "low"
    title: str
    description: str
    affected_component: str
    suggested_action: str
    evidence: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class FeedbackRecord:
    """Record of a processing outcome for feedback loops."""

    timestamp: datetime
    pipeline: str
    artifact_id: str
    success: bool
    processing_time: float
    parameters: Dict
    error_message: Optional[str] = None
    quality_score: Optional[float] = None
    user_feedback: Optional[str] = None


class KnowledgeIntegrationEngine:
    """
    Knowledge integration engine for continuous improvement.

    Features:
    - Pattern analysis (success rates, failure modes, performance trends)
    - Feedback loops (historical outcomes inform decisions)
    - Recommendations (gap analysis, regressions, optimizations)
    - Natural language query interface
    - KPI tracking and visualization
    """

    def __init__(self):
        """Initialize the knowledge engine."""
        self.feedback_records: List[FeedbackRecord] = []
        self.pattern_cache: Dict[str, PatternAnalysis] = {}
        self.recommendations: List[Recommendation] = []
        self.kpi_history: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)

    def add_feedback(
        self,
        pipeline: str,
        artifact_id: str,
        success: bool,
        processing_time: float,
        parameters: Dict,
        error_message: Optional[str] = None,
        quality_score: Optional[float] = None,
        user_feedback: Optional[str] = None,
    ):
        """
        Add a feedback record from a processing run.

        Args:
            pipeline: Pipeline name
            artifact_id: Artifact identifier
            success: Whether processing succeeded
            processing_time: Processing time in seconds
            parameters: Pipeline parameters used
            error_message: Optional error message
            quality_score: Optional quality score (0-1)
            user_feedback: Optional user feedback text
        """
        record = FeedbackRecord(
            timestamp=datetime.now(),
            pipeline=pipeline,
            artifact_id=artifact_id,
            success=success,
            processing_time=processing_time,
            parameters=parameters,
            error_message=error_message,
            quality_score=quality_score,
            user_feedback=user_feedback,
        )
        self.feedback_records.append(record)

        # Clear cached analysis for this pipeline
        if pipeline in self.pattern_cache:
            del self.pattern_cache[pipeline]

        # Update KPIs
        self._update_kpis(pipeline, success, processing_time, quality_score)

    def _update_kpis(
        self,
        pipeline: str,
        success: bool,
        processing_time: float,
        quality_score: Optional[float],
    ):
        """Update KPI tracking."""
        timestamp = datetime.now()

        # Success rate KPI
        kpi_key = f"{pipeline}:success_rate"
        self.kpi_history[kpi_key].append((timestamp, 1.0 if success else 0.0))

        # Processing time KPI
        kpi_key = f"{pipeline}:processing_time"
        self.kpi_history[kpi_key].append((timestamp, processing_time))

        # Quality score KPI
        if quality_score is not None:
            kpi_key = f"{pipeline}:quality_score"
            self.kpi_history[kpi_key].append((timestamp, quality_score))

    def analyze_patterns(self, pipeline: str, days: int = 30) -> PatternAnalysis:
        """
        Analyze patterns for a pipeline.

        Args:
            pipeline: Pipeline name
            days: Number of days to analyze

        Returns:
            Pattern analysis results
        """
        # Check cache
        cache_key = f"{pipeline}:{days}"
        if cache_key in self.pattern_cache:
            return self.pattern_cache[cache_key]

        # Filter records for this pipeline and time window
        cutoff = datetime.now() - timedelta(days=days)
        records = [
            r for r in self.feedback_records
            if r.pipeline == pipeline and r.timestamp >= cutoff
        ]

        if not records:
            return PatternAnalysis(
                pipeline_name=pipeline,
                total_runs=0,
                success_rate=0.0,
                avg_processing_time=0.0,
                median_processing_time=0.0,
                p95_processing_time=0.0,
            )

        # Calculate statistics
        total_runs = len(records)
        successes = sum(1 for r in records if r.success)
        success_rate = successes / total_runs

        processing_times = [r.processing_time for r in records]
        avg_time = statistics.mean(processing_times)
        median_time = statistics.median(processing_times)
        p95_time = statistics.quantiles(processing_times, n=20)[18] if len(processing_times) >= 20 else max(processing_times)

        # Analyze failure modes
        failure_modes = Counter()
        error_patterns = []
        for record in records:
            if not record.success and record.error_message:
                # Extract error type
                import re
                error_type_match = re.search(r'(\w+Error|\w+Exception)', record.error_message)
                if error_type_match:
                    error_type = error_type_match.group(1)
                    failure_modes[error_type] += 1
                    if error_type not in error_patterns:
                        error_patterns.append(record.error_message[:100])  # First 100 chars

        # Analyze trends (compare first half vs second half of period)
        midpoint = len(records) // 2
        first_half = records[:midpoint]
        second_half = records[midpoint:]

        time_trend = self._analyze_trend(
            [r.processing_time for r in first_half],
            [r.processing_time for r in second_half],
            higher_is_worse=True,
        )

        # Quality trend
        quality_trend = "stable"
        first_half_quality = [r.quality_score for r in first_half if r.quality_score is not None]
        second_half_quality = [r.quality_score for r in second_half if r.quality_score is not None]
        if first_half_quality and second_half_quality:
            quality_trend = self._analyze_trend(
                first_half_quality,
                second_half_quality,
                higher_is_worse=False,
            )

        # Find common and optimal parameters
        parameter_values = defaultdict(Counter)
        successful_parameter_values = defaultdict(Counter)

        for record in records:
            for key, value in record.parameters.items():
                # Convert value to string for counting
                value_str = str(value)
                parameter_values[key][value_str] += 1
                if record.success:
                    successful_parameter_values[key][value_str] += 1

        common_parameters = {
            key: counter.most_common(1)[0][0]
            for key, counter in parameter_values.items()
        }

        optimal_parameters = {
            key: counter.most_common(1)[0][0]
            for key, counter in successful_parameter_values.items()
            if counter
        }

        analysis = PatternAnalysis(
            pipeline_name=pipeline,
            total_runs=total_runs,
            success_rate=success_rate,
            avg_processing_time=avg_time,
            median_processing_time=median_time,
            p95_processing_time=p95_time,
            failure_modes=dict(failure_modes),
            error_patterns=error_patterns,
            time_trend=time_trend,
            quality_trend=quality_trend,
            common_parameters=common_parameters,
            optimal_parameters=optimal_parameters,
        )

        # Cache result
        self.pattern_cache[cache_key] = analysis

        return analysis

    def _analyze_trend(
        self,
        first_values: List[float],
        second_values: List[float],
        higher_is_worse: bool = False,
    ) -> str:
        """
        Analyze trend between two sets of values.

        Args:
            first_values: Values from first period
            second_values: Values from second period
            higher_is_worse: If True, higher values indicate degradation

        Returns:
            "improving", "degrading", or "stable"
        """
        if not first_values or not second_values:
            return "stable"

        first_avg = statistics.mean(first_values)
        second_avg = statistics.mean(second_values)

        # Calculate percentage change
        if first_avg == 0:
            return "stable"

        change_pct = (second_avg - first_avg) / first_avg

        # Threshold for significant change (5%)
        threshold = 0.05

        if abs(change_pct) < threshold:
            return "stable"

        if higher_is_worse:
            return "degrading" if change_pct > 0 else "improving"
        else:
            return "improving" if change_pct > 0 else "degrading"

    def generate_recommendations(self, pipeline: Optional[str] = None) -> List[Recommendation]:
        """
        Generate recommendations for improvement.

        Args:
            pipeline: Optional pipeline to focus on (None = all pipelines)

        Returns:
            List of recommendations
        """
        recommendations = []

        # Get list of pipelines to analyze
        pipelines = {r.pipeline for r in self.feedback_records}
        if pipeline:
            pipelines = {pipeline}

        for pipeline_name in pipelines:
            analysis = self.analyze_patterns(pipeline_name)

            # Recommendation: Low success rate
            if analysis.success_rate < 0.9 and analysis.total_runs >= 10:
                recommendations.append(Recommendation(
                    recommendation_type="regression",
                    severity="high" if analysis.success_rate < 0.8 else "medium",
                    title=f"Low success rate for {pipeline_name}",
                    description=f"Success rate is {analysis.success_rate:.1%}, which is below the 90% threshold.",
                    affected_component=pipeline_name,
                    suggested_action="Review recent changes and error patterns. Consider adding more error handling.",
                    evidence=[
                        f"Total runs: {analysis.total_runs}",
                        f"Failures: {analysis.total_runs - int(analysis.total_runs * analysis.success_rate)}",
                        f"Common errors: {', '.join(list(analysis.failure_modes.keys())[:3])}",
                    ],
                    confidence=0.9,
                ))

            # Recommendation: Performance degradation
            if analysis.time_trend == "degrading" and analysis.total_runs >= 20:
                recommendations.append(Recommendation(
                    recommendation_type="optimization",
                    severity="medium",
                    title=f"Performance degradation in {pipeline_name}",
                    description=f"Processing time has increased. Current average: {analysis.avg_processing_time:.2f}s",
                    affected_component=pipeline_name,
                    suggested_action="Profile the pipeline to identify bottlenecks. Consider optimizing slow operations.",
                    evidence=[
                        f"Average time: {analysis.avg_processing_time:.2f}s",
                        f"P95 time: {analysis.p95_processing_time:.2f}s",
                        f"Trend: {analysis.time_trend}",
                    ],
                    confidence=0.8,
                ))

            # Recommendation: Quality degradation
            if analysis.quality_trend == "degrading" and analysis.total_runs >= 20:
                recommendations.append(Recommendation(
                    recommendation_type="regression",
                    severity="high",
                    title=f"Quality degradation in {pipeline_name}",
                    description="Output quality has decreased compared to previous period.",
                    affected_component=pipeline_name,
                    suggested_action="Review recent parameter changes. Consider reverting to previous settings.",
                    evidence=[
                        f"Quality trend: {analysis.quality_trend}",
                        f"Optimal parameters: {analysis.optimal_parameters}",
                    ],
                    confidence=0.85,
                ))

            # Recommendation: Common failure patterns
            if analysis.failure_modes and analysis.total_runs >= 10:
                for error_type, count in list(analysis.failure_modes.items())[:3]:
                    if count >= 3:  # At least 3 occurrences
                        recommendations.append(Recommendation(
                            recommendation_type="missing_test",
                            severity="medium",
                            title=f"Recurring {error_type} in {pipeline_name}",
                            description=f"This error has occurred {count} times.",
                            affected_component=pipeline_name,
                            suggested_action=f"Add test coverage for {error_type}. Implement better error handling.",
                            evidence=[
                                f"Occurrences: {count}",
                                f"Example: {analysis.error_patterns[0] if analysis.error_patterns else 'N/A'}",
                            ],
                            confidence=0.9,
                        ))

        # Check for undocumented features
        # (Features used in parameters but not in documentation)
        # This would require integration with documentation indexer

        self.recommendations = recommendations
        return recommendations

    def query_natural_language(self, query: str) -> str:
        """
        Natural language query interface.

        Args:
            query: Natural language question

        Returns:
            Structured response with relevant information
        """
        query_lower = query.lower()

        # Success rate queries
        if 'success rate' in query_lower or 'success' in query_lower:
            # Find pipeline mentioned in query
            pipelines = {r.pipeline for r in self.feedback_records}
            mentioned_pipeline = None
            for pipeline in pipelines:
                if pipeline.lower() in query_lower:
                    mentioned_pipeline = pipeline
                    break

            if mentioned_pipeline:
                analysis = self.analyze_patterns(mentioned_pipeline)
                return f"The success rate for {mentioned_pipeline} is {analysis.success_rate:.1%} over the last 30 days ({analysis.total_runs} runs)."
            else:
                # Overall success rate
                total_runs = len(self.feedback_records)
                successes = sum(1 for r in self.feedback_records if r.success)
                overall_rate = successes / total_runs if total_runs > 0 else 0
                return f"Overall success rate across all pipelines is {overall_rate:.1%} ({total_runs} total runs)."

        # Performance queries
        if 'performance' in query_lower or 'speed' in query_lower or 'time' in query_lower:
            pipelines = {r.pipeline for r in self.feedback_records}
            mentioned_pipeline = None
            for pipeline in pipelines:
                if pipeline.lower() in query_lower:
                    mentioned_pipeline = pipeline
                    break

            if mentioned_pipeline:
                analysis = self.analyze_patterns(mentioned_pipeline)
                return f"Performance for {mentioned_pipeline}: Average {analysis.avg_processing_time:.2f}s, Median {analysis.median_processing_time:.2f}s, P95 {analysis.p95_processing_time:.2f}s. Trend: {analysis.time_trend}."
            else:
                # Overall performance
                all_times = [r.processing_time for r in self.feedback_records]
                if all_times:
                    avg_time = statistics.mean(all_times)
                    return f"Average processing time across all pipelines is {avg_time:.2f}s."
                else:
                    return "No performance data available."

        # Error queries
        if 'error' in query_lower or 'fail' in query_lower or 'problem' in query_lower:
            pipelines = {r.pipeline for r in self.feedback_records}
            mentioned_pipeline = None
            for pipeline in pipelines:
                if pipeline.lower() in query_lower:
                    mentioned_pipeline = pipeline
                    break

            if mentioned_pipeline:
                analysis = self.analyze_patterns(mentioned_pipeline)
                if analysis.failure_modes:
                    top_errors = ', '.join([f"{k} ({v})" for k, v in list(analysis.failure_modes.items())[:3]])
                    return f"Top errors for {mentioned_pipeline}: {top_errors}"
                else:
                    return f"No errors recorded for {mentioned_pipeline}."
            else:
                # Overall errors
                error_counter = Counter()
                for record in self.feedback_records:
                    if not record.success and record.error_message:
                        import re
                        error_type_match = re.search(r'(\w+Error|\w+Exception)', record.error_message)
                        if error_type_match:
                            error_counter[error_type_match.group(1)] += 1

                if error_counter:
                    top_errors = ', '.join([f"{k} ({v})" for k, v in error_counter.most_common(3)])
                    return f"Top errors across all pipelines: {top_errors}"
                else:
                    return "No errors recorded."

        # Recommendation queries
        if 'recommend' in query_lower or 'suggest' in query_lower or 'improve' in query_lower:
            recommendations = self.generate_recommendations()
            if recommendations:
                # Return top 3 recommendations
                top_recs = recommendations[:3]
                response = "Top recommendations:\n"
                for i, rec in enumerate(top_recs, 1):
                    response += f"{i}. [{rec.severity.upper()}] {rec.title}\n"
                    response += f"   {rec.suggested_action}\n"
                return response
            else:
                return "No specific recommendations at this time. All pipelines are performing well."

        return "I understand queries about success rates, performance, errors, and recommendations. Please rephrase your question."

    def get_kpi_summary(self, pipeline: Optional[str] = None, days: int = 7) -> Dict:
        """
        Get KPI summary for visualization.

        Args:
            pipeline: Optional pipeline to filter (None = all)
            days: Number of days to include

        Returns:
            Dictionary with KPI data points
        """
        cutoff = datetime.now() - timedelta(days=days)
        summary = {}

        for kpi_key, values in self.kpi_history.items():
            # Filter by pipeline if specified
            if pipeline and not kpi_key.startswith(f"{pipeline}:"):
                continue

            # Filter by time
            filtered_values = [
                (ts, val) for ts, val in values
                if ts >= cutoff
            ]

            if filtered_values:
                summary[kpi_key] = {
                    'data_points': filtered_values,
                    'current': filtered_values[-1][1],
                    'average': statistics.mean([v for _, v in filtered_values]),
                    'min': min([v for _, v in filtered_values]),
                    'max': max([v for _, v in filtered_values]),
                }

        return summary

    def export_knowledge_base(self, output_path: str):
        """Export knowledge base to JSON."""
        data = {
            'export_time': datetime.now().isoformat(),
            'total_feedback_records': len(self.feedback_records),
            'pipelines': list({r.pipeline for r in self.feedback_records}),
            'patterns': {},
            'recommendations': [],
            'kpi_summary': {},
        }

        # Export patterns for each pipeline
        pipelines = {r.pipeline for r in self.feedback_records}
        for pipeline in pipelines:
            analysis = self.analyze_patterns(pipeline)
            data['patterns'][pipeline] = {
                'total_runs': analysis.total_runs,
                'success_rate': analysis.success_rate,
                'avg_processing_time': analysis.avg_processing_time,
                'median_processing_time': analysis.median_processing_time,
                'p95_processing_time': analysis.p95_processing_time,
                'failure_modes': analysis.failure_modes,
                'time_trend': analysis.time_trend,
                'quality_trend': analysis.quality_trend,
                'common_parameters': analysis.common_parameters,
                'optimal_parameters': analysis.optimal_parameters,
            }

        # Export recommendations
        recommendations = self.generate_recommendations()
        for rec in recommendations:
            data['recommendations'].append({
                'type': rec.recommendation_type,
                'severity': rec.severity,
                'title': rec.title,
                'description': rec.description,
                'affected_component': rec.affected_component,
                'suggested_action': rec.suggested_action,
                'confidence': rec.confidence,
            })

        # Export KPI summary
        data['kpi_summary'] = self.get_kpi_summary(days=30)
        # Convert timestamps to strings for JSON serialization
        for kpi_key, kpi_data in data['kpi_summary'].items():
            kpi_data['data_points'] = [
                (ts.isoformat(), val) for ts, val in kpi_data['data_points']
            ]

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)


def main():
    """CLI for knowledge engine."""
    import argparse

    parser = argparse.ArgumentParser(description='Knowledge Integration Engine')
    parser.add_argument('--feedback-file', help='JSON file with feedback records')
    parser.add_argument('--query', help='Natural language query')
    parser.add_argument('--analyze-pipeline', help='Analyze specific pipeline')
    parser.add_argument('--recommendations', action='store_true', help='Generate recommendations')
    parser.add_argument('--export', help='Export knowledge base to JSON')
    parser.add_argument('--days', type=int, default=30, help='Days to analyze (default: 30)')

    args = parser.parse_args()

    engine = KnowledgeIntegrationEngine()

    # Load feedback if provided
    if args.feedback_file:
        with open(args.feedback_file) as f:
            feedback_data = json.load(f)

        for record_data in feedback_data.get('records', []):
            engine.add_feedback(
                pipeline=record_data['pipeline'],
                artifact_id=record_data['artifact_id'],
                success=record_data['success'],
                processing_time=record_data['processing_time'],
                parameters=record_data.get('parameters', {}),
                error_message=record_data.get('error_message'),
                quality_score=record_data.get('quality_score'),
            )
        print(f"Loaded {len(engine.feedback_records)} feedback records")

    # Process query
    if args.query:
        response = engine.query_natural_language(args.query)
        print(f"\nQuery: {args.query}")
        print(f"Response: {response}")

    # Analyze pipeline
    if args.analyze_pipeline:
        analysis = engine.analyze_patterns(args.analyze_pipeline, days=args.days)
        print(f"\nPattern Analysis for {args.analyze_pipeline}")
        print(f"  Total runs: {analysis.total_runs}")
        print(f"  Success rate: {analysis.success_rate:.1%}")
        print(f"  Avg time: {analysis.avg_processing_time:.2f}s")
        print(f"  Median time: {analysis.median_processing_time:.2f}s")
        print(f"  P95 time: {analysis.p95_processing_time:.2f}s")
        print(f"  Time trend: {analysis.time_trend}")
        print(f"  Quality trend: {analysis.quality_trend}")

        if analysis.failure_modes:
            print("\n  Failure modes:")
            for error_type, count in analysis.failure_modes.items():
                print(f"    {error_type}: {count}")

    # Generate recommendations
    if args.recommendations:
        recommendations = engine.generate_recommendations()
        print(f"\nGenerated {len(recommendations)} recommendations")
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. [{rec.severity.upper()}] {rec.title}")
            print(f"   {rec.description}")
            print(f"   Action: {rec.suggested_action}")
            print(f"   Confidence: {rec.confidence:.0%}")

    # Export
    if args.export:
        engine.export_knowledge_base(args.export)
        print(f"\nExported knowledge base to {args.export}")


if __name__ == '__main__':
    main()
