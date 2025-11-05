"""
Tests for RAG Knowledge Integration Engine.
"""

import pytest
import sys
from pathlib import Path

# Add agents directory to path
agents_path = Path(__file__).parent.parent / '.github' / 'agents'
sys.path.insert(0, str(agents_path))

from rag_system.knowledge_engine import (  # noqa: E402
    KnowledgeIntegrationEngine,
)


@pytest.fixture
def engine():
    """Create a knowledge engine instance."""
    return KnowledgeIntegrationEngine()


@pytest.fixture
def sample_feedback_data():
    """Create sample feedback data."""
    return [
        {
            'pipeline': 'depth_pipeline',
            'artifact_id': 'img001',
            'success': True,
            'processing_time': 2.5,
            'parameters': {'quality': 'high', 'denoise': 0.5},
            'quality_score': 0.92,
        },
        {
            'pipeline': 'depth_pipeline',
            'artifact_id': 'img002',
            'success': True,
            'processing_time': 2.3,
            'parameters': {'quality': 'high', 'denoise': 0.5},
            'quality_score': 0.91,
        },
        {
            'pipeline': 'depth_pipeline',
            'artifact_id': 'img003',
            'success': False,
            'processing_time': 0.5,
            'parameters': {'quality': 'high', 'denoise': 0.5},
            'error_message': 'ValueError: invalid input shape',
        },
    ]


class TestFeedbackManagement:
    """Test feedback record management."""

    def test_add_feedback(self, engine):
        """Test adding feedback."""
        engine.add_feedback(
            pipeline='depth_pipeline',
            artifact_id='test001',
            success=True,
            processing_time=2.5,
            parameters={'quality': 'high'},
            quality_score=0.9,
        )

        assert len(engine.feedback_records) == 1
        assert engine.feedback_records[0].pipeline == 'depth_pipeline'
        assert engine.feedback_records[0].success is True

    def test_add_multiple_feedback(self, engine, sample_feedback_data):
        """Test adding multiple feedback records."""
        for data in sample_feedback_data:
            engine.add_feedback(**data)

        assert len(engine.feedback_records) == len(sample_feedback_data)

    def test_feedback_updates_kpis(self, engine):
        """Test that feedback updates KPIs."""
        engine.add_feedback(
            pipeline='test_pipeline',
            artifact_id='test001',
            success=True,
            processing_time=1.5,
            parameters={},
            quality_score=0.85,
        )

        # Check KPIs were recorded
        assert 'test_pipeline:success_rate' in engine.kpi_history
        assert 'test_pipeline:processing_time' in engine.kpi_history
        assert 'test_pipeline:quality_score' in engine.kpi_history


class TestPatternAnalysis:
    """Test pattern analysis."""

    def test_analyze_empty_pipeline(self, engine):
        """Test analysis with no data."""
        analysis = engine.analyze_patterns('nonexistent_pipeline')

        assert analysis.pipeline_name == 'nonexistent_pipeline'
        assert analysis.total_runs == 0
        assert analysis.success_rate == 0.0

    def test_analyze_with_data(self, engine, sample_feedback_data):
        """Test analysis with data."""
        for data in sample_feedback_data:
            engine.add_feedback(**data)

        analysis = engine.analyze_patterns('depth_pipeline')

        assert analysis.total_runs == 3
        assert analysis.success_rate == pytest.approx(2/3, rel=0.01)
        assert analysis.avg_processing_time > 0
        assert analysis.median_processing_time > 0

    def test_analyze_failure_modes(self, engine, sample_feedback_data):
        """Test failure mode analysis."""
        for data in sample_feedback_data:
            engine.add_feedback(**data)

        # Add more failures
        engine.add_feedback(
            pipeline='depth_pipeline',
            artifact_id='img004',
            success=False,
            processing_time=0.3,
            parameters={},
            error_message='ValueError: invalid dimensions',
        )

        analysis = engine.analyze_patterns('depth_pipeline')

        assert len(analysis.failure_modes) > 0
        assert 'ValueError' in analysis.failure_modes

    def test_analyze_common_parameters(self, engine, sample_feedback_data):
        """Test common parameter detection."""
        for data in sample_feedback_data:
            engine.add_feedback(**data)

        analysis = engine.analyze_patterns('depth_pipeline')

        assert 'quality' in analysis.common_parameters
        assert analysis.common_parameters['quality'] == 'high'

    def test_analyze_optimal_parameters(self, engine):
        """Test optimal parameter detection."""
        # Add successful runs with one parameter set
        for i in range(5):
            engine.add_feedback(
                pipeline='test_pipeline',
                artifact_id=f'test{i:03d}',
                success=True,
                processing_time=2.0,
                parameters={'quality': 'high', 'denoise': 0.5},
            )

        # Add failed runs with different parameters
        for i in range(2):
            engine.add_feedback(
                pipeline='test_pipeline',
                artifact_id=f'test{i+5:03d}',
                success=False,
                processing_time=1.0,
                parameters={'quality': 'low', 'denoise': 0.1},
                error_message='Processing failed',
            )

        analysis = engine.analyze_patterns('test_pipeline')

        assert 'quality' in analysis.optimal_parameters
        assert analysis.optimal_parameters['quality'] == 'high'

    def test_trend_analysis_stable(self, engine):
        """Test stable trend detection."""
        # Add consistent data
        for i in range(20):
            engine.add_feedback(
                pipeline='test_pipeline',
                artifact_id=f'test{i:03d}',
                success=True,
                processing_time=2.0,
                parameters={},
                quality_score=0.9,
            )

        analysis = engine.analyze_patterns('test_pipeline')

        assert analysis.time_trend == 'stable'
        assert analysis.quality_trend == 'stable'

    def test_trend_analysis_improving(self, engine):
        """Test improving trend detection."""
        # First half: slower/lower quality
        for i in range(10):
            engine.add_feedback(
                pipeline='test_pipeline',
                artifact_id=f'test{i:03d}',
                success=True,
                processing_time=3.0,
                parameters={},
                quality_score=0.7,
            )

        # Second half: faster/higher quality
        for i in range(10, 20):
            engine.add_feedback(
                pipeline='test_pipeline',
                artifact_id=f'test{i:03d}',
                success=True,
                processing_time=2.0,
                parameters={},
                quality_score=0.9,
            )

        analysis = engine.analyze_patterns('test_pipeline')

        assert analysis.time_trend == 'improving'
        assert analysis.quality_trend == 'improving'

    def test_trend_analysis_degrading(self, engine):
        """Test degrading trend detection."""
        # First half: faster/better
        for i in range(10):
            engine.add_feedback(
                pipeline='test_pipeline',
                artifact_id=f'test{i:03d}',
                success=True,
                processing_time=2.0,
                parameters={},
                quality_score=0.9,
            )

        # Second half: slower/worse
        for i in range(10, 20):
            engine.add_feedback(
                pipeline='test_pipeline',
                artifact_id=f'test{i:03d}',
                success=True,
                processing_time=3.5,
                parameters={},
                quality_score=0.6,
            )

        analysis = engine.analyze_patterns('test_pipeline')

        assert analysis.time_trend == 'degrading'
        assert analysis.quality_trend == 'degrading'


class TestRecommendations:
    """Test recommendation generation."""

    def test_generate_recommendations_empty(self, engine):
        """Test recommendations with no data."""
        recommendations = engine.generate_recommendations()
        assert len(recommendations) == 0

    def test_recommend_low_success_rate(self, engine):
        """Test recommendation for low success rate."""
        # Add mostly failures
        for i in range(15):
            success = i % 5 != 0  # Failures every 5th iteration (i.e., 20% failure rate)
            engine.add_feedback(
                pipeline='failing_pipeline',
                artifact_id=f'test{i:03d}',
                success=success,
                processing_time=1.0,
                parameters={},
                error_message='Processing error' if not success else None,
            )

        recommendations = engine.generate_recommendations('failing_pipeline')

        # Should recommend addressing success rate
        assert len(recommendations) > 0
        rec_types = [r.recommendation_type for r in recommendations]
        assert 'regression' in rec_types

    def test_recommend_performance_degradation(self, engine):
        """Test recommendation for performance degradation."""
        # First half: fast
        for i in range(15):
            engine.add_feedback(
                pipeline='slow_pipeline',
                artifact_id=f'test{i:03d}',
                success=True,
                processing_time=1.0,
                parameters={},
            )

        # Second half: slow
        for i in range(15, 30):
            engine.add_feedback(
                pipeline='slow_pipeline',
                artifact_id=f'test{i:03d}',
                success=True,
                processing_time=3.0,
                parameters={},
            )

        recommendations = engine.generate_recommendations('slow_pipeline')

        # Should recommend optimization
        rec_types = [r.recommendation_type for r in recommendations]
        assert 'optimization' in rec_types

    def test_recommend_quality_regression(self, engine):
        """Test recommendation for quality regression."""
        # First half: good quality
        for i in range(15):
            engine.add_feedback(
                pipeline='quality_pipeline',
                artifact_id=f'test{i:03d}',
                success=True,
                processing_time=1.0,
                parameters={},
                quality_score=0.9,
            )

        # Second half: poor quality
        for i in range(15, 30):
            engine.add_feedback(
                pipeline='quality_pipeline',
                artifact_id=f'test{i:03d}',
                success=True,
                processing_time=1.0,
                parameters={},
                quality_score=0.5,
            )

        recommendations = engine.generate_recommendations('quality_pipeline')

        # Should recommend addressing quality
        rec_types = [r.recommendation_type for r in recommendations]
        assert 'regression' in rec_types

    def test_recommend_missing_tests(self, engine):
        """Test recommendation for recurring errors."""
        # Add recurring error
        for i in range(12):
            engine.add_feedback(
                pipeline='error_pipeline',
                artifact_id=f'test{i:03d}',
                success=i % 3 != 0,  # 33% failure rate
                processing_time=1.0,
                parameters={},
                error_message='ValueError: invalid input' if i % 3 == 0 else None,
            )

        recommendations = engine.generate_recommendations('error_pipeline')

        # Should recommend adding tests
        rec_types = [r.recommendation_type for r in recommendations]
        assert 'missing_test' in rec_types or 'regression' in rec_types


class TestNaturalLanguageQuery:
    """Test natural language query interface."""

    def test_query_success_rate(self, engine, sample_feedback_data):
        """Test querying success rate."""
        for data in sample_feedback_data:
            engine.add_feedback(**data)

        response = engine.query_natural_language("What is the success rate for depth_pipeline?")

        assert 'success rate' in response.lower()
        assert 'depth_pipeline' in response

    def test_query_performance(self, engine, sample_feedback_data):
        """Test querying performance."""
        for data in sample_feedback_data:
            engine.add_feedback(**data)

        response = engine.query_natural_language("How is the performance of depth_pipeline?")

        assert 'performance' in response.lower() or 'average' in response.lower()

    def test_query_errors(self, engine, sample_feedback_data):
        """Test querying errors."""
        for data in sample_feedback_data:
            engine.add_feedback(**data)

        response = engine.query_natural_language("What errors occurred in depth_pipeline?")

        assert 'error' in response.lower() or 'valueerror' in response.lower()

    def test_query_recommendations(self, engine):
        """Test querying recommendations."""
        # Add data that triggers recommendations
        for i in range(12):
            engine.add_feedback(
                pipeline='test_pipeline',
                artifact_id=f'test{i:03d}',
                success=i % 2 == 0,  # 50% failure rate
                processing_time=1.0,
                parameters={},
                error_message='Error occurred' if i % 2 != 0 else None,
            )

        response = engine.query_natural_language("What improvements do you recommend?")

        assert 'recommend' in response.lower() or 'improve' in response.lower()

    def test_query_overall_stats(self, engine):
        """Test querying overall statistics."""
        engine.add_feedback(
            pipeline='pipeline1',
            artifact_id='test001',
            success=True,
            processing_time=1.0,
            parameters={},
        )
        engine.add_feedback(
            pipeline='pipeline2',
            artifact_id='test002',
            success=True,
            processing_time=2.0,
            parameters={},
        )

        response = engine.query_natural_language("What is the overall success rate?")

        assert 'success rate' in response.lower()
        assert 'overall' in response.lower()


class TestKPITracking:
    """Test KPI tracking and visualization."""

    def test_kpi_tracking(self, engine):
        """Test KPI updates."""
        engine.add_feedback(
            pipeline='test_pipeline',
            artifact_id='test001',
            success=True,
            processing_time=1.5,
            parameters={},
            quality_score=0.9,
        )

        # Check KPIs
        assert len(engine.kpi_history) > 0
        assert 'test_pipeline:success_rate' in engine.kpi_history
        assert 'test_pipeline:processing_time' in engine.kpi_history
        assert 'test_pipeline:quality_score' in engine.kpi_history

    def test_get_kpi_summary(self, engine):
        """Test KPI summary generation."""
        # Add some data
        for i in range(5):
            engine.add_feedback(
                pipeline='test_pipeline',
                artifact_id=f'test{i:03d}',
                success=True,
                processing_time=1.0 + i * 0.1,
                parameters={},
                quality_score=0.9 - i * 0.01,
            )

        summary = engine.get_kpi_summary(pipeline='test_pipeline', days=7)

        assert len(summary) > 0
        for kpi_key, kpi_data in summary.items():
            assert 'data_points' in kpi_data
            assert 'current' in kpi_data
            assert 'average' in kpi_data
            assert 'min' in kpi_data
            assert 'max' in kpi_data

    def test_kpi_summary_filtering(self, engine):
        """Test KPI summary filtering by pipeline."""
        engine.add_feedback(
            pipeline='pipeline1',
            artifact_id='test001',
            success=True,
            processing_time=1.0,
            parameters={},
        )
        engine.add_feedback(
            pipeline='pipeline2',
            artifact_id='test002',
            success=True,
            processing_time=2.0,
            parameters={},
        )

        summary = engine.get_kpi_summary(pipeline='pipeline1', days=7)

        # Should only include pipeline1 KPIs
        for kpi_key in summary.keys():
            assert kpi_key.startswith('pipeline1:')


class TestExport:
    """Test knowledge base export."""

    def test_export_knowledge_base(self, engine, sample_feedback_data, tmp_path):
        """Test exporting knowledge base."""
        for data in sample_feedback_data:
            engine.add_feedback(**data)

        output_file = tmp_path / "knowledge_base.json"
        engine.export_knowledge_base(str(output_file))

        assert output_file.exists()

        # Verify JSON structure
        import json
        with open(output_file) as f:
            data = json.load(f)

        assert 'export_time' in data
        assert 'total_feedback_records' in data
        assert 'pipelines' in data
        assert 'patterns' in data
        assert 'recommendations' in data
        assert 'kpi_summary' in data

    def test_export_includes_patterns(self, engine, sample_feedback_data, tmp_path):
        """Test that export includes pattern analysis."""
        for data in sample_feedback_data:
            engine.add_feedback(**data)

        output_file = tmp_path / "knowledge_base.json"
        engine.export_knowledge_base(str(output_file))

        import json
        with open(output_file) as f:
            data = json.load(f)

        assert 'depth_pipeline' in data['patterns']
        pattern = data['patterns']['depth_pipeline']
        assert 'success_rate' in pattern
        assert 'avg_processing_time' in pattern


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
