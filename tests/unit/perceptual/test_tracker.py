"""Unit tests for perceptual.tracker.

Covers TrajectoryPoint and EnhancementTrajectory dataclasses, trajectory
arithmetic (improvement, progress, trend), and EnhancementTracker lifecycle
(baseline establishment, step tracking, summary generation) using lightweight
in-process AnalysisResult stubs — no GPU or real image files required.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_perceptual_score(score_value: float, higher_is_better: bool = True):
    from transformation_portal.perceptual.metrics import MetricType, PerceptualScore

    return PerceptualScore(
        metric_type=MetricType.PSNR,
        score=score_value,
        higher_is_better=higher_is_better,
        normalized_score=min(score_value / 50.0, 1.0),
        metadata={},
    )


def _make_image_metadata():
    from transformation_portal.perceptual.image_loader import ImageMetadata

    return ImageMetadata(
        path=Path("/fake/image.jpg"),
        image_type=None,
        width=100,
        height=100,
        channels=3,
        format="JPEG",
        size_bytes=10000,
        bit_depth=8,
        color_space="RGB",
        mean_intensity=0.5,
        std_intensity=0.2,
        dynamic_range=255.0,
        tags={},
    )


def _make_analysis_result(overall_quality: float = 0.7):
    from transformation_portal.perceptual.analyzer import AnalysisResult
    from transformation_portal.perceptual.metrics import MetricType

    return AnalysisResult(
        image_path=Path("/fake/image.jpg"),
        image_metadata=_make_image_metadata(),
        quality_scores={MetricType.PSNR: _make_perceptual_score(35.0)},
        overall_quality=overall_quality,
        analysis_time=0.1,
        timestamp=time.time(),
    )


# ---------------------------------------------------------------------------
# TrajectoryPoint dataclass
# ---------------------------------------------------------------------------


class TestTrajectoryPoint:
    def test_fields_stored(self):
        from transformation_portal.perceptual.tracker import TrajectoryPoint

        pt = TrajectoryPoint(
            step=1,
            timestamp=1000.0,
            overall_quality=0.75,
            metric_scores={"psnr": 35.0},
            description="step 1",
        )
        assert pt.step == 1
        assert pt.overall_quality == pytest.approx(0.75)
        assert pt.metric_scores["psnr"] == pytest.approx(35.0)
        assert pt.description == "step 1"

    def test_default_description_empty(self):
        from transformation_portal.perceptual.tracker import TrajectoryPoint

        pt = TrajectoryPoint(step=0, timestamp=0.0, overall_quality=0.5, metric_scores={})
        assert pt.description == ""


# ---------------------------------------------------------------------------
# EnhancementTrajectory
# ---------------------------------------------------------------------------


class TestEnhancementTrajectory:
    def test_add_point_increases_length(self):
        from transformation_portal.perceptual.tracker import EnhancementTrajectory, TrajectoryPoint

        traj = EnhancementTrajectory(image_name="img", baseline_quality=0.5)
        traj.add_point(TrajectoryPoint(step=1, timestamp=1.0, overall_quality=0.6, metric_scores={}))
        assert len(traj.points) == 1

    def test_get_improvement_positive(self):
        from transformation_portal.perceptual.tracker import EnhancementTrajectory, TrajectoryPoint

        traj = EnhancementTrajectory(image_name="img", baseline_quality=0.5)
        traj.add_point(TrajectoryPoint(step=1, timestamp=1.0, overall_quality=0.7, metric_scores={}))
        assert traj.get_improvement() == pytest.approx(0.2, abs=1e-6)

    def test_get_improvement_zero_with_no_points(self):
        from transformation_portal.perceptual.tracker import EnhancementTrajectory

        traj = EnhancementTrajectory(image_name="img", baseline_quality=0.5)
        assert traj.get_improvement() == pytest.approx(0.0, abs=1e-6)

    def test_is_improving_true_when_quality_rises(self):
        from transformation_portal.perceptual.tracker import EnhancementTrajectory, TrajectoryPoint

        traj = EnhancementTrajectory(image_name="img", baseline_quality=0.5)
        traj.add_point(TrajectoryPoint(step=1, timestamp=1.0, overall_quality=0.6, metric_scores={}))
        traj.add_point(TrajectoryPoint(step=2, timestamp=2.0, overall_quality=0.7, metric_scores={}))
        assert traj.is_improving() is True

    def test_is_improving_false_with_no_points(self):
        from transformation_portal.perceptual.tracker import EnhancementTrajectory

        traj = EnhancementTrajectory(image_name="img", baseline_quality=0.5)
        assert traj.is_improving() is False

    def test_get_progress_with_target(self):
        from transformation_portal.perceptual.tracker import EnhancementTrajectory, TrajectoryPoint

        traj = EnhancementTrajectory(
            image_name="img",
            baseline_quality=0.0,
            target_quality=1.0,
        )
        traj.add_point(TrajectoryPoint(step=1, timestamp=1.0, overall_quality=0.5, metric_scores={}))
        assert traj.get_progress() == pytest.approx(0.5, abs=0.01)


# ---------------------------------------------------------------------------
# EnhancementTracker
# ---------------------------------------------------------------------------


class TestEnhancementTracker:
    def test_establish_baseline_creates_trajectory(self):
        from transformation_portal.perceptual.tracker import EnhancementTracker

        tracker = EnhancementTracker()
        result = _make_analysis_result(overall_quality=0.6)
        tracker.establish_baseline([result])
        assert tracker.get_trajectory(result.image_path.stem) is not None

    def test_track_enhancement_adds_points(self):
        from transformation_portal.perceptual.tracker import EnhancementTracker

        tracker = EnhancementTracker()
        result = _make_analysis_result(overall_quality=0.6)
        tracker.establish_baseline([result])

        improved = _make_analysis_result(overall_quality=0.75)
        tracker.track_enhancement(improved, step=1, description="after depth")
        traj = tracker.get_trajectory(improved.image_path.stem)
        assert len(traj.points) == 2
        assert traj.points[-1].step == 1
        assert traj.points[-1].description == "after depth"

    def test_get_trajectory_returns_none_for_unknown(self):
        from transformation_portal.perceptual.tracker import EnhancementTracker

        tracker = EnhancementTracker()
        assert tracker.get_trajectory("nonexistent") is None

    def test_get_all_trajectories_returns_dict(self):
        from transformation_portal.perceptual.tracker import EnhancementTracker

        tracker = EnhancementTracker()
        result = _make_analysis_result()
        tracker.establish_baseline([result])
        assert isinstance(tracker.get_all_trajectories(), dict)

    def test_get_summary_returns_dict(self):
        from transformation_portal.perceptual.tracker import EnhancementTracker

        tracker = EnhancementTracker()
        result = _make_analysis_result()
        tracker.establish_baseline([result])
        summary = tracker.get_summary()
        assert isinstance(summary, dict)
