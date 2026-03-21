"""Contract tests for PerformanceCapsule schema stability.

These tests ensure the PerformanceCapsule schema remains stable and backward-compatible.
Breaking changes here require explicit version bumps and migration plans.
"""

import json
from datetime import datetime

import pytest

from transformation_portal.metrics.performance_capsule import (

pytestmark = pytest.mark.unit

    DEFAULT_BUCKETS,
    PerformanceBucket,
    PerformanceCapsule,
    compute_config_hash,
    compute_dimension_adjustment,
    compute_specificity,
    get_bucket_for_capsule,
)


@pytest.fixture
def make_capsule():
    """Factory fixture for creating PerformanceCapsule variants.

    Usage:
        def test_something(make_capsule):
            capsule = make_capsule(scene_type="pool", pixel_count=20_000_000)
            assert ...
    """

    def _make(**overrides):
        base = {
            "image_id": "test_image",
            "image_path": "test.tiff",
            "input_hash": "abc123def456",
            "original_shape": (6000, 8000),
            "enforced_shape": (6000, 8000),
            "pixel_count": 48_000_000,
            "dimension_adjustment": "exact",
            "scene_type": None,
            "texture_complexity": None,
            "tile_config": None,
            "tile_count": None,
            "backend_id": "depth_pro",
            "device": "mps",
            "dtype": "float32",
            "cache_hit": False,
            "timings": {"total": 10.0},
            "config_hash": "test_config_hash",
            "pipeline_version": "3.0.0",
            "quality_score": None,
            "firewall_status": "pass",
        }
        base.update(overrides)
        return PerformanceCapsule(**base)

    return _make


class TestPerformanceCapsuleContract:
    """Contract tests for PerformanceCapsule schema."""

    def test_required_fields(self):
        """Test that required fields are enforced."""
        # Missing image_id - TypeError from dataclass
        with pytest.raises(TypeError):
            PerformanceCapsule(
                image_path="test.tiff",
                input_hash="abc123",
                original_shape=(6000, 8000),
                enforced_shape=(5992, 7994),
                pixel_count=47892448,
                dimension_adjustment="cropped_0.2%",
                timings={"total": 10.0},
            )

        # Missing timings - ValueError from __post_init__
        with pytest.raises(ValueError, match="timings dict is required"):
            PerformanceCapsule(
                image_id="test",
                image_path="test.tiff",
                input_hash="abc123",
                original_shape=(6000, 8000),
                enforced_shape=(5992, 7994),
                pixel_count=47892448,
                dimension_adjustment="cropped_0.2%",
            )

    def test_timings_validation(self):
        """Test that timings must include 'total' key."""
        with pytest.raises(ValueError, match="must include 'total' key"):
            PerformanceCapsule(
                image_id="test",
                image_path="test.tiff",
                input_hash="abc123",
                original_shape=(6000, 8000),
                enforced_shape=(5992, 7994),
                pixel_count=47892448,
                dimension_adjustment="cropped_0.2%",
                timings={"inference": 8.0},
            )

    def test_pixel_count_validation(self):
        """Test that pixel_count must be positive."""
        with pytest.raises(ValueError, match="pixel_count must be positive"):
            PerformanceCapsule(
                image_id="test",
                image_path="test.tiff",
                input_hash="abc123",
                original_shape=(6000, 8000),
                enforced_shape=(5992, 7994),
                pixel_count=0,
                dimension_adjustment="cropped_0.2%",
                timings={"total": 10.0},
            )

    def test_minimal_valid_capsule(self, make_capsule):
        """Test that minimal valid capsule can be created."""
        capsule = make_capsule()

        assert capsule.image_id == "test_image"
        assert capsule.timings["total"] == 10.0
        assert capsule.schema_version == "2.0.0"

    def test_full_capsule_roundtrip(self):
        """Test that full capsule can be serialized and deserialized."""
        original = PerformanceCapsule(
            image_id="750_Picacho_Pool",
            image_path="/input/750_Picacho_Pool.tiff",
            input_hash="abc123def456",
            original_shape=(6000, 8000),
            enforced_shape=(5992, 7994),
            pixel_count=47892448,
            dimension_adjustment="cropped_0.2%",
            tile_config={"size": 512, "overlap": 64},
            tile_count=144,
            backend_id="da3",
            model_variant="depth_anything_v3_vits",
            device="mps",
            dtype="float16",
            cache_hit=False,
            cache_key="cache_abc123",
            timings={
                "total": 11.49,
                "load_decode": 0.8,
                "preprocess": 0.2,
                "inference": 8.2,
                "postprocess": 0.5,
                "write_depth": 1.79,
            },
            scene_type="pool",
            texture_complexity="mixed",
            config_hash="abc123",
            pipeline_version="2.0.0",
            quality_score=0.95,
            firewall_status="pass",
        )

        # Serialize
        capsule_dict = original.to_dict()
        json_str = json.dumps(capsule_dict)

        # Deserialize
        loaded_dict = json.loads(json_str)
        reconstructed = PerformanceCapsule.from_dict(loaded_dict)

        # Validate
        assert reconstructed.image_id == original.image_id
        assert reconstructed.timings == original.timings
        assert reconstructed.scene_type == original.scene_type
        assert reconstructed.enforced_shape == original.enforced_shape
        assert reconstructed.tile_count == original.tile_count

    def test_schema_version_stability(self, make_capsule):
        """Test that schema version is stable."""
        capsule = make_capsule()

        # Schema version must be 1.0.0 (contract)
        assert capsule.schema_version == "2.0.0"

    def test_captured_at_auto_populated(self, make_capsule):
        """Test that captured_at is auto-populated with ISO8601 timestamp."""
        capsule = make_capsule()

        # Should be valid ISO8601
        timestamp = datetime.fromisoformat(capsule.captured_at)
        assert timestamp.tzinfo is not None  # Must be timezone-aware


class TestComputeConfigHash:
    """Tests for config hash computation."""

    def test_deterministic_hash(self):
        """Test that hash is deterministic for same config."""
        config = {"backend": "da3", "quality_tier": "stable"}

        hash1 = compute_config_hash(config)
        hash2 = compute_config_hash(config)

        assert hash1 == hash2

    def test_key_order_independent(self):
        """Test that hash is independent of key order."""
        config1 = {"backend": "da3", "quality_tier": "stable"}
        config2 = {"quality_tier": "stable", "backend": "da3"}

        assert compute_config_hash(config1) == compute_config_hash(config2)

    def test_different_configs_different_hash(self):
        """Test that different configs produce different hashes."""
        config1 = {"backend": "da3"}
        config2 = {"backend": "depth_pro"}

        assert compute_config_hash(config1) != compute_config_hash(config2)


class TestComputeDimensionAdjustment:
    """Tests for dimension adjustment computation."""

    def test_exact_match(self):
        """Test exact dimension match."""
        result = compute_dimension_adjustment((6000, 8000), (6000, 8000))
        assert result == "exact"

    def test_cropped(self):
        """Test cropped dimension."""
        result = compute_dimension_adjustment((6000, 8000), (5992, 7994))
        assert result.startswith("cropped_")
        assert "0.2%" in result

    def test_padded(self):
        """Test padded dimension."""
        result = compute_dimension_adjustment((6000, 8000), (6008, 8008))
        assert result.startswith("padded_")


class TestComputeSpecificity:
    """Tests for specificity score computation."""

    def test_empty_filters_zero_score(self):
        """Empty filters should have zero specificity."""
        assert compute_specificity({}) == 0

    def test_scene_type_highest_score(self):
        """scene_type should be highest priority."""
        assert compute_specificity({"scene_type": "pool"}) == 10

    def test_device_medium_score(self):
        """device should be medium priority."""
        assert compute_specificity({"device": "mps"}) == 5

    def test_backend_id_medium_score(self):
        """backend_id should be medium priority."""
        assert compute_specificity({"backend_id": "da3"}) == 5

    def test_pixel_count_range_single_concept(self):
        """pixel_count range should count as ONE concept, not two."""
        score_min_only = compute_specificity({"pixel_count_min": 1000})
        score_max_only = compute_specificity({"pixel_count_max": 5000})
        score_both = compute_specificity({"pixel_count_min": 1000, "pixel_count_max": 5000})

        # All should have same score (3)
        assert score_min_only == 3
        assert score_max_only == 3
        assert score_both == 3

    def test_combined_scores_add(self):
        """Combined filters should add scores."""
        score = compute_specificity(
            {
                "scene_type": "pool",  # +10
                "device": "mps",  # +5
                "pixel_count_min": 1000,  # +3
            }
        )
        assert score == 18


class TestPerformanceBucket:
    """Tests for performance bucket matching."""

    def test_bucket_matches_scene_type(self, make_capsule):
        """Test bucket matching on scene_type."""
        bucket = PerformanceBucket(
            name="pool_test",
            filters={"scene_type": "pool"},
            p50_threshold_sec=10.0,
            p95_threshold_sec=15.0,
        )

        assert bucket.matches(make_capsule(scene_type="pool"))
        assert not bucket.matches(make_capsule(scene_type="aerial"))

    def test_bucket_matches_pixel_count_range(self, make_capsule):
        """Test bucket matching on pixel_count range."""
        bucket = PerformanceBucket(
            name="large_test",
            filters={"pixel_count_min": 20_000_000, "pixel_count_max": 50_000_000},
            p50_threshold_sec=10.0,
            p95_threshold_sec=15.0,
        )

        assert bucket.matches(make_capsule(pixel_count=48_000_000))
        assert not bucket.matches(make_capsule(pixel_count=10_000_000))

    def test_pixel_count_range_boundaries(self, make_capsule):
        """Test boundary conditions for pixel_count ranges (inclusive)."""
        bucket = PerformanceBucket(
            name="test_bucket",
            filters={"pixel_count_min": 20_000_000, "pixel_count_max": 50_000_000},
            p50_threshold_sec=10.0,
            p95_threshold_sec=15.0,
        )

        # Exactly at min (inclusive)
        assert bucket.matches(make_capsule(pixel_count=20_000_000))

        # Exactly at max (inclusive)
        assert bucket.matches(make_capsule(pixel_count=50_000_000))

        # Just below min (exclusive)
        assert not bucket.matches(make_capsule(pixel_count=19_999_999))

        # Just above max (exclusive)
        assert not bucket.matches(make_capsule(pixel_count=50_000_001))

        # Well within range
        assert bucket.matches(make_capsule(pixel_count=35_000_000))

    def test_specificity_property(self):
        """Test that specificity property works."""
        bucket_scene = PerformanceBucket(
            name="scene_bucket",
            filters={"scene_type": "pool"},
            p50_threshold_sec=10.0,
            p95_threshold_sec=15.0,
        )

        assert bucket_scene.specificity == 10

    def test_check_threshold_p50(self):
        """Test threshold checking for p50."""
        bucket = PerformanceBucket(
            name="test",
            filters={},
            p50_threshold_sec=10.0,
            p95_threshold_sec=15.0,
        )

        assert bucket.check_threshold(50, 9.0)
        assert bucket.check_threshold(50, 10.0)
        assert not bucket.check_threshold(50, 11.0)

    def test_check_threshold_p95(self):
        """Test threshold checking for p95."""
        bucket = PerformanceBucket(
            name="test",
            filters={},
            p50_threshold_sec=10.0,
            p95_threshold_sec=15.0,
        )

        assert bucket.check_threshold(95, 14.0)
        assert bucket.check_threshold(95, 15.0)
        assert not bucket.check_threshold(95, 16.0)

    def test_check_threshold_optional_percentiles(self):
        """Test that optional percentiles pass if not set."""
        bucket = PerformanceBucket(
            name="test",
            filters={},
            p50_threshold_sec=10.0,
            p95_threshold_sec=15.0,
        )

        # p90 and p99 not set, should always pass
        assert bucket.check_threshold(90, 999.0)
        assert bucket.check_threshold(99, 999.0)


class TestGetBucketForCapsule:
    """Tests for bucket selection logic."""

    def test_always_returns_bucket_never_none(self, make_capsule):
        """Contract: Always return a bucket, even for unknown scenarios."""
        capsule = make_capsule(
            scene_type="alien_spaceship",  # Not in any specific bucket
            pixel_count=999_999_999,
        )

        bucket = get_bucket_for_capsule(capsule)

        # Should always return a bucket (catch-all)
        assert bucket is not None
        assert bucket.name in {"unknown", "generic_large", "generic_medium"}

    def test_finds_matching_bucket(self, make_capsule):
        """Test that matching bucket is found."""
        capsule = make_capsule(
            scene_type="pool",
            device="mps",
            pixel_count=48_000_000,
        )

        bucket = get_bucket_for_capsule(capsule)

        assert bucket is not None
        assert "pool" in bucket.name.lower()

    def test_returns_most_specific_bucket_by_score(self, make_capsule):
        """Test that most specific bucket is returned based on concept score."""
        capsule = make_capsule(
            scene_type="pool",
            device="mps",
            pixel_count=48_000_000,
        )

        custom_buckets = [
            PerformanceBucket(
                name="generic_device",
                filters={"device": "mps"},  # specificity=5
                p50_threshold_sec=10.0,
                p95_threshold_sec=15.0,
            ),
            PerformanceBucket(
                name="specific_scene",
                filters={"scene_type": "pool"},  # specificity=10
                p50_threshold_sec=11.0,
                p95_threshold_sec=16.0,
            ),
            PerformanceBucket(
                name="catch_all",
                filters={},  # specificity=0
                p50_threshold_sec=60.0,
                p95_threshold_sec=120.0,
            ),
        ]

        bucket = get_bucket_for_capsule(capsule, custom_buckets)

        # Should return scene bucket (specificity=10) over device (specificity=5)
        assert bucket.name == "specific_scene"

    def test_specificity_score_prevents_range_cheating(self, make_capsule):
        """Range buckets don't falsely appear more specific than single-concept."""
        bucket_range = PerformanceBucket(
            name="range_bucket",
            filters={"pixel_count_min": 1000, "pixel_count_max": 5000},
            p50_threshold_sec=10.0,
            p95_threshold_sec=15.0,
        )

        bucket_scene = PerformanceBucket(
            name="scene_bucket",
            filters={"scene_type": "kitchen"},
            p50_threshold_sec=10.0,
            p95_threshold_sec=15.0,
        )

        # scene_type (score=10) beats pixel_count range (score=3)
        assert bucket_scene.specificity > bucket_range.specificity


class TestDefaultBuckets:
    """Tests for default bucket definitions."""

    def test_default_buckets_exist(self):
        """Test that default buckets are defined."""
        assert len(DEFAULT_BUCKETS) > 0

    def test_default_buckets_have_valid_thresholds(self):
        """Test that all default buckets have valid thresholds."""
        for bucket in DEFAULT_BUCKETS:
            assert bucket.p50_threshold_sec > 0
            assert bucket.p95_threshold_sec > 0
            assert bucket.p95_threshold_sec >= bucket.p50_threshold_sec

    def test_default_buckets_cover_apex_scenarios(self):
        """Verify buckets cover APEX scene types by checking filters."""
        # Check filter coverage, not names
        assert any(b.filters.get("scene_type") == "aerial" for b in DEFAULT_BUCKETS)
        assert any(b.filters.get("scene_type") == "pool" for b in DEFAULT_BUCKETS)
        assert any(b.filters.get("scene_type") in {"interior", "great_room", "kitchen"} for b in DEFAULT_BUCKETS)

        # Verify catch-all exists
        assert any(b.filters == {} for b in DEFAULT_BUCKETS), "Missing catch-all bucket"

    def test_catch_all_bucket_has_lenient_thresholds(self):
        """Catch-all bucket should have very lenient thresholds."""
        catch_all = [b for b in DEFAULT_BUCKETS if b.filters == {}]

        assert len(catch_all) == 1, "Should have exactly one catch-all bucket"

        bucket = catch_all[0]
        assert bucket.p50_threshold_sec >= 60.0, "Catch-all p50 should be very lenient"
        assert bucket.p95_threshold_sec >= 120.0, "Catch-all p95 should be very lenient"
