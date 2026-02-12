#!/usr/bin/env python3
"""Example demonstrating performance capsule capture and ledger usage.

This example shows the complete workflow:
1. Create a PerformanceCapsule with timing data
2. Log it to the performance ledger
3. Query historical data
4. Detect regressions
"""

import tempfile
from pathlib import Path

from transformation_portal.metrics import (
    PerformanceCapsule,
    compute_config_hash,
    compute_dimension_adjustment,
    get_bucket_for_capsule,
)
from transformation_portal.metrics.ledger import PerformanceLedger, detect_regression, generate_performance_report


def main():
    """Demonstrate performance ledger workflow."""

    # Create temporary database
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "performance.db"
        ledger = PerformanceLedger(db_path)

        print("=" * 70)
        print("Performance Ledger Example")
        print("=" * 70)
        print()

        # Example 1: Create and log a performance capsule
        print("1. Creating performance capsule for Pool scene...")

        capsule_pool = PerformanceCapsule(
            image_id="750_Picacho_Pool",
            image_path="/input/750_Picacho_Pool.tiff",
            input_hash="abc123def456",
            original_shape=(6000, 8000),
            enforced_shape=(5992, 7994),
            pixel_count=47_892_448,
            dimension_adjustment=compute_dimension_adjustment((6000, 8000), (5992, 7994)),
            backend_id="da3",
            model_variant="depth_anything_v3_vits",
            device="mps",
            dtype="float16",
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
            config_hash=compute_config_hash({"backend": "da3", "quality_tier": "stable"}),
            pipeline_version="2.0.0",
            quality_score=0.95,
            firewall_status="pass",
        )

        # Find matching bucket
        bucket = get_bucket_for_capsule(capsule_pool)

        print(f"   Image: {capsule_pool.image_id}")
        print(f"   Scene: {capsule_pool.scene_type}")
        print(f"   Device: {capsule_pool.device}")
        print(f"   Pixels: {capsule_pool.pixel_count:,}")
        print(f"   Dimension adjustment: {capsule_pool.dimension_adjustment}")
        print(f"   Total runtime: {capsule_pool.timings['total']:.2f}s")
        print(f"   Matched bucket: {bucket.name if bucket else 'none'}")
        if bucket:
            print(f"   Bucket p50: {bucket.p50_threshold_sec:.2f}s")
            print(f"   Bucket p95: {bucket.p95_threshold_sec:.2f}s")
            verdict = "PASS" if capsule_pool.timings["total"] <= bucket.p95_threshold_sec else "BLOCK"
            print(f"   Verdict: {verdict}")
        print()

        # Log to ledger
        print("2. Logging capsule to ledger...")
        ledger.log_capsule(capsule_pool)
        print(f"   ✓ Logged to {db_path}")
        print()

        # Add more sample data (different scene types)
        print("3. Adding more sample data...")

        capsules = [
            PerformanceCapsule(
                image_id="750_Picacho_GreatRoom",
                image_path="/input/750_Picacho_GreatRoom.tiff",
                input_hash="xyz789",
                original_shape=(5000, 6000),
                enforced_shape=(5000, 6000),
                pixel_count=30_000_000,
                dimension_adjustment="exact",
                backend_id="da3",
                device="mps",
                timings={"total": 4.83, "inference": 3.2},
                scene_type="interior",
                config_hash="abc123",
                pipeline_version="2.0.0",
                firewall_status="pass",
            ),
            PerformanceCapsule(
                image_id="750_Picacho_Aerial",
                image_path="/input/750_Picacho_Aerial.tiff",
                input_hash="aerial123",
                original_shape=(6000, 7200),
                enforced_shape=(5992, 7196),
                pixel_count=43_200_000,
                dimension_adjustment="cropped_0.1%",
                backend_id="da3",
                device="mps",
                timings={"total": 8.11, "inference": 6.0},
                scene_type="aerial",
                config_hash="abc123",
                pipeline_version="2.0.0",
                firewall_status="pass",
            ),
        ]

        for capsule in capsules:
            ledger.log_capsule(capsule)
            print(f"   ✓ Logged {capsule.image_id} ({capsule.scene_type})")
        print()

        # Query capsules
        print("4. Querying ledger...")
        all_capsules = ledger.query_capsules()
        print(f"   Total capsules: {len(all_capsules)}")

        pool_capsules = ledger.query_capsules(scene_type="pool")
        print(f"   Pool capsules: {len(pool_capsules)}")

        interior_capsules = ledger.query_capsules(scene_type="interior")
        print(f"   Interior capsules: {len(interior_capsules)}")
        print()

        # Statistics
        print("5. Computing statistics...")
        stats = ledger.get_statistics()
        print(f"   Count: {stats['count']}")
        print(f"   Mean: {stats['mean_sec']:.2f}s")
        print(f"   Median: {stats['median_sec']:.2f}s")
        print(f"   p95: {stats['p95_sec']:.2f}s")
        print(f"   Min: {stats['min_sec']:.2f}s")
        print(f"   Max: {stats['max_sec']:.2f}s")
        print()

        # Regression detection
        print("6. Testing regression detection...")

        # Simulate a "current" run that's slightly slower
        current_pool = PerformanceCapsule(
            image_id="750_Picacho_Pool_v2",
            image_path="/input/750_Picacho_Pool.tiff",
            input_hash="current123",
            original_shape=(6000, 8000),
            enforced_shape=(5992, 7994),
            pixel_count=47_892_448,
            dimension_adjustment="cropped_0.2%",
            backend_id="da3",
            device="mps",
            timings={"total": 12.5},  # Slightly slower than 11.49s
            scene_type="pool",
            config_hash="abc123",
            pipeline_version="2.0.1",
            firewall_status="unknown",
        )

        historical = ledger.query_capsules(scene_type="pool", device="mps")

        regression_result = detect_regression(current_pool, historical)

        print(f"   Current runtime: {regression_result['current_total_sec']:.2f}s")
        print(f"   Historical p50: {regression_result['historical_p50_sec']:.2f}s")
        print(f"   Historical p95: {regression_result['historical_p95_sec']:.2f}s")
        print(f"   Bucket: {regression_result['bucket']}")
        print(f"   Status: {regression_result['status']}")
        if "message" in regression_result:
            print(f"   Message: {regression_result['message']}")
        print()

        # Generate report
        print("7. Generating performance report...")
        report_path = Path(tmpdir) / "performance_report.md"
        generate_performance_report(ledger, report_path)

        print(f"   ✓ Report written to {report_path}")
        print()
        print("Report preview:")
        print("-" * 70)
        print(report_path.read_text()[:500] + "...")
        print("-" * 70)
        print()

        print("✅ Example complete!")
        print()
        print("In production, use the CLI:")
        print("  python -m transformation_portal.metrics.ledger log --capsule capsule.json --ledger-db perf.db")
        print("  python -m transformation_portal.metrics.ledger query --ledger-db perf.db --scene-type pool")
        print("  python -m transformation_portal.metrics.ledger regression --ledger-db perf.db --capsule current.json")
        print("  python -m transformation_portal.metrics.ledger report --ledger-db perf.db --output report.md")


if __name__ == "__main__":
    main()
