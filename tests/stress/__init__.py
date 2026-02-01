"""Stress tests for Transformation Portal.

These tests are designed to validate system behavior under heavy load
and extended operation. They are marked with @pytest.mark.stress and
@pytest.mark.slow to allow selective execution.

Running stress tests:
    # Run all stress tests
    pytest tests/stress/ -v -m stress

    # Run specific stress test file
    pytest tests/stress/test_stress_large_batch.py -v

    # Run with verbose output
    pytest tests/stress/ -v -s -m stress

    # Run fast stress tests only (exclude very slow ones)
    pytest tests/stress/ -v -m "stress and not slow"

Stress tests are NOT run in CI by default. They can be:
1. Run manually during development
2. Triggered on-demand via CI workflow
3. Scheduled as nightly builds
"""
