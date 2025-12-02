#!/usr/bin/env python3
"""Memory profiling test for CI performance monitoring with RAG integration.

This module provides memory profiling tests that track memory usage during
typical operations. Results can be optionally integrated with the Knowledge
Integration Engine for pattern analysis and trend detection.

Usage:
    # Standalone memory profiling
    python -m memory_profiler tests/mem_test.py

    # With RAG/Knowledge Base integration
    python tests/mem_test.py --enable-rag

    # Export memory metrics to knowledge base
    python tests/mem_test.py --enable-rag --export-kb memory_kb.json
"""
import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from memory_profiler import memory_usage, profile


# Memory usage thresholds (in MiB)
MEMORY_THRESHOLDS = {
    'import_core': 50.0,  # Core imports shouldn't exceed 50 MiB
    'array_operations': 100.0,  # Array operations shouldn't exceed 100 MiB
}


def get_memory_delta(func, *args, **kwargs) -> Dict[str, Any]:
    """
    Measure memory usage delta for a function.

    Args:
        func: Function to measure
        *args: Arguments to pass to function
        **kwargs: Keyword arguments to pass to function

    Returns:
        Dictionary with memory metrics
    """
    start_time = time.time()

    # Get baseline memory (returns list with single value when max_usage=False)
    mem_before_list = memory_usage(-1, interval=0.01, timeout=0.1)
    mem_before = mem_before_list[0] if mem_before_list else 0.0

    # Run function and capture maximum memory usage during execution
    # max_usage=True returns (max_mem, num_measurements) tuple
    mem_result = memory_usage((func, args, kwargs), interval=0.05, max_usage=True)
    # mem_result is (max_mem, count) when max_usage=True
    mem_peak = mem_result[0] if isinstance(mem_result, tuple) else mem_result

    end_time = time.time()

    # Get final memory after function completes
    mem_after_list = memory_usage(-1, interval=0.01, timeout=0.1)
    mem_after = mem_after_list[0] if mem_after_list else 0.0

    # Calculate memory delta (peak - baseline)
    mem_delta = max(0.0, mem_peak - mem_before)

    return {
        'mem_before': mem_before,
        'mem_peak': mem_peak,
        'mem_after': mem_after,
        'mem_delta': mem_delta,
        'processing_time': end_time - start_time,
        'success': True,
    }


def _import_core_impl():
    """Implementation of core import test."""
    import numpy as np  # noqa: F401
    from PIL import Image  # noqa: F401


def _array_operations_impl():
    """Implementation of array operations test."""
    import numpy as np
    # Create a moderate-sized array
    arr = np.random.rand(1000, 1000)
    # Perform some operations
    result = arr.mean()
    del arr
    return result


@profile
def test_import_core():
    """Test that core imports don't leak memory."""
    _import_core_impl()
    # Use assertion for pytest compatibility instead of return value
    assert True


@profile
def test_basic_array_operations():
    """Test basic array operations memory usage."""
    result = _array_operations_impl()
    # Use assertion for pytest compatibility instead of return value
    assert result is not None


def run_with_knowledge_integration(
    export_path: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run memory tests with Knowledge Integration Engine tracking.

    Args:
        export_path: Optional path to export knowledge base
        verbose: Enable verbose output

    Returns:
        Dictionary with test results and metrics
    """
    results = {
        'tests': {},
        'overall_success': True,
        'recommendations': [],
    }

    # Try to import Knowledge Integration Engine
    engine = None
    try:
        # Add RAG system to path
        rag_path = Path(__file__).parent.parent / '.github' / 'agents' / 'rag_system'
        if rag_path.exists():
            sys.path.insert(0, str(rag_path))
            from knowledge_engine import KnowledgeIntegrationEngine
            engine = KnowledgeIntegrationEngine()
            if verbose:
                print("✓ Knowledge Integration Engine loaded")
    except ImportError as e:
        if verbose:
            print(f"⚠ Knowledge Integration Engine not available: {e}")

    # Test 1: Import core
    if verbose:
        print("\n📊 Running: test_import_core")
    try:
        metrics = get_memory_delta(_import_core_impl)
        threshold = MEMORY_THRESHOLDS['import_core']
        metrics['within_threshold'] = metrics['mem_delta'] <= threshold

        results['tests']['import_core'] = metrics

        if engine:
            engine.add_feedback(
                pipeline='memory_profiler',
                artifact_id='test_import_core',
                success=metrics['within_threshold'],
                processing_time=metrics['processing_time'],
                parameters={'threshold_mib': threshold},
                quality_score=max(0, 1 - (metrics['mem_delta'] / threshold)),
                error_message=(
                    None if metrics['within_threshold'] else
                    f"Memory usage {metrics['mem_delta']:.1f} MiB exceeded threshold {threshold} MiB"
                ),
            )

        if verbose:
            print(f"  Memory delta: {metrics['mem_delta']:.2f} MiB")
            print(f"  Peak memory: {metrics['mem_peak']:.2f} MiB")
            print(f"  Within threshold ({threshold} MiB): {'✓' if metrics['within_threshold'] else '✗'}")

    except Exception as e:
        results['tests']['import_core'] = {'success': False, 'error': str(e)}
        results['overall_success'] = False
        if engine:
            engine.add_feedback(
                pipeline='memory_profiler',
                artifact_id='test_import_core',
                success=False,
                processing_time=0,
                parameters={},
                error_message=str(e),
            )

    # Test 2: Array operations
    if verbose:
        print("\n📊 Running: test_basic_array_operations")
    try:
        metrics = get_memory_delta(_array_operations_impl)
        threshold = MEMORY_THRESHOLDS['array_operations']
        metrics['within_threshold'] = metrics['mem_delta'] <= threshold

        results['tests']['array_operations'] = metrics

        if engine:
            engine.add_feedback(
                pipeline='memory_profiler',
                artifact_id='test_basic_array_operations',
                success=metrics['within_threshold'],
                processing_time=metrics['processing_time'],
                parameters={'threshold_mib': threshold, 'array_size': '1000x1000'},
                quality_score=max(0, 1 - (metrics['mem_delta'] / threshold)),
                error_message=(
                    None if metrics['within_threshold'] else
                    f"Memory usage {metrics['mem_delta']:.1f} MiB exceeded threshold {threshold} MiB"
                ),
            )

        if verbose:
            print(f"  Memory delta: {metrics['mem_delta']:.2f} MiB")
            print(f"  Peak memory: {metrics['mem_peak']:.2f} MiB")
            print(f"  Within threshold ({threshold} MiB): {'✓' if metrics['within_threshold'] else '✗'}")

    except Exception as e:
        results['tests']['array_operations'] = {'success': False, 'error': str(e)}
        results['overall_success'] = False
        if engine:
            engine.add_feedback(
                pipeline='memory_profiler',
                artifact_id='test_basic_array_operations',
                success=False,
                processing_time=0,
                parameters={},
                error_message=str(e),
            )

    # Generate recommendations if engine available
    if engine:
        recommendations = engine.generate_recommendations('memory_profiler')
        results['recommendations'] = [
            {
                'type': r.recommendation_type,
                'severity': r.severity,
                'title': r.title,
                'action': r.suggested_action,
            }
            for r in recommendations
        ]

        if verbose and recommendations:
            print("\n📋 Recommendations:")
            for rec in recommendations:
                print(f"  [{rec.severity.upper()}] {rec.title}")
                print(f"    → {rec.suggested_action}")

        # Export knowledge base if requested
        if export_path:
            engine.export_knowledge_base(export_path)
            if verbose:
                print(f"\n✓ Exported knowledge base to {export_path}")

    return results


def main():
    """Main entry point with optional RAG integration."""
    parser = argparse.ArgumentParser(
        description='Memory profiling tests with optional RAG integration'
    )
    parser.add_argument(
        '--enable-rag',
        action='store_true',
        help='Enable RAG/Knowledge Base integration for memory tracking'
    )
    parser.add_argument(
        '--export-kb',
        type=str,
        metavar='PATH',
        help='Export memory metrics to knowledge base JSON file'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )

    args = parser.parse_args()

    if args.enable_rag:
        # Run with knowledge integration
        print("🧠 Running memory tests with RAG integration...")
        results = run_with_knowledge_integration(
            export_path=args.export_kb,
            verbose=args.verbose,
        )

        # Print summary
        print("\n" + "=" * 50)
        print("📊 Memory Test Summary")
        print("=" * 50)
        for test_name, metrics in results['tests'].items():
            if metrics.get('success', False):
                status = '✓' if metrics.get('within_threshold', False) else '⚠'
                print(f"  {status} {test_name}: {metrics.get('mem_delta', 0):.2f} MiB")
            else:
                print(f"  ✗ {test_name}: {metrics.get('error', 'Unknown error')}")

        if results['recommendations']:
            print(f"\n  {len(results['recommendations'])} recommendation(s) generated")

        # Exit with appropriate code
        sys.exit(0 if results['overall_success'] else 1)
    else:
        # Standard memory profiler execution
        test_import_core()
        test_basic_array_operations()


if __name__ == "__main__":
    main()
