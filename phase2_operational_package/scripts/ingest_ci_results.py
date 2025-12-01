#!/usr/bin/env python3
"""
Phase 2 RAG System - CI Result Ingestion Helper
===============================================

Standalone script for ingesting CI test results into the Knowledge Engine.
Parses JUnit XML files and updates the knowledge base.

Usage:
    python ingest_ci_results.py --junit results.xml
    python ingest_ci_results.py --dir test-results/
    python ingest_ci_results.py --coverage coverage.xml --junit results.xml
"""

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_junit_xml(junit_path: Path) -> List[Dict[str, Any]]:
    """Parse JUnit XML file and extract test results."""
    results = []
    
    try:
        tree = ET.parse(junit_path)
        root = tree.getroot()
        
        for testsuite in root.iter('testsuite'):
            suite_name = testsuite.get('name', 'unknown')
            
            for testcase in testsuite.iter('testcase'):
                test_name = testcase.get('name', 'unknown')
                classname = testcase.get('classname', '')
                duration = float(testcase.get('time', 0))
                
                # Determine status
                failure = testcase.find('failure')
                error = testcase.find('error')
                skipped = testcase.find('skipped')
                
                if failure is not None:
                    status = 'failed'
                    message = failure.get('message', '')
                elif error is not None:
                    status = 'error'
                    message = error.get('message', '')
                elif skipped is not None:
                    status = 'skipped'
                    message = skipped.get('message', '')
                else:
                    status = 'passed'
                    message = ''
                
                results.append({
                    'test_id': f"{classname}::{test_name}",
                    'test_file': classname.replace('.', '/') + '.py',
                    'test_name': test_name,
                    'status': status,
                    'duration_ms': duration * 1000,
                    'error_message': message if message else None,
                    'suite': suite_name,
                    'timestamp': datetime.now().isoformat()
                })
                
    except ET.ParseError as e:
        print(f"Error parsing {junit_path}: {e}", file=sys.stderr)
    except FileNotFoundError:
        print(f"File not found: {junit_path}", file=sys.stderr)
    
    return results


def parse_coverage_xml(coverage_path: Path) -> Optional[Dict[str, Any]]:
    """Parse coverage XML file and extract metrics."""
    try:
        tree = ET.parse(coverage_path)
        root = tree.getroot()
        
        # Get overall coverage
        line_rate = float(root.get('line-rate', 0)) * 100
        branch_rate = float(root.get('branch-rate', 0)) * 100
        
        return {
            'metric_id': 'code_coverage',
            'metric_type': 'coverage',
            'value': line_rate,
            'unit': 'percent',
            'timestamp': datetime.now().isoformat(),
            'context': {
                'line_rate': line_rate,
                'branch_rate': branch_rate
            }
        }
        
    except (ET.ParseError, FileNotFoundError) as e:
        print(f"Error parsing coverage: {e}", file=sys.stderr)
        return None


def update_knowledge_base(
    cache_dir: Path,
    test_results: List[Dict],
    metrics: List[Dict],
    run_id: str = "local"
) -> Dict[str, Any]:
    """Update knowledge base with new results."""
    knowledge_dir = cache_dir / "knowledge"
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    
    # Load existing results
    results_file = knowledge_dir / "ci_test_results.json"
    existing_results = []
    if results_file.exists():
        with open(results_file) as f:
            existing_results = json.load(f)
    
    # Append new results
    existing_results.extend(test_results)
    
    # Keep last 1000 results
    if len(existing_results) > 1000:
        existing_results = existing_results[-1000:]
    
    with open(results_file, "w") as f:
        json.dump(existing_results, f, indent=2)
    
    # Load existing metrics
    metrics_file = knowledge_dir / "quality_metrics.json"
    existing_metrics = []
    if metrics_file.exists():
        with open(metrics_file) as f:
            existing_metrics = json.load(f)
    
    # Calculate summary metrics from test results
    total_tests = len(test_results)
    total_passed = sum(1 for r in test_results if r['status'] == 'passed')
    total_failed = sum(1 for r in test_results if r['status'] in ('failed', 'error'))
    total_skipped = sum(1 for r in test_results if r['status'] == 'skipped')
    total_duration = sum(r.get('duration_ms', 0) for r in test_results) / 1000
    
    # Add new metrics
    new_metrics = [
        {
            'metric_id': 'test_pass_rate',
            'metric_type': 'test_health',
            'value': (total_passed / total_tests * 100) if total_tests > 0 else 0,
            'unit': 'percent',
            'source': f'CI Run #{run_id}',
            'timestamp': datetime.now().isoformat(),
            'context': {
                'total': total_tests,
                'passed': total_passed,
                'failed': total_failed,
                'skipped': total_skipped
            }
        },
        {
            'metric_id': 'test_execution_time',
            'metric_type': 'performance',
            'value': total_duration,
            'unit': 'seconds',
            'source': f'CI Run #{run_id}',
            'timestamp': datetime.now().isoformat(),
            'context': {'test_count': total_tests}
        }
    ]
    
    # Add any additional metrics (e.g., coverage)
    new_metrics.extend(metrics)
    
    existing_metrics.extend(new_metrics)
    
    # Keep last 100 metric entries
    if len(existing_metrics) > 100:
        existing_metrics = existing_metrics[-100:]
    
    with open(metrics_file, "w") as f:
        json.dump(existing_metrics, f, indent=2)
    
    # Update knowledge state
    state_file = knowledge_dir / "knowledge_state.json"
    state = {
        'test_results_count': len(existing_results),
        'metrics_count': len(existing_metrics),
        'last_ingestion': datetime.now().isoformat(),
        'last_run_id': run_id,
        'storage_path': str(knowledge_dir)
    }
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)
    
    return {
        'tests_ingested': len(test_results),
        'metrics_added': len(new_metrics),
        'total_results': len(existing_results),
        'total_metrics': len(existing_metrics),
        'summary': {
            'total': total_tests,
            'passed': total_passed,
            'failed': total_failed,
            'skipped': total_skipped,
            'pass_rate': (total_passed / total_tests * 100) if total_tests > 0 else 0,
            'duration': total_duration
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description="Ingest CI test results into Knowledge Engine"
    )
    parser.add_argument(
        "--junit",
        type=Path,
        help="Path to JUnit XML file"
    )
    parser.add_argument(
        "--dir",
        type=Path,
        help="Directory containing JUnit XML files"
    )
    parser.add_argument(
        "--coverage",
        type=Path,
        help="Path to coverage XML file"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".rag_cache"),
        help="RAG cache directory"
    )
    parser.add_argument(
        "--run-id",
        default="local",
        help="CI run identifier"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Collect JUnit files
    junit_files = []
    if args.junit:
        junit_files.append(args.junit)
    if args.dir:
        junit_files.extend(args.dir.glob("**/junit*.xml"))
    
    if not junit_files:
        parser.error("At least one --junit file or --dir is required")
    
    # Parse all JUnit files
    all_results = []
    for junit_file in junit_files:
        if args.verbose:
            print(f"Parsing {junit_file}...", file=sys.stderr)
        results = parse_junit_xml(junit_file)
        all_results.extend(results)
    
    if args.verbose:
        print(f"Parsed {len(all_results)} test results", file=sys.stderr)
    
    # Parse coverage if provided
    metrics = []
    if args.coverage:
        coverage_metric = parse_coverage_xml(args.coverage)
        if coverage_metric:
            metrics.append(coverage_metric)
    
    # Update knowledge base
    summary = update_knowledge_base(
        args.cache_dir,
        all_results,
        metrics,
        args.run_id
    )
    
    # Print summary
    print("=" * 60)
    print("KNOWLEDGE ENGINE INGESTION COMPLETE")
    print("=" * 60)
    print(f"  Tests Ingested: {summary['tests_ingested']}")
    print(f"  Passed: {summary['summary']['passed']}")
    print(f"  Failed: {summary['summary']['failed']}")
    print(f"  Skipped: {summary['summary']['skipped']}")
    print(f"  Duration: {summary['summary']['duration']:.2f}s")
    print(f"  Pass Rate: {summary['summary']['pass_rate']:.1f}%")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
