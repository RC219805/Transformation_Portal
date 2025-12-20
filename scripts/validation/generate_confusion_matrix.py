#!/usr/bin/env python3
"""
Generate Confusion Matrix for Scene Classification Validation

Compares predicted scene types against expected ground truth.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple


# Ground truth scene types (based on stratification)
EXPECTED_SCENE_TYPES = {
    # Structure-dominated (interiors with strong lines)
    '750Picacho_Kitchen': 'structure_dominated',
    '750Picacho_GreatRoom': 'structure_dominated',
    '750Picacho_MasterBath': 'structure_dominated',
    '750Picacho_MasterBedroom': 'structure_dominated',
    '750Picacho_Office': 'structure_dominated',
    
    # Texture-dominated (water, glass, reflective surfaces)
    '750Picacho_Pool': 'texture_dominated',
    '750Picacho_Ocean': 'texture_dominated',
    'Montecito-Shores-10': 'texture_dominated',
    
    # Mixed/borderline cases
    '750Picacho_Exterior': 'structure_dominated',  # has strong architectural lines
    '750Picacho_Entry': 'structure_dominated',
    
    # Additional images (update as needed)
}


def load_metrics(output_dir: Path) -> Dict[str, dict]:
    """Load all metrics files from output directory."""
    metrics_files = sorted(output_dir.glob("*_metrics.json"))
    
    results = {}
    for mf in metrics_files:
        # Extract image name (remove _metrics.json suffix)
        image_name = mf.stem.replace('_metrics', '')
        
        with open(mf) as f:
            data = json.load(f)
        
        results[image_name] = data
    
    return results


def generate_confusion_matrix(
    metrics: Dict[str, dict],
    expected: Dict[str, str]
) -> Tuple[Dict, int, int]:
    """
    Generate confusion matrix and accuracy stats.
    
    Returns:
        (confusion_matrix, correct_count, total_count)
    """
    confusion = defaultdict(lambda: defaultdict(int))
    correct = 0
    total = 0
    
    for image_name, data in metrics.items():
        predicted = data.get('scene_type', 'unknown')
        actual = expected.get(image_name, 'unknown')
        
        confusion[actual][predicted] += 1
        total += 1
        
        if predicted == actual:
            correct += 1
    
    return dict(confusion), correct, total


def print_confusion_matrix(confusion: Dict, correct: int, total: int, metrics: Dict[str, dict]):
    """Pretty-print confusion matrix."""
    print("\n" + "="*80)
    print("SCENE CLASSIFICATION CONFUSION MATRIX")
    print("="*80)
    
    # Get all unique labels
    all_labels = sorted(set(
        list(confusion.keys()) + 
        [pred for preds in confusion.values() for pred in preds.keys()]
    ))
    
    # Print header
    header_label = "Actual \\ Predicted"
    print(f"\n{header_label:<25}", end="")
    for label in all_labels:
        print(f"{label[:20]:>22}", end="")
    print()
    print("-" * 80)
    
    # Print rows
    for actual in all_labels:
        print(f"{actual:<25}", end="")
        for predicted in all_labels:
            count = confusion.get(actual, {}).get(predicted, 0)
            print(f"{count:>22}", end="")
        print()
    
    # Print summary
    print("\n" + "="*80)
    print(f"Classification Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
    print("="*80)
    
    # Print misclassifications
    print("\nMisclassifications:")
    for image_name, data in sorted(metrics.items()):
        predicted = data.get('scene_type', 'unknown')
        actual = EXPECTED_SCENE_TYPES.get(image_name, 'unknown')
        
        if predicted != actual and actual != 'unknown':
            factors = data.get('classification_factors', {})
            print(f"  ❌ {image_name}")
            print(f"     Expected: {actual}, Got: {predicted}")
            print(f"     Factors: ratio={factors.get('ratio', 0):.2f}, "
                  f"depth_var={factors.get('depth_variance', 0):.4f}, "
                  f"edge_density={factors.get('edge_density', 0):.4f}")
            print(f"     Decision: {factors.get('decision_rule', 'unknown')}")
    
    # Print correct classifications
    print("\nCorrect Classifications:")
    for image_name, data in sorted(metrics.items()):
        predicted = data.get('scene_type', 'unknown')
        actual = EXPECTED_SCENE_TYPES.get(image_name, 'unknown')
        
        if predicted == actual and actual != 'unknown':
            factors = data.get('classification_factors', {})
            print(f"  ✅ {image_name}: {predicted}")
            print(f"     Factors: ratio={factors.get('ratio', 0):.2f}, "
                  f"depth_var={factors.get('depth_variance', 0):.4f}, "
                  f"edge_density={factors.get('edge_density', 0):.4f}")


def print_quality_summary(metrics: Dict[str, dict]):
    """Print quality gate summary."""
    print("\n" + "="*80)
    print("QUALITY GATE SUMMARY")
    print("="*80)
    
    by_scene = defaultdict(lambda: {
        'total': 0,
        'lenient_pass': 0,
        'strict_pass': 0,
        'edge_f1': []
    })
    
    for image_name, data in metrics.items():
        scene_type = data.get('scene_type', 'unknown')
        
        by_scene[scene_type]['total'] += 1
        if data.get('lenient_pass'):
            by_scene[scene_type]['lenient_pass'] += 1
        if data.get('strict_pass'):
            by_scene[scene_type]['strict_pass'] += 1
        
        if 'edge_f1' in data:
            by_scene[scene_type]['edge_f1'].append(data['edge_f1'])
    
    for scene_type, stats in sorted(by_scene.items()):
        print(f"\n{scene_type.upper()} ({stats['total']} images):")
        print(f"  Lenient pass: {stats['lenient_pass']}/{stats['total']} "
              f"({100*stats['lenient_pass']/stats['total']:.1f}%)")
        print(f"  Strict pass:  {stats['strict_pass']}/{stats['total']} "
              f"({100*stats['strict_pass']/stats['total']:.1f}%)")
        
        if stats['edge_f1']:
            import numpy as np
            print(f"  Edge F1: mean={np.mean(stats['edge_f1']):.3f}, "
                  f"min={np.min(stats['edge_f1']):.3f}, "
                  f"max={np.max(stats['edge_f1']):.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate confusion matrix for scene classification"
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Output directory containing metrics JSON files'
    )
    parser.add_argument(
        '--expected',
        type=Path,
        help='Optional JSON file with expected scene types (overrides defaults)'
    )
    
    args = parser.parse_args()
    
    # Load expected scene types
    expected = EXPECTED_SCENE_TYPES.copy()
    if args.expected:
        with open(args.expected) as f:
            expected.update(json.load(f))
    
    # Load metrics
    print(f"Loading metrics from: {args.output_dir}")
    metrics = load_metrics(args.output_dir)
    print(f"Found {len(metrics)} metrics files")
    
    # Generate confusion matrix
    confusion, correct, total = generate_confusion_matrix(metrics, expected)
    
    # Print results
    print_confusion_matrix(confusion, correct, total, metrics)
    print_quality_summary(metrics)
    
    # Print acceptance criteria
    print("\n" + "="*80)
    print("ACCEPTANCE CRITERIA")
    print("="*80)
    
    accuracy = 100 * correct / total if total > 0 else 0
    
    checks = [
        (f"{total}/{total} images processed", total == len(metrics), "✅" if total == len(metrics) else "❌"),
        (f"Classification accuracy ≥85%", accuracy >= 85, "✅" if accuracy >= 85 else "❌"),
        (f"Classification accuracy ≥90%", accuracy >= 90, "✅" if accuracy >= 90 else "❌"),
    ]
    
    for description, passed, symbol in checks:
        status = "PASS" if passed else "FAIL"
        print(f"{symbol} {description}: {status}")
    
    # Exit code
    if accuracy >= 85:
        print("\n✅ Validation PASSED - Proceed to Materials V3")
        return 0
    else:
        print("\n❌ Validation FAILED - Classification accuracy too low")
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
