#!/usr/bin/env python3
"""
Codebase Health Monitor
Tracks and prevents recurring quality issues
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


class CodebaseHealthMonitor:
    """Monitor and track codebase health metrics"""

    def __init__(self, repo_root: Path = None):
        self.repo_root = repo_root or Path('.')
        self.health_file = self.repo_root / '.codebase_health.json'
        self.load_health_data()

    def load_health_data(self):
        """Load historical health data"""
        if self.health_file.exists():
            with open(self.health_file, 'r') as f:
                self.data = json.load(f)
        else:
            self.data = {
                'checks': [],
                'recurring_issues': {},
                'improvements': [],
                'quality_score': 100
            }

    def save_health_data(self):
        """Save health data"""
        with open(self.health_file, 'w') as f:
            json.dump(self.data, f, indent=2)

    def check_undefined_names(self) -> Tuple[bool, List[str]]:
        """Check for undefined names (F821)"""
        result = subprocess.run(
            ['flake8', '.', '--select=F821',
             '--exclude=deprecated,src/transformation_portal,scripts,.venv'],
            capture_output=True,
            text=True
        )

        issues = []
        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                if line and 'F821' in line:
                    issues.append(line)

        return len(issues) == 0, issues

    def check_import_issues(self) -> Tuple[bool, List[str]]:
        """Check for import problems (E402, F401, W0404)"""
        result = subprocess.run(
            ['flake8', '.', '--select=E402,F401',
             '--exclude=deprecated,src/transformation_portal,scripts,.venv'],
            capture_output=True,
            text=True
        )

        issues = []
        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                if line:
                    issues.append(line)

        return len(issues) == 0, issues

    def check_trailing_whitespace(self) -> Tuple[bool, Dict[str, int]]:
        """Check for excessive trailing whitespace"""
        result = subprocess.run(
            ['flake8', '.', '--select=W291,W293',
             '--exclude=deprecated,src/transformation_portal,scripts,.venv'],
            capture_output=True,
            text=True
        )

        file_counts = {}
        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                if line and ':' in line:
                    filepath = line.split(':')[0]
                    file_counts[filepath] = file_counts.get(filepath, 0) + 1

        return len(file_counts) == 0, file_counts

    def check_docstring_coverage(self) -> Tuple[bool, float]:
        """Check docstring coverage using pylint"""
        result = subprocess.run(
            ['pylint', '--disable=all', '--enable=missing-docstring',
             '--exclude-directories=deprecated,src/transformation_portal,scripts,.venv',
             '.'],
            capture_output=True,
            text=True
        )

        # Count missing docstrings
        missing_count = result.stdout.count('missing-docstring')

        # Estimate total public functions/classes
        py_files = list(self.repo_root.rglob('*.py'))
        py_files = [f for f in py_files if not any(
            exclude in str(f) for exclude in ['deprecated', 'src/transformation_portal', 'scripts', '.venv']
        )]

        estimated_total = len(py_files) * 10  # Rough estimate
        coverage = max(0, (1 - missing_count / max(estimated_total, 1)) * 100)

        return coverage > 70, coverage

    def check_test_coverage(self) -> Tuple[bool, float]:
        """Check test coverage"""
        try:
            result = subprocess.run(
                ['pytest', '--cov=.', '--cov-report=term', '--tb=no', '-q'],
                capture_output=True,
                text=True,
                timeout=60
            )

            # Parse coverage percentage
            for line in result.stdout.split('\n'):
                if 'TOTAL' in line and '%' in line:
                    coverage_str = line.split()[-1].rstrip('%')
                    try:
                        coverage = float(coverage_str)
                        return coverage > 60, coverage
                    except ValueError:
                        pass
        except (subprocess.TimeoutExpired, Exception):
            pass

        return False, 0.0

    def run_comprehensive_check(self) -> Dict:
        """Run all health checks"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'checks': {}
        }

        print("Running comprehensive health check...")
        print("=" * 60)

        # Check undefined names
        passed, issues = self.check_undefined_names()
        results['checks']['undefined_names'] = {
            'passed': passed,
            'count': len(issues),
            'issues': issues[:10]  # First 10
        }
        print(f"{'✅' if passed else '❌'} Undefined names: {len(issues)} issues")

        # Check imports
        passed, issues = self.check_import_issues()
        results['checks']['imports'] = {
            'passed': passed,
            'count': len(issues),
            'issues': issues[:10]
        }
        print(f"{'✅' if passed else '❌'} Import issues: {len(issues)} issues")

        # Check trailing whitespace
        passed, file_counts = self.check_trailing_whitespace()
        total_ws = sum(file_counts.values())
        results['checks']['trailing_whitespace'] = {
            'passed': passed,
            'count': total_ws,
            'affected_files': len(file_counts)
        }
        print(f"{'✅' if passed else '⚠️'} Trailing whitespace: {total_ws} lines in {len(file_counts)} files")

        # Check docstring coverage
        passed, coverage = self.check_docstring_coverage()
        results['checks']['docstring_coverage'] = {
            'passed': passed,
            'coverage': coverage
        }
        print(f"{'✅' if passed else '⚠️'} Docstring coverage: {coverage:.1f}%")

        # Calculate overall quality score
        score = 100
        if not results['checks']['undefined_names']['passed']:
            score -= 30
        if not results['checks']['imports']['passed']:
            score -= 20
        if not results['checks']['trailing_whitespace']['passed']:
            score -= 10
        if not results['checks']['docstring_coverage']['passed']:
            score -= 10

        results['quality_score'] = max(0, score)

        print("=" * 60)
        print(f"Overall Quality Score: {results['quality_score']}/100")

        # Track recurring issues
        self.track_recurring_issues(results)

        # Save results
        self.data['checks'].append(results)
        self.data['quality_score'] = results['quality_score']
        self.save_health_data()

        return results

    def track_recurring_issues(self, results: Dict):
        """Track issues that keep recurring"""
        for check_name, check_data in results['checks'].items():
            if not check_data['passed']:
                if check_name not in self.data['recurring_issues']:
                    self.data['recurring_issues'][check_name] = {
                        'count': 0,
                        'first_seen': results['timestamp'],
                        'last_seen': results['timestamp']
                    }
                else:
                    self.data['recurring_issues'][check_name]['count'] += 1
                    self.data['recurring_issues'][check_name]['last_seen'] = results['timestamp']

    def generate_report(self) -> str:
        """Generate health report"""
        report = []
        report.append("=" * 60)
        report.append("CODEBASE HEALTH REPORT")
        report.append("=" * 60)

        if not self.data['checks']:
            report.append("No health checks recorded yet.")
            return '\n'.join(report)

        latest = self.data['checks'][-1]

        report.append(f"\nLatest Check: {latest['timestamp']}")
        report.append(f"Quality Score: {latest['quality_score']}/100")
        report.append("")

        report.append("Check Results:")
        for check_name, check_data in latest['checks'].items():
            status = "✅ PASS" if check_data['passed'] else "❌ FAIL"
            report.append(f"  {status} {check_name}")
            if 'count' in check_data:
                report.append(f"       Issues: {check_data['count']}")
            if 'coverage' in check_data:
                report.append(f"       Coverage: {check_data['coverage']:.1f}%")

        if self.data['recurring_issues']:
            report.append("\nRecurring Issues:")
            for issue_name, issue_data in self.data['recurring_issues'].items():
                report.append(f"  ⚠️  {issue_name}:")
                report.append(f"       Occurred: {issue_data['count']} times")
                report.append(f"       Last seen: {issue_data['last_seen']}")

        # Recommendations
        report.append("\nRecommendations:")
        if not latest['checks']['undefined_names']['passed']:
            report.append("  🔧 Fix undefined name errors (F821) - CRITICAL")
        if not latest['checks']['imports']['passed']:
            report.append("  🔧 Clean up import statements")
        if latest['checks']['trailing_whitespace']['count'] > 50:
            report.append("  🔧 Run: autopep8 --in-place --select=W291,W293 .")

        return '\n'.join(report)


def main():
    """Run health monitor"""
    monitor = CodebaseHealthMonitor()
    results = monitor.run_comprehensive_check()

    print("\n" + monitor.generate_report())

    # Exit with error if quality score too low
    if results['quality_score'] < 70:
        print("\n⚠️  Quality score below threshold (70)")
        return 1

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
