#!/usr/bin/env python3
"""
Module Boundary Validation Script

Enforces layered architecture by checking that modules don't import
from layers above them in the hierarchy:

Architecture Layers (bottom to top):
1. utils - Shared utility functions (lowest layer)
2. interfaces - Abstract base classes
3. processors - Core transformation engines
4. enhancers - Specialized improvement algorithms
5. pipelines - Orchestration & multi-step workflows (highest layer)

Rules:
- Lower layers cannot import from higher layers
- utils cannot import from processors, enhancers, or pipelines
- processors cannot import from pipelines
- interfaces should only be imported by implementations

This script is meant to be run in CI to catch architectural violations.

Usage:
    python scripts/validation/check_module_boundaries.py
    python scripts/validation/check_module_boundaries.py --fail-on-violations

Exit codes:
    0 - No violations found
    1 - Violations found

See Also:
    - docs/architecture/adr/ADR-001-module-interface-contracts.md
    - docs/ARCHITECTURE.md
"""

import ast
import sys
from pathlib import Path
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class ImportViolation:
    """Represents an import boundary violation."""
    file: Path
    line: int
    importing_module: str
    imported_module: str
    reason: str


# Define layer hierarchy (lower index = lower in hierarchy)
# Note: Layer names match actual directory names in src/transformation_portal/
LAYER_HIERARCHY = {
    'utils': 0,          # Shared utility functions
    'interfaces': 1,     # Abstract base classes
    'processors': 2,     # Core transformation engines
    'enhancers': 2,      # Same level as processors
    'depth': 2,          # Depth processing (same level as processors)
    'segmentation': 2,   # Segmentation utilities (same level as processors)
    'pipelines': 3,      # Multi-stage orchestration
    'cli': 4,            # Highest layer - user-facing interfaces
}


class ModuleBoundaryChecker:
    """Checks module import boundaries."""
    
    def __init__(self, src_dir: Path):
        self.src_dir = src_dir
        self.violations: List[ImportViolation] = []
    
    def check_file(self, filepath: Path) -> List[ImportViolation]:
        """
        Check a single Python file for boundary violations.
        
        Args:
            filepath: Path to Python file to check
            
        Returns:
            List of ImportViolation objects
        """
        if not filepath.exists():
            return []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read(), filename=str(filepath))
        except SyntaxError:
            print(f"Warning: Syntax error in {filepath}, skipping")
            return []
        
        violations = []
        importing_layer = self._get_layer(filepath)
        
        if importing_layer is None:
            return []  # Not in transformation_portal package
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    violation = self._check_import(
                        filepath, node.lineno, importing_layer, alias.name
                    )
                    if violation:
                        violations.append(violation)
            
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    violation = self._check_import(
                        filepath, node.lineno, importing_layer, node.module
                    )
                    if violation:
                        violations.append(violation)
        
        return violations
    
    def _get_layer(self, filepath: Path) -> str:
        """
        Determine which architectural layer a file belongs to.
        
        Args:
            filepath: Path to Python file
            
        Returns:
            Layer name or None if not in transformation_portal
        """
        try:
            rel_path = filepath.relative_to(self.src_dir / 'transformation_portal')
        except ValueError:
            return None
        
        parts = rel_path.parts
        if len(parts) == 0:
            return None
        
        # First directory component determines layer
        layer = parts[0]
        
        # Map directory names to layers
        if layer in LAYER_HIERARCHY:
            return layer
        
        return None
    
    def _check_import(
        self,
        filepath: Path,
        lineno: int,
        importing_layer: str,
        imported_module: str
    ) -> ImportViolation:
        """
        Check if an import violates boundary rules.
        
        Args:
            filepath: File doing the import
            lineno: Line number of import
            importing_layer: Layer of importing file
            imported_module: Name of imported module
            
        Returns:
            ImportViolation if violation found, None otherwise
        """
        # Only check transformation_portal imports
        if not imported_module.startswith('transformation_portal.'):
            return None
        
        # Extract layer from imported module
        parts = imported_module.split('.')
        if len(parts) < 2:
            return None
        
        imported_layer = parts[1]
        
        if imported_layer not in LAYER_HIERARCHY:
            return None  # Unknown layer, skip
        
        importing_level = LAYER_HIERARCHY[importing_layer]
        imported_level = LAYER_HIERARCHY[imported_layer]
        
        # Violation: Lower layer importing from higher layer
        if importing_level < imported_level:
            return ImportViolation(
                file=filepath,
                line=lineno,
                importing_module=importing_layer,
                imported_module=imported_layer,
                reason=f"Layer '{importing_layer}' (level {importing_level}) "
                       f"cannot import from '{imported_layer}' (level {imported_level})"
            )
        
        return None
    
    def check_all(self) -> List[ImportViolation]:
        """
        Check all Python files in src directory.
        
        Returns:
            List of all violations found
        """
        python_files = list(self.src_dir.glob('**/*.py'))
        
        for filepath in python_files:
            violations = self.check_file(filepath)
            self.violations.extend(violations)
        
        return self.violations
    
    def print_report(self) -> None:
        """Print violation report."""
        if not self.violations:
            print("✅ No module boundary violations found!")
            return
        
        print(f"❌ Found {len(self.violations)} module boundary violation(s):\n")
        
        # Group by importing module
        by_module = defaultdict(list)
        for v in self.violations:
            by_module[v.importing_module].append(v)
        
        for module, violations in sorted(by_module.items()):
            print(f"\n{module}/ violations ({len(violations)}):")
            for v in violations:
                rel_path = v.file.relative_to(self.src_dir.parent.parent)
                print(f"  {rel_path}:{v.line}")
                print(f"    ⚠️  {v.reason}")
        
        print(f"\n\nTotal violations: {len(self.violations)}")
        print("\nTo fix:")
        print("1. Refactor code to use interfaces instead of direct imports")
        print("2. Move shared code to lower layers (utils)")
        print("3. Use dependency injection for cross-layer dependencies")
        print("\nSee docs/architecture/adr/ADR-001-module-interface-contracts.md")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Check module boundary violations in transformation_portal"
    )
    parser.add_argument(
        '--src-dir',
        type=Path,
        default=Path(__file__).parent.parent.parent / 'src',
        help="Source directory to check (default: src/)"
    )
    parser.add_argument(
        '--fail-on-violations',
        action='store_true',
        help="Exit with code 1 if violations found (for CI)"
    )
    
    args = parser.parse_args()
    
    if not args.src_dir.exists():
        print(f"Error: Source directory not found: {args.src_dir}")
        sys.exit(1)
    
    checker = ModuleBoundaryChecker(args.src_dir)
    violations = checker.check_all()
    checker.print_report()
    
    if violations and args.fail_on_violations:
        sys.exit(1)
    
    sys.exit(0)


if __name__ == '__main__':
    main()
