#!/usr/bin/env python3
"""
Transformation Portal - Developer Setup Script
===============================================

Unified setup script for development environment with RAG hooks integration.

This script provides:
- Virtual environment setup
- Dependency installation (core + optional ML/TIFF extras)
- RAG system git hooks installation
- Development tool configuration
- Pre-commit hooks setup

Usage:
    python scripts/setup/dev_setup.py              # Full setup
    python scripts/setup/dev_setup.py --minimal    # Core deps only
    python scripts/setup/dev_setup.py --with-ml    # Include ML extras
    python scripts/setup/dev_setup.py --with-rag   # Include RAG hooks
    python scripts/setup/dev_setup.py --all        # Everything

Author: Transformation Portal
Version: 1.0.0
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def get_repo_root() -> Path:
    """Get the repository root directory."""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("Could not find repository root (.git directory)")


def run_command(cmd: list[str], cwd: Path | None = None, check: bool = True) -> int:
    """Run a command and return exit code."""
    print(f"  → Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=False)
    if check and result.returncode != 0:
        print(f"  ✗ Command failed with exit code {result.returncode}")
    return result.returncode


def check_python_version() -> bool:
    """Check if Python version is 3.10+."""
    if sys.version_info < (3, 10):
        print(f"✗ Python 3.10+ required. Current: {sys.version}")
        return False
    print(f"✓ Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True


def setup_venv(repo_root: Path) -> Path:
    """Create virtual environment if it doesn't exist."""
    venv_path = repo_root / ".venv"
    
    if venv_path.exists():
        print(f"✓ Virtual environment exists: {venv_path}")
    else:
        print("Creating virtual environment...")
        run_command([sys.executable, "-m", "venv", str(venv_path)])
        print(f"✓ Created virtual environment: {venv_path}")
    
    return venv_path


def get_pip_executable(venv_path: Path) -> str:
    """Get the pip executable path for the virtual environment."""
    if sys.platform == "win32":
        return str(venv_path / "Scripts" / "pip.exe")
    return str(venv_path / "bin" / "pip")


def install_dependencies(pip_exe: str, repo_root: Path, options: argparse.Namespace) -> bool:
    """Install project dependencies."""
    print("\n📦 Installing dependencies...")
    
    # Upgrade pip
    run_command([pip_exe, "install", "--upgrade", "pip", "wheel"])
    
    # Install core dependencies
    print("\nInstalling core dependencies...")
    requirements_file = repo_root / "requirements.txt"
    if requirements_file.exists():
        run_command([pip_exe, "install", "-r", str(requirements_file)])
    
    # Install dev dependencies
    dev_requirements = repo_root / "requirements-dev.txt"
    if dev_requirements.exists():
        print("\nInstalling development dependencies...")
        run_command([pip_exe, "install", "-r", str(dev_requirements)])
    
    # Install package in editable mode
    print("\nInstalling package in editable mode...")
    extras = []
    if options.with_ml or options.all:
        extras.append("ml")
    if options.with_tiff or options.all:
        extras.append("tiff")
    if options.with_dev or options.all:
        extras.append("dev")
    
    if extras:
        extras_str = ",".join(extras)
        run_command([pip_exe, "install", "-e", f".[{extras_str}]"], cwd=repo_root)
    else:
        run_command([pip_exe, "install", "-e", "."], cwd=repo_root)
    
    print("✓ Dependencies installed")
    return True


def install_rag_hooks(repo_root: Path) -> bool:
    """Install RAG system git hooks for incremental indexing."""
    print("\n🔗 Installing RAG git hooks...")
    
    git_hooks_script = repo_root / ".github" / "agents" / "rag_system" / "git_hooks.py"
    
    if not git_hooks_script.exists():
        print(f"✗ RAG git hooks script not found: {git_hooks_script}")
        return False
    
    # Run the git hooks installer
    result = run_command(
        [sys.executable, str(git_hooks_script), "install"],
        cwd=repo_root,
        check=False
    )
    
    if result == 0:
        print("✓ RAG git hooks installed successfully")
        print("  - post-commit: Automatic index updates")
        print("  - post-merge: Index sync after pulls")
        print("  - pre-push: Cache consistency validation")
        return True
    else:
        print("⚠ RAG git hooks installation failed (non-critical)")
        return False


def setup_pre_commit(repo_root: Path, pip_exe: str) -> bool:
    """Install and configure pre-commit hooks."""
    print("\n🪝 Setting up pre-commit hooks...")
    
    pre_commit_config = repo_root / ".pre-commit-config.yaml"
    
    if not pre_commit_config.exists():
        print("⚠ No .pre-commit-config.yaml found, skipping pre-commit setup")
        return True
    
    # Install pre-commit
    run_command([pip_exe, "install", "pre-commit"])
    
    # Install hooks
    result = run_command(
        ["pre-commit", "install"],
        cwd=repo_root,
        check=False
    )
    
    if result == 0:
        print("✓ Pre-commit hooks installed")
        return True
    else:
        print("⚠ Pre-commit setup failed (non-critical)")
        return False


def print_summary(options: argparse.Namespace) -> None:
    """Print setup summary and next steps."""
    print("\n" + "=" * 60)
    print("🎉 Development Environment Setup Complete!")
    print("=" * 60)
    
    print("\n📝 Next Steps:")
    print("  1. Activate the virtual environment:")
    print("     source .venv/bin/activate  # Linux/macOS")
    print("     .venv\\Scripts\\activate     # Windows")
    print()
    print("  2. Run tests:")
    print("     make test-fast             # Quick tests")
    print("     pytest tests/              # Full test suite")
    print()
    print("  3. Run linting:")
    print("     make lint                  # Full lint check")
    print()
    
    if options.with_rag or options.all:
        print("  4. RAG System commands:")
        print("     python .github/agents/rag_system/git_hooks.py validate  # Check cache")
        print("     python .github/agents/rag_system/git_hooks.py update    # Manual update")
        print()
    
    print("📚 Documentation:")
    print("   - README.md                 - Project overview")
    print("   - .github/copilot-instructions.md - Development guidelines")
    print("   - .github/agents/           - RAG and agent documentation")
    print()


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Transformation Portal Developer Setup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/setup/dev_setup.py              # Standard setup
  python scripts/setup/dev_setup.py --minimal    # Core deps only
  python scripts/setup/dev_setup.py --with-ml    # Include ML extras
  python scripts/setup/dev_setup.py --with-rag   # Include RAG hooks
  python scripts/setup/dev_setup.py --all        # Everything
        """
    )
    
    parser.add_argument(
        "--minimal", action="store_true",
        help="Install only core dependencies (no extras)"
    )
    parser.add_argument(
        "--with-ml", action="store_true",
        help="Include ML extras (PyTorch, transformers, etc.)"
    )
    parser.add_argument(
        "--with-tiff", action="store_true",
        help="Include TIFF processing extras (tifffile, imagecodecs)"
    )
    parser.add_argument(
        "--with-dev", action="store_true",
        help="Include development tools (pytest, flake8, etc.)"
    )
    parser.add_argument(
        "--with-rag", action="store_true",
        help="Install RAG system git hooks for incremental indexing"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Install everything (all extras + RAG hooks)"
    )
    parser.add_argument(
        "--skip-venv", action="store_true",
        help="Skip virtual environment creation (use current Python)"
    )
    
    options = parser.parse_args()
    
    print("=" * 60)
    print("🚀 Transformation Portal - Developer Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Get repository root
    try:
        repo_root = get_repo_root()
        print(f"✓ Repository root: {repo_root}")
    except RuntimeError as e:
        print(f"✗ {e}")
        return 1
    
    # Setup virtual environment
    if options.skip_venv:
        pip_exe = "pip"
        print("⚠ Skipping virtual environment (using current Python)")
    else:
        venv_path = setup_venv(repo_root)
        pip_exe = get_pip_executable(venv_path)
    
    # Install dependencies
    if not options.minimal:
        options.with_dev = True  # Always include dev deps unless minimal
    
    install_dependencies(pip_exe, repo_root, options)
    
    # Install RAG hooks
    if options.with_rag or options.all:
        install_rag_hooks(repo_root)
    
    # Setup pre-commit (if not minimal)
    if not options.minimal:
        setup_pre_commit(repo_root, pip_exe)
    
    # Print summary
    print_summary(options)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
