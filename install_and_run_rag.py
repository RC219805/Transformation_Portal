#!/usr/bin/env python3
"""
Transformation Portal — RAG System Installer (Python 3.11 + Architecture-Aware FAISS)

This script:
1. Locates or validates Python 3.11
2. Creates and activates a venv at /Users/rc/Transformation_Portal/venv
3. Detects Apple Silicon vs Intel and installs correct FAISS build
4. Installs dependencies + RAG system in editable mode
5. Verifies full functionality (indexing + retrieval)
"""

import os
import sys
import platform
import subprocess
from pathlib import Path


# === Configuration ===
REPO_ROOT = Path("/Users/rc/Transformation_Portal")
VENV_PATH = REPO_ROOT / "venv"
RAG_PATH = REPO_ROOT / ".github" / "agents" / "rag_system"
REQUIREMENTS_FILE = RAG_PATH / "requirements.txt"


def run(cmd, cwd=None, env=None):
    """Execute a command visibly."""
    print(f"\n> {' '.join(str(x) for x in cmd)}")
    subprocess.check_call(cmd, cwd=cwd, env=env)


def locate_python311():
    """Find a Python 3.11 interpreter."""
    print("🔍 Locating Python 3.11...")
    candidates = [
        "python3.11",
        "/opt/homebrew/bin/python3.11",  # Apple Silicon
        "/usr/local/bin/python3.11"      # Intel Mac / Homebrew (x86)
    ]
    for c in candidates:
        try:
            out = subprocess.check_output([c, "--version"], stderr=subprocess.STDOUT).decode()
            if "3.11" in out:
                print(f"✅ Found Python 3.11 at {c}")
                return c
        except Exception:
            continue
    print("❌ Python 3.11 not found. Install via:")
    print("   brew install python@3.11")
    sys.exit(1)


def ensure_venv(python311_path):
    """Create venv and install packages."""
    if not VENV_PATH.exists():
        print(f"⚙️  Creating venv at {VENV_PATH} using {python311_path} ...")
        run([python311_path, "-m", "venv", str(VENV_PATH)])
    else:
        print(f"✅ Existing venv detected at {VENV_PATH}")

    pip = VENV_PATH / "bin" / "pip"
    python = VENV_PATH / "bin" / "python"

    print("🔼 Upgrading pip...")
    run([str(pip), "install", "--upgrade", "pip"])

    # Detect architecture
    arch = platform.machine().lower()
    is_arm = "arm" in arch or "aarch" in arch
    faiss_pkg = "faiss-metal" if is_arm else "faiss-cpu"
    print(f"🧩 Architecture: {'Apple Silicon (ARM)' if is_arm else 'Intel / x86_64'}")
    print(f"📦 Installing FAISS package: {faiss_pkg}")

    print("📦 Installing dependencies...")
    if REQUIREMENTS_FILE.exists():
        run([str(pip), "install", "-r", str(REQUIREMENTS_FILE)])
    else:
        run([
            str(pip), "install",
            "numpy", "pandas", "scikit-learn",
            faiss_pkg, "openai", "tiktoken", "pytest"
        ])

    print("🔗 Installing RAG system in editable (-e) mode...")
    run([str(pip), "install", "-e", str(REPO_ROOT)])

    return python


def verify_install(python_path):
    """Run basic indexing + retrieval verification."""
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{RAG_PATH}:{env.get('PYTHONPATH', '')}"

    print("\n🧠 Verifying RAG system installation...")
    code = """
from citation import CitationGenerator
from retriever import HybridRetriever
from indexer import RepositoryIndexer
from pathlib import Path
repo = Path('/Users/rc/Transformation_Portal')
indexer = RepositoryIndexer(str(repo))
chunks = indexer.index_repository()
retriever = HybridRetriever()
retriever.index(chunks)
results = retriever.retrieve('depth pipeline', top_k=3)
print(f'✅ Verification complete — {len(chunks)} chunks indexed, {len(results)} results retrieved.')
"""
    run([str(python_path), "-c", code], env=env)
    print("\n🎯 RAG system verified successfully (Python 3.11, FAISS optimized).")


def main():
    print("=" * 90)
    print("Transformation Portal — RAG System Setup (Apple Silicon Optimized)")
    print("=" * 90)

    python311 = locate_python311()
    python_path = ensure_venv(python311)
    verify_install(python_path)

    print("\n✨ Setup complete!")
    print(f"📁 Repository: {REPO_ROOT}")
    print(f"🐍 Virtual Env: {VENV_PATH}")
    print("=" * 90)


if __name__ == "__main__":
    main()