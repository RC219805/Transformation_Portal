#!/usr/bin/env python3
"""
Wrapper script for RAG system CLI that handles import paths correctly.

Usage:
    python scripts/utilities/rag_cli_wrapper.py index --repo-root . --output stats.json
    python scripts/utilities/rag_cli_wrapper.py search "depth pipeline" --top-k 5
"""

import sys
from pathlib import Path

# Add RAG system to path
rag_system_path = Path(__file__).parent.parent.parent / ".github" / "agents"
sys.path.insert(0, str(rag_system_path))

# Now import and run the CLI
from rag_system.cli import main

if __name__ == "__main__":
    main()
