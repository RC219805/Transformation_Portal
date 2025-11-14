#!/usr/bin/env python3
"""
Transformation Portal — RAG Environment Verifier

Checks:
  1) Python version and virtualenv status
  2) Torch with Metal (MPS) backend availability and a tiny MPS matmul
  3) FAISS availability and a minimal index/search roundtrip
  4) Sentence-Transformers availability (no model download)
  5) RAG modules importability from .github/agents/rag_system

Usage:
  python verify_rag_install.py [--repo /Users/rc/Transformation_Portal] [--fix-imports] [--verbose]

Exit codes:
  0 = all good
  1 = warnings only (non-fatal)
  2 = failure (at least one required check failed)
"""

from __future__ import annotations
import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple

REPO_DEFAULT = "/Users/rc/Transformation_Portal"

# --- UI helpers ---
class UI:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    @staticmethod
    def ok(msg: str) -> None:
        print(f"{UI.GREEN}✅ {msg}{UI.RESET}")

    @staticmethod
    def warn(msg: str) -> None:
        print(f"{UI.YELLOW}⚠️  {msg}{UI.RESET}")

    @staticmethod
    def err(msg: str) -> None:
        print(f"{UI.RED}❌ {msg}{UI.RESET}")

    @staticmethod
    def info(msg: str) -> None:
        print(f"{UI.DIM}{msg}{UI.RESET}")


def check_python(args) -> Tuple[bool, str]:
    ok = True
    msgs: List[str] = []
    ver = sys.version.split()[0]
    in_venv = (hasattr(sys, "real_prefix") or (getattr(sys, "base_prefix", "") != sys.prefix) or bool(os.environ.get("VIRTUAL_ENV")))
    msgs.append(f"Python: {ver}")
    msgs.append(f"Virtualenv active: {in_venv} (sys.prefix={sys.prefix})")

    # Expect 3.11.x
    major_minor = ".".join(ver.split(".")[:2])
    if major_minor != "3.11":
        ok = False
        msgs.append("Expected Python 3.11.x for best compatibility on macOS ARM.")
    if not in_venv:
        ok = False
        msgs.append("Not running inside a virtual environment.")
    return ok, "\n".join(msgs)


def check_torch(args) -> Tuple[bool, str]:
    try:
        import torch  # type: ignore
        msgs = [f"Torch: {torch.__version__}"]
        mps_built = getattr(torch.backends.mps, "is_built", lambda: False)()
        mps_avail = getattr(torch.backends.mps, "is_available", lambda: False)()
        msgs.append(f"MPS built: {mps_built}, available: {mps_avail}")
        ok = True

        # Tiny runtime test if available
        if mps_avail:
            try:
                a = torch.randn((2, 2), device="mps")
                b = torch.randn((2, 2), device="mps")
                _ = a @ b
                msgs.append("MPS runtime matmul: OK")
            except Exception as e:
                ok = False
                msgs.append(f"MPS runtime matmul FAILED: {e}")
        else:
            UI.warn("MPS not available. Torch will fall back to CPU.")

        return ok, "\n".join(msgs)
    except Exception as e:
        return False, f"Torch import failed: {e}"


def check_faiss(args) -> Tuple[bool, str]:
    try:
        import faiss  # type: ignore
        msgs = [f"FAISS: {getattr(faiss, '__version__', 'unknown')}"]
        # Minimal index roundtrip
        import numpy as np
        d = 4
        idx = faiss.IndexFlatL2(d)
        x = np.array([[0.0, 0.1, 0.2, 0.3],
                      [0.9, 0.8, 0.7, 0.6]], dtype="float32")
        idx.add(x)
        D, I = idx.search(x, 1)
        ok = I.shape == (2, 1)
        msgs.append(f"FAISS search: OK (neighbors={I.ravel().tolist()})" if ok else "FAISS search: FAILED")
        return ok, "\n".join(msgs)
    except Exception as e:
        return False, f"FAISS check failed: {e}"


def check_sentence_transformers(args) -> Tuple[bool, str]:
    try:
        import sentence_transformers  # type: ignore
        ver = getattr(sentence_transformers, "__version__", "unknown")
        return True, f"Sentence-Transformers: {ver} (import OK)"
    except Exception as e:
        return False, f"Sentence-Transformers import failed: {e}"


def ensure_repo_on_path(repo_root: Path) -> None:
    # Ensure the repo root is on sys.path so imports resolve without global config
    repo_str = str(repo_root)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def maybe_fix_hidden_github(repo_root: Path, verbose: bool = False) -> Tuple[bool, str]:
    """
    Make hidden .github importable as 'github' by ensuring:
      - __init__.py files exist
      - a symlink 'github' -> '.github' exists
    """
    msgs: List[str] = []
    gh = repo_root / ".github"
    if not gh.exists():
        return False, f"Missing {gh} directory."

    # __init__.py files
    for p in [gh, gh / "agents", gh / "agents" / "rag_system"]:
        if p.exists():
            init = p / "__init__.py"
            if not init.exists():
                try:
                    init.touch()
                    msgs.append(f"Created: {init}")
                except Exception as e:
                    return False, f"Failed to create {init}: {e}"

    # Symlink github -> .github
    link = repo_root / "github"
    if not link.exists():
        try:
            link.symlink_to(".github")
            msgs.append(f"Created symlink: {link} -> .github")
        except FileExistsError:
            pass
        except Exception as e:
            return False, f"Failed to create symlink {link}: {e}"
    else:
        if verbose:
            msgs.append(f"Symlink already present: {link} -> .github")

    return True, "\n".join(msgs) if msgs else "Import alias already configured."


def check_rag_imports(args, repo_root: Path) -> Tuple[bool, str]:
    ensure_repo_on_path(repo_root)
    try:
        # Try canonical path first
        from github.agents.rag_system.retriever import HybridRetriever  # type: ignore
        from github.agents.rag_system.indexer import RepositoryIndexer  # type: ignore
        from github.agents.rag_system.citation import CitationGenerator  # type: ignore
        return True, "RAG modules import OK (github.agents.rag_system.*)"
    except ModuleNotFoundError as e:
        if args.fix_imports:
            ok, msg = maybe_fix_hidden_github(repo_root, verbose=args.verbose)
            if not ok:
                return False, f"RAG import fix failed: {msg}"
            # Retry after fix
            try:
                from github.agents.rag_system.retriever import HybridRetriever  # type: ignore
                from github.agents.rag_system.indexer import RepositoryIndexer  # type: ignore
                from github.agents.rag_system.citation import CitationGenerator  # type: ignore
                return True, f"RAG modules import OK after fix.\n{msg}"
            except Exception as e2:
                return False, f"RAG modules still not importable after fix: {e2}"
        return False, f"RAG import failed: {e}"
    except Exception as e:
        return False, f"RAG import failed: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=REPO_DEFAULT, help="Repository root path")
    parser.add_argument("--fix-imports", action="store_true", help="Create 'github'->'.github' symlink and missing __init__.py files if needed")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo).resolve()
    if not repo_root.exists():
        UI.err(f"Repository path does not exist: {repo_root}")
        sys.exit(2)

    UI.info(f"Verifying environment for repo: {repo_root}")

    failures = 0
    warnings = 0

    # 1) Python / venv
    ok, msg = check_python(args)
    (UI.ok if ok else UI.err)(msg)
    if not ok:
        failures += 1

    # 2) Torch + MPS
    ok, msg = check_torch(args)
    (UI.ok if ok else UI.err)(msg)
    if not ok:
        failures += 1

    # 3) FAISS
    ok, msg = check_faiss(args)
    (UI.ok if ok else UI.err)(msg)
    if not ok:
        failures += 1

    # 4) Sentence-Transformers
    ok, msg = check_sentence_transformers(args)
    if ok:
        UI.ok(msg)
    else:
        UI.warn(msg)
        warnings += 1

    # 5) RAG imports
    ok, msg = check_rag_imports(args, repo_root)
    (UI.ok if ok else UI.err)(msg)
    if not ok:
        failures += 1

    # Summary / exit code
    print()
    if failures == 0 and warnings == 0:
        UI.ok("System Ready — all checks passed.")
        sys.exit(0)
    elif failures == 0 and warnings > 0:
        UI.warn(f"System Ready with warnings ({warnings}).")
        sys.exit(1)
    else:
        UI.err(f"System NOT ready — failures: {failures}, warnings: {warnings}.")
        sys.exit(2)


if __name__ == "__main__":
    main()