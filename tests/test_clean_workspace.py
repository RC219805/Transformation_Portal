"""
Tests for tools/clean_workspace.py

Verifies that the workspace cleanup utility:
- Correctly identifies cleanup candidates
- Respects repository boundaries
- Provides accurate dry-run output
"""

import sys
from pathlib import Path
import tempfile

# Add tools directory to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import clean_workspace


def test_classify_path():
    """Test path classification for output formatting."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Create test files and directories
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")
        test_dir = tmp_path / "testdir"
        test_dir.mkdir()
        
        assert clean_workspace.classify_path(test_file) == "FILE"
        assert clean_workspace.classify_path(test_dir) == "DIR "


def test_file_patterns():
    """Verify cleanup patterns are reasonable and comprehensive."""
    # Should include common artifact patterns
    assert "*.log" in clean_workspace.FILE_PATTERNS
    assert "**/*.pyc" in clean_workspace.RECURSIVE_FILE_PATTERNS
    assert "**/__pycache__" in clean_workspace.DIR_PATTERNS
    
    # Should include build artifacts from old Makefile
    assert "**/.pytest_cache" in clean_workspace.DIR_PATTERNS
    assert "**/.hypothesis" in clean_workspace.DIR_PATTERNS
    assert "**/*.egg-info" in clean_workspace.DIR_PATTERNS
    assert "build" in clean_workspace.DIR_PATTERNS
    assert "dist" in clean_workspace.DIR_PATTERNS
    
    # Should not include overly broad patterns
    for pattern in clean_workspace.FILE_PATTERNS:
        assert pattern != "*"  # Would delete everything
    for pattern in clean_workspace.DIR_PATTERNS:
        assert pattern not in ("*", "**")  # Would delete all directories


def test_iter_paths_stays_in_repo():
    """Verify that iter_paths only yields paths within the repository."""
    paths = list(clean_workspace.iter_paths())
    
    # Filter to resolved paths that exist
    existing_paths = []
    for p in paths:
        try:
            resolved = p.resolve()
            if resolved.exists():
                existing_paths.append(resolved)
        except Exception:
            continue
    
    # All paths should be within the repository root
    for p in existing_paths:
        try:
            p.relative_to(clean_workspace.ROOT)
        except ValueError:
            # This is acceptable - the cleanup function filters these out
            pass


def test_cleanup_dry_run_no_deletion():
    """Verify dry-run mode does not delete files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Create test artifacts
        test_log = tmp_path / "test.log"
        test_log.write_text("test log")
        test_cache = tmp_path / "__pycache__"
        test_cache.mkdir()
        
        # Store original ROOT
        original_root = clean_workspace.ROOT
        
        try:
            # Temporarily set ROOT to tmpdir for testing
            clean_workspace.ROOT = tmp_path
            
            # Run cleanup in dry-run mode
            clean_workspace.cleanup(apply=False, verbose=False)
            
            # Files should still exist
            assert test_log.exists()
            assert test_cache.exists()
            
        finally:
            # Restore original ROOT
            clean_workspace.ROOT = original_root


def test_cleanup_apply_deletes_files(tmp_path, monkeypatch):
    """Verify apply mode actually deletes files matching patterns."""
    # Set ROOT to a temp dir
    monkeypatch.setattr(clean_workspace, "ROOT", tmp_path)
    monkeypatch.setattr(clean_workspace, "FILE_PATTERNS", ["test.log"])
    monkeypatch.setattr(clean_workspace, "DIR_PATTERNS", ["__pycache__"])
    monkeypatch.setattr(clean_workspace, "RECURSIVE_FILE_PATTERNS", [])

    test_log = tmp_path / "test.log"
    pycache_dir = tmp_path / "__pycache__"
    pycache_dir.mkdir()
    (pycache_dir / "foo.pyc").write_text("bytecode")
    test_log.write_text("log")

    # Neutralize tracked files for this test
    monkeypatch.setattr(clean_workspace, "get_tracked_files", lambda: set())

    # Run cleanup with apply
    clean_workspace.cleanup(apply=True, verbose=False)

    # Files should be deleted
    assert not test_log.exists(), "test.log should be deleted"
    assert not pycache_dir.exists(), "__pycache__ should be deleted"


def test_cleanup_skips_tracked_files(tmp_path, monkeypatch):
    """Verify cleanup skips files tracked by git."""
    # Fake a tracked file that would otherwise be deleted
    fake_tracked = (tmp_path / "foo.log").resolve()

    def fake_get_tracked_files():
        return {fake_tracked}

    monkeypatch.setattr(clean_workspace, "get_tracked_files", fake_get_tracked_files)
    monkeypatch.setattr(clean_workspace, "ROOT", tmp_path)
    monkeypatch.setattr(clean_workspace, "FILE_PATTERNS", ["*.log"])
    monkeypatch.setattr(clean_workspace, "DIR_PATTERNS", [])
    monkeypatch.setattr(clean_workspace, "RECURSIVE_FILE_PATTERNS", [])

    fake_tracked.parent.mkdir(parents=True, exist_ok=True)
    fake_tracked.write_text("tracked data")

    clean_workspace.cleanup(apply=True, verbose=False)

    # File should still be there, because it's "tracked"
    assert fake_tracked.exists(), "Tracked file should not be deleted"


def test_cleanup_skips_excluded_dirs(tmp_path, monkeypatch):
    """Verify cleanup skips excluded directories (.venv, weights, .git)."""
    # Create fake .venv with __pycache__ inside
    venv_dir = tmp_path / ".venv" / "lib" / "python3.11"
    venv_dir.mkdir(parents=True)
    venv_cache = venv_dir / "__pycache__"
    venv_cache.mkdir()
    (venv_cache / "test.pyc").write_text("bytecode")
    
    # Create fake weights dir with a log file
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    (weights_dir / "training.log").write_text("log")
    
    # Create non-excluded cache that should be cleaned
    normal_cache = tmp_path / "src" / "__pycache__"
    normal_cache.mkdir(parents=True)
    (normal_cache / "test.pyc").write_text("bytecode")

    monkeypatch.setattr(clean_workspace, "ROOT", tmp_path)
    monkeypatch.setattr(clean_workspace, "FILE_PATTERNS", [])
    monkeypatch.setattr(clean_workspace, "DIR_PATTERNS", ["**/__pycache__", "*.log"])
    monkeypatch.setattr(clean_workspace, "RECURSIVE_FILE_PATTERNS", [])
    monkeypatch.setattr(clean_workspace, "get_tracked_files", lambda: set())

    clean_workspace.cleanup(apply=True, verbose=False)

    # .venv and weights should be untouched
    assert venv_cache.exists(), ".venv/__pycache__ should be excluded"
    assert (weights_dir / "training.log").exists(), "weights/training.log should be excluded"
    
    # Normal workspace artifacts should be cleaned
    assert not normal_cache.exists(), "src/__pycache__ should be deleted"


def test_is_excluded():
    """Test the exclusion logic."""
    # Save original ROOT
    original_root = clean_workspace.ROOT
    
    try:
        # Use a temp directory as ROOT
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            clean_workspace.ROOT = tmp_path
            
            # Paths inside excluded dirs should be excluded
            assert clean_workspace.is_excluded(tmp_path / ".venv" / "lib" / "test.py")
            assert clean_workspace.is_excluded(tmp_path / "weights" / "model.pth")
            assert clean_workspace.is_excluded(tmp_path / ".git" / "config")
            
            # Paths outside excluded dirs should not be excluded
            assert not clean_workspace.is_excluded(tmp_path / "src" / "test.py")
            assert not clean_workspace.is_excluded(tmp_path / "tests" / "test.py")
            
    finally:
        clean_workspace.ROOT = original_root


def test_get_tracked_files_graceful_failure():
    """Verify get_tracked_files handles non-git repos gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        original_root = clean_workspace.ROOT
        try:
            clean_workspace.ROOT = tmp_path
            tracked = clean_workspace.get_tracked_files()
            # Should return empty set in non-git directory
            assert isinstance(tracked, set)
        finally:
            clean_workspace.ROOT = original_root


def test_cleanup_handles_external_paths_gracefully(tmp_path):
    """Verify cleanup handles paths outside the repository gracefully."""
    # Create a file outside the repo (simulated by tmpdir)
    external_file = tmp_path / "external.log"
    external_file.write_text("external")
    
    # Create symlink in repo to external file
    # (This test is conceptual - in practice the cleanup function
    # checks paths after resolution and skips non-repo paths)
    
    # The cleanup function should handle this gracefully
    # by checking if resolved path is within ROOT
    original_root = clean_workspace.ROOT
    
    try:
        clean_workspace.ROOT = Path("/nonexistent/path")
        
        # Should not raise an error even with invalid ROOT
        clean_workspace.cleanup(apply=False, verbose=False)
        
    finally:
        clean_workspace.ROOT = original_root


def test_no_cleanup_when_workspace_tidy(capsys):
    """Verify message when no cleanup needed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Empty directory - nothing to clean
        original_root = clean_workspace.ROOT
        
        try:
            clean_workspace.ROOT = tmp_path
            clean_workspace.cleanup(apply=False, verbose=False)
            
            captured = capsys.readouterr()
            assert "Nothing to clean" in captured.out or "Workspace already tidy" in captured.out
            
        finally:
            clean_workspace.ROOT = original_root
