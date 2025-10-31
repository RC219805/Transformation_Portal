from pathlib import Path

from tools.montecito_manifest import iter_files, write_manifest


def test_iter_files_orders_results(tmp_path: Path) -> None:
    """Test that iter_files yields files in sorted order with correct metadata."""
    (tmp_path / "a").write_text("alpha", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "b").write_bytes(b"beta")

    results = list(iter_files(tmp_path))

    # Should return (relative_path, size_bytes, md5_hash) tuples
    assert [entry[0] for entry in results] == [Path("a"), Path("nested/b")]
    assert results[0][1] == 5  # "alpha" is 5 bytes
    assert len(results[0][2]) == 32  # MD5 hash length


def test_write_manifest_creates_csv(tmp_path: Path) -> None:
    """Test that write_manifest creates a proper CSV file."""
    (tmp_path / "file.txt").write_text("content", encoding="utf-8")
    destination = tmp_path / "manifest.csv"

    write_manifest(tmp_path, destination)

    lines = destination.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "filename,bytes,md5"
    assert len(lines) == 2
    # Check that the line contains the filename, byte count, and MD5 hash
    parts = lines[1].split(",")
    assert parts[0] == "file.txt"
    assert parts[1] == "7"  # "content" is 7 bytes
    assert len(parts[2]) == 32  # MD5 hash


def test_iter_files_handles_empty_directory(tmp_path: Path) -> None:
    """Test that iter_files handles an empty directory without errors."""
    results = list(iter_files(tmp_path))
    assert not results


def test_iter_files_ignores_directories(tmp_path: Path) -> None:
    """Test that iter_files only yields files, not directories."""
    (tmp_path / "file.txt").write_text("test", encoding="utf-8")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "file2.txt").write_text("test2", encoding="utf-8")

    results = list(iter_files(tmp_path))

    # Should only include files, not the directory itself
    paths = [entry[0] for entry in results]
    assert Path("subdir") not in paths
    assert Path("file.txt") in paths
    assert Path("subdir/file2.txt") in paths


def test_write_manifest_creates_parent_directories(tmp_path: Path) -> None:
    """Test that write_manifest creates parent directories if needed."""
    (tmp_path / "file.txt").write_text("content", encoding="utf-8")
    destination = tmp_path / "nested" / "dir" / "manifest.csv"

    write_manifest(tmp_path, destination)

    assert destination.exists()
    assert destination.parent.exists()


def test_iter_files_handles_nested_directories(tmp_path: Path) -> None:
    """Test that iter_files correctly handles files in deeply nested directories."""
    nested = tmp_path / "level1" / "level2"
    nested.mkdir(parents=True)
    (nested / "file.txt").write_text("test", encoding="utf-8")

    results = list(iter_files(tmp_path))

    # Should find the file in nested directories with correct relative path
    assert len(results) == 1
    relative_path = results[0][0]
    assert relative_path == Path("level1/level2/file.txt")
