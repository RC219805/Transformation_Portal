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
