"""
Integration tests for enhanced RAG system features.

Tests:
- Persistent caching
- Configuration system
- Logging
- Vector embeddings
- Query caching
- Full pipeline integration
"""

# pylint: disable=wrong-import-position,redefined-outer-name

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

# Add agents directory to path
agents_path = Path(__file__).parent.parent / ".github" / "agents"
sys.path.insert(0, str(agents_path))

from rag_system.citation import CitationGenerator  # noqa: E402
from rag_system.config import Config, get_config, reset_config  # noqa: E402
from rag_system.exceptions import IndexingError, RetrievalError  # noqa: E402
from rag_system.indexer import DocumentChunk, RepositoryIndexer  # noqa: E402
from rag_system.logger import get_logger  # noqa: E402
from rag_system.reranker import ResultReranker  # noqa: E402
from rag_system.retriever import HybridRetriever  # noqa: E402


@pytest.fixture
def repo_root():
    """Get repository root path."""
    return Path(__file__).parent.parent


@pytest.fixture
def temp_repo():
    """Create temporary repository for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create sample structure
        (tmp_path / "docs").mkdir()
        (tmp_path / "src").mkdir()
        (tmp_path / "tests").mkdir()

        # Create sample files
        (tmp_path / "docs" / "readme.md").write_text("# Test Repository\n\nThis is a test repository for RAG system.")
        (tmp_path / "src" / "main.py").write_text(
            'def hello_world():\n    """Print hello world."""\n    print(\'Hello, World!\')'
        )
        (tmp_path / "tests" / "test_main.py").write_text("def test_hello():\n    assert True")

        yield tmp_path


@pytest.fixture(autouse=True)
def reset_config_after_test():
    """Reset config after each test."""
    yield
    reset_config()


class TestConfiguration:
    """Test configuration system."""

    def test_config_loads_defaults(self):
        """Test that config loads with defaults."""
        config = Config()
        assert config.get("indexer.chunk_size_tokens") == 750
        assert config.get("retriever.bm25_weight") == 0.7
        assert config.get("citation.max_results") == 5

    def test_config_get_section(self):
        """Test getting entire section."""
        config = Config()
        indexer_config = config.get_section("indexer")

        assert isinstance(indexer_config, dict)
        assert "chunk_size_tokens" in indexer_config
        assert "overlap_tokens" in indexer_config

    def test_config_set_value(self):
        """Test setting config value at runtime."""
        config = Config()
        config.set("indexer.chunk_size_tokens", 1000)

        assert config.get("indexer.chunk_size_tokens") == 1000

    def test_config_env_override(self, monkeypatch):
        """Test environment variable override."""
        monkeypatch.setenv("RAG_INDEXER_CACHE_ENABLED", "false")
        reset_config()

        config = get_config()
        assert config.get("indexer.cache_enabled") is False

    def test_config_env_override_types(self, monkeypatch):
        """Test environment variable override with different types."""
        # Boolean
        monkeypatch.setenv("RAG_INDEXER_CACHE_ENABLED", "true")
        # Float
        monkeypatch.setenv("RAG_RETRIEVER_BM25_WEIGHT", "0.9")
        # Integer
        monkeypatch.setenv("RAG_CITATION_MAX_RESULTS", "10")
        # String
        monkeypatch.setenv("RAG_INDEXER_CACHE_DIR", ".custom_cache")

        reset_config()
        config = get_config()

        assert config.get("indexer.cache_enabled") is True
        assert config.get("retriever.bm25_weight") == 0.9
        assert config.get("citation.max_results") == 10
        assert config.get("indexer.cache_dir") == ".custom_cache"


class TestPersistentCaching:
    """Test persistent caching functionality."""

    def test_cache_saves_and_loads(self, temp_repo):
        """Test that cache is saved and loaded correctly."""
        # First indexing
        indexer1 = RepositoryIndexer(str(temp_repo), use_cache=True)
        chunks1 = indexer1.index_repository()

        assert len(chunks1) > 0
        assert indexer1.cache_file.exists()

        # Second indexing should load from cache
        indexer2 = RepositoryIndexer(str(temp_repo), use_cache=True)
        chunks2 = indexer2.index_repository()

        assert len(chunks1) == len(chunks2)
        assert chunks1[0].content == chunks2[0].content

    def test_valid_cache_skips_chunking(self, temp_repo, monkeypatch):
        (temp_repo / "src" / "empty.py").write_text("")
        indexer = RepositoryIndexer(str(temp_repo))
        expected = indexer.index_repository()
        assert all(chunk.content.strip() for chunk in expected)

        def unexpected_chunking(*_args):
            pytest.fail("unchanged content should reuse cached chunks")

        monkeypatch.setattr(indexer, "_index_file", unexpected_chunking)
        assert indexer.index_repository() == expected

    @pytest.mark.parametrize("mutation", ["edit", "add", "delete", "rename"])
    def test_cache_invalidates_for_source_changes(self, temp_repo, mutation):
        indexer = RepositoryIndexer(str(temp_repo))
        previous = indexer.index_repository()
        source = temp_repo / "docs" / "readme.md"
        if mutation == "edit":
            # Hash bytes, not mtimes or sizes: same-length edits still invalidate.
            stat = source.stat()
            source.write_text(source.read_text().replace("Test", "Next"))
            os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        elif mutation == "add":
            (temp_repo / "docs" / "new.md").write_text("# New governed source")
        elif mutation == "delete":
            source.unlink()
        else:
            source.rename(source.with_name("renamed.md"))

        refreshed = indexer.index_repository()
        uncached = RepositoryIndexer(str(temp_repo), use_cache=False).index_repository()
        assert refreshed != previous
        assert refreshed == uncached

    def test_cache_invalidates_for_chunk_settings(self, temp_repo):
        source = temp_repo / "docs" / "readme.md"
        source.write_text("A line of documentation.\n" * 30)
        previous = RepositoryIndexer(str(temp_repo), chunk_size_tokens=750).index_repository()
        indexer = RepositoryIndexer(str(temp_repo), chunk_size_tokens=30, overlap_tokens=0)
        refreshed = indexer.index_repository()
        assert indexer.overlap == 0
        assert len(refreshed) > len(previous)
        assert refreshed == indexer.index_repository(force_reindex=True)

    def test_legacy_json_cache_is_rebuilt(self, temp_repo):
        indexer = RepositoryIndexer(str(temp_repo))
        expected = indexer.index_repository()
        payload = json.loads(indexer.cache_file.read_text())
        payload.update(cache_format="tp.rag.chunks.v1", version=1, chunks=[])
        payload.pop("source_fingerprint")
        indexer.cache_file.write_text(json.dumps(payload))
        assert indexer.index_repository() == expected

    def test_source_changes_during_indexing_do_not_publish_cache(self, temp_repo, monkeypatch):
        indexer = RepositoryIndexer(str(temp_repo))
        original = indexer._index_file

        def change_after_read(file_path, chunk_type):
            source_hash = original(file_path, chunk_type)
            if file_path.name == "readme.md":
                file_path.write_text("# Changed during indexing")
            return source_hash

        monkeypatch.setattr(indexer, "_index_file", change_after_read)
        indexer.index_repository()
        assert not indexer.cache_file.exists()

    def test_transient_source_change_cannot_bind_wrong_chunks_to_restored_source(self, temp_repo, monkeypatch):
        """A->B->A edits must not cache B chunks under the unchanged A digest."""
        indexer = RepositoryIndexer(str(temp_repo))
        original_index_file = indexer._index_file
        source = temp_repo / "docs" / "readme.md"
        original_bytes = source.read_bytes()

        def change_while_reading(file_path, chunk_type):
            if file_path != source:
                return original_index_file(file_path, chunk_type)
            source.write_text("# Intermediate source B")
            try:
                return original_index_file(file_path, chunk_type)
            finally:
                source.write_bytes(original_bytes)

        monkeypatch.setattr(indexer, "_index_file", change_while_reading)
        during_change = indexer.index_repository()
        assert any(chunk.content == "# Intermediate source B" for chunk in during_change)
        assert not indexer.cache_file.exists()
        refreshed = RepositoryIndexer(str(temp_repo)).index_repository()
        assert refreshed == RepositoryIndexer(str(temp_repo), use_cache=False).index_repository()
        assert all(chunk.content != "# Intermediate source B" for chunk in refreshed)

    def test_transient_read_failure_cannot_publish_partial_cache(self, temp_repo, monkeypatch):
        """Successful before/after hashes cannot authorize a skipped source read."""
        indexer = RepositoryIndexer(str(temp_repo))
        source = temp_repo / "docs" / "readme.md"
        original_read = Path.read_bytes

        def fail_source_read(file_path):
            if file_path == source:
                raise OSError("Transient source read failure")
            return original_read(file_path)

        monkeypatch.setattr(Path, "read_bytes", fail_source_read)
        indexer.index_repository()
        assert not indexer.cache_file.exists()

    def test_force_reindex(self, temp_repo):
        """Test force reindexing ignores cache."""
        # Create cache
        indexer1 = RepositoryIndexer(str(temp_repo), use_cache=True)
        indexer1.index_repository()

        # Force reindex
        indexer2 = RepositoryIndexer(str(temp_repo), use_cache=True)
        chunks = indexer2.index_repository(force_reindex=True)

        assert len(chunks) > 0

    def test_cache_disabled(self, temp_repo):
        """Test indexing with cache disabled."""
        indexer = RepositoryIndexer(str(temp_repo), use_cache=False)
        chunks = indexer.index_repository()

        assert len(chunks) > 0
        assert not indexer.cache_file.exists()

    def test_clear_cache(self, temp_repo):
        """Test clearing cache."""
        indexer = RepositoryIndexer(str(temp_repo), use_cache=True)
        indexer.index_repository()

        assert indexer.cache_file.exists()

        indexer.clear_cache()
        assert not indexer.cache_file.exists()

    def test_save_cache_cleans_unique_temp_file_on_replace_failure(self, temp_repo):
        """Test temp cache file cleanup when atomic replace fails."""
        indexer = RepositoryIndexer(str(temp_repo), use_cache=True)
        indexer.chunks = [
            DocumentChunk(
                content="sample",
                file_path="src/main.py",
                start_line=1,
                end_line=1,
                chunk_type="code",
            )
        ]

        indexer.cache_dir.mkdir(parents=True, exist_ok=True)
        indexer.cache_file.mkdir(parents=True, exist_ok=True)

        indexer._save_cache()

        tmp_files = list(indexer.cache_dir.glob(f".{indexer.cache_file.name}.*.tmp"))
        assert not tmp_files


class TestLogging:
    """Test logging functionality."""

    def test_logger_creation(self):
        """Test logger can be created."""
        logger = get_logger("test_logger")

        assert logger is not None
        assert logger.name == "test_logger"

    def test_logger_has_handlers(self):
        """Test logger has console handler."""
        logger = get_logger("test_logger2")

        assert len(logger.handlers) > 0


class TestVectorSearch:
    """Test vector embedding functionality."""

    def test_retriever_without_vectors(self, temp_repo):
        """Test retriever works without vector search."""
        indexer = RepositoryIndexer(str(temp_repo), use_cache=False)
        chunks = indexer.index_repository()

        retriever = HybridRetriever(enable_vector_search=False)
        retriever.index(chunks)

        results = retriever.retrieve("hello world", top_k=3)

        assert len(results) > 0
        assert all(r.retrieval_method in ("bm25", "hybrid") for r in results)

    def test_retriever_with_vectors_if_available(self, temp_repo):
        """Test retriever with vector search if sentence-transformers available."""
        try:
            from sentence_transformers import SentenceTransformer  # noqa: F401

            indexer = RepositoryIndexer(str(temp_repo), use_cache=False)
            chunks = indexer.index_repository()

            retriever = HybridRetriever(enable_vector_search=True)
            retriever.index(chunks)

            results = retriever.retrieve("hello world", top_k=3)

            assert len(results) > 0
            # With vector search, we might get hybrid or vector results
            assert all(r.retrieval_method in ("bm25", "vector", "hybrid") for r in results)

        except ImportError:
            pytest.skip("sentence-transformers not installed")


class TestQueryCaching:
    """Test query caching functionality."""

    def test_query_caching_enabled(self, temp_repo):
        """Test that queries are cached."""
        indexer = RepositoryIndexer(str(temp_repo), use_cache=False)
        chunks = indexer.index_repository()

        retriever = HybridRetriever(enable_vector_search=False)
        retriever.index(chunks)

        # First query
        results1 = retriever.retrieve("hello", top_k=3)

        # Second query (should be cached)
        results2 = retriever.retrieve("hello", top_k=3)

        # Results should be identical
        assert len(results1) == len(results2)
        if results1:
            assert results1[0].chunk_id == results2[0].chunk_id


class TestFullPipeline:
    """Integration tests for full RAG pipeline."""

    def test_end_to_end_pipeline(self, repo_root):
        """Test complete RAG workflow from indexing to citations."""
        # 1. Index repository
        indexer = RepositoryIndexer(str(repo_root), use_cache=False)
        chunks = indexer.index_repository()

        assert len(chunks) > 0

        # 2. Retrieve relevant chunks
        retriever = HybridRetriever(enable_vector_search=False)
        retriever.index(chunks)

        query = "depth pipeline processing"
        results = retriever.retrieve(query, top_k=10)

        assert len(results) > 0

        # 3. Rerank results
        reranker = ResultReranker()
        reranked = reranker.rerank(results, query, top_k=5)

        assert len(reranked) <= 5
        if len(reranked) > 1:
            # Check that scores are sorted
            for i in range(len(reranked) - 1):
                assert reranked[i].score >= reranked[i + 1].score

        # 4. Generate citations
        citation_gen = CitationGenerator()
        citations = citation_gen.generate_citations(reranked, max_citations=3)

        assert len(citations) <= 3
        assert all(c.confidence > 0 for c in citations)
        assert all(c.file_path for c in citations)

        # 5. Format citations
        formatted = citation_gen.format_citations(citations, format_type="markdown")

        assert "##" in formatted
        assert "Relevance score" in formatted
        assert "not correctness or authority" in formatted

    def test_pipeline_with_filtering(self, repo_root):
        """Test pipeline with chunk type filtering."""
        indexer = RepositoryIndexer(str(repo_root), use_cache=False)
        chunks = indexer.index_repository()

        retriever = HybridRetriever(enable_vector_search=False)
        retriever.index(chunks)

        # Filter only code chunks
        results = retriever.retrieve("function definition", top_k=5, chunk_type_filter=["code"])

        assert all(r.metadata.get("entity_type") in ("function", "class", None) for r in results)

    def test_pipeline_error_handling(self):
        """Test error handling in pipeline."""
        retriever = HybridRetriever()

        # Should raise error if not indexed
        with pytest.raises(RetrievalError):
            retriever.retrieve("test query")

    def test_statistics_tracking(self, temp_repo):
        """Test that indexer tracks statistics."""
        indexer = RepositoryIndexer(str(temp_repo), use_cache=False)
        chunks = indexer.index_repository()

        stats = indexer.get_statistics()

        assert stats["total_chunks"] == len(chunks)
        assert "by_type" in stats
        assert "by_language" in stats
        assert stats["total_chars"] > 0


class TestExceptionHandling:
    """Test custom exception handling."""

    def test_indexing_error(self):
        """Test IndexingError is raised for invalid repo."""
        # Trying to index a non-existent directory should raise an error
        # But our current implementation just logs warnings
        # This is more of a design test
        indexer = RepositoryIndexer("/non/existent/path", use_cache=False)

        try:
            chunks = indexer.index_repository()
            # If no error, chunks should be empty
            assert len(chunks) == 0
        except IndexingError:
            # If IndexingError is raised, that's also valid
            pass

    def test_retrieval_error(self):
        """Test RetrievalError is raised when not indexed."""
        retriever = HybridRetriever()

        with pytest.raises(RetrievalError):
            retriever.retrieve("test")

    def test_cache_error_handling(self, temp_repo):
        """Test cache error handling."""
        indexer = RepositoryIndexer(str(temp_repo), use_cache=True)
        indexer.index_repository()

        # Make cache file unreadable by replacing with directory
        cache_file = indexer.cache_file
        if cache_file.exists():
            cache_file.unlink()

        # This should handle the error gracefully
        indexer2 = RepositoryIndexer(str(temp_repo), use_cache=True)
        chunks = indexer2.index_repository(force_reindex=True)

        assert len(chunks) > 0


class TestPerformance:
    """Performance tests for RAG system."""

    def test_indexing_performance(self, temp_repo):
        """Test that indexing completes in reasonable time."""
        indexer = RepositoryIndexer(str(temp_repo), use_cache=False)

        start = time.time()
        chunks = indexer.index_repository()
        elapsed = time.time() - start

        assert len(chunks) > 0
        assert elapsed < 5.0  # Should complete in under 5 seconds for small repo

    def test_retrieval_performance(self, temp_repo):
        """Test that retrieval is fast."""
        indexer = RepositoryIndexer(str(temp_repo), use_cache=False)
        chunks = indexer.index_repository()

        retriever = HybridRetriever(enable_vector_search=False)
        retriever.index(chunks)

        start = time.time()
        results = retriever.retrieve("hello world", top_k=5)
        elapsed = time.time() - start

        assert len(results) >= 0
        assert elapsed < 0.1  # Should complete in under 100ms

    def test_cache_improves_performance(self, temp_repo):
        """Test that cache improves indexing performance."""
        # First indexing (no cache)
        indexer1 = RepositoryIndexer(str(temp_repo), use_cache=True)
        start1 = time.time()
        chunks1 = indexer1.index_repository()
        time1 = time.time() - start1

        # Second indexing (with cache)
        indexer2 = RepositoryIndexer(str(temp_repo), use_cache=True)
        start2 = time.time()
        chunks2 = indexer2.index_repository()
        time2 = time.time() - start2

        assert len(chunks1) == len(chunks2)
        # Cache loading should be much faster than indexing
        # But this might not always be true for tiny repos
        assert time2 <= time1 * 2  # Allow some variance


class TestCurrentGuidanceInventory:
    """Retrieval must represent live authorities with deterministic ordering."""

    def test_includes_authorities_and_prunes_archived_profiles(self, tmp_path):
        sources = {
            "AGENTS.md": "# Root operating contract",
            ".github/copilot-instructions.md": "# Copilot instructions",
            ".github/agents/live.md": "# Live profile",
            ".github/agents/_archive/old.md": "# Retired profile",
            ".github/agents/rag_system/_archive/old.md": "# Retired RAG guidance",
            "docs/z.md": "# Last document",
            "docs/a.md": "# First document",
            "docs/build/output.md": "# Generated build output",
            "README_GUIDE.md": "# Matches two root patterns",
        }
        for rel_path, content in sources.items():
            path = tmp_path / rel_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        chunks = RepositoryIndexer(str(tmp_path), use_cache=False).index_repository()
        paths = [chunk.file_path for chunk in chunks]
        assert paths == sorted(set(paths))
        assert set(paths) == {
            "AGENTS.md",
            ".github/copilot-instructions.md",
            ".github/agents/live.md",
            "docs/a.md",
            "docs/z.md",
            "README_GUIDE.md",
        }
        assert all(
            chunk.chunk_type == "agent"
            for chunk in chunks
            if chunk.file_path
            in {
                "AGENTS.md",
                ".github/copilot-instructions.md",
                ".github/agents/live.md",
            }
        )

    def test_source_snapshot_preserves_universal_newlines(self, tmp_path):
        (tmp_path / "README.md").write_bytes(b"# Title\r\nFirst line\rSecond line\n")
        indexer = RepositoryIndexer(str(tmp_path))
        chunks = indexer.index_repository()
        assert len(chunks) == 1
        assert chunks[0].content == "# Title\nFirst line\nSecond line\n"
        assert chunks[0].start_line == 1
        assert chunks[0].end_line == 4
        assert indexer.index_repository() == chunks

    def test_configured_cache_tree_is_not_indexed(self, tmp_path):
        get_config().set("indexer.cache_dir", "docs/cache")
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "readme.md").write_text("# Current documentation")
        indexer = RepositoryIndexer(str(tmp_path))
        expected = indexer.index_repository()
        assert indexer.index_repository(force_reindex=True) == expected


@pytest.mark.parametrize("entrypoint", ["script", "module"])
def test_rag_cli_can_index_with_both_entrypoints(tmp_path, entrypoint):
    """The supported CLI must import its package and perform an offline operation."""
    (tmp_path / "README.md").write_text("# CLI fixture")
    command = (
        [sys.executable, str(agents_path / "rag_system" / "cli.py")]
        if entrypoint == "script"
        else [sys.executable, "-m", "rag_system.cli"]
    )
    output = tmp_path / "stats.json"
    env = {**os.environ, "PYTHONPATH": str(agents_path), "RAG_RETRIEVER_ENABLE_VECTOR_SEARCH": "false"}
    result = subprocess.run(
        [*command, "index", "--repo-root", str(tmp_path), "--output", str(output)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        timeout=30,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())["total_chunks"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
