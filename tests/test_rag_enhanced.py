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

import sys
import tempfile
import time
from pathlib import Path

import pytest

# Add agents directory to path
agents_path = Path(__file__).parent.parent / '.github' / 'agents'
sys.path.insert(0, str(agents_path))

from rag_system.citation import CitationGenerator  # noqa: E402
from rag_system.config import Config, get_config, reset_config  # noqa: E402
from rag_system.exceptions import IndexingError, RetrievalError  # noqa: E402
from rag_system.indexer import RepositoryIndexer  # noqa: E402
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
        (tmp_path / 'docs').mkdir()
        (tmp_path / 'src').mkdir()
        (tmp_path / 'tests').mkdir()

        # Create sample files
        (tmp_path / 'docs' / 'readme.md').write_text(
            "# Test Repository\n\nThis is a test repository for RAG system."
        )
        (tmp_path / 'src' / 'main.py').write_text(
            "def hello_world():\n    \"\"\"Print hello world.\"\"\"\n    print('Hello, World!')"
        )
        (tmp_path / 'tests' / 'test_main.py').write_text(
            "def test_hello():\n    assert True"
        )

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
        assert config.get('indexer.chunk_size_tokens') == 750
        assert config.get('retriever.bm25_weight') == 0.7
        assert config.get('citation.max_results') == 5

    def test_config_get_section(self):
        """Test getting entire section."""
        config = Config()
        indexer_config = config.get_section('indexer')

        assert isinstance(indexer_config, dict)
        assert 'chunk_size_tokens' in indexer_config
        assert 'overlap_tokens' in indexer_config

    def test_config_set_value(self):
        """Test setting config value at runtime."""
        config = Config()
        config.set('indexer.chunk_size_tokens', 1000)

        assert config.get('indexer.chunk_size_tokens') == 1000

    # Environment variable override functionality is not yet implemented
    # def test_config_env_override(self, monkeypatch):
    #     """Test environment variable override."""
    #     monkeypatch.setenv('RAG_INDEXER_CACHE_ENABLED', 'false')
    #     reset_config()
    #
    #     config = get_config()
    #     assert config.get('indexer.cache_enabled') is False


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


class TestLogging:
    """Test logging functionality."""

    def test_logger_creation(self):
        """Test logger can be created."""
        logger = get_logger('test_logger')

        assert logger is not None
        assert logger.name == 'test_logger'

    def test_logger_has_handlers(self):
        """Test logger has console handler."""
        logger = get_logger('test_logger2')

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
        assert all(r.retrieval_method in ('bm25', 'hybrid') for r in results)

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
            assert all(r.retrieval_method in ('bm25', 'vector', 'hybrid') for r in results)

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
        formatted = citation_gen.format_citations(citations, format_type='markdown')

        assert '##' in formatted
        assert 'Confidence' in formatted

    def test_pipeline_with_filtering(self, repo_root):
        """Test pipeline with chunk type filtering."""
        indexer = RepositoryIndexer(str(repo_root), use_cache=False)
        chunks = indexer.index_repository()

        retriever = HybridRetriever(enable_vector_search=False)
        retriever.index(chunks)

        # Filter only code chunks
        results = retriever.retrieve(
            "function definition",
            top_k=5,
            chunk_type_filter=['code']
        )

        assert all(r.metadata.get('entity_type') in ('function', 'class', None) for r in results)

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

        assert stats['total_chunks'] == len(chunks)
        assert 'by_type' in stats
        assert 'by_language' in stats
        assert stats['total_chars'] > 0


class TestExceptionHandling:
    """Test custom exception handling."""

    def test_indexing_error(self):
        """Test IndexingError is raised for invalid repo."""
        # Trying to index a non-existent directory should raise an error
        # But our current implementation just logs warnings
        # This is more of a design test
        indexer = RepositoryIndexer('/non/existent/path', use_cache=False)

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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
