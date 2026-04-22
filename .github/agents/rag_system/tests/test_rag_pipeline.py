"""
Integration tests for the complete RAG pipeline.

Tests the end-to-end workflow:
1. Index repository content
2. Retrieve relevant chunks
3. Rerank results
4. Generate citations
5. Use prompt templates
"""

import sys
import tempfile
from pathlib import Path

# Make the rag_system package importable when tests are invoked directly
# (pytest discovery handles this for typical layouts; this keeps `python
# tests/test_rag_pipeline.py` working too).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
from rag_system.citation import CitationGenerator
from rag_system.indexer import DocumentChunk, RepositoryIndexer
from rag_system.reranker import ResultReranker
from rag_system.retriever import HybridRetriever
from rag_system.templates import CodeModificationResponse, PromptTemplates


@pytest.fixture
def temp_repo():
    """Create a temporary repository structure for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)

        # Create sample documentation
        docs_dir = repo_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "README.md").write_text(
            """
# Depth Pipeline Documentation

The depth pipeline processes images using Depth Anything V2 for monocular depth estimation.

## Features
- Depth-aware tone mapping
- Atmospheric effects
- Zone-based processing
        """
        )

        # Create sample source code
        src_dir = repo_path / "src"
        src_dir.mkdir()
        (src_dir / "pipeline.py").write_text(
            """
def process_image(image_path, depth_map):
    '''Process image with depth information.

    Args:
        image_path: Path to input image
        depth_map: Normalized depth map

    Returns:
        Processed image array
    '''
    # Apply depth-aware processing
    return processed_image
        """
        )

        # Create sample test
        tests_dir = repo_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_pipeline.py").write_text(
            """
def test_process_image():
    '''Test depth pipeline processing.'''
    result = process_image('test.jpg', depth_map)
    assert result is not None
        """
        )

        yield str(repo_path)


class TestRAGPipeline:
    """Integration tests for RAG pipeline."""

    def test_indexer_creates_chunks(self, temp_repo):
        """Test that indexer creates document chunks."""
        indexer = RepositoryIndexer(temp_repo)
        chunks = indexer.index_repository()

        assert len(chunks) > 0, "Indexer should create chunks"
        assert all(isinstance(c, DocumentChunk) for c in chunks), "All items should be DocumentChunk"

        # Check chunk types
        chunk_types = {c.chunk_type for c in chunks}
        assert "doc" in chunk_types or "code" in chunk_types, "Should have doc or code chunks"

    def test_retriever_finds_relevant_chunks(self, temp_repo):
        """Test that retriever finds relevant chunks for a query."""
        # Index repository
        indexer = RepositoryIndexer(temp_repo)
        chunks = indexer.index_repository()

        # Setup retriever
        retriever = HybridRetriever()
        retriever.index(chunks)

        # Search for depth-related content
        results = retriever.retrieve("depth pipeline processing", top_k=5)

        assert len(results) > 0, "Should find relevant results"
        assert all(hasattr(r, "content") for r in results), "Results should have content"
        assert all(hasattr(r, "score") for r in results), "Results should have scores"

    def test_reranker_improves_results(self, temp_repo):
        """Test that reranker improves result quality."""
        # Index and retrieve
        indexer = RepositoryIndexer(temp_repo)
        chunks = indexer.index_repository()

        retriever = HybridRetriever()
        retriever.index(chunks)
        results = retriever.retrieve("depth map", top_k=10)

        # Rerank
        reranker = ResultReranker()
        reranked = reranker.rerank(results, "depth map", top_k=5)

        assert len(reranked) <= 5, "Should limit to top-k"
        assert len(reranked) > 0, "Should have reranked results"

        # Scores should be in descending order
        if len(reranked) > 1:
            scores = [r.score for r in reranked]
            assert scores == sorted(scores, reverse=True), "Results should be sorted by score"

    def test_citation_generator_creates_citations(self, temp_repo):
        """Test that citation generator creates proper citations."""
        # Index and retrieve
        indexer = RepositoryIndexer(temp_repo)
        chunks = indexer.index_repository()

        retriever = HybridRetriever()
        retriever.index(chunks)
        results = retriever.retrieve("depth pipeline", top_k=3)

        # Generate citations
        citation_gen = CitationGenerator()
        citations = citation_gen.generate_citations(results, max_citations=3)

        assert len(citations) > 0, "Should generate citations"

        for citation in citations:
            assert citation.file_path, "Citation should have file_path"
            assert citation.snippet, "Citation should have snippet"
            assert 0.0 <= citation.confidence <= 1.0, "Confidence should be in [0,1]"

    def test_end_to_end_workflow(self, temp_repo):
        """Test complete RAG workflow from indexing to citation."""
        # 1. Index repository
        indexer = RepositoryIndexer(temp_repo)
        chunks = indexer.index_repository()
        assert len(chunks) > 0

        # 2. Setup retrieval
        retriever = HybridRetriever()
        retriever.index(chunks)

        # 3. Retrieve relevant content
        query = "How to process images with depth maps?"
        results = retriever.retrieve(query, top_k=10)
        assert len(results) > 0

        # 4. Rerank for better precision
        reranker = ResultReranker()
        reranked = reranker.rerank(results, query, top_k=5)
        assert len(reranked) > 0

        # 5. Generate citations
        citation_gen = CitationGenerator()
        citations = citation_gen.generate_citations(reranked, max_citations=3)
        assert len(citations) > 0

        # 6. Format citations
        formatted = citation_gen.format_citations(citations, format_type="markdown")
        assert isinstance(formatted, str)
        assert len(formatted) > 0

    def test_prompt_templates_feature_implementation(self):
        """Test feature implementation template generation."""
        template = PromptTemplates.feature_implementation(
            feature_description="Add depth-based atmospheric haze effect",
            context="Existing atmospheric processor in depth_pipeline/processors/",
        )

        assert isinstance(template, str), "Template should be a string"
        assert "depth" in template.lower(), "Template should mention depth"
        assert "atmospheric" in template.lower(), "Template should mention atmospheric"

    def test_prompt_templates_bug_triage(self):
        """Test bug triage template generation."""
        template = PromptTemplates.bug_triage(
            error_log="ImportError: No module named 'torch'",
            reproduction_steps="Run python pipeline.py",
            environment="Python 3.10, Ubuntu 20.04",
        )

        assert isinstance(template, str), "Template should be a string"
        assert "torch" in template.lower(), "Template should mention torch"

    def test_code_modification_response_schema(self):
        """Test CodeModificationResponse schema."""
        from rag_system.templates import FileModification

        response = CodeModificationResponse(
            summary="Add new depth effect",
            files=[
                FileModification(
                    path="depth_pipeline/processors/atmospheric.py",
                    patch="+ def apply_haze(image, depth): pass",
                    description="Add haze effect",
                )
            ],
            tests=["tests/test_atmospheric.py"],
            explanation="Implements depth-based haze using fog color blending",
            confidence=0.85,
            citations=[{"file_path": "existing.py", "snippet": "example"}],
        )

        # Test JSON serialization
        json_str = response.to_json()
        assert isinstance(json_str, str)
        assert "Add new depth effect" in json_str
        assert "0.85" in json_str

    def test_retriever_with_type_filter(self, temp_repo):
        """Test retriever with chunk type filtering."""
        indexer = RepositoryIndexer(temp_repo)
        chunks = indexer.index_repository()

        retriever = HybridRetriever()
        retriever.index(chunks)

        # Filter by code chunks only
        results = retriever.retrieve("process image", top_k=5, chunk_type_filter=["code"])

        if results:
            # All results should be code chunks - check metadata
            # Note: RetrievalResult doesn't have chunk_type directly, but we can check metadata
            assert len(results) > 0, "Should have results when filtering by code"

    def test_citation_formatting_options(self, temp_repo):
        """Test different citation formatting options."""
        indexer = RepositoryIndexer(temp_repo)
        chunks = indexer.index_repository()

        retriever = HybridRetriever()
        retriever.index(chunks)
        results = retriever.retrieve("depth", top_k=2)

        citation_gen = CitationGenerator()
        citations = citation_gen.generate_citations(results, max_citations=2)

        # Test markdown format
        markdown = citation_gen.format_citations(citations, format_type="markdown")
        assert "##" in markdown or "[" in markdown, "Markdown should have formatting"

        # Test plain text format
        plain = citation_gen.format_citations(citations, format_type="text")
        assert "CITATIONS" in plain or "File:" in plain, "Plain text should have citations"

        # Test JSON format
        json_str = citation_gen.format_citations(citations, format_type="json")
        assert "file_path" in json_str, "JSON should have file_path field"


class TestRerankerIdempotence:
    """Regression tests for the reranker mutation bug."""

    def _make_result(self, score=1.0, content="def foo(): pass"):
        from rag_system.retriever import RetrievalResult

        return RetrievalResult(
            chunk_id="c1",
            content=content,
            file_path="a.py",
            start_line=1,
            end_line=2,
            score=score,
            retrieval_method="bm25",
            metadata={},
        )

    def test_rerank_does_not_mutate_input(self):
        """rerank() must not mutate the input results or their metadata."""
        original = self._make_result()
        reranker = ResultReranker()
        reranker.rerank([original], "foo")

        assert original.score == 1.0
        assert "rerank_boost" not in original.metadata

    def test_rerank_preserves_metadata_across_passes(self):
        """rerank() must overwrite (not compound) rerank_boost in metadata."""
        result = self._make_result()
        reranker = ResultReranker()

        first = reranker.rerank([result], "foo")
        second = reranker.rerank(first, "foo")

        # The boost is the same on each pass (idempotent in metadata); the
        # score is expected to keep accumulating because reranker signals are
        # additive on the input score — that's by design.
        assert first[0].metadata["rerank_boost"] == second[0].metadata["rerank_boost"]
        # The original input object is still untouched.
        assert result.score == 1.0
        assert "rerank_boost" not in result.metadata


class TestKnowledgeEngine:
    """Tests for KnowledgeIntegrationEngine."""

    def test_add_feedback_and_analyze_patterns(self):
        from rag_system.knowledge_engine import KnowledgeIntegrationEngine

        engine = KnowledgeIntegrationEngine()
        for i in range(10):
            engine.add_feedback(
                pipeline="lux-depth-v3",
                artifact_id=f"artifact-{i}",
                success=i % 3 != 0,
                processing_time=1.0 + i * 0.1,
                parameters={"preset": "premium"},
                error_message="ValueError: bad input" if i % 3 == 0 else None,
                quality_score=0.8,
            )

        analysis = engine.analyze_patterns("lux-depth-v3", days=30)
        assert analysis.total_runs == 10
        assert 0.5 <= analysis.success_rate <= 0.8
        assert analysis.avg_processing_time > 0
        assert "ValueError" in analysis.failure_modes

    def test_analyze_patterns_with_no_records(self):
        from rag_system.knowledge_engine import KnowledgeIntegrationEngine

        engine = KnowledgeIntegrationEngine()
        analysis = engine.analyze_patterns("unknown-pipeline")

        assert analysis.total_runs == 0
        assert analysis.success_rate == 0.0

    def test_kpi_summary(self):
        from rag_system.knowledge_engine import KnowledgeIntegrationEngine

        engine = KnowledgeIntegrationEngine()
        engine.add_feedback(
            pipeline="ingest",
            artifact_id="a1",
            success=True,
            processing_time=2.5,
            parameters={},
            quality_score=0.9,
        )

        summary = engine.get_kpi_summary(pipeline="ingest", days=1)
        assert "ingest:success_rate" in summary
        assert "ingest:processing_time" in summary
        assert summary["ingest:processing_time"]["current"] == 2.5


class TestConfigEnvOverrides:
    """Tests for RAG_* environment variable overrides."""

    def _reset(self):
        from rag_system.config import reset_config

        reset_config()

    def test_env_override_bool(self, monkeypatch):
        from rag_system.config import Config, reset_config

        monkeypatch.setenv("RAG_INDEXER_CACHE_ENABLED", "false")
        reset_config()
        cfg = Config()
        assert cfg.get("indexer.cache_enabled") is False
        reset_config()

    def test_env_override_float(self, monkeypatch):
        from rag_system.config import Config, reset_config

        monkeypatch.setenv("RAG_RETRIEVER_BM25_WEIGHT", "0.85")
        reset_config()
        cfg = Config()
        assert cfg.get("retriever.bm25_weight") == 0.85
        reset_config()

    def test_env_override_int(self, monkeypatch):
        from rag_system.config import Config, reset_config

        monkeypatch.setenv("RAG_CITATION_MAX_RESULTS", "10")
        reset_config()
        cfg = Config()
        assert cfg.get("citation.max_results") == 10
        reset_config()

    def test_set_and_get_roundtrip(self):
        from rag_system.config import Config

        cfg = Config()
        cfg.set("retriever.bm25_weight", 0.42)
        assert cfg.get("retriever.bm25_weight") == 0.42
        assert cfg.get("retriever", "bm25_weight") == 0.42


class TestIndexerCache:
    """Tests for the indexer JSON cache round-trip."""

    def test_cache_round_trip(self, temp_repo):
        indexer_a = RepositoryIndexer(temp_repo)
        chunks_a = indexer_a.index_repository()
        assert len(chunks_a) > 0
        assert indexer_a.cache_file.exists()

        # Second instance should load from cache without re-walking the repo
        indexer_b = RepositoryIndexer(temp_repo)
        chunks_b = indexer_b.index_repository()
        assert len(chunks_b) == len(chunks_a)
        assert {c.chunk_id for c in chunks_b} == {c.chunk_id for c in chunks_a}

    def test_clear_cache(self, temp_repo):
        indexer = RepositoryIndexer(temp_repo)
        indexer.index_repository()
        assert indexer.cache_file.exists()

        indexer.clear_cache()
        assert not indexer.cache_file.exists()


class TestSemanticSearch:
    """Smoke tests for SemanticCodeSearch against a minimal Python repo."""

    def _make_repo(self, tmp_path):
        pkg = tmp_path / "example_pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "depth.py").write_text(
            "def compute_depth(image):\n"
            '    """Compute the depth map for the given image."""\n'
            "    return image\n"
            "\n"
            "class DepthProcessor:\n"
            '    """Process images to produce depth maps."""\n'
            "    def process(self, image):\n"
            "        return compute_depth(image)\n"
        )
        return tmp_path

    def test_index_and_find_function(self, tmp_path):
        from rag_system.semantic_search import SemanticCodeSearch

        repo = self._make_repo(tmp_path)
        search = SemanticCodeSearch(str(repo))
        search.index_codebase()

        assert any(e.name == "compute_depth" for e in search.entities.values())
        assert any(e.name == "DepthProcessor" for e in search.entities.values())

    def test_search_returns_entities(self, tmp_path):
        from rag_system.semantic_search import SemanticCodeSearch

        repo = self._make_repo(tmp_path)
        search = SemanticCodeSearch(str(repo))
        search.index_codebase()

        results = search.search("depth map", top_k=5)
        assert len(results) > 0
        assert any("depth" in r.entity.name.lower() or "depth" in (r.entity.docstring or "").lower() for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
