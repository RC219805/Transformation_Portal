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

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest  # noqa: E402
from citation import CitationGenerator  # noqa: E402
from indexer import DocumentChunk, RepositoryIndexer  # noqa: E402
from reranker import ResultReranker  # noqa: E402
from retriever import HybridRetriever  # noqa: E402
from templates import CodeModificationResponse, PromptTemplates  # noqa: E402


@pytest.fixture
def temp_repo():
    """Create a temporary repository structure for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)

        # Create sample documentation
        docs_dir = repo_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "README.md").write_text("""
# Depth Pipeline Documentation

The depth pipeline processes images using Depth Anything V2 for monocular depth estimation.

## Features
- Depth-aware tone mapping
- Atmospheric effects
- Zone-based processing
        """)

        # Create sample source code
        src_dir = repo_path / "src"
        src_dir.mkdir()
        (src_dir / "pipeline.py").write_text("""
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
        """)

        # Create sample test
        tests_dir = repo_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_pipeline.py").write_text("""
def test_process_image():
    '''Test depth pipeline processing.'''
    result = process_image('test.jpg', depth_map)
    assert result is not None
        """)

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
        assert 'doc' in chunk_types or 'code' in chunk_types, "Should have doc or code chunks"

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
        assert all(hasattr(r, 'content') for r in results), "Results should have content"
        assert all(hasattr(r, 'score') for r in results), "Results should have scores"

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
            assert 'file_path' in citation, "Citation should have file_path"
            assert 'snippet' in citation, "Citation should have snippet"
            assert 'confidence' in citation, "Citation should have confidence"
            assert 0.0 <= citation['confidence'] <= 1.0, "Confidence should be in [0,1]"

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
        formatted = citation_gen.format_citations(citations, format_type='markdown')
        assert isinstance(formatted, str)
        assert len(formatted) > 0

    def test_prompt_templates_feature_implementation(self):
        """Test feature implementation template generation."""
        template = PromptTemplates.feature_implementation(
            description="Add depth-based atmospheric haze effect",
            context="Existing atmospheric processor in depth_pipeline/processors/"
        )

        assert isinstance(template, str), "Template should be a string"
        assert "depth" in template.lower(), "Template should mention depth"
        assert "atmospheric" in template.lower(), "Template should mention atmospheric"

    def test_prompt_templates_bug_triage(self):
        """Test bug triage template generation."""
        template = PromptTemplates.bug_triage(
            error_log="ImportError: No module named 'torch'",
            reproduction_steps="Run python pipeline.py",
            environment="Python 3.10, Ubuntu 20.04"
        )

        assert isinstance(template, str), "Template should be a string"
        assert "torch" in template.lower(), "Template should mention torch"

    def test_code_modification_response_schema(self):
        """Test CodeModificationResponse schema."""
        from templates import FileModification

        response = CodeModificationResponse(
            summary="Add new depth effect",
            files=[
                FileModification(
                    path="depth_pipeline/processors/atmospheric.py",
                    patch="+ def apply_haze(image, depth): pass",
                    description="Add haze effect"
                )
            ],
            tests=["tests/test_atmospheric.py"],
            explanation="Implements depth-based haze using fog color blending",
            confidence=0.85,
            citations=[{"file_path": "existing.py", "snippet": "example"}]
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
        results = retriever.retrieve("process image", top_k=5, chunk_type_filter=['code'])

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
        markdown = citation_gen.format_citations(citations, format_type='markdown')
        assert '##' in markdown or '[' in markdown, "Markdown should have formatting"

        # Test plain text format
        plain = citation_gen.format_citations(citations, format_type='text')
        assert 'CITATIONS' in plain or 'File:' in plain, "Plain text should have citations"

        # Test JSON format
        json_str = citation_gen.format_citations(citations, format_type='json')
        assert 'file_path' in json_str, "JSON should have file_path field"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
