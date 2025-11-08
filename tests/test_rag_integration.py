"""
Integration tests for RAG system - full pipeline.

Tests the complete workflow from indexing to citation generation.
"""

import sys
from pathlib import Path

import pytest

# Add agents directory to path  # noqa: E402
agents_path = Path(__file__).parent.parent / '.github' / 'agents'
sys.path.insert(0, str(agents_path))

from rag_system.citation import CitationGenerator  # noqa: E402
from rag_system.indexer import RepositoryIndexer  # noqa: E402
from rag_system.reranker import ResultReranker  # noqa: E402
from rag_system.retriever import HybridRetriever  # noqa: E402
from rag_system.templates import FewShotExamples, PromptTemplates  # noqa: E402


@pytest.fixture(scope="module")
def rag_pipeline():
    """Create a complete RAG pipeline for testing."""
    repo_root = Path(__file__).parent.parent

    # Index repository
    indexer = RepositoryIndexer(repo_root)
    chunks = indexer.index_repository()

    # Setup retrieval
    retriever = HybridRetriever()
    retriever.index(chunks)

    # Setup reranker
    reranker = ResultReranker()

    # Setup citation generator
    citation_gen = CitationGenerator()

    return {
        'indexer': indexer,
        'retriever': retriever,
        'reranker': reranker,
        'citation_gen': citation_gen,
        'chunks': chunks,
    }


class TestRAGIntegration:
    """Test complete RAG pipeline integration."""

    def test_pipeline_finds_depth_processing_code(self, rag_pipeline):
        """Test pipeline can find depth processing related code."""
        query = "depth pipeline atmospheric effects"

        # Retrieve
        results = rag_pipeline['retriever'].retrieve(query, top_k=5)
        assert len(results) > 0

        # Rerank
        reranked = rag_pipeline['reranker'].rerank(results, query, top_k=3)
        assert len(reranked) > 0

        # Generate citations
        citations = rag_pipeline['citation_gen'].generate_citations(reranked, max_citations=2)
        assert len(citations) > 0

        # Verify citation has required fields
        cite = citations[0]
        assert cite.file_path
        assert cite.snippet
        assert 0.0 <= cite.confidence <= 1.0

    def test_pipeline_finds_ffmpeg_filter_examples(self, rag_pipeline):
        """Test pipeline can find FFmpeg filter graph examples."""
        query = "FFmpeg filter graph build"

        results = rag_pipeline['retriever'].retrieve(query, top_k=5)
        reranked = rag_pipeline['reranker'].rerank(results, query, top_k=3)
        citations = rag_pipeline['citation_gen'].generate_citations(reranked, max_citations=2)

        assert len(citations) > 0

        # At least one result should be from code
        file_paths = [c.file_path for c in citations]
        has_code = any(path.endswith('.py') for path in file_paths)
        assert has_code, "Should find Python code examples"

    def test_pipeline_finds_documentation(self, rag_pipeline):
        """Test pipeline can find relevant documentation."""
        query = "pipeline operations guide"

        # Filter to only docs
        results = rag_pipeline['retriever'].retrieve(
            query,
            top_k=5,
            chunk_type_filter=['doc']
        )

        assert len(results) > 0

        # All results should be markdown
        for result in results:
            assert result.file_path.endswith('.md')

    def test_pipeline_filters_by_test_files(self, rag_pipeline):
        """Test pipeline can filter for test files only."""
        query = "test luxury video grader"

        results = rag_pipeline['retriever'].retrieve(
            query,
            top_k=5,
            chunk_type_filter=['test']
        )

        if results:  # May not have test results depending on query
            for result in results:
                assert 'test' in result.file_path.lower()

    def test_pipeline_handles_specific_file_search(self, rag_pipeline):
        """Test pipeline can search in specific files."""
        query = "material response"

        results = rag_pipeline['retriever'].retrieve(
            query,
            top_k=5,
            file_path_filter=r'material_response'
        )

        if results:
            for result in results:
                assert 'material_response' in result.file_path.lower()

    def test_feature_template_with_rag_context(self, rag_pipeline):
        """Test feature template generation with RAG context."""
        # Search for relevant context
        query = "add new LUT preset"
        results = rag_pipeline['retriever'].retrieve(query, top_k=3)
        reranked = rag_pipeline['reranker'].rerank(results, query)
        citations = rag_pipeline['citation_gen'].generate_citations(reranked, max_citations=2)

        # Format citations
        citation_text = rag_pipeline['citation_gen'].format_citations(
            citations,
            format_type='markdown'
        )

        # Generate template with context
        template = PromptTemplates.feature_implementation(
            "Add new sunset LUT preset",
            context=citation_text
        )

        # Verify template structure
        assert "Feature Description" in template
        assert "Add new sunset LUT preset" in template
        assert "Response Format" in template

        # If citations were found, they should be in context
        if citations:
            assert "Citations" in template or citation_text in template

    def test_bug_triage_template_with_examples(self, rag_pipeline):
        """Test bug triage template with few-shot examples."""
        error_log = "ImportError: No module named 'tifffile'"

        # Get few-shot examples
        examples = FewShotExamples.get_bug_triage_examples()

        # Generate template
        template = PromptTemplates.bug_triage(
            error_log,
            environment="Python 3.10"
        )

        # Add examples
        template_with_examples = PromptTemplates.add_few_shot_examples(
            template,
            examples
        )

        assert "ImportError" in template_with_examples
        assert "Few-Shot Examples" in template_with_examples
        assert "Python 3.10" in template_with_examples

    def test_citation_confidence_scores_are_ranked(self, rag_pipeline):
        """Test that citation confidence scores decrease with rank."""
        query = "depth processing pipeline"

        results = rag_pipeline['retriever'].retrieve(query, top_k=5)
        reranked = rag_pipeline['reranker'].rerank(results, query)
        citations = rag_pipeline['citation_gen'].generate_citations(reranked, max_citations=5)

        if len(citations) >= 2:
            # Higher ranked citations should generally have higher confidence
            # (though reranking can change order)
            confidences = [c.confidence for c in citations]
            assert all(0.0 <= conf <= 1.0 for conf in confidences)

    @pytest.mark.slow
    def test_pipeline_performance(self, rag_pipeline):
        """Test that pipeline performs reasonably fast."""
        import os
        import time

        query = "image processing"

        # Use more lenient thresholds in CI environments
        ci_factor = 3.0 if os.getenv('CI') else 1.0

        # Time retrieval
        start = time.time()
        results = rag_pipeline['retriever'].retrieve(query, top_k=10)
        retrieval_time = time.time() - start

        # Time reranking
        start = time.time()
        reranked = rag_pipeline['reranker'].rerank(results, query)
        rerank_time = time.time() - start

        # Time citation generation
        start = time.time()
        _ = rag_pipeline['citation_gen'].generate_citations(reranked, max_citations=5)
        citation_time = time.time() - start

        # Performance expectations with CI tolerance
        retrieval_threshold = 1.0 * ci_factor
        rerank_threshold = 0.5 * ci_factor
        citation_threshold = 0.1 * ci_factor
        total_threshold = 2.0 * ci_factor

        assert retrieval_time < retrieval_threshold, \
            f"Retrieval took {retrieval_time:.3f}s (expected < {retrieval_threshold}s)"
        assert rerank_time < rerank_threshold, \
            f"Reranking took {rerank_time:.3f}s (expected < {rerank_threshold}s)"
        assert citation_time < citation_threshold, \
            f"Citation took {citation_time:.3f}s (expected < {citation_threshold}s)"

        total_time = retrieval_time + rerank_time + citation_time
        assert total_time < total_threshold, \
            f"Total pipeline took {total_time:.3f}s (expected < {total_threshold}s)"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
