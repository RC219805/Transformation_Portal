"""
Tests for RAG system components.
"""

import sys
from pathlib import Path

import pytest

# Add agents directory to path for imports  # noqa: E402
agents_path = Path(__file__).parent.parent / '.github' / 'agents'
sys.path.insert(0, str(agents_path))

from rag_system.citation import CitationGenerator  # noqa: E402
from rag_system.indexer import DocumentChunk, RepositoryIndexer  # noqa: E402
from rag_system.reranker import ResultReranker  # noqa: E402
from rag_system.retriever import BM25Retriever, HybridRetriever  # noqa: E402
from rag_system.templates import (  # noqa: E402
    CodeModificationResponse,
    FewShotExamples,
    FileModification,
    PromptTemplates,
    validate_response_schema,
)


@pytest.fixture
def repo_root():
    """Get repository root path."""
    return Path(__file__).parent.parent


@pytest.fixture
def sample_chunks():
    """Create sample document chunks for testing."""
    return [
        DocumentChunk(
            content="def process_image(img):\n    \"\"\"Process an image\"\"\"\n    return img",
            file_path="src/processor.py",
            start_line=1,
            end_line=3,
            chunk_type='code',
            language='python',
            metadata={'function_name': 'process_image', 'entity_type': 'function'}
        ),
        DocumentChunk(
            content="# Depth Pipeline\n\nProcessing depth information for images.",
            file_path="docs/depth.md",
            start_line=1,
            end_line=3,
            chunk_type='doc',
            language='markdown',
            metadata={'title': 'Depth Pipeline', 'document_type': 'guide'}
        ),
        DocumentChunk(
            content="def test_processor():\n    assert process_image(img) is not None",
            file_path="tests/test_processor.py",
            start_line=1,
            end_line=2,
            chunk_type='test',
            language='python',
            metadata={'function_name': 'test_processor', 'entity_type': 'function'}
        ),
    ]


class TestIndexer:
    """Test RepositoryIndexer."""

    def test_indexer_initialization(self, repo_root):
        """Test indexer can be initialized."""
        indexer = RepositoryIndexer(repo_root)
        assert indexer.repo_root == Path(repo_root)
        assert indexer.chunk_size > 0
        assert indexer.overlap > 0

    def test_indexer_indexes_repository(self, repo_root):
        """Test indexer can index the repository."""
        indexer = RepositoryIndexer(repo_root)
        chunks = indexer.index_repository()

        assert len(chunks) > 0
        assert all(isinstance(c, DocumentChunk) for c in chunks)

        # Should have different chunk types
        chunk_types = {c.chunk_type for c in chunks}
        assert 'doc' in chunk_types or 'code' in chunk_types

    def test_indexer_statistics(self, repo_root):
        """Test indexer provides statistics."""
        indexer = RepositoryIndexer(repo_root)
        chunks = indexer.index_repository()
        stats = indexer.get_statistics()

        assert 'total_chunks' in stats
        assert 'by_type' in stats
        assert 'by_language' in stats
        assert stats['total_chunks'] == len(chunks)

    def test_chunk_has_metadata(self, repo_root):
        """Test chunks have proper metadata."""
        indexer = RepositoryIndexer(repo_root)
        chunks = indexer.index_repository()

        for chunk in chunks[:10]:  # Check first 10
            assert chunk.chunk_id is not None
            assert chunk.file_path
            assert chunk.start_line > 0
            assert chunk.end_line >= chunk.start_line
            assert chunk.chunk_type in ('doc', 'code', 'test', 'agent', 'config')


class TestBM25Retriever:
    """Test BM25Retriever."""

    def test_bm25_initialization(self):
        """Test BM25 can be initialized."""
        bm25 = BM25Retriever()
        assert bm25.k1 > 0
        assert bm25.b > 0

    def test_bm25_fit_and_search(self):
        """Test BM25 fit and search."""
        documents = [
            "The quick brown fox jumps over the lazy dog",
            "A fast brown fox leaps over a sleeping dog",
            "Python programming language tutorial",
        ]

        bm25 = BM25Retriever()
        bm25.fit(documents)

        # Search for fox-related content
        results = bm25.search("brown fox", top_k=2)

        assert len(results) <= 2
        assert all(isinstance(r, tuple) for r in results)
        assert all(len(r) == 2 for r in results)  # (index, score)

        # First results should be fox-related
        assert results[0][0] in (0, 1)
        assert results[0][1] > 0


class TestHybridRetriever:
    """Test HybridRetriever."""

    def test_retriever_initialization(self):
        """Test retriever can be initialized."""
        retriever = HybridRetriever()
        assert retriever.bm25_weight >= 0
        assert retriever.vector_weight >= 0

    def test_retriever_index_and_retrieve(self, sample_chunks):
        """Test retriever indexing and retrieval."""
        retriever = HybridRetriever()
        retriever.index(sample_chunks)

        results = retriever.retrieve("process image", top_k=2)

        assert len(results) <= 2
        assert all(hasattr(r, 'content') for r in results)
        assert all(hasattr(r, 'score') for r in results)

    def test_retriever_filtering(self, sample_chunks):
        """Test retriever can filter by type and path."""
        retriever = HybridRetriever()
        retriever.index(sample_chunks)

        # Filter by type
        results = retriever.retrieve("process", top_k=5, chunk_type_filter=['code'])

        if results:
            assert all(r.file_path.endswith('.py') for r in results)

        # Filter by file path
        results = retriever.retrieve("process", top_k=5, file_path_filter=r'test_')

        if results:
            assert all('test_' in r.file_path for r in results)


class TestReranker:
    """Test ResultReranker."""

    def test_reranker_initialization(self):
        """Test reranker can be initialized."""
        reranker = ResultReranker()
        assert reranker.signals is not None

    def test_reranker_reranks_results(self, sample_chunks):
        """Test reranker can rerank results."""
        # First retrieve
        retriever = HybridRetriever()
        retriever.index(sample_chunks)
        results = retriever.retrieve("image processing", top_k=3)

        # Then rerank
        reranker = ResultReranker()
        reranked = reranker.rerank(results, "image processing", top_k=2)

        assert len(reranked) <= len(results)
        assert all(hasattr(r, 'metadata') for r in reranked)

        # Should have rerank_boost in metadata
        if reranked:
            assert 'rerank_boost' in reranked[0].metadata


class TestCitationGenerator:
    """Test CitationGenerator."""

    def test_citation_generator_initialization(self):
        """Test citation generator can be initialized."""
        gen = CitationGenerator()
        assert gen.snippet_max_lines > 0
        assert gen.snippet_max_chars > 0

    def test_generate_citations(self, sample_chunks):
        """Test citation generation."""
        # Retrieve and generate citations
        retriever = HybridRetriever()
        retriever.index(sample_chunks)
        results = retriever.retrieve("process", top_k=2)

        gen = CitationGenerator()
        citations = gen.generate_citations(results, max_citations=2)

        assert len(citations) <= 2

        for cite in citations:
            assert cite.file_path
            assert cite.start_line > 0
            assert cite.snippet
            assert 0.0 <= cite.confidence <= 1.0

    def test_format_citations(self, sample_chunks):
        """Test citation formatting."""
        retriever = HybridRetriever()
        retriever.index(sample_chunks)
        results = retriever.retrieve("process", top_k=1)

        gen = CitationGenerator()
        citations = gen.generate_citations(results, max_citations=1)

        # Test different formats
        markdown = gen.format_citations(citations, format_type='markdown')
        assert '##' in markdown or len(citations) == 0

        text = gen.format_citations(citations, format_type='text')
        assert 'CITATIONS' in text or len(citations) == 0

        json_str = gen.format_citations(citations, format_type='json')
        assert 'citations' in json_str


class TestPromptTemplates:
    """Test PromptTemplates."""

    def test_feature_implementation_template(self):
        """Test feature implementation template generation."""
        template = PromptTemplates.feature_implementation(
            "Add new feature X",
            context="Some context"
        )

        assert "Feature Description" in template
        assert "Add new feature X" in template
        assert "Some context" in template
        assert "Response Format" in template
        assert "json" in template

    def test_bug_triage_template(self):
        """Test bug triage template generation."""
        template = PromptTemplates.bug_triage(
            "ImportError: No module named X",
            reproduction_steps="Run python script.py"
        )

        assert "Error Log" in template
        assert "ImportError" in template
        assert "Reproduction Steps" in template
        assert "Root Cause Analysis" in template

    def test_ci_change_template(self):
        """Test CI change template generation."""
        template = PromptTemplates.ci_change(
            "build.yml",
            "Add Python 3.12 to matrix"
        )

        assert "Workflow Name" in template
        assert "build.yml" in template
        assert "Python 3.12" in template
        assert "Testing Strategy" in template

    def test_add_few_shot_examples(self):
        """Test adding few-shot examples to templates."""
        base_template = "# Base Template\n\nContent here"
        examples = [
            {'input': 'example input', 'output': 'example output'}
        ]

        enhanced = PromptTemplates.add_few_shot_examples(base_template, examples)

        assert "Few-Shot Examples" in enhanced
        assert "example input" in enhanced
        assert "example output" in enhanced


class TestFewShotExamples:
    """Test FewShotExamples."""

    def test_get_feature_examples(self):
        """Test getting feature examples."""
        examples = FewShotExamples.get_feature_examples()

        assert len(examples) > 0
        assert all('input' in ex for ex in examples)
        assert all('output' in ex for ex in examples)

    def test_get_bug_triage_examples(self):
        """Test getting bug triage examples."""
        examples = FewShotExamples.get_bug_triage_examples()

        assert len(examples) > 0
        assert all('input' in ex for ex in examples)
        assert all('output' in ex for ex in examples)

    def test_get_ci_change_examples(self):
        """Test getting CI change examples."""
        examples = FewShotExamples.get_ci_change_examples()

        assert len(examples) > 0
        assert all('input' in ex for ex in examples)
        assert all('output' in ex for ex in examples)


class TestResponseSchema:
    """Test response schema validation."""

    def test_valid_schema(self):
        """Test valid response schema."""
        valid_json = '''
        {
          "summary": "Add feature X",
          "files": [
            {"path": "file.py", "patch": "diff content", "description": "desc"}
          ],
          "tests": ["test_file.py"],
          "explanation": "Detailed explanation",
          "confidence": 0.85
        }
        '''

        assert validate_response_schema(valid_json)

    def test_invalid_schema_missing_field(self):
        """Test invalid schema with missing field."""
        invalid_json = '''
        {
          "summary": "Add feature X",
          "files": []
        }
        '''

        assert not validate_response_schema(invalid_json)

    def test_code_modification_response(self):
        """Test CodeModificationResponse dataclass."""
        response = CodeModificationResponse(
            summary="Test summary",
            files=[
                FileModification(
                    path="test.py",
                    patch="dif",
                    description="Test change"
                )
            ],
            tests=["test.py"],
            explanation="Test explanation",
            confidence=0.9,
        )

        # Convert to JSON
        json_str = response.to_json()
        assert "Test summary" in json_str

        # Parse back
        parsed = CodeModificationResponse.from_json(json_str)
        assert parsed.summary == response.summary
        assert len(parsed.files) == 1
        assert parsed.confidence == 0.9


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
