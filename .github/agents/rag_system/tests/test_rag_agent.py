"""
Tests for RAG Integration Agent
================================

Comprehensive tests for the RAGAgent orchestration system including:
- Query orchestration with multiple strategies
- Knowledge fusion and conflict resolution
- Confidence scoring and gap analysis
- Cross-agent coordination
- Adaptive learning and feedback
"""

import sys
import tempfile
from pathlib import Path

# Add parent directory to path
rag_system_path = Path(__file__).parent.parent
sys.path.insert(0, str(rag_system_path))

import pytest

# Import with proper module path handling
try:
    from rag_agent import (
        ConfidenceLevel,
        KnowledgeSource,
        QueryContext,
        RAGAgent,
        RAGResponse,
        RetrievalStrategy,
        UserIntent,
    )
except ImportError:
    # Try alternative import method
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "rag_agent",
        rag_system_path / "rag_agent.py"
    )
    if spec and spec.loader:
        rag_agent_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rag_agent_module)
        ConfidenceLevel = rag_agent_module.ConfidenceLevel
        KnowledgeSource = rag_agent_module.KnowledgeSource
        QueryContext = rag_agent_module.QueryContext
        RAGAgent = rag_agent_module.RAGAgent
        RAGResponse = rag_agent_module.RAGResponse
        RetrievalStrategy = rag_agent_module.RetrievalStrategy
        UserIntent = rag_agent_module.UserIntent
    else:
        raise


@pytest.fixture
def temp_repo():
    """Create a temporary repository for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        
        # Create sample files
        (repo_path / "docs").mkdir()
        (repo_path / "docs" / "README.md").write_text("""
# Depth Pipeline Documentation

The depth pipeline processes images using Depth Anything V2.

## Features
- Depth-aware tone mapping
- Atmospheric effects
- Zone-based processing
        """)
        
        (repo_path / "src").mkdir()
        (repo_path / "src" / "pipeline.py").write_text("""
def process_image(image_path, depth_map):
    '''Process image with depth information.'''
    return processed_image
        """)
        
        (repo_path / "tests").mkdir()
        (repo_path / "tests" / "test_pipeline.py").write_text("""
def test_process_image():
    '''Test depth pipeline processing.'''
    assert True
        """)
        
        yield str(repo_path)


class TestRAGAgent:
    """Test RAG agent core functionality."""
    
    def test_agent_initialization(self, temp_repo):
        """Test agent initializes correctly."""
        agent = RAGAgent(temp_repo, enable_cache=True, enable_learning=True)
        assert agent.repo_root == Path(temp_repo)
        assert agent.enable_cache is True
        assert agent.enable_learning is True
        assert agent.query_count == 0
    
    def test_agent_initialize_indexes_repo(self, temp_repo):
        """Test initialize() indexes repository."""
        agent = RAGAgent(temp_repo)
        num_chunks = agent.initialize()
        
        assert num_chunks > 0
        assert agent.indexed_chunks is not None
        assert len(agent.indexed_chunks) == num_chunks
    
    def test_simple_query_execution(self, temp_repo):
        """Test simple query execution."""
        agent = RAGAgent(temp_repo)
        agent.initialize()
        
        response = agent.query("depth pipeline processing")
        
        assert isinstance(response, RAGResponse)
        assert response.query == "depth pipeline processing"
        assert len(response.answer) > 0
        assert isinstance(response.confidence, ConfidenceLevel)
        assert len(response.sources) > 0
    
    def test_query_with_context(self, temp_repo):
        """Test query with context."""
        agent = RAGAgent(temp_repo)
        agent.initialize()
        
        context = QueryContext(
            user_intent=UserIntent.IMPLEMENTATION,
            priority="high"
        )
        
        response = agent.query(
            "How to add atmospheric effects?",
            context=context
        )
        
        assert isinstance(response, RAGResponse)
        assert "Implementation Guidance" in response.answer or len(response.sources) > 0
    
    def test_multi_source_strategy(self, temp_repo):
        """Test multi-source retrieval strategy."""
        agent = RAGAgent(temp_repo)
        agent.initialize()
        
        context = QueryContext(user_intent=UserIntent.IMPLEMENTATION)
        
        response = agent.query(
            "depth map processing",
            context=context,
            strategy=RetrievalStrategy.MULTI_SOURCE
        )
        
        assert isinstance(response, RAGResponse)
        assert response.metrics.retrieval_strategy == RetrievalStrategy.MULTI_SOURCE.value
        
        # Should retrieve from multiple chunk types
        chunk_types = {s.chunk_type for s in response.sources}
        # May have code, doc, or test chunks depending on indexing
        assert len(chunk_types) >= 1
    
    def test_query_caching(self, temp_repo):
        """Test query results are cached."""
        agent = RAGAgent(temp_repo, enable_cache=True)
        agent.initialize()
        
        query = "depth pipeline"
        
        # First query
        response1 = agent.query(query)
        assert agent.cache_hits == 0
        
        # Second query (should hit cache)
        response2 = agent.query(query)
        assert agent.cache_hits == 1
        
        # Responses should be identical
        assert response1.query == response2.query
        assert response1.answer == response2.answer
    
    def test_intent_classification(self, temp_repo):
        """Test user intent classification."""
        agent = RAGAgent(temp_repo)
        
        # Implementation intent
        intent1 = agent._classify_intent("add new atmospheric effect")
        assert intent1 == UserIntent.IMPLEMENTATION
        
        # Bug fix intent
        intent2 = agent._classify_intent("fix error in depth processing")
        assert intent2 == UserIntent.BUG_FIX
        
        # Exploration intent
        intent3 = agent._classify_intent("how does depth pipeline work?")
        assert intent3 == UserIntent.EXPLORATION
        
        # Optimization intent
        intent4 = agent._classify_intent("optimize memory usage")
        assert intent4 == UserIntent.OPTIMIZATION
    
    def test_strategy_selection(self, temp_repo):
        """Test automatic strategy selection."""
        agent = RAGAgent(temp_repo)
        
        # Implementation should use multi-source
        context1 = QueryContext(user_intent=UserIntent.IMPLEMENTATION)
        strategy1 = agent._select_strategy("add feature", context1)
        assert strategy1 == RetrievalStrategy.MULTI_SOURCE
        
        # Complex query should use chain reasoning
        context2 = QueryContext(user_intent=UserIntent.EXPLORATION)
        strategy2 = agent._select_strategy("first do X and then Y", context2)
        assert strategy2 == RetrievalStrategy.CHAIN_REASONING
        
        # Simple query should use single
        context3 = QueryContext(user_intent=UserIntent.EXPLORATION)
        strategy3 = agent._select_strategy("what is depth map", context3)
        assert strategy3 == RetrievalStrategy.SINGLE_QUERY
    
    def test_confidence_assessment(self, temp_repo):
        """Test confidence level assessment."""
        agent = RAGAgent(temp_repo)
        agent.initialize()
        
        # High-quality sources
        high_sources = [
            KnowledgeSource(
                chunk_id="test1",
                content="test",
                file_path="test.py",
                chunk_type="code",
                start_line=1,
                end_line=10,
                score=0.9,
                retrieval_method="bm25",
                recency_score=0.9,
                quality_score=0.9
            )
        ]
        confidence_high = agent._assess_confidence(high_sources, "test")
        assert confidence_high == ConfidenceLevel.HIGH
        
        # Medium-quality sources
        medium_sources = [
            KnowledgeSource(
                chunk_id="test2",
                content="test",
                file_path="test.py",
                chunk_type="code",
                start_line=1,
                end_line=10,
                score=0.6,
                retrieval_method="bm25",
                recency_score=0.6,
                quality_score=0.6
            )
        ]
        confidence_medium = agent._assess_confidence(medium_sources, "test")
        assert confidence_medium == ConfidenceLevel.MEDIUM
        
        # No sources
        confidence_low = agent._assess_confidence([], "test")
        assert confidence_low == ConfidenceLevel.LOW
    
    def test_gap_identification(self, temp_repo):
        """Test knowledge gap identification."""
        agent = RAGAgent(temp_repo)
        
        # Code without docs
        sources_no_docs = [
            KnowledgeSource(
                chunk_id="test",
                content="def func(): pass",
                file_path="test.py",
                chunk_type="code",
                start_line=1,
                end_line=2,
                score=0.8,
                retrieval_method="bm25",
                recency_score=0.8,
                quality_score=0.2
            )
        ]
        gaps = agent._identify_gaps(sources_no_docs, "test")
        assert any("lacks documentation" in gap for gap in gaps)
        assert any("low code quality" in gap for gap in gaps)
        
        # No sources
        gaps_empty = agent._identify_gaps([], "test")
        assert any("No relevant documentation" in gap for gap in gaps_empty)
    
    def test_conflict_detection(self, temp_repo):
        """Test conflict detection between sources."""
        agent = RAGAgent(temp_repo)
        
        # Sources with varying recency
        sources_conflict = [
            KnowledgeSource(
                chunk_id="old",
                content="old",
                file_path="test.py",
                chunk_type="code",
                start_line=1,
                end_line=2,
                score=0.8,
                retrieval_method="bm25",
                recency_score=0.2,
                quality_score=0.8
            ),
            KnowledgeSource(
                chunk_id="new",
                content="new",
                file_path="test.py",
                chunk_type="code",
                start_line=5,
                end_line=6,
                score=0.8,
                retrieval_method="bm25",
                recency_score=0.9,
                quality_score=0.8
            )
        ]
        conflicts = agent._detect_conflicts(sources_conflict)
        assert any("varying recency" in conflict for conflict in conflicts)
    
    def test_prepare_context_for_agent(self, temp_repo):
        """Test preparing context for another agent."""
        agent = RAGAgent(temp_repo)
        agent.initialize()
        
        context = agent.prepare_context_for_agent(
            "transformation-portal-specialist",
            "Add atmospheric haze effect"
        )
        
        assert context['agent'] == "transformation-portal-specialist"
        assert context['task'] == "Add atmospheric haze effect"
        assert 'retrieved_sources' in context
        assert 'confidence' in context
        assert 'citations' in context
        assert 'timestamp' in context
    
    def test_feedback_recording(self, temp_repo):
        """Test feedback recording for learning."""
        agent = RAGAgent(temp_repo, enable_learning=True)
        agent.initialize()
        
        # Execute query
        query = "depth processing"
        agent.query(query)
        
        # Add feedback
        agent.add_feedback(query, helpful=True, comment="Very helpful!")
        
        assert len(agent.feedback_history) == 1
        assert agent.feedback_history[0]['helpful'] is True
        assert agent.feedback_history[0]['comment'] == "Very helpful!"
    
    def test_statistics_tracking(self, temp_repo):
        """Test agent statistics tracking."""
        agent = RAGAgent(temp_repo)
        agent.initialize()
        
        # Execute some queries
        agent.query("test 1")
        agent.query("test 2")
        agent.query("test 1")  # Cache hit
        
        stats = agent.get_statistics()
        
        assert stats['total_queries'] == 3
        assert stats['cache_hits'] == 1
        assert stats['cache_hit_rate'] == pytest.approx(1/3)
        assert stats['avg_query_time_ms'] > 0
        assert stats['indexed_chunks'] > 0
        assert stats['conversation_turns'] == 3
    
    def test_response_serialization(self, temp_repo):
        """Test RAGResponse serialization."""
        agent = RAGAgent(temp_repo)
        agent.initialize()
        
        response = agent.query("test query")
        
        # Test to_dict
        response_dict = response.to_dict()
        assert isinstance(response_dict, dict)
        assert 'query' in response_dict
        assert 'answer' in response_dict
        assert 'confidence' in response_dict
        
        # Test to_json
        response_json = response.to_json()
        assert isinstance(response_json, str)
        assert 'test query' in response_json
    
    def test_knowledge_fusion(self, temp_repo):
        """Test knowledge fusion from multiple sources."""
        agent = RAGAgent(temp_repo)
        
        # Create overlapping sources
        sources = [
            KnowledgeSource(
                chunk_id="1",
                content="line 1\nline 2",
                file_path="test.py",
                chunk_type="code",
                start_line=1,
                end_line=2,
                score=0.8,
                retrieval_method="bm25",
                recency_score=0.8,
                quality_score=0.8
            ),
            KnowledgeSource(
                chunk_id="2",
                content="line 3\nline 4",
                file_path="test.py",
                chunk_type="code",
                start_line=3,
                end_line=4,
                score=0.7,
                retrieval_method="bm25",
                recency_score=0.7,
                quality_score=0.7
            )
        ]
        
        fused = agent._fuse_knowledge(sources)
        
        # Should merge adjacent chunks
        assert len(fused) <= len(sources)
        assert all(isinstance(s, KnowledgeSource) for s in fused)
    
    def test_query_decomposition(self, temp_repo):
        """Test complex query decomposition."""
        agent = RAGAgent(temp_repo)
        context = QueryContext()
        
        # Complex query with "and"
        sub_queries = agent._decompose_query(
            "add atmospheric effect and update documentation",
            context
        )
        
        assert len(sub_queries) >= 1
        assert isinstance(sub_queries, list)
        assert all(isinstance(q, str) for q in sub_queries)
    
    def test_adaptive_strategy_with_feedback(self, temp_repo):
        """Test adaptive strategy selection improves with feedback."""
        agent = RAGAgent(temp_repo, enable_learning=True)
        agent.initialize()
        
        # Execute query with multi-source strategy
        context = QueryContext(user_intent=UserIntent.IMPLEMENTATION)
        agent.query(
            "add new feature",
            context=context,
            strategy=RetrievalStrategy.MULTI_SOURCE
        )
        
        # Provide positive feedback
        agent.add_feedback("add new feature", helpful=True)
        
        # Check that strategy is tracked
        assert RetrievalStrategy.MULTI_SOURCE.value in agent.successful_strategies
        assert len(agent.successful_strategies[RetrievalStrategy.MULTI_SOURCE.value]) > 0
    
    def test_citation_generation_in_response(self, temp_repo):
        """Test citations are generated in response."""
        agent = RAGAgent(temp_repo)
        agent.initialize()
        
        response = agent.query(
            "depth processing",
            include_citations=True,
            max_citations=3
        )
        
        assert 'citations' in response.__dict__
        # Citations may be empty if no sources found, but structure exists
        assert hasattr(response, 'citations')
        assert isinstance(response.citations, list)


class TestQueryContext:
    """Test QueryContext data class."""
    
    def test_query_context_creation(self):
        """Test QueryContext creation."""
        context = QueryContext(
            conversation_history=["previous query"],
            user_intent=UserIntent.IMPLEMENTATION,
            priority="high"
        )
        
        assert context.conversation_history == ["previous query"]
        assert context.user_intent == UserIntent.IMPLEMENTATION
        assert context.priority == "high"
    
    def test_query_context_defaults(self):
        """Test QueryContext defaults."""
        context = QueryContext()
        
        assert context.conversation_history == []
        assert context.user_intent is None
        assert context.priority == "medium"
        assert isinstance(context.constraints, dict)
        assert isinstance(context.metadata, dict)


class TestKnowledgeSource:
    """Test KnowledgeSource data class."""
    
    def test_knowledge_source_creation(self):
        """Test KnowledgeSource creation."""
        source = KnowledgeSource(
            chunk_id="test123",
            content="test content",
            file_path="test.py",
            chunk_type="code",
            start_line=10,
            end_line=20,
            score=0.85,
            retrieval_method="bm25",
            recency_score=0.9,
            quality_score=0.8
        )
        
        assert source.chunk_id == "test123"
        assert source.content == "test content"
        assert source.file_path == "test.py"
        assert source.chunk_type == "code"
        assert source.score == 0.85


class TestEnums:
    """Test enum definitions."""
    
    def test_retrieval_strategy_enum(self):
        """Test RetrievalStrategy enum."""
        assert RetrievalStrategy.SINGLE_QUERY.value == "single"
        assert RetrievalStrategy.MULTI_SOURCE.value == "multi_source"
        assert RetrievalStrategy.CHAIN_REASONING.value == "chain"
        assert RetrievalStrategy.ADAPTIVE.value == "adaptive"
    
    def test_user_intent_enum(self):
        """Test UserIntent enum."""
        assert UserIntent.IMPLEMENTATION.value == "implementation"
        assert UserIntent.BUG_FIX.value == "bug_fix"
        assert UserIntent.EXPLORATION.value == "exploration"
    
    def test_confidence_level_enum(self):
        """Test ConfidenceLevel enum."""
        assert ConfidenceLevel.HIGH.value == "high"
        assert ConfidenceLevel.MEDIUM.value == "medium"
        assert ConfidenceLevel.LOW.value == "low"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
