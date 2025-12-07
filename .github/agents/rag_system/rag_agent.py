"""
RAG Integration Agent - Autonomous RAG Query Orchestrator
==========================================================

This module provides the programmatic interface for the RAG Integration Agent,
enabling autonomous RAG query orchestration, knowledge fusion, and context-aware
code assistance.

Features:
- Intelligent query orchestration with multi-step retrieval
- Knowledge fusion from multiple sources (code, docs, tests)
- Context-aware caching and retrieval optimization
- Confidence scoring and validation
- Adaptive learning from feedback
- Cross-agent coordination protocol

Usage:
    from rag_agent import RAGAgent, QueryContext, RetrievalStrategy
    
    # Initialize agent
    agent = RAGAgent(repo_root='.')
    
    # Simple query
    response = agent.query("How to add a new LUT preset?")
    
    # Complex query with context
    context = QueryContext(
        conversation_history=["Previous query about video processing"],
        user_intent="implementation",
        priority="high"
    )
    response = agent.query(
        "Add sunset LUT to video grader",
        context=context,
        strategy=RetrievalStrategy.MULTI_SOURCE
    )
    
    # Cross-agent coordination
    specialist_context = agent.prepare_context_for_agent(
        "transformation-portal-specialist",
        task="Implement depth-based vignetting"
    )

Author: Transformation Portal
Version: 1.0.0
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Import RAG system components
from .cache_manager import CacheManager
from .citation import CitationGenerator
from .config import get_config
from .enhanced_retriever import EnhancedHybridRetriever, RetrieverConfig
from .indexer import DocumentChunk, RepositoryIndexer
from .knowledge_engine import KnowledgeIntegrationEngine
from .logger import get_logger
from .reranker import ResultReranker
from .retriever import HybridRetriever, RetrievalResult

logger = get_logger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

# Configuration constants
DEFAULT_RECENCY_SCORE = 0.5  # Default when file mtime unavailable
DEFAULT_QUALITY_SCORE = 0.5  # Default when code quality signals unavailable
ADJACENT_CHUNK_THRESHOLD = 5  # Lines between chunks to consider for merging


class RetrievalStrategy(Enum):
    """Strategy for RAG retrieval."""
    
    SINGLE_QUERY = "single"           # Simple single-step retrieval
    MULTI_SOURCE = "multi_source"     # Query code, docs, tests separately
    CHAIN_REASONING = "chain"         # Chain multiple queries for complex tasks
    ADAPTIVE = "adaptive"             # Agent decides optimal strategy


class UserIntent(Enum):
    """User intent classification."""
    
    IMPLEMENTATION = "implementation"  # Implement new feature
    BUG_FIX = "bug_fix"               # Debug/fix issue
    EXPLORATION = "exploration"        # Understand code
    REFACTORING = "refactoring"       # Refactor existing code
    DOCUMENTATION = "documentation"    # Document code
    OPTIMIZATION = "optimization"      # Performance optimization


class ConfidenceLevel(Enum):
    """Confidence level in retrieval results."""
    
    HIGH = "high"         # >0.8 - Strong evidence, recent, consistent
    MEDIUM = "medium"     # 0.5-0.8 - Good evidence, may have gaps
    LOW = "low"           # <0.5 - Weak evidence, outdated, conflicting


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class QueryContext:
    """Context for a RAG query."""
    
    conversation_history: List[str] = field(default_factory=list)
    user_intent: Optional[UserIntent] = None
    priority: str = "medium"  # low, medium, high, critical
    constraints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalMetrics:
    """Metrics for a retrieval operation."""
    
    query_time_ms: float
    num_sources: int
    num_results: int
    retrieval_strategy: str
    cache_hit: bool
    confidence_score: float
    timestamp: datetime = field(default_factory=lambda: datetime.now())


@dataclass
class KnowledgeSource:
    """A source of knowledge with metadata."""
    
    chunk_id: str
    content: str
    file_path: str
    chunk_type: str  # code, doc, test, config
    start_line: int
    end_line: int
    score: float
    retrieval_method: str
    recency_score: float  # 0-1, based on file modification time
    quality_score: float  # 0-1, based on code quality signals
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGResponse:
    """Response from RAG agent."""
    
    query: str
    answer: str
    sources: List[KnowledgeSource]
    confidence: ConfidenceLevel
    citations: List[Dict[str, Any]]
    metrics: RetrievalMetrics
    recommendations: List[str] = field(default_factory=list)
    gaps_identified: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'query': self.query,
            'answer': self.answer,
            'sources': [asdict(s) for s in self.sources],
            'confidence': self.confidence.value,
            'citations': self.citations,
            'metrics': asdict(self.metrics),
            'recommendations': self.recommendations,
            'gaps_identified': self.gaps_identified,
            'conflicts': self.conflicts,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


# =============================================================================
# RAG Agent Core
# =============================================================================


class RAGAgent:
    """
    Autonomous RAG Integration Agent for intelligent knowledge retrieval.
    
    This agent orchestrates RAG queries, fuses knowledge from multiple sources,
    manages context, validates results, and coordinates with other agents.
    
    Features:
    - Multi-strategy retrieval (single, multi-source, chain, adaptive)
    - Knowledge fusion with conflict resolution
    - Context-aware caching and query optimization
    - Confidence scoring and gap analysis
    - Adaptive learning from feedback
    - Cross-agent coordination
    
    Example:
        agent = RAGAgent(repo_root='.')
        response = agent.query("How to add atmospheric haze effect?")
        print(response.answer)
        print(f"Confidence: {response.confidence.value}")
        for citation in response.citations:
            print(f"  - {citation['file_path']}:{citation['line_range']}")
    """
    
    def __init__(
        self,
        repo_root: str,
        config_path: Optional[str] = None,
        enable_cache: bool = True,
        enable_learning: bool = True,
    ):
        """
        Initialize RAG agent.
        
        Args:
            repo_root: Repository root directory
            config_path: Optional path to config file
            enable_cache: Enable query and embedding caching
            enable_learning: Enable adaptive learning from feedback
        """
        self.repo_root = Path(repo_root)
        self.config = get_config(config_path)
        self.enable_cache = enable_cache
        self.enable_learning = enable_learning
        
        # Initialize components
        self.indexer = RepositoryIndexer(str(self.repo_root))
        self.retriever = None  # Lazy initialization
        self.enhanced_retriever = None
        self.reranker = ResultReranker()
        self.citation_gen = CitationGenerator()
        self.knowledge_engine = KnowledgeIntegrationEngine()
        self.cache_manager = CacheManager() if enable_cache else None
        
        # Context management
        self.conversation_history: List[Tuple[str, RAGResponse]] = []
        self.query_cache: Dict[str, RAGResponse] = {}
        self.indexed_chunks: Optional[List[DocumentChunk]] = None
        
        # Metrics tracking
        self.query_count = 0
        self.cache_hits = 0
        self.total_query_time_ms = 0.0
        
        # Learning components
        self.feedback_history: List[Dict[str, Any]] = []
        self.query_patterns: Dict[str, int] = defaultdict(int)
        self.successful_strategies: Dict[str, List[str]] = defaultdict(list)
        
        logger.info(f"RAGAgent initialized for repo: {self.repo_root}")
    
    def initialize(self, force_reindex: bool = False) -> int:
        """
        Initialize the agent by indexing the repository.
        
        Args:
            force_reindex: Force reindexing even if cache exists
            
        Returns:
            Number of chunks indexed
        """
        start_time = time.time()
        logger.info("Initializing RAG agent...")
        
        # Index repository
        self.indexed_chunks = self.indexer.index_repository(force_reindex=force_reindex)
        
        # Initialize retriever
        if self.config.get('retriever', {}).get('enable_vector_search', False):
            # Use enhanced retriever with vector search
            retriever_config = RetrieverConfig(
                enable_vector_search=True,
                bm25_weight=self.config.get('retriever', {}).get('bm25_weight', 0.6),
                vector_weight=self.config.get('retriever', {}).get('vector_weight', 0.4),
            )
            self.enhanced_retriever = EnhancedHybridRetriever(retriever_config)
            self.enhanced_retriever.index(self.indexed_chunks)
            logger.info("Initialized enhanced hybrid retriever with vector search")
        else:
            # Use basic hybrid retriever
            self.retriever = HybridRetriever()
            self.retriever.index(self.indexed_chunks)
            logger.info("Initialized basic hybrid retriever")
        
        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(f"RAG agent initialized in {elapsed_ms:.1f}ms with {len(self.indexed_chunks)} chunks")
        
        return len(self.indexed_chunks)
    
    def query(
        self,
        query_text: str,
        context: Optional[QueryContext] = None,
        strategy: RetrievalStrategy = RetrievalStrategy.ADAPTIVE,
        top_k: int = 10,
        include_citations: bool = True,
        max_citations: int = 5,
    ) -> RAGResponse:
        """
        Execute a RAG query with intelligent orchestration.
        
        Args:
            query_text: Natural language query
            context: Optional query context
            strategy: Retrieval strategy to use
            top_k: Number of results to retrieve
            include_citations: Include formatted citations
            max_citations: Maximum number of citations
            
        Returns:
            RAGResponse with answer, sources, and metadata
        """
        if not self.indexed_chunks:
            self.initialize()
        
        self.query_count += 1
        start_time = time.time()
        
        # Check cache
        cache_key = self._get_cache_key(query_text, strategy, top_k)
        if self.enable_cache and cache_key in self.query_cache:
            self.cache_hits += 1
            cached_response = self.query_cache[cache_key]
            logger.info(f"Cache hit for query: {query_text[:50]}...")
            return cached_response
        
        # Classify user intent if not provided
        if context is None:
            context = QueryContext()
        if context.user_intent is None:
            context.user_intent = self._classify_intent(query_text)
        
        # Determine optimal strategy
        if strategy == RetrievalStrategy.ADAPTIVE:
            strategy = self._select_strategy(query_text, context)
        
        # Execute retrieval based on strategy
        if strategy == RetrievalStrategy.SINGLE_QUERY:
            sources = self._retrieve_single(query_text, top_k)
        elif strategy == RetrievalStrategy.MULTI_SOURCE:
            sources = self._retrieve_multi_source(query_text, top_k, context)
        elif strategy == RetrievalStrategy.CHAIN_REASONING:
            sources = self._retrieve_chain(query_text, top_k, context)
        else:
            sources = self._retrieve_single(query_text, top_k)
        
        # Fuse knowledge and resolve conflicts
        fused_sources = self._fuse_knowledge(sources)
        
        # Generate answer
        answer = self._generate_answer(query_text, fused_sources, context)
        
        # Assess confidence
        confidence = self._assess_confidence(fused_sources, query_text)
        
        # Generate citations
        # Note: Citation generator expects RetrievalResult objects
        # This conversion maintains compatibility with existing citation system
        citations = []
        if include_citations and fused_sources:
            retrieval_results = [self._source_to_result(s) for s in fused_sources]
            citations = self.citation_gen.generate_citations(
                retrieval_results[:max_citations],
                max_citations=max_citations
            )
        
        # Identify gaps and conflicts
        gaps = self._identify_gaps(fused_sources, query_text)
        conflicts = self._detect_conflicts(fused_sources)
        recommendations = self._generate_recommendations(fused_sources, gaps, conflicts)
        
        # Create metrics
        elapsed_ms = (time.time() - start_time) * 1000
        self.total_query_time_ms += elapsed_ms
        metrics = RetrievalMetrics(
            query_time_ms=elapsed_ms,
            num_sources=len(sources),
            num_results=len(fused_sources),
            retrieval_strategy=strategy.value,
            cache_hit=False,
            confidence_score=self._confidence_to_score(confidence),
        )
        
        # Create response
        response = RAGResponse(
            query=query_text,
            answer=answer,
            sources=fused_sources,
            confidence=confidence,
            citations=citations,
            metrics=metrics,
            recommendations=recommendations,
            gaps_identified=gaps,
            conflicts=conflicts,
        )
        
        # Cache response
        if self.enable_cache:
            self.query_cache[cache_key] = response
        
        # Track for learning
        if self.enable_learning:
            self._record_query(query_text, strategy, context, response)
        
        # Store in conversation history
        self.conversation_history.append((query_text, response))
        
        logger.info(
            f"Query completed in {elapsed_ms:.1f}ms: {query_text[:50]}... "
            f"(confidence: {confidence.value}, sources: {len(fused_sources)})"
        )
        
        return response
    
    def _retrieve_single(self, query: str, top_k: int) -> List[KnowledgeSource]:
        """Execute single-query retrieval."""
        retriever = self.enhanced_retriever or self.retriever
        results = retriever.retrieve(query, top_k=top_k * 2)  # Retrieve more for reranking
        
        # Rerank results
        reranked = self.reranker.rerank(results, query, top_k=top_k)
        
        # Convert to KnowledgeSource
        sources = [self._result_to_source(r) for r in reranked]
        return sources
    
    def _retrieve_multi_source(
        self,
        query: str,
        top_k: int,
        context: QueryContext
    ) -> List[KnowledgeSource]:
        """
        Execute multi-source retrieval (code, docs, tests).
        
        Queries different chunk types separately and combines results.
        """
        sources = []
        retriever = self.enhanced_retriever or self.retriever
        
        # Query code
        code_results = retriever.retrieve(query, top_k=top_k, chunk_type_filter=['code'])
        sources.extend([self._result_to_source(r) for r in code_results])
        
        # Query documentation
        doc_results = retriever.retrieve(query, top_k=top_k // 2, chunk_type_filter=['doc'])
        sources.extend([self._result_to_source(r) for r in doc_results])
        
        # Query tests if relevant
        if context.user_intent in (UserIntent.IMPLEMENTATION, UserIntent.BUG_FIX):
            test_results = retriever.retrieve(query, top_k=top_k // 2, chunk_type_filter=['test'])
            sources.extend([self._result_to_source(r) for r in test_results])
        
        # Deduplicate and sort by score
        sources = self._deduplicate_sources(sources)
        sources.sort(key=lambda s: s.score, reverse=True)
        
        return sources[:top_k]
    
    def _retrieve_chain(
        self,
        query: str,
        top_k: int,
        context: QueryContext
    ) -> List[KnowledgeSource]:
        """
        Execute chain reasoning retrieval.
        
        Breaks down complex query into sub-queries and chains results.
        """
        # Decompose query into sub-queries
        sub_queries = self._decompose_query(query, context)
        
        all_sources = []
        for sub_query in sub_queries:
            sources = self._retrieve_single(sub_query, top_k // len(sub_queries))
            all_sources.extend(sources)
        
        # Deduplicate and sort
        all_sources = self._deduplicate_sources(all_sources)
        all_sources.sort(key=lambda s: s.score, reverse=True)
        
        return all_sources[:top_k]
    
    def _fuse_knowledge(self, sources: List[KnowledgeSource]) -> List[KnowledgeSource]:
        """
        Fuse knowledge from multiple sources.
        
        Combines related sources, resolves conflicts, and ranks by quality.
        """
        if not sources:
            return []
        
        # Group sources by file path
        by_file: Dict[str, List[KnowledgeSource]] = defaultdict(list)
        for source in sources:
            by_file[source.file_path].append(source)
        
        # Merge adjacent chunks from same file
        merged_sources = []
        for file_path, file_sources in by_file.items():
            file_sources.sort(key=lambda s: s.start_line)
            
            current_merge = file_sources[0]
            for source in file_sources[1:]:
                # Merge if adjacent (within threshold lines)
                if source.start_line - current_merge.end_line <= ADJACENT_CHUNK_THRESHOLD:
                    # Create new instance to avoid mutating original
                    current_merge = replace(
                        current_merge,
                        content=current_merge.content + "\n" + source.content,
                        end_line=source.end_line,
                        score=max(current_merge.score, source.score)
                    )
                else:
                    merged_sources.append(current_merge)
                    current_merge = source
            
            merged_sources.append(current_merge)
        
        # Sort by combined score (retrieval + recency + quality)
        for source in merged_sources:
            source.score = self._compute_combined_score(source)
        
        merged_sources.sort(key=lambda s: s.score, reverse=True)
        
        return merged_sources
    
    def _generate_answer(
        self,
        query: str,
        sources: List[KnowledgeSource],
        context: QueryContext
    ) -> str:
        """
        Generate answer by synthesizing information from sources.
        
        This is a template that should be enhanced with LLM integration.
        """
        if not sources:
            return (
                f"I couldn't find specific information about: {query}\n\n"
                "This may be a documentation gap. Consider adding documentation or examples."
            )
        
        # Build answer from sources
        answer_parts = []
        
        # Add main findings
        answer_parts.append(f"Based on {len(sources)} relevant sources:\n")
        
        for i, source in enumerate(sources[:3], 1):
            snippet = source.content[:200] + "..." if len(source.content) > 200 else source.content
            answer_parts.append(
                f"\n{i}. **{source.file_path}** ({source.chunk_type}):\n"
                f"   {snippet}\n"
            )
        
        # Add context-specific guidance
        if context.user_intent == UserIntent.IMPLEMENTATION:
            answer_parts.append(
                "\n**Implementation Guidance:**\n"
                "- Follow patterns from the retrieved examples\n"
                "- Add tests similar to those in test sources\n"
                "- Update documentation if adding new features\n"
            )
        elif context.user_intent == UserIntent.BUG_FIX:
            answer_parts.append(
                "\n**Debug Guidance:**\n"
                "- Check error handling in retrieved code\n"
                "- Look for similar bug fixes in git history\n"
                "- Verify edge cases in test files\n"
            )
        
        return "".join(answer_parts)
    
    def _assess_confidence(
        self,
        sources: List[KnowledgeSource],
        query: str
    ) -> ConfidenceLevel:
        """
        Assess confidence in retrieval results.
        
        Considers: retrieval scores, recency, quality, consistency.
        """
        if not sources:
            return ConfidenceLevel.LOW
        
        # Average scores
        avg_retrieval_score = np.mean([s.score for s in sources])
        avg_recency_score = np.mean([s.recency_score for s in sources])
        avg_quality_score = np.mean([s.quality_score for s in sources])
        
        # Combined confidence score
        confidence_score = (
            0.5 * avg_retrieval_score +
            0.25 * avg_recency_score +
            0.25 * avg_quality_score
        )
        
        # Map to confidence level
        if confidence_score >= 0.8:
            return ConfidenceLevel.HIGH
        elif confidence_score >= 0.5:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    def _identify_gaps(
        self,
        sources: List[KnowledgeSource],
        query: str
    ) -> List[str]:
        """Identify documentation or knowledge gaps."""
        gaps = []
        
        if not sources:
            gaps.append("No relevant documentation found for this query")
            return gaps
        
        # Check for missing chunk types
        chunk_types = {s.chunk_type for s in sources}
        if 'code' in chunk_types and 'doc' not in chunk_types:
            gaps.append("Code exists but lacks documentation")
        if 'code' in chunk_types and 'test' not in chunk_types:
            gaps.append("Code exists but lacks comprehensive tests")
        
        # Check for low-quality sources
        low_quality = [s for s in sources if s.quality_score < 0.3]
        if low_quality:
            gaps.append(f"{len(low_quality)} sources have low code quality (missing docstrings/type hints)")
        
        return gaps
    
    def _detect_conflicts(self, sources: List[KnowledgeSource]) -> List[str]:
        """Detect conflicts between sources."""
        conflicts = []
        
        # Check for version conflicts (simplified)
        # In a full implementation, this would analyze semantic differences
        
        if len(sources) < 2:
            return conflicts
        
        # Group by file path
        by_file: Dict[str, List[KnowledgeSource]] = defaultdict(list)
        for source in sources:
            by_file[source.file_path].append(source)
        
        # Check for recency mismatches
        recency_scores = [s.recency_score for s in sources]
        if len(recency_scores) > 1 and max(recency_scores) - min(recency_scores) > 0.5:
            conflicts.append("Sources have varying recency - some may be outdated")
        
        return conflicts
    
    def _generate_recommendations(
        self,
        sources: List[KnowledgeSource],
        gaps: List[str],
        conflicts: List[str]
    ) -> List[str]:
        """Generate recommendations for improving knowledge coverage."""
        recommendations = []
        
        if "lacks documentation" in " ".join(gaps):
            code_files = {s.file_path for s in sources if s.chunk_type == 'code'}
            for file_path in code_files:
                recommendations.append(f"Add documentation for {file_path}")
        
        if "lacks comprehensive tests" in " ".join(gaps):
            code_files = {s.file_path for s in sources if s.chunk_type == 'code'}
            for file_path in code_files:
                recommendations.append(f"Add test coverage for {file_path}")
        
        if conflicts:
            recommendations.append("Update outdated documentation to match current code")
        
        return recommendations
    
    def prepare_context_for_agent(
        self,
        agent_name: str,
        task: str,
        include_history: bool = False
    ) -> Dict[str, Any]:
        """
        Prepare RAG context for another agent.
        
        Args:
            agent_name: Name of target agent (e.g., "transformation-portal-specialist")
            task: Task description for the agent
            include_history: Include conversation history
            
        Returns:
            Context dictionary for agent
        """
        # Query for task-relevant context
        response = self.query(
            task,
            strategy=RetrievalStrategy.MULTI_SOURCE,
            top_k=10
        )
        
        context = {
            'agent': agent_name,
            'task': task,
            'retrieved_sources': len(response.sources),
            'confidence': response.confidence.value,
            'citations': response.citations,
            'recommendations': response.recommendations,
            'timestamp': datetime.now().isoformat(),
        }
        
        if include_history and self.conversation_history:
            context['conversation_history'] = [
                {'query': q, 'confidence': r.confidence.value}
                for q, r in self.conversation_history[-3:]  # Last 3 interactions
            ]
        
        return context
    
    def add_feedback(
        self,
        query: str,
        helpful: bool,
        comment: Optional[str] = None
    ):
        """
        Add feedback for adaptive learning.
        
        Args:
            query: Query that was executed
            helpful: Whether the response was helpful
            comment: Optional feedback comment
        """
        if not self.enable_learning:
            return
        
        feedback = {
            'query': query,
            'helpful': helpful,
            'comment': comment,
            'timestamp': datetime.now().isoformat(),
        }
        self.feedback_history.append(feedback)
        
        # Update learning metrics
        if helpful:
            # Find the query in conversation history
            for hist_query, response in self.conversation_history:
                if hist_query == query:
                    strategy = response.metrics.retrieval_strategy
                    self.successful_strategies[strategy].append(query)
                    break
        
        logger.info(f"Feedback recorded for query: {query[:50]}... (helpful: {helpful})")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            'total_queries': self.query_count,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': self.cache_hits / self.query_count if self.query_count > 0 else 0.0,
            'avg_query_time_ms': self.total_query_time_ms / self.query_count if self.query_count > 0 else 0.0,
            'indexed_chunks': len(self.indexed_chunks) if self.indexed_chunks else 0,
            'conversation_turns': len(self.conversation_history),
            'feedback_count': len(self.feedback_history),
            'successful_strategies': {k: len(v) for k, v in self.successful_strategies.items()},
        }
    
    # Helper methods
    
    def _get_cache_key(self, query: str, strategy: RetrievalStrategy, top_k: int) -> str:
        """Generate cache key for query."""
        return f"{query}|{strategy.value}|{top_k}"
    
    def _classify_intent(self, query: str) -> UserIntent:
        """Classify user intent from query text."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['add', 'create', 'implement', 'new']):
            return UserIntent.IMPLEMENTATION
        elif any(word in query_lower for word in ['fix', 'bug', 'error', 'failing']):
            return UserIntent.BUG_FIX
        elif any(word in query_lower for word in ['how', 'what', 'where', 'understand', 'explain']):
            return UserIntent.EXPLORATION
        elif any(word in query_lower for word in ['refactor', 'improve', 'clean']):
            return UserIntent.REFACTORING
        elif any(word in query_lower for word in ['document', 'doc', 'readme']):
            return UserIntent.DOCUMENTATION
        elif any(word in query_lower for word in ['optimize', 'performance', 'speed', 'memory']):
            return UserIntent.OPTIMIZATION
        else:
            return UserIntent.EXPLORATION
    
    def _select_strategy(self, query: str, context: QueryContext) -> RetrievalStrategy:
        """Select optimal retrieval strategy based on query and context."""
        # Simple heuristic-based selection
        # In production, this could use ML models
        
        query_lower = query.lower()
        
        # Use multi-source for implementation tasks
        if context.user_intent == UserIntent.IMPLEMENTATION:
            return RetrievalStrategy.MULTI_SOURCE
        
        # Use chain reasoning for complex queries
        if any(word in query_lower for word in ['and', 'then', 'after', 'before', 'workflow']):
            return RetrievalStrategy.CHAIN_REASONING
        
        # Use single query for simple lookups
        return RetrievalStrategy.SINGLE_QUERY
    
    def _decompose_query(self, query: str, context: QueryContext) -> List[str]:
        """
        Decompose complex query into sub-queries.
        
        Note: This is a simplified heuristic-based decomposition.
        Production systems would use:
        - NLP techniques (spaCy, NLTK)
        - Dependency parsing
        - Intent recognition
        - Query reformulation
        
        Current limitations:
        - Only handles simple ' and ' splits
        - May miss complex query patterns
        - Does not handle nested conditions
        """
        sub_queries = []
        
        # Split on "and", "then", etc.
        parts = query.split(' and ')
        if len(parts) > 1:
            sub_queries.extend(parts)
        else:
            sub_queries.append(query)
        
        return sub_queries
    
    def _result_to_source(self, result: RetrievalResult) -> KnowledgeSource:
        """Convert RetrievalResult to KnowledgeSource."""
        return KnowledgeSource(
            chunk_id=result.chunk_id,
            content=result.content,
            file_path=result.file_path,
            chunk_type=result.metadata.get('chunk_type', 'unknown'),
            start_line=result.start_line,
            end_line=result.end_line,
            score=result.score,
            retrieval_method=result.retrieval_method,
            recency_score=DEFAULT_RECENCY_SCORE,  # TODO: Compute from file mtime
            quality_score=DEFAULT_QUALITY_SCORE,  # TODO: Extract code quality signals
            metadata=result.metadata,
        )
    
    def _source_to_result(self, source: KnowledgeSource) -> RetrievalResult:
        """Convert KnowledgeSource to RetrievalResult."""
        return RetrievalResult(
            chunk_id=source.chunk_id,
            content=source.content,
            file_path=source.file_path,
            start_line=source.start_line,
            end_line=source.end_line,
            score=source.score,
            retrieval_method=source.retrieval_method,
            metadata=source.metadata,
        )
    
    def _deduplicate_sources(self, sources: List[KnowledgeSource]) -> List[KnowledgeSource]:
        """Remove duplicate sources."""
        seen = set()
        unique_sources = []
        
        for source in sources:
            key = (source.file_path, source.start_line, source.end_line)
            if key not in seen:
                seen.add(key)
                unique_sources.append(source)
        
        return unique_sources
    
    def _compute_combined_score(self, source: KnowledgeSource) -> float:
        """Compute combined score from retrieval, recency, and quality."""
        return (
            0.6 * source.score +
            0.2 * source.recency_score +
            0.2 * source.quality_score
        )
    
    def _confidence_to_score(self, confidence: ConfidenceLevel) -> float:
        """Convert ConfidenceLevel to numeric score."""
        if confidence == ConfidenceLevel.HIGH:
            return 0.9
        elif confidence == ConfidenceLevel.MEDIUM:
            return 0.6
        else:
            return 0.3
    
    def _record_query(
        self,
        query: str,
        strategy: RetrievalStrategy,
        context: QueryContext,
        response: RAGResponse
    ):
        """Record query for learning."""
        self.query_patterns[strategy.value] += 1
