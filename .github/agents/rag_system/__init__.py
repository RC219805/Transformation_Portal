"""
RAG (Retrieval-Augmented Generation) System for Transformation Portal Specialist Agent

This module provides retrieval-augmented capabilities to reduce hallucinations
and increase relevance to repository-specific patterns.
"""

from .citation import CitationGenerator
from .classifier import ArtifactClassifier, ArtifactType, PipelineType
from .indexer import RepositoryIndexer
from .knowledge_engine import KnowledgeIntegrationEngine, PatternAnalysis, Recommendation
from .reranker import ResultReranker
from .retriever import HybridRetriever

__all__ = [
    'RepositoryIndexer',
    'HybridRetriever',
    'ResultReranker',
    'CitationGenerator',
    'ArtifactClassifier',
    'ArtifactType',
    'PipelineType',
    'KnowledgeIntegrationEngine',
    'PatternAnalysis',
    'Recommendation',
]
