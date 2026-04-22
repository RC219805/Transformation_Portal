"""
RAG (Retrieval-Augmented Generation) System for Transformation Portal Specialist Agent

This module provides retrieval-augmented capabilities to reduce hallucinations
and increase relevance to repository-specific patterns.
"""

from .citation import CitationGenerator
from .classifier import ArtifactClassifier, ArtifactType, PipelineType
from .indexer import RepositoryIndexer
from .intelligent_completion import CompletionSuggestion, IntelligentCompletion
from .interactive_docs import APIDocumentation, InteractiveDocumentationSystem
from .knowledge_engine import KnowledgeIntegrationEngine, PatternAnalysis, Recommendation
from .reranker import ResultReranker
from .retriever import HybridRetriever
from .semantic_search import CodeEntity, CodeParser, SemanticCodeSearch, SemanticSearchResult

__all__ = [
    "APIDocumentation",
    "ArtifactClassifier",
    "ArtifactType",
    "CitationGenerator",
    "CodeEntity",
    "CodeParser",
    "CompletionSuggestion",
    "HybridRetriever",
    "IntelligentCompletion",
    "InteractiveDocumentationSystem",
    "KnowledgeIntegrationEngine",
    "PatternAnalysis",
    "PipelineType",
    "Recommendation",
    "RepositoryIndexer",
    "ResultReranker",
    "SemanticCodeSearch",
    "SemanticSearchResult",
]
