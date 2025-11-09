"""
Custom exceptions for RAG System.

Provides specific exception types for better error handling and debugging.
"""


class RAGSystemError(Exception):
    """Base exception for all RAG system errors."""
    pass


class IndexingError(RAGSystemError):
    """Raised when indexing fails."""
    pass


class RetrievalError(RAGSystemError):
    """Raised when retrieval fails."""
    pass


class RerankingError(RAGSystemError):
    """Raised when reranking fails."""
    pass


class CitationError(RAGSystemError):
    """Raised when citation generation fails."""
    pass


class ConfigurationError(RAGSystemError):
    """Raised when configuration is invalid."""
    pass


class CacheError(RAGSystemError):
    """Raised when cache operations fail."""
    pass


class ValidationError(RAGSystemError):
    """Raised when input validation fails."""
    pass
