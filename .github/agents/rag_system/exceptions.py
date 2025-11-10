"""
Custom exceptions for RAG System.
"""


class RAGSystemError(Exception):
    """Base exception for RAG system errors."""
    pass


class IndexingError(RAGSystemError):
    """Raised when indexing fails."""
    pass


class RetrievalError(RAGSystemError):
    """Raised when retrieval fails."""
    pass


class CacheError(RAGSystemError):
    """Raised when cache operations fail."""
    pass


class ConfigError(RAGSystemError):
    """Raised when configuration is invalid or missing."""
    pass
