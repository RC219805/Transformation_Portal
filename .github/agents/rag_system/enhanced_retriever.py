"""
Transformation Portal RAG System - Enhanced Hybrid Retriever
=============================================================
Phase 1 Implementation: Activated semantic vector search with embedding caching.

This module provides:
- BM25 sparse retrieval (keyword matching)
- Dense vector embeddings via Sentence Transformers
- Hybrid scoring with configurable weights
- Embedding caching for persistence across sessions
- Query result caching with LRU eviction
- GPU/MPS/CPU automatic device selection

Architecture:
    HybridRetriever
    ├── BM25Retriever (sparse, keyword-based)
    ├── VectorRetriever (dense, semantic)
    │   ├── SentenceTransformer model
    │   └── EmbeddingCache (numpy persistence)
    ├── HybridScorer (weight combination)
    └── QueryCache (LRU result caching)

Performance Characteristics:
    - Model loading: ~2-3 seconds (one-time)
    - Embedding 1000 chunks: ~3-5 seconds
    - BM25 query: <10ms
    - Vector query: ~15-25ms
    - Hybrid query: ~20-30ms
    - Cached query: <1ms

Usage:
    from enhanced_retriever import EnhancedHybridRetriever, RetrieverConfig

    config = RetrieverConfig(
        enable_vector_search=True,
        bm25_weight=0.6,
        vector_weight=0.4,
    )
    retriever = EnhancedHybridRetriever(config)

    # Index chunks
    retriever.index(chunks)

    # Search with hybrid retrieval
    results = retriever.retrieve("atmospheric depth effects", top_k=10)

    # Save embeddings for persistence
    retriever.save_embeddings(cache_manager)

Author: Transformation Portal
Version: 2.0.0 (Phase 1)
"""

from __future__ import annotations

import logging
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np

# Configure module logger
logger = logging.getLogger("rag_system.retriever")


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class RetrieverConfig:
    """Configuration for the enhanced hybrid retriever."""

    # Vector search activation (Phase 1 Core)
    enable_vector_search: bool = True

    # Hybrid weights (tuned for code/documentation corpus)
    bm25_weight: float = 0.6
    vector_weight: float = 0.4

    # BM25 parameters (Okapi BM25)
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # Vector search configuration
    vector_model: str = "all-MiniLM-L6-v2"
    vector_dimensions: int = 384
    similarity_metric: str = "cosine"  # cosine, euclidean, dot_product

    # Performance
    embedding_batch_size: int = 32
    use_gpu: str = "auto"  # auto, true, false
    model_warmup: bool = True

    # Query caching
    query_cache_size: int = 100
    query_cache_enabled: bool = True

    # Default retrieval
    top_k_default: int = 10

    def __post_init__(self):
        """Validate configuration."""
        assert 0 <= self.bm25_weight <= 1, "bm25_weight must be in [0, 1]"
        assert 0 <= self.vector_weight <= 1, "vector_weight must be in [0, 1]"
        total = self.bm25_weight + self.vector_weight
        if not math.isclose(total, 1.0, rel_tol=1e-5):
            logger.warning(f"Weights sum to {total}, normalizing")
            self.bm25_weight /= total
            self.vector_weight /= total


@dataclass
class RetrievalResult:
    """A single retrieval result with scoring metadata."""

    chunk_id: str
    chunk: Any  # The actual chunk object
    score: float
    bm25_score: float = 0.0
    vector_score: float = 0.0
    retrieval_method: str = "hybrid"  # bm25, vector, hybrid

    # Metadata
    file_path: str = ""
    line_start: int = 0
    line_end: int = 0
    chunk_type: str = ""  # code, doc, test, config


@dataclass
class RetrievalStats:
    """Statistics about retrieval operations."""

    total_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    avg_query_time_ms: float = 0.0
    avg_results_returned: float = 0.0

    # Per-method breakdown
    bm25_queries: int = 0
    vector_queries: int = 0
    hybrid_queries: int = 0


# =============================================================================
# BM25 Retriever
# =============================================================================


class BM25Retriever:
    """
    BM25 (Okapi BM25) sparse retrieval implementation.

    Provides fast keyword-based matching with TF-IDF weighting.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 retriever.

        Args:
            k1: Term frequency saturation parameter
            b: Length normalization parameter
        """
        self.k1 = k1
        self.b = b

        # Index structures
        self.documents: List[List[str]] = []  # Tokenized documents
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0.0
        self.doc_freqs: Counter = Counter()  # Document frequency per term
        self.idf: Dict[str, float] = {}
        self.n_docs: int = 0

        self._indexed = False

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        # Lowercase and split on non-alphanumeric
        import re
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens

    def index(self, documents: List[str]) -> None:
        """
        Index documents for BM25 retrieval.

        Args:
            documents: List of document strings to index
        """
        start_time = time.time()

        self.documents = []
        self.doc_lengths = []
        self.doc_freqs = Counter()

        for doc in documents:
            tokens = self._tokenize(doc)
            self.documents.append(tokens)
            self.doc_lengths.append(len(tokens))

            # Count document frequency (unique terms per doc)
            unique_terms = set(tokens)
            for term in unique_terms:
                self.doc_freqs[term] += 1

        self.n_docs = len(documents)
        self.avg_doc_length = (
            sum(self.doc_lengths) / self.n_docs if self.n_docs > 0 else 0
        )

        # Compute IDF for all terms
        self._compute_idf()

        self._indexed = True
        elapsed = (time.time() - start_time) * 1000
        logger.info(f"BM25 indexed {self.n_docs} documents in {elapsed:.1f}ms")

    def _compute_idf(self) -> None:
        """Compute IDF scores for all terms."""
        self.idf = {}
        for term, df in self.doc_freqs.items():
            # IDF with smoothing
            idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)
            self.idf[term] = idf

    def _score_document(
        self,
        query_tokens: List[str],
        doc_idx: int,
    ) -> float:
        """Compute BM25 score for a single document."""
        doc = self.documents[doc_idx]
        doc_length = self.doc_lengths[doc_idx]

        # Term frequency in document
        tf_doc = Counter(doc)

        score = 0.0
        for term in query_tokens:
            if term not in self.idf:
                continue

            tf = tf_doc.get(term, 0)
            if tf == 0:
                continue

            idf = self.idf[term]

            # BM25 scoring formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (
                1 - self.b + self.b * (doc_length / self.avg_doc_length)
            )
            score += idf * (numerator / denominator)

        return score

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Tuple[int, float]]:
        """
        Retrieve top-k documents for query.

        Args:
            query: Search query string
            top_k: Number of results to return

        Returns:
            List of (doc_index, score) tuples, sorted by score descending
        """
        if not self._indexed:
            raise RuntimeError("Index not built. Call index() first.")

        query_tokens = self._tokenize(query)

        # Score all documents
        scores = []
        for doc_idx in range(self.n_docs):
            score = self._score_document(query_tokens, doc_idx)
            if score > 0:
                scores.append((doc_idx, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]


# =============================================================================
# Vector Retriever
# =============================================================================


class VectorRetriever:
    """
    Dense vector retrieval using Sentence Transformers.

    Provides semantic similarity search beyond keyword matching.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "auto",
        batch_size: int = 32,
    ):
        """
        Initialize vector retriever.

        Args:
            model_name: Sentence Transformer model name
            device: Device to use (auto, cuda, mps, cpu)
            batch_size: Batch size for encoding
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = self._resolve_device(device)

        self.model = None
        self.embeddings: Optional[np.ndarray] = None
        self.chunk_ids: List[str] = []

        self._model_loaded = False

    def _resolve_device(self, device: str) -> str:
        """Resolve device string to actual device."""
        if device != "auto":
            return device

        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            # Check MPS availability with proper version handling
            try:
                if (
                    hasattr(torch.backends, "mps") and
                    hasattr(torch.backends.mps, "is_available") and
                    torch.backends.mps.is_available()
                ):
                    return "mps"
            except (AttributeError, RuntimeError):
                # Older PyTorch versions or MPS not supported
                pass
        except ImportError:
            pass

        return "cpu"

    def _load_model(self) -> None:
        """Load the sentence transformer model."""
        if self._model_loaded:
            return

        try:
            from sentence_transformers import SentenceTransformer

            start_time = time.time()
            self.model = SentenceTransformer(self.model_name, device=self.device)
            elapsed = time.time() - start_time

            logger.info(
                f"Loaded {self.model_name} on {self.device} in {elapsed:.2f}s"
            )
            self._model_loaded = True

        except ImportError:
            raise ImportError(
                "sentence-transformers required for vector search. "
                "Install with: pip install sentence-transformers"
            )

    def index(
        self,
        documents: List[str],
        chunk_ids: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Index documents by computing embeddings.

        Args:
            documents: List of document strings to embed
            chunk_ids: Optional list of chunk IDs

        Returns:
            numpy array of embeddings (n_docs, embedding_dim)
        """
        self._load_model()

        start_time = time.time()

        # Encode in batches
        embeddings = self.model.encode(
            documents,
            batch_size=self.batch_size,
            show_progress_bar=len(documents) > 100,
            convert_to_numpy=True,
            normalize_embeddings=True,  # For cosine similarity
        )

        self.embeddings = embeddings
        self.chunk_ids = chunk_ids or [str(i) for i in range(len(documents))]

        elapsed = (time.time() - start_time) * 1000
        logger.info(
            f"Embedded {len(documents)} documents in {elapsed:.1f}ms "
            f"(shape: {embeddings.shape})"
        )

        return embeddings

    def set_embeddings(
        self,
        embeddings: np.ndarray,
        chunk_ids: List[str],
    ) -> None:
        """
        Set pre-computed embeddings (from cache).

        Args:
            embeddings: Pre-computed embeddings array
            chunk_ids: Corresponding chunk IDs
        """
        self.embeddings = embeddings
        self.chunk_ids = chunk_ids
        logger.info(f"Loaded {len(chunk_ids)} cached embeddings")

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Tuple[int, float]]:
        """
        Retrieve top-k documents by semantic similarity.

        Args:
            query: Search query string
            top_k: Number of results to return

        Returns:
            List of (doc_index, score) tuples, sorted by score descending
        """
        if self.embeddings is None:
            raise RuntimeError("Embeddings not computed. Call index() first.")

        self._load_model()

        # Encode query
        query_embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        # Compute cosine similarity (embeddings are normalized)
        similarities = np.dot(self.embeddings, query_embedding)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = [
            (int(idx), float(similarities[idx]))
            for idx in top_indices
        ]

        return results

    def get_embeddings(self) -> Optional[np.ndarray]:
        """Get computed embeddings."""
        return self.embeddings

    def get_chunk_ids(self) -> List[str]:
        """Get chunk IDs corresponding to embeddings."""
        return self.chunk_ids


# =============================================================================
# Enhanced Hybrid Retriever
# =============================================================================


class EnhancedHybridRetriever:
    """
    Enhanced hybrid retriever combining BM25 and vector search.

    Features:
    - BM25 sparse retrieval for keyword matching
    - Dense vector embeddings for semantic similarity
    - Configurable hybrid scoring
    - Query result caching
    - Embedding persistence support
    """

    def __init__(self, config: Optional[RetrieverConfig] = None):
        """
        Initialize the enhanced hybrid retriever.

        Args:
            config: Retriever configuration
        """
        self.config = config or RetrieverConfig()

        # Initialize retrievers
        self.bm25 = BM25Retriever(
            k1=self.config.bm25_k1,
            b=self.config.bm25_b,
        )

        self.vector: Optional[VectorRetriever] = None
        if self.config.enable_vector_search:
            self.vector = VectorRetriever(
                model_name=self.config.vector_model,
                device=self.config.use_gpu,
                batch_size=self.config.embedding_batch_size,
            )

        # Chunk storage
        self.chunks: List[Any] = []
        self.chunk_texts: List[str] = []
        self.chunk_ids: List[str] = []

        # Statistics
        self.stats = RetrievalStats()

        # Query cache - using OrderedDict for proper LRU behavior
        from collections import OrderedDict
        self._query_cache: OrderedDict[str, List[RetrievalResult]] = OrderedDict()

        self._indexed = False

        logger.info(
            f"EnhancedHybridRetriever initialized "
            f"(vector_search={self.config.enable_vector_search})"
        )

    def _extract_text(self, chunk: Any) -> str:
        """Extract text content from chunk object."""
        if isinstance(chunk, str):
            return chunk
        elif hasattr(chunk, "content"):
            return chunk.content
        elif hasattr(chunk, "text"):
            return chunk.text
        elif isinstance(chunk, dict):
            return chunk.get("content", chunk.get("text", str(chunk)))
        else:
            return str(chunk)

    def _extract_chunk_id(self, chunk: Any, index: int) -> str:
        """Extract or generate chunk ID."""
        if hasattr(chunk, "chunk_id"):
            return chunk.chunk_id
        elif hasattr(chunk, "id"):
            return chunk.id
        elif isinstance(chunk, dict):
            return chunk.get("chunk_id", chunk.get("id", str(index)))
        else:
            return str(index)

    def index(self, chunks: List[Any]) -> None:
        """
        Index chunks for retrieval.

        Args:
            chunks: List of chunk objects to index
        """
        start_time = time.time()

        self.chunks = chunks
        self.chunk_texts = [self._extract_text(c) for c in chunks]
        self.chunk_ids = [
            self._extract_chunk_id(c, i) for i, c in enumerate(chunks)
        ]

        # Index BM25
        self.bm25.index(self.chunk_texts)

        # Index vectors if enabled
        if self.config.enable_vector_search and self.vector:
            self.vector.index(self.chunk_texts, self.chunk_ids)

        # Clear query cache on re-index
        self._query_cache.clear()

        self._indexed = True

        elapsed = (time.time() - start_time) * 1000
        logger.info(f"Indexed {len(chunks)} chunks in {elapsed:.1f}ms")

    def load_cached_embeddings(
        self,
        embeddings: np.ndarray,
        chunk_ids: List[str],
    ) -> None:
        """
        Load pre-computed embeddings from cache.

        Args:
            embeddings: Cached embeddings array
            chunk_ids: Corresponding chunk IDs

        Returns:
            True if embeddings were loaded successfully, False otherwise
        """
        if not self.config.enable_vector_search or not self.vector:
            logger.warning("Vector search disabled, ignoring cached embeddings")
            return False

        # Validate chunk IDs match
        if chunk_ids != self.chunk_ids:
            logger.warning(
                "Cached embeddings chunk IDs don't match indexed chunks. "
                "Recomputing embeddings."
            )
            return False

        self.vector.set_embeddings(embeddings, chunk_ids)
        logger.info("Loaded cached embeddings successfully")
        return True

    def _normalize_scores(
        self,
        scores: List[Tuple[int, float]],
    ) -> Dict[int, float]:
        """Normalize scores to [0, 1] range."""
        if not scores:
            return {}

        max_score = max(s[1] for s in scores) if scores else 1.0
        if max_score == 0:
            max_score = 1.0

        return {idx: score / max_score for idx, score in scores}

    def _combine_scores(
        self,
        bm25_scores: Dict[int, float],
        vector_scores: Dict[int, float],
    ) -> Dict[int, float]:
        """Combine BM25 and vector scores with configured weights."""
        all_indices = set(bm25_scores.keys()) | set(vector_scores.keys())

        combined = {}
        for idx in all_indices:
            bm25_score = bm25_scores.get(idx, 0.0)
            vector_score = vector_scores.get(idx, 0.0)

            combined[idx] = (
                self.config.bm25_weight * bm25_score +
                self.config.vector_weight * vector_score
            )

        return combined

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        chunk_type_filter: Optional[Set[str]] = None,
        file_path_filter: Optional[str] = None,
        method: Optional[str] = None,  # bm25, vector, hybrid
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks for query.

        Args:
            query: Search query string
            top_k: Number of results to return
            chunk_type_filter: Filter by chunk type (code, doc, test)
            file_path_filter: Filter by file path regex
            method: Force retrieval method (bm25, vector, hybrid)

        Returns:
            List of RetrievalResult objects, sorted by score descending
        """
        if not self._indexed:
            raise RuntimeError("Index not built. Call index() first.")

        top_k = top_k or self.config.top_k_default
        start_time = time.time()

        # Check cache
        cache_key = f"{query}:{top_k}:{method}"
        if self.config.query_cache_enabled and cache_key in self._query_cache:
            self.stats.cache_hits += 1
            return self._query_cache[cache_key]

        self.stats.cache_misses += 1

        # Determine retrieval method
        use_vector = (
            self.config.enable_vector_search and
            self.vector is not None and
            method != "bm25"
        )
        use_bm25 = method != "vector"

        # BM25 retrieval
        bm25_results = []
        if use_bm25:
            bm25_results = self.bm25.retrieve(query, top_k=top_k * 2)
            self.stats.bm25_queries += 1

        # Vector retrieval
        vector_results = []
        if use_vector:
            vector_results = self.vector.retrieve(query, top_k=top_k * 2)
            self.stats.vector_queries += 1

        # Combine scores
        if use_bm25 and use_vector:
            bm25_normalized = self._normalize_scores(bm25_results)
            vector_normalized = self._normalize_scores(vector_results)
            combined_scores = self._combine_scores(bm25_normalized, vector_normalized)
            retrieval_method = "hybrid"
            self.stats.hybrid_queries += 1
        elif use_vector:
            combined_scores = dict(vector_results)
            retrieval_method = "vector"
        else:
            combined_scores = dict(bm25_results)
            retrieval_method = "bm25"

        # Sort by combined score
        sorted_indices = sorted(
            combined_scores.keys(),
            key=lambda x: combined_scores[x],
            reverse=True,
        )

        # Build results
        results = []
        for idx in sorted_indices[:top_k * 2]:  # Get extra for filtering
            chunk = self.chunks[idx]

            # Extract metadata
            file_path = ""
            line_start = 0
            line_end = 0
            chunk_type = ""

            if hasattr(chunk, "file_path"):
                file_path = chunk.file_path
            elif isinstance(chunk, dict):
                file_path = chunk.get("file_path", "")

            if hasattr(chunk, "line_start"):
                line_start = chunk.line_start
            elif isinstance(chunk, dict):
                line_start = chunk.get("line_start", 0)

            if hasattr(chunk, "line_end"):
                line_end = chunk.line_end
            elif isinstance(chunk, dict):
                line_end = chunk.get("line_end", 0)

            if hasattr(chunk, "chunk_type"):
                chunk_type = chunk.chunk_type
            elif isinstance(chunk, dict):
                chunk_type = chunk.get("chunk_type", "")

            # Apply filters
            if chunk_type_filter and chunk_type not in chunk_type_filter:
                continue

            if file_path_filter:
                import re
                if not re.search(file_path_filter, file_path):
                    continue

            # Get individual scores
            bm25_score = dict(bm25_results).get(idx, 0.0) if bm25_results else 0.0
            vector_score = (
                dict(vector_results).get(idx, 0.0) if vector_results else 0.0
            )

            result = RetrievalResult(
                chunk_id=self.chunk_ids[idx],
                chunk=chunk,
                score=combined_scores[idx],
                bm25_score=bm25_score,
                vector_score=vector_score,
                retrieval_method=retrieval_method,
                file_path=file_path,
                line_start=line_start,
                line_end=line_end,
                chunk_type=chunk_type,
            )
            results.append(result)

            if len(results) >= top_k:
                break

        # Update statistics
        elapsed_ms = (time.time() - start_time) * 1000
        self.stats.total_queries += 1
        self.stats.avg_query_time_ms = (
            (
                self.stats.avg_query_time_ms * (self.stats.total_queries - 1)
                + elapsed_ms
            )
            / self.stats.total_queries
        )
        self.stats.avg_results_returned = (
            (
                self.stats.avg_results_returned * (self.stats.total_queries - 1)
                + len(results)
            )
            / self.stats.total_queries
        )

        # Cache results using OrderedDict for proper LRU
        if self.config.query_cache_enabled:
            # Move to end if exists (most recently used)
            if cache_key in self._query_cache:
                self._query_cache.move_to_end(cache_key)
            else:
                # Evict oldest (first) if cache is full
                if len(self._query_cache) >= self.config.query_cache_size:
                    self._query_cache.popitem(last=False)
                self._query_cache[cache_key] = results

        logger.debug(
            f"Retrieved {len(results)} results for '{query[:50]}...' "
            f"in {elapsed_ms:.1f}ms ({retrieval_method})"
        )

        return results

    def get_embeddings(self) -> Optional[Tuple[np.ndarray, List[str]]]:
        """
        Get computed embeddings for persistence.

        Returns:
            Tuple of (embeddings_array, chunk_ids) or None
        """
        if not self.vector:
            return None

        embeddings = self.vector.get_embeddings()
        if embeddings is None:
            return None

        return embeddings, self.chunk_ids

    def save_embeddings(self, cache_manager: Any) -> bool:
        """
        Save embeddings to cache manager.

        Args:
            cache_manager: CacheManager instance

        Returns:
            True if save was successful
        """
        result = self.get_embeddings()
        if result is None:
            logger.warning("No embeddings to save")
            return False

        embeddings, chunk_ids = result
        return cache_manager.save_embeddings(embeddings, chunk_ids)

    def get_statistics(self) -> Dict[str, Any]:
        """Get retrieval statistics."""
        return {
            "total_queries": self.stats.total_queries,
            "cache_hits": self.stats.cache_hits,
            "cache_misses": self.stats.cache_misses,
            "cache_hit_rate": (
                self.stats.cache_hits /
                max(1, self.stats.cache_hits + self.stats.cache_misses)
            ),
            "avg_query_time_ms": self.stats.avg_query_time_ms,
            "avg_results_returned": self.stats.avg_results_returned,
            "bm25_queries": self.stats.bm25_queries,
            "vector_queries": self.stats.vector_queries,
            "hybrid_queries": self.stats.hybrid_queries,
            "indexed_chunks": len(self.chunks),
            "vector_search_enabled": self.config.enable_vector_search,
        }

    def clear_cache(self) -> None:
        """Clear query result cache."""
        self._query_cache.clear()
        logger.info("Query cache cleared")


# =============================================================================
# Convenience Functions
# =============================================================================


def create_retriever(
    enable_vector_search: bool = True,
    bm25_weight: float = 0.6,
    vector_weight: float = 0.4,
    model_name: str = "all-MiniLM-L6-v2",
) -> EnhancedHybridRetriever:
    """
    Create an enhanced hybrid retriever with common defaults.

    Args:
        enable_vector_search: Whether to enable semantic search
        bm25_weight: Weight for BM25 scores
        vector_weight: Weight for vector scores
        model_name: Sentence Transformer model name

    Returns:
        Configured EnhancedHybridRetriever instance
    """
    config = RetrieverConfig(
        enable_vector_search=enable_vector_search,
        bm25_weight=bm25_weight,
        vector_weight=vector_weight,
        vector_model=model_name,
    )
    return EnhancedHybridRetriever(config)


# =============================================================================
# CLI Interface
# =============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Enhanced Hybrid Retriever CLI"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run basic functionality test",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Test query string",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results",
    )
    parser.add_argument(
        "--no-vector",
        action="store_true",
        help="Disable vector search",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.test:
        # Basic functionality test
        print("\n=== Enhanced Hybrid Retriever Test ===\n")

        # Create test documents
        test_docs = [
            "The depth pipeline processes images using Depth Anything V2",
            "Material response technology enhances wood, metal, and glass surfaces",
            "Atmospheric effects include fog, haze, and depth-based blur",
            "Color grading uses LUTs for film emulation and tone mapping",
            "The batch processor handles 400-600 images per hour",
            "CoreML provides 3-5x speedup on Apple Silicon M-series chips",
            "FFmpeg filter graphs process HDR video with tone mapping",
            "pytest fixtures provide common setup for testing pipelines",
        ]

        # Create retriever
        retriever = create_retriever(
            enable_vector_search=not args.no_vector,
            bm25_weight=0.6,
            vector_weight=0.4,
        )

        # Index documents
        retriever.index(test_docs)

        # Test query
        query = args.query or "depth-based atmospheric effects"
        print(f"Query: {query}\n")

        results = retriever.retrieve(query, top_k=args.top_k)

        print("Results:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Score: {result.score:.4f} ({result.retrieval_method})")
            print(f"   BM25: {result.bm25_score:.4f}, Vector: {result.vector_score:.4f}")
            print(f"   Content: {result.chunk[:80]}...")

        # Show statistics
        print("\n=== Statistics ===")
        stats = retriever.get_statistics()
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

        print("\n=== Test Complete ===")
