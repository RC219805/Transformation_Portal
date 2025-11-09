"""
Hybrid Retriever for RAG System

Implements hybrid retrieval using BM25 (sparse) and dense vector embeddings
to ensure both recall and precision.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import get_config
from exceptions import RetrievalError
from logger import get_logger

logger = get_logger(__name__)

# Optional sentence-transformers import
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning(
        "sentence-transformers not installed. Vector search will be disabled. "
        "Install with: pip install sentence-transformers"
    )


@dataclass
class RetrievalResult:
    """Result from retrieval with scoring information."""

    chunk_id: str
    content: str
    file_path: str
    start_line: int
    end_line: int
    score: float
    retrieval_method: str  # 'bm25', 'vector', 'hybrid'
    metadata: Dict


class BM25Retriever:
    """
    BM25 sparse retrieval implementation.

    BM25 is a probabilistic ranking function that considers:
    - Term frequency (TF)
    - Inverse document frequency (IDF)
    - Document length normalization
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 retriever.

        Args:
            k1: Controls term frequency saturation (typically 1.2-2.0)
            b: Controls length normalization (typically 0.75)
        """
        self.k1 = k1
        self.b = b
        self.corpus = []
        self.doc_freqs = Counter()
        self.idf = {}
        self.doc_lens = []
        self.avgdl = 0
        self.N = 0

    def fit(self, documents: List[str]):
        """
        Fit the BM25 model on a corpus of documents.

        Args:
            documents: List of document strings
        """
        self.corpus = documents
        self.N = len(documents)

        # Tokenize and compute document frequencies
        tokenized_docs = [self._tokenize(doc) for doc in documents]
        self.doc_lens = [len(doc) for doc in tokenized_docs]
        self.avgdl = sum(self.doc_lens) / self.N if self.N > 0 else 0

        # Compute document frequencies
        for tokens in tokenized_docs:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] += 1

        # Compute IDF scores using Robertson-Sparck Jones (RSJ) IDF formula
        # The 0.5 smoothing constants prevent negative or undefined IDF values
        # for very rare or very common terms
        for token, freq in self.doc_freqs.items():
            self.idf[token] = math.log((self.N - freq + 0.5) / (freq + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Search for documents matching the query.

        Args:
            query: Search query string
            top_k: Number of top results to return

        Returns:
            List of (document_index, score) tuples
        """
        query_tokens = self._tokenize(query)
        scores = []

        for doc_idx, doc in enumerate(self.corpus):
            doc_tokens = self._tokenize(doc)
            token_freqs = Counter(doc_tokens)

            score = 0.0
            doc_len = self.doc_lens[doc_idx]

            for token in query_tokens:
                if token not in token_freqs:
                    continue

                tf = token_freqs[token]
                idf = self.idf.get(token, 0)

                # BM25 formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
                score += idf * (numerator / denominator)

            scores.append((doc_idx, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into lowercase terms."""
        # Simple tokenization: lowercase, split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens


class HybridRetriever:
    """
    Hybrid retrieval combining BM25 and vector similarity.

    For this implementation, we use BM25 as the primary retrieval method,
    with support for extending to dense vector embeddings when available.
    """

    def __init__(
        self,
        bm25_weight: Optional[float] = None,
        vector_weight: Optional[float] = None,
        enable_vector_search: Optional[bool] = None,
        vector_model: Optional[str] = None,
    ):
        """
        Initialize hybrid retriever.

        Args:
            bm25_weight: Weight for BM25 scores (0-1, uses config if None)
            vector_weight: Weight for vector similarity (0-1, uses config if None)
            enable_vector_search: Enable vector embeddings (uses config if None)
            vector_model: Sentence transformer model name (uses config if None)
        """
        # Load config
        config = get_config()
        retriever_config = config.get_section('retriever')
        citation_config = config.get_section('citation')

        self.bm25_weight = bm25_weight if bm25_weight is not None else retriever_config.get('bm25_weight', 0.7)
        self.vector_weight = vector_weight if vector_weight is not None else retriever_config.get('vector_weight', 0.3)
        self.enable_vector_search = (
            enable_vector_search
            if enable_vector_search is not None
            else retriever_config.get('enable_vector_search', False)
        )
        
        # Get max expected score for normalization
        self.max_expected_score = citation_config.get('max_expected_score', 20.0)

        # BM25 retriever
        bm25_k1 = retriever_config.get('bm25_k1', 1.5)
        bm25_b = retriever_config.get('bm25_b', 0.75)
        self.bm25 = BM25Retriever(k1=bm25_k1, b=bm25_b)

        # Vector search components
        self.encoder = None
        self.embeddings = None

        if self.enable_vector_search and SENTENCE_TRANSFORMERS_AVAILABLE:
            model_name = vector_model or retriever_config.get('vector_model', 'all-MiniLM-L6-v2')
            try:
                logger.info(f"Loading sentence transformer model: {model_name}")
                self.encoder = SentenceTransformer(model_name)
                logger.info("Vector search enabled")
            except Exception as e:
                logger.warning(f"Failed to load sentence transformer: {e}")
                self.enable_vector_search = False

        # State
        self.chunks = []
        self.indexed = False

        # Query cache size
        cache_size = retriever_config.get('query_cache_size', 100)
        if cache_size > 0:
            # Wrap retrieve method with LRU cache
            self._retrieve_cached = lru_cache(maxsize=cache_size)(self._retrieve_impl)
            logger.debug(f"Query caching enabled: max_size={cache_size}")
        else:
            self._retrieve_cached = self._retrieve_impl

        logger.debug(
            f"Initialized retriever: bm25_weight={self.bm25_weight}, "
            f"vector_weight={self.vector_weight}, "
            f"vector_search={self.enable_vector_search}"
        )

    def index(self, chunks: List):
        """
        Index document chunks for retrieval.

        Args:
            chunks: List of DocumentChunk objects
        """
        logger.info(f"Indexing {len(chunks)} chunks...")

        self.chunks = chunks
        documents = [chunk.content for chunk in chunks]

        # BM25 indexing
        self.bm25.fit(documents)

        # Vector indexing (if enabled)
        if self.enable_vector_search and self.encoder is not None:
            try:
                logger.info("Computing vector embeddings...")
                self.embeddings = self.encoder.encode(
                    documents,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                logger.info(f"Computed embeddings: shape={self.embeddings.shape}")
            except Exception as e:
                logger.warning(f"Failed to compute embeddings: {e}")
                self.embeddings = None

        self.indexed = True
        logger.info("Indexing complete")

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        chunk_type_filter: Optional[List[str]] = None,
        file_path_filter: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: Search query
            top_k: Number of results to return
            chunk_type_filter: Filter by chunk types (e.g., ['code', 'doc'])
            file_path_filter: Filter by file path pattern (regex)

        Returns:
            List of RetrievalResult objects
        """
        if not self.indexed:
            raise RetrievalError("Retriever not indexed. Call index() first.")

        # Create cache key for filters (make hashable)
        filter_key = (
            tuple(sorted(chunk_type_filter)) if chunk_type_filter else None,
            file_path_filter
        )

        # Call cached implementation
        return self._retrieve_cached(query, top_k, filter_key)

    def _retrieve_impl(
        self,
        query: str,
        top_k: int,
        filter_key: Optional[Tuple],
    ) -> List[RetrievalResult]:
        """
        Internal retrieval implementation (cacheable).

        Args:
            query: Search query
            top_k: Number of results to return
            filter_key: Hashable filter key (chunk_types, file_path_filter)

        Returns:
            List of RetrievalResult objects
        """
        # Extract filters from key
        chunk_type_filter = list(filter_key[0]) if filter_key and filter_key[0] else None
        file_path_filter = filter_key[1] if filter_key else None

        # Apply filters
        filtered_indices = self._apply_filters(chunk_type_filter, file_path_filter)

        if not filtered_indices:
            logger.debug("No chunks match filters")
            return []

        # Create filtered corpus
        filtered_chunks = [self.chunks[i] for i in filtered_indices]
        filtered_docs = [chunk.content for chunk in filtered_chunks]

        # Get BM25 results
        bm25_scores = self._bm25_search(filtered_docs, query, top_k)

        # Get vector results (if enabled)
        vector_scores = None
        if self.enable_vector_search and self.embeddings is not None:
            vector_scores = self._vector_search(filtered_indices, query, top_k)

        # Combine results
        results = self._combine_results(
            filtered_indices,
            bm25_scores,
            vector_scores,
            top_k
        )

        logger.debug(f"Retrieved {len(results)} results for query: '{query[:50]}'")
        return results

    def _bm25_search(
        self,
        documents: List[str],
        query: str,
        top_k: int
    ) -> Dict[int, float]:
        """
        Perform BM25 search.

        Returns:
            Dict mapping local index to BM25 score
        """
        temp_bm25 = BM25Retriever(k1=self.bm25.k1, b=self.bm25.b)
        temp_bm25.fit(documents)
        bm25_results = temp_bm25.search(query, top_k=top_k * 2)  # Get more for hybrid

        scores = {}
        for local_idx, score in bm25_results:
            if score > 0:
                scores[local_idx] = score

        return scores

    def _vector_search(
        self,
        filtered_indices: List[int],
        query: str,
        top_k: int
    ) -> Dict[int, float]:
        """
        Perform vector similarity search.

        Returns:
            Dict mapping local index to similarity score
        """
        try:
            # Encode query
            query_embedding = self.encoder.encode([query], convert_to_numpy=True)[0]

            # Get filtered embeddings
            filtered_embeddings = self.embeddings[filtered_indices]

            # Compute cosine similarity
            query_norm = np.linalg.norm(query_embedding)
            doc_norms = np.linalg.norm(filtered_embeddings, axis=1)

            # Avoid division by zero
            similarities = np.dot(filtered_embeddings, query_embedding) / (doc_norms * query_norm + 1e-8)

            # Get top-k indices
            top_indices = np.argsort(similarities)[::-1][:top_k * 2]

            scores = {}
            for local_idx in top_indices:
                score = float(similarities[local_idx])
                if score > 0:
                    scores[int(local_idx)] = score

            return scores

        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            return {}

    def _combine_results(
        self,
        filtered_indices: List[int],
        bm25_scores: Dict[int, float],
        vector_scores: Optional[Dict[int, float]],
        top_k: int
    ) -> List[RetrievalResult]:
        """
        Combine BM25 and vector scores.

        Returns:
            List of RetrievalResult objects sorted by combined score
        """
        combined_scores = {}

        # Add BM25 scores
        for local_idx, score in bm25_scores.items():
            combined_scores[local_idx] = self.bm25_weight * score

        # Add vector scores
        if vector_scores:
            for local_idx, score in vector_scores.items():
                # Normalize vector score to roughly match BM25 range
                normalized_score = score * self.max_expected_score
                combined_scores[local_idx] = (
                    combined_scores.get(local_idx, 0) +
                    self.vector_weight * normalized_score
                )

        # Sort by combined score
        sorted_indices = sorted(
            combined_scores.keys(),
            key=lambda idx: combined_scores[idx],
            reverse=True
        )[:top_k]

        # Create results
        results = []
        for local_idx in sorted_indices:
            original_idx = filtered_indices[local_idx]
            chunk = self.chunks[original_idx]

            # Determine retrieval method
            has_bm25 = local_idx in bm25_scores
            has_vector = vector_scores and local_idx in vector_scores

            if has_bm25 and has_vector:
                method = 'hybrid'
            elif has_vector:
                method = 'vector'
            else:
                method = 'bm25'

            results.append(RetrievalResult(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                file_path=chunk.file_path,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                score=combined_scores[local_idx],
                retrieval_method=method,
                metadata=chunk.metadata
            ))

        return results

    def _apply_filters(
        self,
        chunk_type_filter: Optional[List[str]],
        file_path_filter: Optional[str],
    ) -> List[int]:
        """Apply filters and return valid chunk indices."""
        valid_indices = []

        for i, chunk in enumerate(self.chunks):
            # Check chunk type filter
            if chunk_type_filter and chunk.chunk_type not in chunk_type_filter:
                continue

            # Check file path filter
            if file_path_filter and not re.search(file_path_filter, chunk.file_path):
                continue

            valid_indices.append(i)

        return valid_indices

    def get_context_window(
        self,
        chunk_id: str,
        window_size: int = 2,
    ) -> List[RetrievalResult]:
        """
        Get surrounding chunks for context.

        Args:
            chunk_id: ID of the central chunk
            window_size: Number of chunks to include before and after

        Returns:
            List of chunks in context window
        """
        # Find the chunk
        target_idx = None
        for i, chunk in enumerate(self.chunks):
            if chunk.chunk_id == chunk_id:
                target_idx = i
                break

        if target_idx is None:
            return []

        # Get surrounding chunks from same file
        target_file = self.chunks[target_idx].file_path
        context = []

        # Look backward
        for i in range(max(0, target_idx - window_size), target_idx):
            if self.chunks[i].file_path == target_file:
                context.append(self._chunk_to_result(self.chunks[i], 0.0, 'context'))

        # Add target
        context.append(self._chunk_to_result(self.chunks[target_idx], 1.0, 'target'))

        # Look forward
        for i in range(target_idx + 1, min(len(self.chunks), target_idx + window_size + 1)):
            if self.chunks[i].file_path == target_file:
                context.append(self._chunk_to_result(self.chunks[i], 0.0, 'context'))

        return context

    def _chunk_to_result(self, chunk, score: float, method: str) -> RetrievalResult:
        """Convert a chunk to a RetrievalResult."""
        return RetrievalResult(
            chunk_id=chunk.chunk_id,
            content=chunk.content,
            file_path=chunk.file_path,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            score=score,
            retrieval_method=method,
            metadata=chunk.metadata
        )


def main():
    """CLI for testing retrieval."""
    import argparse
    import os
    import sys

    # Add parent directory to path for imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from rag_system.indexer import RepositoryIndexer

    parser = argparse.ArgumentParser(description='Test RAG retrieval')
    parser.add_argument('--repo-root', default='.', help='Repository root directory')
    parser.add_argument('--query', required=True, help='Search query')
    parser.add_argument('--top-k', type=int, default=5, help='Number of results')
    parser.add_argument('--type', nargs='+', help='Filter by chunk type')
    parser.add_argument('--file', help='Filter by file path pattern')

    args = parser.parse_args()

    # Index repository
    print("Indexing repository...")
    indexer = RepositoryIndexer(args.repo_root)
    chunks = indexer.index_repository()
    print(f"Indexed {len(chunks)} chunks")

    # Create retriever
    print("\nPerforming retrieval...")
    retriever = HybridRetriever()
    retriever.index(chunks)

    # Search
    results = retriever.retrieve(
        query=args.query,
        top_k=args.top_k,
        chunk_type_filter=args.type,
        file_path_filter=args.file,
    )

    # Display results
    print(f"\nFound {len(results)} results for query: '{args.query}'")
    print("=" * 80)

    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result.file_path}:{result.start_line}-{result.end_line}")
        print(f"   Score: {result.score:.3f} | Method: {result.retrieval_method}")
        if result.metadata:
            print(f"   Metadata: {result.metadata}")
        print(f"   Preview: {result.content[:200]}...")


if __name__ == '__main__':
    main()
