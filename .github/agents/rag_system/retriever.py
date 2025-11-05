"""
Hybrid Retriever for RAG System

Implements hybrid retrieval using BM25 (sparse) and dense vector embeddings
to ensure both recall and precision.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import re
import math
from collections import Counter


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
        bm25_weight: float = 0.7,
        vector_weight: float = 0.3,
    ):
        """
        Initialize hybrid retriever.

        Args:
            bm25_weight: Weight for BM25 scores (0-1)
            vector_weight: Weight for vector similarity scores (0-1)
        """
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.bm25 = BM25Retriever()
        self.chunks = []
        self.indexed = False

    def index(self, chunks: List):
        """
        Index document chunks for retrieval.

        Args:
            chunks: List of DocumentChunk objects
        """
        self.chunks = chunks
        documents = [chunk.content for chunk in chunks]
        self.bm25.fit(documents)
        self.indexed = True

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
            raise ValueError("Retriever not indexed. Call index() first.")

        # Apply filters
        filtered_indices = self._apply_filters(chunk_type_filter, file_path_filter)

        if not filtered_indices:
            return []

        # Create filtered corpus for BM25
        filtered_chunks = [self.chunks[i] for i in filtered_indices]
        filtered_docs = [chunk.content for chunk in filtered_chunks]

        # Perform BM25 search on filtered corpus
        temp_bm25 = BM25Retriever()
        temp_bm25.fit(filtered_docs)
        bm25_results = temp_bm25.search(query, top_k=top_k)

        # Convert to RetrievalResult objects
        results = []
        for local_idx, score in bm25_results:
            if score > 0:  # Only include results with positive scores
                original_idx = filtered_indices[local_idx]
                chunk = self.chunks[original_idx]
                results.append(RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    content=chunk.content,
                    file_path=chunk.file_path,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    score=score,
                    retrieval_method='bm25',
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
    import sys
    import os

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
