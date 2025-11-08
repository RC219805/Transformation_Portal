"""
Result Reranker for RAG System

Reranks retrieval results to improve precision using additional signals.
"""

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RerankingSignal:
    """Signals used for reranking."""

    exact_match_bonus: float = 2.0
    recency_bonus: float = 0.5
    code_quality_bonus: float = 0.3
    documentation_bonus: float = 0.2
    test_bonus: float = 0.1


class ResultReranker:
    """
    Reranks retrieval results using multiple signals.

    Signals considered:
    - Exact phrase matches in content
    - Recency (for changelogs, docs)
    - Code quality indicators (docstrings, type hints)
    - Documentation completeness
    - Test coverage relevance
    """

    def __init__(self, signals: Optional[RerankingSignal] = None):
        """
        Initialize reranker.

        Args:
            signals: Reranking signal weights
        """
        self.signals = signals or RerankingSignal()

    def rerank(
        self,
        results: List,
        query: str,
        top_k: Optional[int] = None,
    ) -> List:
        """
        Rerank retrieval results.

        Args:
            results: List of RetrievalResult objects
            query: Original query string
            top_k: Number of top results to return (None = all)

        Returns:
            Reranked list of results
        """
        if not results:
            return results

        # Compute reranking scores
        reranked = []
        for result in results:
            rerank_score = self._compute_rerank_score(result, query)
            # Combine with original retrieval score
            final_score = result.score + rerank_score

            # Create modified result with updated score
            result.score = final_score
            result.metadata['rerank_boost'] = rerank_score
            reranked.append(result)

        # Sort by final score
        reranked.sort(key=lambda x: x.score, reverse=True)

        if top_k:
            return reranked[:top_k]
        return reranked

    def _compute_rerank_score(self, result, query: str) -> float:
        """Compute reranking score for a result."""
        score = 0.0

        # Exact match bonus
        score += self._exact_match_score(result.content, query)

        # Code quality bonus
        if result.metadata.get('entity_type') in ('function', 'class'):
            score += self._code_quality_score(result)

        # Documentation bonus
        if result.file_path.endswith('.md'):
            score += self._documentation_score(result)

        # Test relevance bonus
        if 'test' in result.file_path:
            score += self._test_relevance_score(result, query)

        return score

    def _exact_match_score(self, content: str, query: str) -> float:
        """Score based on exact phrase matches."""
        content_lower = content.lower()
        query_lower = query.lower()

        # Exact query match
        if query_lower in content_lower:
            return self.signals.exact_match_bonus

        # Partial matches (query terms)
        query_terms = re.findall(r'\b\w+\b', query_lower)
        if len(query_terms) < 2:
            return 0.0

        matches = sum(1 for term in query_terms if term in content_lower)
        match_ratio = matches / len(query_terms)

        return self.signals.exact_match_bonus * match_ratio * 0.5

    def _code_quality_score(self, result) -> float:
        """Score based on code quality indicators."""
        score = 0.0
        content = result.content

        # Has docstring - check for docstrings at start of functions/classes
        # Use regex to match docstrings after def/class declarations to avoid false positives
        docstring_pattern = r'(?:def|class)\s+\w+[^:]*:\s*(?:"""|\'\'\')(.*?)(?:"""|\'\'\')'
        if re.search(docstring_pattern, content, re.DOTALL):
            score += self.signals.code_quality_bonus * 0.5

        # Has type hints
        if '->' in content or ': ' in content:
            score += self.signals.code_quality_bonus * 0.3

        # Has meaningful variable names (longer than 3 chars)
        vars_match = re.findall(r'\b[a-z_][a-z0-9_]{3,}\b', content)
        if len(vars_match) > 5:
            score += self.signals.code_quality_bonus * 0.2

        return score

    def _documentation_score(self, result) -> float:
        """Score based on documentation completeness."""
        score = 0.0
        content = result.content

        # Has title
        if result.metadata.get('title'):
            score += self.signals.documentation_bonus * 0.3

        # Has code examples
        if '```' in content:
            score += self.signals.documentation_bonus * 0.4

        # Has links
        if re.search(r'\[.*?\]\(.*?\)', content):
            score += self.signals.documentation_bonus * 0.3

        return score

    def _test_relevance_score(self, result, query: str) -> float:
        """Score based on test relevance."""
        score = 0.0

        # Test function name matches query
        if result.metadata.get('function_name'):
            func_name = result.metadata['function_name']
            if any(term in func_name.lower() for term in query.lower().split()):
                score += self.signals.test_bonus

        return score


def main():
    """CLI for testing reranking."""
    import argparse
    import os
    import sys

    # Add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from rag_system.indexer import RepositoryIndexer
    from rag_system.retriever import HybridRetriever

    parser = argparse.ArgumentParser(description='Test RAG reranking')
    parser.add_argument('--repo-root', default='.', help='Repository root directory')
    parser.add_argument('--query', required=True, help='Search query')
    parser.add_argument('--top-k', type=int, default=5, help='Number of results')

    args = parser.parse_args()

    # Index and retrieve
    print("Indexing repository...")
    indexer = RepositoryIndexer(args.repo_root)
    chunks = indexer.index_repository()

    print("Retrieving...")
    retriever = HybridRetriever()
    retriever.index(chunks)
    results = retriever.retrieve(args.query, top_k=args.top_k * 2)

    print(f"\nBefore reranking ({len(results)} results):")
    for i, r in enumerate(results[:args.top_k], 1):
        print(f"{i}. {r.file_path}:{r.start_line} - Score: {r.score:.3f}")

    # Rerank
    print("\nReranking...")
    reranker = ResultReranker()
    reranked = reranker.rerank(results, args.query, top_k=args.top_k)

    print(f"\nAfter reranking ({len(reranked)} results):")
    for i, r in enumerate(reranked, 1):
        boost = r.metadata.get('rerank_boost', 0.0)
        print(f"{i}. {r.file_path}:{r.start_line} - Score: {r.score:.3f} (boost: {boost:+.3f})")


if __name__ == '__main__':
    main()
