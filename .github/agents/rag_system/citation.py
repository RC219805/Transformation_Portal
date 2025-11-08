"""
Citation Generator for RAG System

Generates citations with file paths, snippets, and confidence scores.
"""

import textwrap
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Citation:
    """A citation with source information and confidence."""

    file_path: str
    start_line: int
    end_line: int
    snippet: str
    confidence: float  # 0.0 to 1.0
    relevance_note: Optional[str] = None


class CitationGenerator:
    """
    Generates structured citations from retrieval results.

    Citations include:
    - File path and line numbers
    - Code/text snippet (trimmed for readability)
    - Confidence score based on retrieval rank and score
    - Relevance notes explaining why the citation is relevant
    """

    # Maximum expected BM25 score for normalization (typical scores range 0-20)
    MAX_EXPECTED_SCORE = 20.0

    def __init__(
        self,
        snippet_max_lines: int = 10,
        snippet_max_chars: int = 500,
    ):
        """
        Initialize citation generator.

        Args:
            snippet_max_lines: Maximum lines to include in snippet
            snippet_max_chars: Maximum characters in snippet
        """
        self.snippet_max_lines = snippet_max_lines
        self.snippet_max_chars = snippet_max_chars

    def generate_citations(
        self,
        results: List,
        max_citations: int = 5,
    ) -> List[Citation]:
        """
        Generate citations from retrieval results.

        Args:
            results: List of RetrievalResult objects
            max_citations: Maximum number of citations to generate

        Returns:
            List of Citation objects
        """
        citations = []

        for i, result in enumerate(results[:max_citations]):
            # Compute confidence based on rank and score
            confidence = self._compute_confidence(i, result.score, len(results))

            # Extract snippet
            snippet = self._extract_snippet(result.content)

            # Generate relevance note
            relevance_note = self._generate_relevance_note(result)

            citations.append(Citation(
                file_path=result.file_path,
                start_line=result.start_line,
                end_line=result.end_line,
                snippet=snippet,
                confidence=confidence,
                relevance_note=relevance_note,
            ))

        return citations

    def _compute_confidence(
        self,
        rank: int,
        score: float,
        total_results: int,
    ) -> float:
        """
        Compute confidence score for a result.

        Confidence is based on:
        - Rank in results (higher rank = higher confidence)
        - Retrieval score magnitude
        - Relative score compared to others
        """
        # Rank-based confidence (exponential decay)
        rank_confidence = 1.0 / (1.0 + rank * 0.3)

        # Score-based confidence (normalized to 0-1)
        score_confidence = min(1.0, score / self.MAX_EXPECTED_SCORE)

        # Combine with weights
        confidence = (rank_confidence * 0.6 + score_confidence * 0.4)

        return round(confidence, 2)

    def _extract_snippet(self, content: str) -> str:
        """Extract a readable snippet from content."""
        lines = content.split('\n')

        # Limit lines
        if len(lines) > self.snippet_max_lines:
            lines = lines[:self.snippet_max_lines]
            truncated = True
        else:
            truncated = False

        snippet = '\n'.join(lines)

        # Limit characters
        if len(snippet) > self.snippet_max_chars:
            snippet = snippet[:self.snippet_max_chars]
            truncated = True

        # Add truncation indicator
        if truncated:
            snippet += '\n...'

        return snippet

    def _generate_relevance_note(self, result) -> str:
        """Generate a note explaining why this result is relevant."""
        notes = []

        # Check metadata for specific indicators
        if result.metadata.get('entity_type') == 'function':
            func_name = result.metadata.get('function_name', 'unknown')
            notes.append(f"Function: {func_name}")
        elif result.metadata.get('entity_type') == 'class':
            class_name = result.metadata.get('class_name', 'unknown')
            notes.append(f"Class: {class_name}")

        if result.metadata.get('docstring'):
            notes.append("Has documentation")

        if result.metadata.get('document_type'):
            doc_type = result.metadata['document_type']
            notes.append(f"Type: {doc_type}")

        if result.retrieval_method == 'bm25':
            notes.append("Text match")
        elif result.retrieval_method == 'vector':
            notes.append("Semantic match")
        elif result.retrieval_method == 'hybrid':
            notes.append("Hybrid match")

        return " | ".join(notes) if notes else "Relevant match"

    def format_citations(
        self,
        citations: List[Citation],
        format_type: str = 'markdown',
    ) -> str:
        """
        Format citations for display.

        Args:
            citations: List of Citation objects
            format_type: 'markdown', 'text', or 'json'

        Returns:
            Formatted citation string
        """
        if format_type == 'markdown':
            return self._format_markdown(citations)
        elif format_type == 'text':
            return self._format_text(citations)
        elif format_type == 'json':
            return self._format_json(citations)
        else:
            raise ValueError(f"Unknown format type: {format_type}")

    def _format_markdown(self, citations: List[Citation]) -> str:
        """Format citations as markdown."""
        lines = ["## Citations\n"]

        for i, cite in enumerate(citations, 1):
            lines.append(f"### [{i}] {cite.file_path}:{cite.start_line}-{cite.end_line}")
            lines.append(f"**Confidence**: {cite.confidence:.0%}")
            if cite.relevance_note:
                lines.append(f"**Relevance**: {cite.relevance_note}")
            lines.append("\n```")
            lines.append(cite.snippet)
            lines.append("```\n")

        return "\n".join(lines)

    def _format_text(self, citations: List[Citation]) -> str:
        """Format citations as plain text."""
        lines = ["CITATIONS\n" + "=" * 80 + "\n"]

        for i, cite in enumerate(citations, 1):
            lines.append(f"[{i}] {cite.file_path}:{cite.start_line}-{cite.end_line}")
            lines.append(f"    Confidence: {cite.confidence:.0%}")
            if cite.relevance_note:
                lines.append(f"    Relevance: {cite.relevance_note}")
            lines.append("\n    Snippet:")
            # Indent snippet
            indented = textwrap.indent(cite.snippet, "    ")
            lines.append(indented)
            lines.append("")

        return "\n".join(lines)

    def _format_json(self, citations: List[Citation]) -> str:
        """Format citations as JSON."""
        import json

        citations_dict = [
            {
                'file_path': cite.file_path,
                'start_line': cite.start_line,
                'end_line': cite.end_line,
                'snippet': cite.snippet,
                'confidence': cite.confidence,
                'relevance_note': cite.relevance_note,
            }
            for cite in citations
        ]

        return json.dumps({'citations': citations_dict}, indent=2)


def main():
    """CLI for testing citation generation."""
    import argparse
    import os
    import sys

    # Add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from rag_system.indexer import RepositoryIndexer
    from rag_system.reranker import ResultReranker
    from rag_system.retriever import HybridRetriever

    parser = argparse.ArgumentParser(description='Test citation generation')
    parser.add_argument('--repo-root', default='.', help='Repository root directory')
    parser.add_argument('--query', required=True, help='Search query')
    parser.add_argument('--max-citations', type=int, default=5, help='Max citations')
    parser.add_argument('--format', choices=['markdown', 'text', 'json'],
                        default='markdown', help='Output format')

    args = parser.parse_args()

    # Full RAG pipeline
    print("Indexing repository...")
    indexer = RepositoryIndexer(args.repo_root)
    chunks = indexer.index_repository()

    print("Retrieving...")
    retriever = HybridRetriever()
    retriever.index(chunks)
    results = retriever.retrieve(args.query, top_k=args.max_citations * 2)

    print("Reranking...")
    reranker = ResultReranker()
    reranked = reranker.rerank(results, args.query)

    print("Generating citations...\n")
    generator = CitationGenerator()
    citations = generator.generate_citations(reranked, max_citations=args.max_citations)

    # Format and display
    formatted = generator.format_citations(citations, format_type=args.format)
    print(formatted)


if __name__ == '__main__':
    main()
