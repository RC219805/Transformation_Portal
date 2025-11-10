#!/usr/bin/env python3
"""
Simple wrapper to query the RAG system with proper imports.
Usage: ./rag_query.py "your question here"
"""

import sys
from pathlib import Path

# Add RAG system to path
sys.path.insert(0, str(Path(__file__).parent / ".github" / "agents"))

from rag_system.citation import CitationGenerator
from rag_system.retriever import HybridRetriever
from rag_system.indexer import RepositoryIndexer

def main():
    if len(sys.argv) < 2:
        print("Usage: ./rag_query.py 'your question here'")
        sys.exit(1)

    query = " ".join(sys.argv[1:])

    # Load index
    repo = Path(__file__).parent
    indexer = RepositoryIndexer(str(repo))
    chunks = indexer.index_repository()

    # Retrieve results
    retriever = HybridRetriever()
    retriever.index(chunks)
    results = retriever.retrieve(query, top_k=5)

    # Generate citations
    citation_gen = CitationGenerator()

    print(f"\n{'='*80}")
    print(f"Query: {query}")
    print(f"{'='*80}\n")

    for i, result in enumerate(results, 1):
        print(f"\n[Result {i}] Score: {result.score:.3f}")
        print(f"File: {result.file_path}")
        print(f"Lines: {result.start_line}-{result.end_line}")
        print(f"Method: {result.retrieval_method}")
        print(f"\nContent:\n{result.content[:500]}...")
        print("-" * 80)

    print(f"\n✅ Found {len(results)} relevant results from {len(chunks)} indexed chunks\n")

if __name__ == "__main__":
    main()
