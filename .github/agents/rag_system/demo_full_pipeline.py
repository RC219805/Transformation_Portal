#!/usr/bin/env python3
"""
Comprehensive RAG Pipeline Demonstration

This script demonstrates the complete RAG (Retrieval-Augmented Generation) pipeline
for the Transformation Portal, showcasing all components working together.

Components demonstrated:
1. Repository Indexing
2. Hybrid Retrieval (BM25)
3. Result Reranking
4. Citation Generation
5. Prompt Templates
6. Knowledge Integration

Usage:
    python3 demo_full_pipeline.py --repo-root /path/to/repo
"""

import argparse
import sys
from pathlib import Path
from typing import List

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from indexer import RepositoryIndexer, DocumentChunk
from retriever import HybridRetriever, RetrievalResult
from reranker import ResultReranker
from citation import CitationGenerator, Citation
from templates import PromptTemplates, CodeModificationResponse, FileModification


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def demo_indexing(repo_root: str) -> List[DocumentChunk]:
    """Demonstrate repository indexing."""
    print_section("STEP 1: Repository Indexing")
    
    print(f"📁 Indexing repository: {repo_root}\n")
    
    indexer = RepositoryIndexer(
        repo_root=repo_root,
        chunk_size_tokens=750,
        overlap_tokens=75,
    )
    
    chunks = indexer.index_repository()
    
    print(f"✅ Indexed {len(chunks)} chunks")
    
    # Show statistics
    stats = indexer.get_statistics()
    print(f"\n📊 Statistics:")
    print(f"   Total chunks: {stats['total_chunks']:,}")
    print(f"   Total characters: {stats['total_chars']:,}")
    
    print(f"\n   By type:")
    for chunk_type, count in sorted(stats['by_type'].items()):
        percentage = (count / stats['total_chunks']) * 100
        print(f"      {chunk_type:10s}: {count:4d} ({percentage:5.1f}%)")
    
    print(f"\n   By language:")
    for language, count in sorted(stats['by_language'].items()):
        percentage = (count / stats['total_chunks']) * 100
        print(f"      {language:15s}: {count:4d} ({percentage:5.1f}%)")
    
    # Show sample chunks
    print(f"\n📄 Sample chunks:")
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"\n   {i}. {chunk.file_path}:{chunk.start_line}-{chunk.end_line}")
        print(f"      Type: {chunk.chunk_type} | Language: {chunk.language}")
        if chunk.metadata:
            print(f"      Metadata: {chunk.metadata}")
        preview = chunk.content[:100].replace('\n', ' ')
        print(f"      Preview: {preview}...")
    
    return chunks


def demo_retrieval(chunks: List[DocumentChunk], query: str) -> List[RetrievalResult]:
    """Demonstrate hybrid retrieval."""
    print_section("STEP 2: Hybrid Retrieval (BM25)")
    
    print(f"🔍 Query: \"{query}\"\n")
    
    retriever = HybridRetriever()
    retriever.index(chunks)
    
    print(f"✅ Retriever indexed {len(chunks)} chunks")
    
    # Perform retrieval
    results = retriever.retrieve(query, top_k=10)
    
    print(f"\n📋 Retrieved {len(results)} results:")
    
    for i, result in enumerate(results[:5], 1):
        print(f"\n   {i}. {result.file_path}:{result.start_line}-{result.end_line}")
        print(f"      Score: {result.score:.3f} | Method: {result.retrieval_method}")
        if result.metadata:
            metadata_str = ', '.join(f"{k}={v}" for k, v in result.metadata.items() if k != 'rerank_boost')
            if metadata_str:
                print(f"      Metadata: {metadata_str[:80]}...")
        preview = result.content[:150].replace('\n', ' ')
        print(f"      Preview: {preview}...")
    
    return results


def demo_reranking(results: List[RetrievalResult], query: str) -> List[RetrievalResult]:
    """Demonstrate result reranking."""
    print_section("STEP 3: Result Reranking")
    
    print(f"🎯 Reranking {len(results)} results with quality signals\n")
    
    reranker = ResultReranker()
    reranked = reranker.rerank(results, query, top_k=5)
    
    print(f"✅ Reranked to top {len(reranked)} results:")
    
    for i, result in enumerate(reranked, 1):
        boost = result.metadata.get('rerank_boost', 0.0)
        original_score = result.score - boost
        print(f"\n   {i}. {result.file_path}:{result.start_line}-{result.end_line}")
        print(f"      Original score: {original_score:.3f}")
        print(f"      Rerank boost:   {boost:+.3f}")
        print(f"      Final score:    {result.score:.3f}")
        
        # Explain the boost
        boost_reasons = []
        if result.metadata.get('entity_type') in ('function', 'class'):
            boost_reasons.append("code structure")
        if result.metadata.get('docstring'):
            boost_reasons.append("documented")
        if result.file_path.endswith('.md'):
            boost_reasons.append("documentation")
        
        if boost_reasons:
            print(f"      Boost from: {', '.join(boost_reasons)}")
    
    return reranked


def demo_citations(results: List[RetrievalResult]) -> List[Citation]:
    """Demonstrate citation generation."""
    print_section("STEP 4: Citation Generation")
    
    print(f"📚 Generating citations from {len(results)} results\n")
    
    citation_gen = CitationGenerator(
        snippet_max_lines=10,
        snippet_max_chars=500
    )
    
    citations = citation_gen.generate_citations(results, max_citations=3)
    
    print(f"✅ Generated {len(citations)} citations:")
    
    for i, cite in enumerate(citations, 1):
        print(f"\n   [{i}] {cite.file_path}:{cite.start_line}-{cite.end_line}")
        print(f"       Confidence: {cite.confidence:.0%}")
        print(f"       Relevance: {cite.relevance_note}")
        print(f"\n       Snippet:")
        for line in cite.snippet.split('\n')[:5]:
            print(f"         {line}")
        if cite.snippet.count('\n') > 5:
            print(f"         ...")
    
    # Show formatted output
    print(f"\n📝 Formatted Citations (Markdown):")
    print("-" * 80)
    formatted = citation_gen.format_citations(citations, format_type='markdown')
    # Show first 500 chars of formatted output
    print(formatted[:500])
    if len(formatted) > 500:
        print("\n... (truncated)")
    
    return citations


def demo_templates(query: str):
    """Demonstrate prompt templates."""
    print_section("STEP 5: Prompt Templates")
    
    print(f"📝 Generating prompt templates for: \"{query}\"\n")
    
    # Feature implementation template
    print("1️⃣  Feature Implementation Template:")
    feature_template = PromptTemplates.feature_implementation(
        feature_description=query,
        context="Existing pipeline infrastructure in depth_pipeline/"
    )
    # Show first 400 chars
    print(feature_template[:400])
    print("... (truncated)\n")
    
    # Bug triage template
    print("2️⃣  Bug Triage Template:")
    bug_template = PromptTemplates.bug_triage(
        error_log="ImportError: No module named 'torch'",
        reproduction_steps="Run python pipeline.py",
        environment="Python 3.10, Ubuntu 20.04"
    )
    # Show first 300 chars
    print(bug_template[:300])
    print("... (truncated)\n")
    
    # Demonstrate structured response
    print("3️⃣  Structured Code Modification Response:")
    response = CodeModificationResponse(
        summary=f"Implementation of {query}",
        files=[
            FileModification(
                path="depth_pipeline/processors/new_feature.py",
                patch="+ def new_feature(): pass",
                description="Add new feature processor"
            )
        ],
        tests=["tests/test_new_feature.py"],
        explanation=f"This implements {query} using depth-aware processing",
        confidence=0.85,
        citations=[{"file_path": "existing.py", "snippet": "example"}]
    )
    
    json_output = response.to_json()
    print(json_output[:400])
    print("... (truncated)")


def demo_end_to_end(repo_root: str):
    """Demonstrate the complete end-to-end RAG pipeline."""
    print_section("🚀 COMPLETE RAG PIPELINE DEMONSTRATION")
    
    print("This demonstration showcases the full RAG pipeline:")
    print("  1. Repository Indexing")
    print("  2. Hybrid Retrieval (BM25)")
    print("  3. Result Reranking")
    print("  4. Citation Generation")
    print("  5. Prompt Templates")
    print("\nRunning all components in sequence...\n")
    
    # Example query
    query = "depth-aware image processing with Material Response"
    
    # Step 1: Index
    chunks = demo_indexing(repo_root)
    
    # Step 2: Retrieve
    results = demo_retrieval(chunks, query)
    
    # Step 3: Rerank
    reranked = demo_reranking(results, query)
    
    # Step 4: Generate citations
    citations = demo_citations(reranked)
    
    # Step 5: Show templates
    demo_templates(query)
    
    # Summary
    print_section("✅ PIPELINE DEMONSTRATION COMPLETE")
    
    print("Summary:")
    print(f"  • Indexed: {len(chunks)} chunks")
    print(f"  • Retrieved: {len(results)} results")
    print(f"  • Reranked: {len(reranked)} top results")
    print(f"  • Generated: {len(citations)} citations")
    print(f"  • Query: \"{query}\"")
    
    print("\n💡 The RAG pipeline is fully operational and ready for use!")
    print("\nKey Features:")
    print("  ✓ Intelligent code and documentation retrieval")
    print("  ✓ Multi-signal reranking for precision")
    print("  ✓ Structured citations with confidence scores")
    print("  ✓ Canonical workflow templates")
    print("  ✓ Repository-specific knowledge integration")
    
    print("\n📚 For more information:")
    print("  • See .github/agents/rag_system/README.md")
    print("  • Run: python3 cli.py --help")
    print("  • Tests: pytest tests/test_rag_pipeline.py")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Demonstrate the complete RAG pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full demonstration
  python3 demo_full_pipeline.py --repo-root .
  
  # Custom query
  python3 demo_full_pipeline.py --repo-root . --query "FFmpeg video processing"
        """
    )
    
    parser.add_argument(
        '--repo-root',
        default='.',
        help='Repository root directory (default: current directory)'
    )
    
    parser.add_argument(
        '--query',
        help='Custom query to test (uses default if not provided)'
    )
    
    args = parser.parse_args()
    
    try:
        demo_end_to_end(args.repo_root)
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
