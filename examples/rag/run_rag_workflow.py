#!/usr/bin/env python3
"""
Run RAG workflow end-to-end for the Transformation Portal.
This wrapper handles proper imports and demonstrates all RAG capabilities.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RAG_AGENTS_PATH = REPO_ROOT / ".github" / "agents"
sys.path.insert(0, str(RAG_AGENTS_PATH))

from rag_system.citation import CitationGenerator
from rag_system.classifier import ArtifactClassifier

# Now import RAG components
from rag_system.indexer import RepositoryIndexer
from rag_system.knowledge_engine import KnowledgeIntegrationEngine
from rag_system.reranker import ResultReranker
from rag_system.retriever import HybridRetriever
from rag_system.templates import CodeModificationResponse, FileModification, PromptTemplates


def run_workflow():
    """Execute complete RAG workflow."""
    print("=" * 80)
    print("RAG WORKFLOW - TRANSFORMATION PORTAL")
    print("=" * 80)

    repo_root = REPO_ROOT
    print(f"\nRepository: {repo_root}")

    # Step 1: Index Repository
    print("\n[1/7] INDEXING REPOSITORY...")
    print("-" * 80)
    indexer = RepositoryIndexer(str(repo_root))
    chunks = indexer.index_repository()
    print(f"✓ Created {len(chunks)} chunks from repository")

    # Step 2: Setup Retrieval
    print("\n[2/7] SETTING UP HYBRID RETRIEVAL...")
    print("-" * 80)
    retriever = HybridRetriever()
    retriever.index(chunks)
    print(f"✓ Indexed {len(chunks)} chunks for retrieval")

    # Step 3: Search Query
    print("\n[3/7] EXECUTING SEARCH QUERIES...")
    print("-" * 80)
    queries = ["depth pipeline atmospheric effects", "material response enhancement", "FFmpeg video processing HDR"]

    all_results = {}
    for query in queries:
        results = retriever.retrieve(query, top_k=10)
        all_results[query] = results
        print(f"✓ Query: '{query}' → {len(results)} results")

    # Step 4: Rerank Results
    print("\n[4/7] RERANKING RESULTS...")
    print("-" * 80)
    reranker = ResultReranker()
    reranked_results = {}

    for query, results in all_results.items():
        reranked = reranker.rerank(results, query, top_k=5)
        reranked_results[query] = reranked
        print(f"✓ Reranked '{query[:40]}...' → {len(reranked)} top results")

    # Step 5: Generate Citations
    print("\n[5/7] GENERATING CITATIONS...")
    print("-" * 80)
    citation_gen = CitationGenerator()

    query = queries[0]  # Use first query for detailed example
    citations = citation_gen.generate_citations(reranked_results[query], max_citations=3)
    print(f"✓ Generated {len(citations)} citations for: '{query}'")

    markdown_citations = citation_gen.format_citations(citations, format_type="markdown")
    print("\nSample Citations (Markdown):")
    print(markdown_citations)

    # Step 6: Prompt Template with Context
    print("\n[6/7] GENERATING PROMPT TEMPLATE WITH CONTEXT...")
    print("-" * 80)

    context = citation_gen.format_citations(citations, format_type="text")
    template = PromptTemplates.feature_implementation(
        feature_description="Add real-time depth map visualization overlay", context=context
    )
    print(f"✓ Generated feature implementation template ({len(template)} chars)")
    print("\nTemplate Preview (first 400 chars):")
    print(template[:400] + "...")

    # Step 7: Artifact Classification (if output directory exists)
    print("\n[7/7] ARTIFACT CLASSIFICATION...")
    print("-" * 80)

    output_dir = repo_root / "output"
    if output_dir.exists():
        classifier = ArtifactClassifier()
        artifact_count = 0
        for artifact_path in output_dir.rglob("*"):
            if artifact_path.is_file():
                artifact = classifier.classify_artifact(str(artifact_path))
                if artifact:
                    artifact_count += 1
        stats = classifier.get_statistics()
        print(f"✓ Classified {artifact_count} artifacts")
        print(f"  By type: {stats.get('by_type', {})}")
        print(f"  By pipeline: {stats.get('by_pipeline', {})}")
    else:
        print(f"⊘ Output directory not found: {output_dir}")
        print("  (This is optional - skip if no pipeline outputs exist)")

    # Summary
    print("\n" + "=" * 80)
    print("RAG WORKFLOW COMPLETE")
    print("=" * 80)
    print(f"\n✓ Indexed {len(chunks)} code chunks")
    print(f"✓ Executed {len(queries)} search queries")
    print(f"✓ Generated {len(citations)} citations")
    print(f"✓ Created prompt template with context")
    print("\nThe RAG system is fully operational and ready for use!")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(run_workflow())
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
