#!/usr/bin/env python3
"""
Example of using the RAG system with the Transformation Portal repository.

This script demonstrates:
1. Indexing repository content
2. Searching for relevant chunks
3. Reranking results
4. Generating citations
5. Using prompt templates with context
6. Classifying artifacts
7. Analyzing pipeline performance
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RAG_AGENTS_PATH = REPO_ROOT / ".github" / "agents"
sys.path.insert(0, str(RAG_AGENTS_PATH))

try:
    from rag_system.citation import CitationGenerator
    from rag_system.classifier import ArtifactClassifier
    from rag_system.indexer import RepositoryIndexer
    from rag_system.knowledge_engine import KnowledgeIntegrationEngine
    from rag_system.reranker import ResultReranker
    from rag_system.retriever import HybridRetriever
    from rag_system.templates import CodeModificationResponse, FileModification, PromptTemplates
except ImportError as e:
    print(f"Error: Could not import RAG components: {e}")
    print("Make sure you're running from the repository root directory")
    sys.exit(1)


def example_basic_search():
    """Example 1: Basic search workflow."""
    print("=" * 80)
    print("Example 1: Basic Search Workflow")
    print("=" * 80)

    repo_root = REPO_ROOT

    print(f"\n1. Indexing repository: {repo_root}")
    indexer = RepositoryIndexer(str(repo_root))
    chunks = indexer.index_repository()
    print(f"   Created {len(chunks)} chunks")

    print("\n2. Setting up retrieval")
    retriever = HybridRetriever()
    retriever.index(chunks)
    print("   Retriever ready")

    print("\n3. Searching for 'depth pipeline'")
    query = "depth pipeline atmospheric effects"
    results = retriever.retrieve(query, top_k=5)
    print(f"   Found {len(results)} results")

    print("\n4. Top results:")
    for i, result in enumerate(results[:3], 1):
        print(f"   [{i}] {result.file_path}:{result.start_line}")
        print(f"       Score: {result.score:.3f}, Method: {result.retrieval_method}")
        print(f"       Preview: {result.content[:100]}...")


def example_with_reranking():
    """Example 2: Search with reranking."""
    print("\n" + "=" * 80)
    print("Example 2: Search with Reranking")
    print("=" * 80)

    repo_root = REPO_ROOT

    # Index and retrieve
    indexer = RepositoryIndexer(str(repo_root))
    chunks = indexer.index_repository()

    retriever = HybridRetriever()
    retriever.index(chunks)

    query = "How to add a new LUT preset?"
    print(f"\nQuery: {query}")

    # Get initial results
    results = retriever.retrieve(query, top_k=10)
    print(f"Initial retrieval: {len(results)} results")

    # Rerank
    reranker = ResultReranker()
    reranked = reranker.rerank(results, query, top_k=5)
    print(f"After reranking: {len(reranked)} results")

    print("\nTop reranked results:")
    for i, result in enumerate(reranked[:3], 1):
        print(f"[{i}] {result.file_path}:{result.start_line}-{result.end_line}")
        print(f"    Score: {result.score:.3f}")


def example_with_citations():
    """Example 3: Generate citations for evidence."""
    print("\n" + "=" * 80)
    print("Example 3: Generate Citations")
    print("=" * 80)

    repo_root = REPO_ROOT

    # Index, retrieve, and rerank
    indexer = RepositoryIndexer(str(repo_root))
    chunks = indexer.index_repository()

    retriever = HybridRetriever()
    retriever.index(chunks)

    query = "material response enhancement"
    results = retriever.retrieve(query, top_k=10)

    reranker = ResultReranker()
    reranked = reranker.rerank(results, query, top_k=5)

    # Generate citations
    citation_gen = CitationGenerator()
    citations = citation_gen.generate_citations(reranked, max_citations=3)

    print(f"\nGenerated {len(citations)} citations for: '{query}'\n")

    # Format as markdown
    markdown = citation_gen.format_citations(citations, format_type="markdown")
    print(markdown)


def example_prompt_template():
    """Example 4: Using prompt templates with citations."""
    print("\n" + "=" * 80)
    print("Example 4: Prompt Template with Context")
    print("=" * 80)

    repo_root = REPO_ROOT

    # Get context from repository
    indexer = RepositoryIndexer(str(repo_root))
    chunks = indexer.index_repository()

    retriever = HybridRetriever()
    retriever.index(chunks)

    # Find relevant context for feature request
    query = "FFmpeg filter graph video processing"
    results = retriever.retrieve(query, top_k=5)

    reranker = ResultReranker()
    reranked = reranker.rerank(results, query, top_k=3)

    citation_gen = CitationGenerator()
    citations = citation_gen.generate_citations(reranked, max_citations=3)
    context = citation_gen.format_citations(citations, format_type="text")

    # Generate template with context
    print("\nGenerating feature implementation template...")
    template = PromptTemplates.feature_implementation(
        feature_description="Add HDR tone mapping with custom transfer function", context=context
    )

    print("\nTemplate (first 500 chars):")
    print(template[:500] + "...\n")


def example_code_modification_response():
    """Example 5: Creating structured code modification response."""
    print("\n" + "=" * 80)
    print("Example 5: Structured Code Modification Response")
    print("=" * 80)

    # Create a structured response
    response = CodeModificationResponse(
        summary="Add atmospheric haze effect to depth pipeline",
        files=[
            FileModification(
                path="depth_pipeline/processors/atmospheric.py",
                patch="""
@@ -10,6 +10,15 @@

 class AtmosphericProcessor:
+    def apply_haze(self, image, depth_map, intensity=0.3):
+        '''Apply depth-based atmospheric haze.
+
+        Args:
+            image: Input image array
+            depth_map: Normalized depth map
+            intensity: Haze intensity (0.0-1.0)
+        '''
+        fog_color = np.array([0.8, 0.85, 0.9])
+        return image * (1 - depth_map * intensity) + fog_color * depth_map * intensity
""",
                description="Add haze effect implementation",
            ),
            FileModification(
                path="config/presets/exterior.yaml",
                patch="""
@@ -5,6 +5,7 @@
   tone_mapping: agx
   denoising: 0.2
+  atmospheric_haze: 0.3
""",
                description="Enable haze in exterior preset",
            ),
        ],
        tests=["tests/test_atmospheric_processor.py"],
        explanation=(
            "Atmospheric haze is implemented by blending fog color proportional to depth. "
            "Distant objects (high depth values) receive more fog, creating realistic "
            "atmospheric perspective. The effect is configurable via the intensity parameter."
        ),
        confidence=0.85,
        citations=[
            {
                "file_path": "depth_pipeline/processors/clarity.py",
                "snippet": "Similar depth-based processing pattern",
                "relevance": "Shows existing depth map usage",
            }
        ],
    )

    print("\nStructured Response:")
    print(response.to_json())


def example_artifact_classification():
    """Example 6: Classify pipeline artifacts."""
    print("\n" + "=" * 80)
    print("Example 6: Artifact Classification")
    print("=" * 80)

    # Note: This requires an actual output directory with artifacts
    output_dir = REPO_ROOT / "output"

    if not output_dir.exists():
        print(f"\nSkipping: {output_dir} does not exist")
        print("Create some pipeline outputs first, then run this example")
        return

    classifier = ArtifactClassifier()
    artifacts = classifier.classify_directory(str(output_dir), recursive=True)

    print(f"\nClassified {len(artifacts)} artifacts in {output_dir}")

    stats = classifier.get_statistics()
    print("\nStatistics:")
    print(f"  By type: {stats.get('by_type', {})}")
    print(f"  By pipeline: {stats.get('by_pipeline', {})}")


def example_knowledge_engine():
    """Example 7: Knowledge engine analysis."""
    print("\n" + "=" * 80)
    print("Example 7: Knowledge Integration Engine")
    print("=" * 80)

    # Create sample feedback data
    engine = KnowledgeIntegrationEngine()

    # Add some sample feedback
    sample_feedback = [
        ("depth_pipeline", "img001", True, 0.045, {"model": "depth_anything_v2", "tone_mapping": "agx"}),
        ("depth_pipeline", "img002", True, 0.052, {"model": "depth_anything_v2", "tone_mapping": "agx"}),
        ("material_response", "img003", True, 0.120, {"strength": 0.7, "surfaces": ["wood", "metal"]}),
    ]

    for pipeline, artifact_id, success, time, params in sample_feedback:
        engine.add_feedback(
            pipeline=pipeline, artifact_id=artifact_id, success=success, processing_time=time, parameters=params
        )

    # Analyze pipeline
    print("\nAnalyzing depth_pipeline...")
    analysis = engine.analyze_patterns("depth_pipeline")

    print(f"  Success rate: {analysis.success_rate:.1%}")
    print(f"  Avg processing time: {analysis.avg_processing_time:.3f}s")
    print(f"  Common parameters: {analysis.common_parameters}")

    # Natural language query
    print("\nNatural language query:")
    answer = engine.query_natural_language("What is the success rate for depth_pipeline?")
    print(f"  {answer}")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("RAG System Usage Examples")
    print("Transformation Portal Repository")
    print("=" * 80)

    try:
        example_basic_search()
        example_with_reranking()
        example_with_citations()
        example_prompt_template()
        example_code_modification_response()
        example_artifact_classification()
        example_knowledge_engine()

        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
