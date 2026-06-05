#!/usr/bin/env python3
"""
RAG System Workflow Demonstration
==================================
Comprehensive demonstration of the RAG system following the integration guide.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RAG_AGENTS_PATH = REPO_ROOT / ".github" / "agents"
OUTPUT_ROOT = Path("/tmp/tp-rag-workflow-demo")
sys.path.insert(0, str(RAG_AGENTS_PATH))

from rag_system.citation import CitationGenerator
from rag_system.classifier import ArtifactClassifier
from rag_system.indexer import RepositoryIndexer
from rag_system.knowledge_engine import KnowledgeIntegrationEngine
from rag_system.reranker import ResultReranker
from rag_system.retriever import HybridRetriever
from rag_system.templates import CodeModificationResponse, FileModification, PromptTemplates


def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def step1_basic_workflow():
    """Step 1: Python API Usage - Basic Workflow (lines 82-118)"""
    print_section("STEP 1: Basic Workflow - Index, Search, Rerank, Cite")

    # 1. Index repository
    print("1. Indexing repository...")
    indexer = RepositoryIndexer(str(REPO_ROOT))
    chunks = indexer.index_repository()
    print(f"   ✓ Indexed {len(chunks)} chunks")

    # Show chunk type distribution
    chunk_types = {}
    for chunk in chunks:
        chunk_types[chunk.chunk_type] = chunk_types.get(chunk.chunk_type, 0) + 1
    print(f"   Chunk distribution: {chunk_types}")

    # 2. Setup retrieval
    print("\n2. Setting up hybrid retrieval...")
    retriever = HybridRetriever()
    retriever.index(chunks)
    print(f"   ✓ Indexed {len(chunks)} chunks in retriever")

    # 3. Search for content
    print("\n3. Searching: 'How to add a new LUT preset?'")
    results = retriever.retrieve("How to add a new LUT preset?", top_k=10)
    print(f"   ✓ Found {len(results)} results")

    # Show top 3 results
    print("\n   Top 3 results:")
    for i, result in enumerate(results[:3], 1):
        print(f"   {i}. {result.file_path} (Score: {result.score:.3f})")
        chunk_type = result.metadata.get("chunk_type", "unknown")
        print(f"      Type: {chunk_type}, Lines: {result.start_line}-{result.end_line}")

    # 4. Rerank for better precision
    print("\n4. Reranking results...")
    reranker = ResultReranker()
    reranked = reranker.rerank(results, "How to add a new LUT preset?", top_k=5)
    print(f"   ✓ Reranked to top {len(reranked)} results")

    print("\n   Reranked results:")
    for i, result in enumerate(reranked, 1):
        rerank_boost = result.metadata.get("rerank_boost", 0.0)
        print(f"   {i}. {result.file_path} (Final Score: {result.score:.3f}, Boost: {rerank_boost:.3f})")

    # 5. Generate citations
    print("\n5. Generating citations...")
    citation_gen = CitationGenerator()
    citations = citation_gen.generate_citations(reranked, max_citations=3)
    print(f"   ✓ Generated {len(citations)} citations")

    # 6. Format citations in markdown
    print("\n6. Formatting citations (markdown):")
    formatted = citation_gen.format_citations(citations, format_type="markdown")
    print(formatted)

    # Save to file
    OUTPUT_ROOT.mkdir(exist_ok=True)
    output_file = OUTPUT_ROOT / "step1_citations.md"
    with open(output_file, "w") as f:
        f.write(formatted)
    print(f"\n   ✓ Saved citations to {output_file}")

    return chunks, retriever


def step2_prompt_templates():
    """Step 2: Prompt Templates Usage (lines 122-150)"""
    print_section("STEP 2: Prompt Templates - Feature Implementation & Code Modification")

    # Generate feature implementation template
    print("1. Generating feature implementation template...")
    template = PromptTemplates.feature_implementation(
        feature_description="Add HDR tone mapping with custom transfer function",
        context="Existing tone mapping in tonemapper_agx_filmic.py",
    )
    print("\n   Template preview (first 500 chars):")
    print("   " + template[:500].replace("\n", "\n   "))

    # Save template
    OUTPUT_ROOT.mkdir(exist_ok=True)
    template_file = OUTPUT_ROOT / "step2_feature_template.md"
    with open(template_file, "w") as f:
        f.write(template)
    print(f"\n   ✓ Saved template to {template_file}")

    # Create structured code modification response
    print("\n2. Creating CodeModificationResponse example...")
    response = CodeModificationResponse(
        summary="Add atmospheric haze effect to depth pipeline",
        files=[
            FileModification(
                path="depth_pipeline/processors/atmospheric.py",
                patch="+ def apply_haze(image, depth, intensity=0.3): ...",
                description="Add haze effect implementation",
            ),
            FileModification(
                path="config/exterior_preset.yaml",
                patch="+ haze_intensity: 0.3",
                description="Configure default haze for exteriors",
            ),
        ],
        tests=["tests/test_atmospheric_processor.py"],
        explanation="Atmospheric haze blends fog color proportional to depth distance",
        confidence=0.85,
        citations=[
            {"file_path": "depth_pipeline/processors/clarity.py", "relevance": "Similar pattern for depth-based processing"}
        ],
    )

    print(f"   ✓ Created response with {len(response.files)} file modifications")
    print(f"   Confidence: {response.confidence}")
    print(f"   Tests: {response.tests}")

    # Convert to JSON
    print("\n3. Exporting to JSON...")
    json_output = response.to_json()

    # Pretty print JSON (first 800 chars)
    json_str = json.dumps(json.loads(json_output), indent=2)
    print("\n   JSON output preview:")
    print("   " + json_str[:800].replace("\n", "\n   "))

    # Save JSON
    OUTPUT_ROOT.mkdir(exist_ok=True)
    json_file = OUTPUT_ROOT / "step2_code_modification.json"
    with open(json_file, "w") as f:
        f.write(json_str)
    print(f"\n   ✓ Saved JSON to {json_file}")


def step3_artifact_classification():
    """Step 3: Artifact Classification (lines 154-168)"""
    print_section("STEP 3: Artifact Classification - Organize Pipeline Outputs")

    # Create sample output directory if it doesn't exist
    output_dir = OUTPUT_ROOT / "output"
    output_dir.mkdir(exist_ok=True)

    # Create some sample artifacts for demonstration
    print("1. Creating sample artifacts in output/ directory...")
    sample_files = [
        (output_dir / "render_enhanced.jpg", None),
        (output_dir / "depth_map.png", None),
        (output_dir / "graded_video.mp4", None),
        (output_dir / "test_result.log", "ERROR: Processing failed\nException: ValueError"),
        (output_dir / "metrics.json", '{"processing_time": 2.5, "success": true, "memory_usage": 1024}'),
    ]

    for file_path, content in sample_files:
        Path(file_path).touch()
        if content:
            Path(file_path).write_text(content)
    print(f"   ✓ Created {len(sample_files)} sample artifacts")

    # Classify artifacts
    print("\n2. Classifying artifacts in output/ directory...")
    classifier = ArtifactClassifier()

    # Add each artifact
    artifact_nodes = []
    for file_path, content in sample_files:
        node = classifier.add_artifact(file_path, content=content)
        artifact_nodes.append(node)

    print(f"   ✓ Classified {len(artifact_nodes)} artifacts")

    # Show some examples
    print("\n   Sample classifications:")
    for node in artifact_nodes[:5]:
        pipeline_name = node.metadata.pipeline.value
        print(f"   - {node.file_path}: {node.artifact_type.value} ({pipeline_name})")
        if node.tags:
            sample_tags = list(node.tags)[:3]
            print(f"     Tags: {', '.join(sample_tags)}")

    # Get statistics
    print("\n3. Getting statistics...")
    stats = classifier.get_statistics()
    print(f"   Artifacts by type: {stats['by_type']}")
    print(f"   Artifacts by pipeline: {stats['by_pipeline']}")
    print(f"   Total artifacts: {stats['total_artifacts']}")
    print(f"   Success rate: {stats['success_rate']:.1%}")
    if stats["avg_processing_time"] > 0:
        print(f"   Avg processing time: {stats['avg_processing_time']:.3f}s")

    # Export to JSON
    print("\n4. Exporting to JSON...")
    catalog_file = OUTPUT_ROOT / "artifacts_catalog.json"
    classifier.export_to_json(str(catalog_file))
    print(f"   ✓ Saved catalog to {catalog_file}")

    # Show JSON preview
    with open(catalog_file, "r") as f:
        catalog = json.load(f)
    print(f"\n   Catalog contains {len(catalog['artifacts'])} artifacts")
    print(f"   Statistics: {catalog['statistics']}")


def step4_knowledge_engine():
    """Step 4: Knowledge Engine Demo (lines 172-199)"""
    print_section("STEP 4: Knowledge Engine - Performance Analysis & Recommendations")

    # Create engine
    print("1. Creating knowledge integration engine...")
    engine = KnowledgeIntegrationEngine()
    print("   ✓ Engine initialized")

    # Add sample feedback for depth_pipeline
    print("\n2. Adding sample feedback for depth_pipeline...")
    feedback_samples = [
        ("depth_pipeline", "art_001", True, 0.045, {"model": "depth_anything_v2", "tone_mapping": "agx"}),
        ("depth_pipeline", "art_002", True, 0.038, {"model": "depth_anything_v2", "tone_mapping": "agx"}),
        (
            "depth_pipeline",
            "art_003",
            False,
            0.0,
            {"model": "depth_anything_v2", "tone_mapping": "custom"},
            "Custom tone mapping not found",
        ),
        ("depth_pipeline", "art_004", True, 0.042, {"model": "depth_anything_v2", "tone_mapping": "agx"}),
        ("lux_render", "art_005", True, 2.5, {"model": "sdxl", "controlnet": "canny"}),
    ]

    for pipeline, artifact_id, success, proc_time, params, *rest in feedback_samples:
        error_msg = rest[0] if rest else None
        engine.add_feedback(
            pipeline=pipeline,
            artifact_id=artifact_id,
            success=success,
            processing_time=proc_time,
            parameters=params,
            error_message=error_msg,
        )
    print(f"   ✓ Added {len(feedback_samples)} feedback entries")

    # Analyze pipeline performance
    print("\n3. Analyzing depth_pipeline performance...")
    analysis = engine.analyze_patterns("depth_pipeline", days=30)
    print(f"   Success rate: {analysis.success_rate:.1%}")
    print(f"   Avg processing time: {analysis.avg_processing_time:.3f}s")
    print(f"   Median processing time: {analysis.median_processing_time:.3f}s")
    print(f"   P95 processing time: {analysis.p95_processing_time:.3f}s")
    print(f"   Total executions: {analysis.total_runs}")
    if analysis.failure_modes:
        print(f"   Failure modes: {analysis.failure_modes}")
    if analysis.common_parameters:
        print(f"   Common parameters: {analysis.common_parameters}")

    # Generate recommendations
    print("\n4. Generating recommendations...")
    recommendations = engine.generate_recommendations()
    print(f"   ✓ Generated {len(recommendations)} recommendations")

    for i, rec in enumerate(recommendations[:3], 1):  # Show first 3
        print(f"\n   Recommendation {i}:")
        print(f"   Type: {rec.recommendation_type}")
        print(f"   Severity: {rec.severity}")
        print(f"   Title: {rec.title}")
        print(f"   Description: {rec.description}")
        print(f"   Suggested action: {rec.suggested_action}")
        if rec.evidence:
            print(f"   Evidence: {rec.evidence[:2]}")  # Show first 2 evidence items

    # Natural language query
    print("\n5. Demonstrating natural language query...")
    queries = [
        "What is the success rate for depth_pipeline?",
        "How many pipelines have been executed?",
        "What is the average processing time?",
    ]

    for query in queries:
        answer = engine.query_natural_language(query)
        print(f"\n   Q: {query}")
        print(f"   A: {answer}")


def step5_example_workflows(chunks, retriever):
    """Step 5: Example Workflows (lines 216-254)"""
    print_section("STEP 5: Example Workflows - Real-world Scenarios")

    # Scenario 1: Find similar code patterns for LUT processing
    print("Scenario 1: Finding Similar Code Patterns for LUT Processing")
    print("-" * 60)

    query1 = "LUT application video processing"
    print(f"Query: '{query1}'")
    print("Filtering: code chunks only")

    # Filter code chunks and create a new retriever
    code_chunks = [c for c in chunks if c.chunk_type == "code"]
    print(f"Found {len(code_chunks)} code chunks")

    code_retriever = HybridRetriever()
    code_retriever.index(code_chunks)

    results1 = code_retriever.retrieve(query1, top_k=5)
    print(f"\n✓ Found {len(results1)} code examples")

    print("\nTop results:")
    for i, result in enumerate(results1[:5], 1):
        print(f"{i}. {result.file_path}:{result.start_line}-{result.end_line}")
        print(f"   Score: {result.score:.3f}")
        preview = result.content[:100].replace("\n", " ")
        print(f"   Preview: {preview}...")

    # Save to file
    OUTPUT_ROOT.mkdir(exist_ok=True)
    lut_examples_path = OUTPUT_ROOT / "step5_lut_examples.txt"
    with open(lut_examples_path, "w") as f:
        f.write("LUT Processing Code Examples\n")
        f.write(f"Query: {query1}\n\n")
        for i, result in enumerate(results1, 1):
            f.write(f"\n{'='*60}\n")
            f.write(f"Example {i}: {result.file_path}\n")
            f.write(f"Lines: {result.start_line}-{result.end_line}\n")
            f.write(f"Score: {result.score:.3f}\n")
            f.write(f"\n{result.content}\n")
    print(f"\n✓ Saved examples to {lut_examples_path}")

    # Scenario 2: Documentation lookup for depth estimation
    print("\n\nScenario 2: Documentation Lookup for Depth Estimation")
    print("-" * 60)

    query2 = "depth estimation CoreML"
    print(f"Query: '{query2}'")
    print("Filtering: documentation chunks only")

    # Filter doc chunks
    doc_chunks = [c for c in chunks if c.chunk_type == "doc"]
    print(f"Found {len(doc_chunks)} documentation chunks")

    doc_retriever = HybridRetriever()
    doc_retriever.index(doc_chunks)

    results2 = doc_retriever.retrieve(query2, top_k=3)
    print(f"\n✓ Found {len(results2)} documentation results")

    # Generate citations
    citation_gen = CitationGenerator()
    citations2 = citation_gen.generate_citations(results2, max_citations=5)
    formatted2 = citation_gen.format_citations(citations2, format_type="markdown")

    print("\nGenerated citations:")
    print(formatted2)

    # Save citations
    depth_docs_path = OUTPUT_ROOT / "step5_depth_docs.md"
    with open(depth_docs_path, "w") as f:
        f.write("# Depth Estimation Documentation\n\n")
        f.write(f"Query: {query2}\n\n")
        f.write(formatted2)
    print(f"✓ Saved documentation citations to {depth_docs_path}")

    # Scenario 3: Feature implementation with context
    print("\n\nScenario 3: Feature Implementation with Context")
    print("-" * 60)

    query3 = "atmospheric effects depth map"
    print(f"Query: '{query3}'")
    print("1. Searching for related code...")

    results3 = retriever.retrieve(query3, top_k=5)
    print(f"   ✓ Found {len(results3)} relevant chunks")

    # Generate context citations
    citations3 = citation_gen.generate_citations(results3, max_citations=3)
    context_text = citation_gen.format_citations(citations3, format_type="text")

    print("\n2. Generating feature template with context...")
    feature_template = PromptTemplates.feature_implementation(
        feature_description="Add fog density parameter to atmospheric effects", context=context_text
    )

    # Save feature plan
    feature_plan_path = OUTPUT_ROOT / "step5_feature_plan.md"
    with open(feature_plan_path, "w") as f:
        f.write(feature_template)
    print(f"   ✓ Saved feature plan to {feature_plan_path}")

    print("\n   Feature plan preview (first 600 chars):")
    print("   " + feature_template[:600].replace("\n", "\n   "))


def main():
    """Run complete RAG workflow demonstration"""
    print("\n" + "🎯" * 40)
    print("  RAG System Workflow Demonstration")
    print("  Following the repository RAG integration guide")
    print("🎯" * 40)

    try:
        # Step 1: Basic workflow
        chunks, retriever = step1_basic_workflow()

        # Step 2: Prompt templates
        step2_prompt_templates()

        # Step 3: Artifact classification
        step3_artifact_classification()

        # Step 4: Knowledge engine
        step4_knowledge_engine()

        # Step 5: Example workflows
        step5_example_workflows(chunks, retriever)

        # Summary
        print_section("✅ DEMONSTRATION COMPLETE")
        print("All steps executed successfully!\n")
        print("Generated files:")
        output_files = [
            OUTPUT_ROOT / "step1_citations.md",
            OUTPUT_ROOT / "step2_feature_template.md",
            OUTPUT_ROOT / "step2_code_modification.json",
            OUTPUT_ROOT / "artifacts_catalog.json",
            OUTPUT_ROOT / "step5_lut_examples.txt",
            OUTPUT_ROOT / "step5_depth_docs.md",
            OUTPUT_ROOT / "step5_feature_plan.md",
        ]
        for f in output_files:
            if Path(f).exists():
                size = Path(f).stat().st_size
                print(f"  ✓ {f} ({size} bytes)")

        print("\n" + "=" * 80)
        print("RAG system is fully operational and ready for integration!")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
