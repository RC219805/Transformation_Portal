#!/usr/bin/env python3
"""
Command-line interface for the RAG system.

Provides commands for:
- Indexing repository content
- Searching for relevant chunks
- Generating citations
- Using prompt templates
- Running the complete pipeline
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from indexer import RepositoryIndexer
    from retriever import HybridRetriever
    from reranker import ResultReranker
    from citation import CitationGenerator
    from templates import PromptTemplates
    from classifier import ArtifactClassifier
    from knowledge_engine import KnowledgeIntegrationEngine
except ImportError as e:
    print(f"Error importing RAG components: {e}", file=sys.stderr)
    print("Make sure you're running from the correct directory", file=sys.stderr)
    sys.exit(1)


def cmd_index(args):
    """Index repository content."""
    print(f"Indexing repository: {args.repo_root}")

    indexer = RepositoryIndexer(
        repo_root=args.repo_root,
        chunk_size_tokens=args.chunk_size,
        overlap_tokens=args.chunk_overlap
    )

    chunks = indexer.index_repository()

    print(f"\nIndexing complete:")
    print(f"  Total chunks: {len(chunks)}")

    # Show chunk type distribution
    chunk_types = {}
    for chunk in chunks:
        chunk_types[chunk.chunk_type] = chunk_types.get(chunk.chunk_type, 0) + 1

    print(f"  Chunk types:")
    for chunk_type, count in sorted(chunk_types.items()):
        print(f"    {chunk_type}: {count}")

    # Save index stats if requested
    if args.output:
        stats = {
            'total_chunks': len(chunks),
            'chunk_types': chunk_types,
            'repo_root': args.repo_root,
            'chunk_size_tokens': args.chunk_size,
            'overlap_tokens': args.chunk_overlap
        }

        with open(args.output, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"\nStats saved to: {args.output}")

    return chunks


def cmd_search(args):
    """Search for relevant chunks."""
    print(f"Searching repository: {args.repo_root}")
    print(f"Query: {args.query}")

    # Index repository
    indexer = RepositoryIndexer(args.repo_root)
    chunks = indexer.index_repository()
    print(f"Indexed {len(chunks)} chunks")

    # Setup retrieval
    retriever = HybridRetriever()
    retriever.index(chunks)

    # Retrieve
    chunk_types = args.types.split(',') if args.types else None
    results = retriever.retrieve(
        query=args.query,
        top_k=args.top_k * 2,  # Get more for reranking
        chunk_type_filter=chunk_types
    )

    # Rerank if requested
    if not args.no_rerank:
        reranker = ResultReranker()
        results = reranker.rerank(results, args.query, top_k=args.top_k)
    else:
        results = results[:args.top_k]

    print(f"\nFound {len(results)} results:")
    print("=" * 80)

    for i, result in enumerate(results, 1):
        print(f"\n[{i}] {result.file_path}:{result.start_line}-{result.end_line}")
        print(f"Score: {result.score:.3f}")
        print(f"Method: {result.retrieval_method}")
        if result.metadata:
            print(f"Metadata: {result.metadata}")
        print(f"\nContent preview:")
        # Show first 200 characters
        preview = result.content[:200]
        if len(result.content) > 200:
            preview += "..."
        print(preview)
        print("-" * 80)

    return results


def cmd_cite(args):
    """Generate citations for a query."""
    print(f"Generating citations for: {args.query}")

    # Index and retrieve
    indexer = RepositoryIndexer(args.repo_root)
    chunks = indexer.index_repository()

    retriever = HybridRetriever()
    retriever.index(chunks)

    results = retriever.retrieve(args.query, top_k=args.max_citations * 2)

    # Rerank
    reranker = ResultReranker()
    results = reranker.rerank(results, args.query, top_k=args.max_citations)

    # Generate citations
    citation_gen = CitationGenerator()
    citations = citation_gen.generate_citations(results, max_citations=args.max_citations)

    # Format output
    formatted = citation_gen.format_citations(citations, format_type=args.format)

    print("\n" + formatted)

    # Save if requested
    if args.output:
        with open(args.output, 'w') as f:
            if args.format == 'json':
                json.dump(citations, f, indent=2)
            else:
                f.write(formatted)
        print(f"\nCitations saved to: {args.output}")

    return citations


def cmd_template(args):
    """Generate prompt template."""
    if args.type == 'feature':
        template = PromptTemplates.feature_implementation(
            feature_description=args.description,
            context=args.context or ""
        )
    elif args.type == 'bug':
        template = PromptTemplates.bug_triage(
            error_log=args.description,
            reproduction_steps=args.context or "Not provided",
            environment=args.environment or "Not specified"
        )
    elif args.type == 'ci':
        template = PromptTemplates.ci_change(
            workflow_name=args.workflow or "build.yml",
            change_description=args.description,
            reason=args.context or "Not provided"
        )
    else:
        print(f"Unknown template type: {args.type}", file=sys.stderr)
        return None

    print(template)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(template)
        print(f"\nTemplate saved to: {args.output}", file=sys.stderr)

    return template


def cmd_classify(args):
    """Classify artifacts in a directory."""
    print(f"Classifying artifacts in: {args.input_dir}")

    classifier = ArtifactClassifier()

    # Scan directory and classify artifacts
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Directory not found: {input_dir}")
        return []

    artifact_count = 0
    for file_path in input_dir.rglob('*') if args.recursive else input_dir.glob('*'):
        if file_path.is_file():
            # Try to read content if it's a text file
            content = None
            if file_path.suffix in {'.json', '.log', '.txt', '.md'}:
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                except Exception:
                    pass

            classifier.add_artifact(str(file_path), content)
            artifact_count += 1

    # Show statistics
    stats = classifier.get_statistics()
    print(f"\nClassified {stats['total_artifacts']} artifacts")
    print(f"\nBy type:")
    for artifact_type, count in sorted(stats.get('by_type', {}).items()):
        print(f"  {artifact_type}: {count}")
    print(f"\nBy pipeline:")
    for pipeline, count in sorted(stats.get('by_pipeline', {}).items()):
        print(f"  {pipeline}: {count}")

    # Save if requested
    if args.output:
        classifier.export_to_json(args.output)
        print(f"\nArtifacts saved to: {args.output}")

    return classifier.artifacts


def cmd_analyze(args):
    """Analyze pipeline performance and generate recommendations."""
    print(f"Analyzing feedback from: {args.feedback_file}")

    # Load feedback
    with open(args.feedback_file, 'r') as f:
        feedback_data = json.load(f)

    engine = KnowledgeIntegrationEngine()

    # Add feedback
    for entry in feedback_data:
        engine.add_feedback(entry)

    if args.pipeline:
        # Analyze specific pipeline
        print(f"\nAnalyzing pipeline: {args.pipeline}")
        analysis = engine.analyze_pipeline(args.pipeline)

        print(f"\nSuccess rate: {analysis.success_rate:.1%}")
        print(f"Processing time: avg={analysis.avg_processing_time:.2f}s, "
              f"p95={analysis.p95_processing_time:.2f}s")
        print(f"Common parameters: {analysis.common_parameters}")

        if analysis.failure_modes:
            print(f"\nFailure modes:")
            for mode, count in analysis.failure_modes.items():
                print(f"  {mode}: {count}")

    elif args.recommendations:
        # Generate recommendations
        print("\nGenerating recommendations...")
        recommendations = engine.generate_recommendations()

        for rec in recommendations:
            print(f"\n[{rec.type.upper()}] {rec.title}")
            print(f"  Priority: {rec.priority}")
            print(f"  Description: {rec.description}")
            if rec.suggested_action:
                print(f"  Action: {rec.suggested_action}")

    elif args.query:
        # Natural language query
        print(f"\nQuery: {args.query}")
        answer = engine.query(args.query)
        print(f"\nAnswer:\n{answer}")

    # Export if requested
    if args.export:
        knowledge_base = engine.export_knowledge_base()
        with open(args.export, 'w') as f:
            json.dump(knowledge_base, f, indent=2)
        print(f"\nKnowledge base exported to: {args.export}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='RAG System CLI for Transformation Portal',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Index command
    index_parser = subparsers.add_parser('index', help='Index repository content')
    index_parser.add_argument('--repo-root', default='.', help='Repository root directory')
    index_parser.add_argument('--chunk-size', type=int, default=1000, help='Chunk size in tokens')
    index_parser.add_argument('--chunk-overlap', type=int, default=100, help='Chunk overlap in tokens')
    index_parser.add_argument('--output', help='Save index stats to JSON file')
    index_parser.add_argument('--verbose', action='store_true', help='Verbose output')

    # Search command
    search_parser = subparsers.add_parser('search', help='Search for relevant chunks')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('--repo-root', default='.', help='Repository root directory')
    search_parser.add_argument('--top-k', type=int, default=5, help='Number of results')
    search_parser.add_argument('--types', help='Comma-separated chunk types (code,doc,test)')
    search_parser.add_argument('--no-rerank', action='store_true', help='Skip reranking')

    # Citation command
    cite_parser = subparsers.add_parser('cite', help='Generate citations')
    cite_parser.add_argument('query', help='Search query')
    cite_parser.add_argument('--repo-root', default='.', help='Repository root directory')
    cite_parser.add_argument('--max-citations', type=int, default=5, help='Max citations')
    cite_parser.add_argument('--format', choices=['markdown', 'text', 'json'],
                             default='markdown', help='Output format')
    cite_parser.add_argument('--output', help='Save citations to file')

    # Template command
    template_parser = subparsers.add_parser('template', help='Generate prompt template')
    template_parser.add_argument('type', choices=['feature', 'bug', 'ci'],
                                 help='Template type')
    template_parser.add_argument('description', help='Feature/bug/change description')
    template_parser.add_argument('--context', help='Additional context')
    template_parser.add_argument('--environment', help='Environment info (for bug triage)')
    template_parser.add_argument('--workflow', help='Workflow name (for CI changes)')
    template_parser.add_argument('--output', help='Save template to file')

    # Classify command
    classify_parser = subparsers.add_parser('classify', help='Classify artifacts')
    classify_parser.add_argument('input_dir', help='Input directory with artifacts')
    classify_parser.add_argument('--output', help='Save results to JSON file')
    classify_parser.add_argument('--recursive', action='store_true', help='Recursive search')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze pipeline performance')
    analyze_parser.add_argument('--feedback-file', required=True, help='Feedback JSON file')
    analyze_parser.add_argument('--pipeline', help='Analyze specific pipeline')
    analyze_parser.add_argument('--recommendations', action='store_true',
                               help='Generate recommendations')
    analyze_parser.add_argument('--query', help='Natural language query')
    analyze_parser.add_argument('--export', help='Export knowledge base to JSON')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Execute command
    try:
        if args.command == 'index':
            cmd_index(args)
        elif args.command == 'search':
            cmd_search(args)
        elif args.command == 'cite':
            cmd_cite(args)
        elif args.command == 'template':
            cmd_template(args)
        elif args.command == 'classify':
            cmd_classify(args)
        elif args.command == 'analyze':
            cmd_analyze(args)
        else:
            print(f"Unknown command: {args.command}", file=sys.stderr)
            return 1

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose if hasattr(args, 'verbose') else False:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
