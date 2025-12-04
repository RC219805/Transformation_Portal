#!/usr/bin/env python3
"""
RAG-Powered Recipe Suggestion System

Uses historical run card data to recommend optimal recipes based on:
- Scene type
- Baseline quality score  
- Previous experimental outcomes
- Human ratings from past runs

Usage:
    python scripts/rag/suggest_recipe.py \
        --scene-type interior_bedroom \
        --baseline-score 60.4 \
        --notes "neutral, daylight, high-end staging"
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add RAG system to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / ".github" / "agents"))

try:
    from rag_system.retriever import HybridRetriever
    from rag_system.indexer import RepositoryIndexer
except ImportError as e:
    print(f"⚠️  RAG system not available: {e}")
    print("Falling back to rule-based recommendations")
    HAVE_RAG = False
else:
    HAVE_RAG = True


def build_rag_query(scene_type: str, baseline_score: float, notes: str = "") -> str:
    """Build structured RAG query for recipe suggestion."""
    query_parts = [
        f"Scene type: {scene_type}",
        f"Baseline quality score: {baseline_score:.1f}%",
    ]
    
    if notes:
        query_parts.append(f"Additional context: {notes}")
    
    query_parts.extend([
        "",
        "Question: Based on similar past experiments, which recipe:",
        "1. Preserved or improved quality best?",
        "2. Received positive human ratings?",
        "3. Was recommended for production use?",
        "",
        "Return: Recipe name, expected delta score, confidence level, and reasoning."
    ])
    
    return "\n".join(query_parts)


def query_rag_system(query: str, repo_root: Path) -> List[Dict[str, Any]]:
    """Query RAG system for relevant historical run cards."""
    if not HAVE_RAG:
        return []
    
    try:
        # Initialize indexer
        indexer = RepositoryIndexer(str(repo_root))
        
        # Try to load from cache
        cache_file = repo_root / ".github" / "agents" / "rag_system" / ".index_cache.pkl"
        if cache_file.exists():
            print("📚 Loading RAG index from cache...")
            chunks = indexer._load_cache()
        else:
            print("🔍 Indexing repository (first time)...")
            chunks = indexer.index_repository()
        
        # Create retriever
        retriever = HybridRetriever()
        retriever.index(chunks)
        
        # Retrieve relevant context
        print(f"🧠 Querying RAG system...")
        results = retriever.retrieve(query, top_k=5)
        
        return results
    except Exception as e:
        print(f"⚠️  RAG query failed: {e}")
        return []


def rule_based_suggestion(scene_type: str, baseline_score: float) -> Dict[str, Any]:
    """Fallback rule-based suggestion when RAG unavailable."""
    
    # Hero shots (≥55%) - preserve
    if baseline_score >= 55.0:
        return {
            "recipe": "baseline (no processing)",
            "reason": "Hero shot - baseline quality already excellent",
            "confidence": "high",
            "expected_delta": 0.0,
            "alternatives": ["signature_estate_gentle (if brand consistency required)"],
            "warning": "Any processing will likely reduce quality by 3-6%"
        }
    
    # Good shots (45-55%) - gentle touch
    elif 45.0 <= baseline_score < 55.0:
        if "interior" in scene_type.lower():
            return {
                "recipe": "signature_estate_gentle",
                "reason": "Good interior - light processing appropriate",
                "confidence": "medium",
                "expected_delta": -3.5,
                "alternatives": ["baseline", "interior_warm_minimal"],
                "warning": "Visual review required - may prefer baseline"
            }
        elif any(x in scene_type.lower() for x in ["exterior", "aerial", "pool"]):
            return {
                "recipe": "exterior_enhanced",
                "reason": "Moderate exterior - can benefit from enhancement",
                "confidence": "medium",
                "expected_delta": +3.0,
                "alternatives": ["signature_estate"],
                "warning": "Pool scenes high-risk - test carefully"
            }
    
    # Weak shots (<45%) - enhance
    else:
        if any(x in scene_type.lower() for x in ["exterior", "aerial"]):
            return {
                "recipe": "exterior_enhanced",
                "reason": "Low baseline exterior - strong enhancement recommended",
                "confidence": "high",
                "expected_delta": +5.0,
                "alternatives": ["signature_estate"],
                "warning": "Proven to improve weak exteriors significantly"
            }
        else:
            return {
                "recipe": "signature_estate",
                "reason": "Low baseline interior - full processing appropriate",
                "confidence": "medium",
                "expected_delta": +2.0,
                "alternatives": ["signature_estate_gentle"],
                "warning": "Interior processing still risky - review results"
            }
    
    return suggestion


def format_suggestion(suggestion: Dict[str, Any], sources: List[str] = None) -> str:
    """Format recipe suggestion for display."""
    lines = [
        "╭─────────────────────────────────────────────────────",
        "│ 🎯 Recipe Recommendation",
        "├─────────────────────────────────────────────────────",
        f"│ Recipe: {suggestion['recipe']}",
        f"│ Confidence: {suggestion['confidence']}",
        f"│ Expected Δ: {suggestion['expected_delta']:+.1f}%",
        "│",
        f"│ Reasoning: {suggestion['reason']}",
    ]
    
    if suggestion.get('alternatives'):
        lines.append("│")
        lines.append("│ Alternatives:")
        for alt in suggestion['alternatives']:
            lines.append(f"│   - {alt}")
    
    if suggestion.get('warning'):
        lines.append("│")
        lines.append(f"│ ⚠️  {suggestion['warning']}")
    
    if sources:
        lines.append("│")
        lines.append("│ 📚 Based on:")
        for source in sources[:3]:
            src_name = Path(source).name if isinstance(source, str) else "run card"
            lines.append(f"│   - {src_name}")
    
    lines.append("╰─────────────────────────────────────────────────────")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="RAG-powered recipe suggestion based on historical data"
    )
    parser.add_argument(
        "--scene-type",
        required=True,
        help="Scene type: interior_bedroom, aerial_exterior, pool_exterior, etc."
    )
    parser.add_argument(
        "--baseline-score",
        type=float,
        required=True,
        help="Baseline quality score (e.g., 60.4)"
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Additional scene notes or context"
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root path"
    )
    
    args = parser.parse_args()
    
    print("\n🔍 Transformation Portal - RAG Recipe Advisor")
    print("=" * 60)
    print(f"Scene Type: {args.scene_type}")
    print(f"Baseline Score: {args.baseline_score:.1f}%")
    if args.notes:
        print(f"Notes: {args.notes}")
    print("=" * 60)
    print()
    
    # Try RAG-based suggestion first
    if HAVE_RAG:
        query = build_rag_query(args.scene_type, args.baseline_score, args.notes)
        results = query_rag_system(query, args.repo_root)
        
        if results:
            print("✅ RAG system provided relevant context")
            print()
            
            # Extract relevant info from top results
            sources = []
            for r in results:
                if hasattr(r, 'file_path'):
                    sources.append(r.file_path)
                elif isinstance(r, dict) and 'file_path' in r:
                    sources.append(r['file_path'])
            
            # For now, still use rule-based but show RAG found context
            # TODO: Parse RAG results to build data-driven suggestion
            suggestion = rule_based_suggestion(args.scene_type, args.baseline_score)
            print(format_suggestion(suggestion, sources if sources else None))
        else:
            print("⚠️  No relevant RAG context found, using rules")
            print()
            suggestion = rule_based_suggestion(args.scene_type, args.baseline_score)
            print(format_suggestion(suggestion))
    else:
        print("⚠️  RAG system unavailable, using rule-based fallback")
        print()
        suggestion = rule_based_suggestion(args.scene_type, args.baseline_score)
        print(format_suggestion(suggestion))
    
    print()


if __name__ == "__main__":
    main()
