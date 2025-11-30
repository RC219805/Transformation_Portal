"""
Transformation Portal RAG System - Phase 1 Integration Module
==============================================================
Unified interface for the enhanced RAG system with persistence and vector search.

This module provides:
- Unified RAG system initialization
- Automatic persistence management
- Seamless cache integration
- Configuration-driven behavior
- CLI tools for management

Architecture:
    RAGSystem (Unified Interface)
    ├── CacheManager (persistence)
    ├── RepositoryIndexer (chunking)
    ├── EnhancedHybridRetriever (search)
    │   ├── BM25Retriever (sparse)
    │   └── VectorRetriever (dense)
    ├── ResultReranker (scoring)
    └── CitationGenerator (output)

Usage:
    from phase1_integration import RAGSystem, RAGConfig

    # Initialize with defaults
    rag = RAGSystem()

    # Or with custom configuration
    config = RAGConfig(
        repo_root="/path/to/repo",
        enable_vector_search=True,
        cache_enabled=True,
    )
    rag = RAGSystem(config)

    # Index repository (uses cache if valid)
    rag.index()

    # Search with hybrid retrieval
    results = rag.search("atmospheric depth effects", top_k=10)

    # Generate citations
    citations = rag.cite(results, format="markdown")

    # Persist state
    rag.save()

Author: Transformation Portal
Version: 2.0.0 (Phase 1)
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import yaml

# Configure module logger
logger = logging.getLogger("rag_system.integration")


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class RAGConfig:
    """
    Unified configuration for the Phase 1 RAG system.

    Combines indexer, retriever, and cache configurations.
    """

    # Repository settings
    repo_root: str = "."

    # Cache settings (Phase 1 Core)
    cache_enabled: bool = True
    cache_dir: str = ".rag_cache"

    # Vector search settings (Phase 1 Core)
    enable_vector_search: bool = True
    vector_model: str = "all-MiniLM-L6-v2"

    # Hybrid retrieval weights
    bm25_weight: float = 0.6
    vector_weight: float = 0.4

    # Indexer settings
    chunk_size_tokens: int = 750
    overlap_tokens: int = 75

    # Retriever settings
    top_k_default: int = 10
    query_cache_size: int = 100

    # Directories to index
    index_directories: List[str] = field(default_factory=lambda: [
        "docs/",
        "depth_pipeline/",
        "tests/",
        ".github/agents/",
        "config/",
        "tools/",
    ])

    # File patterns
    include_patterns: List[str] = field(default_factory=lambda: [
        "*.py", "*.md", "*.yaml", "*.yml", "*.json", "*.txt",
    ])

    exclude_patterns: List[str] = field(default_factory=lambda: [
        "deprecated/*", ".venv/*", "__pycache__/*", "*.pyc",
        ".git/*", "node_modules/*", ".rag_cache/*",
    ])

    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "RAGConfig":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        # Flatten nested configuration
        flat = {}
        if "indexer" in data:
            flat.update(data["indexer"])
        if "retriever" in data:
            flat.update(data["retriever"])
        if "persistence" in data:
            flat["cache_dir"] = data["persistence"].get("index_path", ".rag_cache")
        if "feature_flags" in data:
            flat["enable_vector_search"] = data["feature_flags"].get(
                "vector_search", True
            )
            flat["cache_enabled"] = data["feature_flags"].get("persistent_cache", True)

        # Map to dataclass fields
        config_dict = {}
        for key, value in flat.items():
            if hasattr(cls, key.replace("-", "_")):
                config_dict[key.replace("-", "_")] = value

        return cls(**config_dict)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        data = {
            "indexer": {
                "chunk_size_tokens": self.chunk_size_tokens,
                "overlap_tokens": self.overlap_tokens,
                "cache_enabled": self.cache_enabled,
                "cache_dir": self.cache_dir,
                "include_patterns": self.include_patterns,
                "exclude_patterns": self.exclude_patterns,
                "index_directories": self.index_directories,
            },
            "retriever": {
                "enable_vector_search": self.enable_vector_search,
                "vector_model": self.vector_model,
                "bm25_weight": self.bm25_weight,
                "vector_weight": self.vector_weight,
                "top_k_default": self.top_k_default,
                "query_cache_size": self.query_cache_size,
            },
            "feature_flags": {
                "vector_search": self.enable_vector_search,
                "persistent_cache": self.cache_enabled,
                "embedding_cache": self.enable_vector_search,
                "query_cache": True,
            },
        }
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


# =============================================================================
# Chunk Data Structure
# =============================================================================


@dataclass
class Chunk:
    """
    Represents an indexed chunk of repository content.
    """

    chunk_id: str
    content: str
    file_path: str
    line_start: int
    line_end: int
    chunk_type: str  # code, doc, test, config, agent
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Optional fields
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    docstring: Optional[str] = None

    def __hash__(self):
        return hash(self.chunk_id)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "file_path": self.file_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "chunk_type": self.chunk_type,
            "metadata": self.metadata,
            "function_name": self.function_name,
            "class_name": self.class_name,
            "docstring": self.docstring,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        """Create from dictionary."""
        return cls(**data)


# =============================================================================
# Simple Indexer (Placeholder for Full Implementation)
# =============================================================================


class SimpleIndexer:
    """
    Simple repository indexer for Phase 1.

    This is a simplified implementation. The full RepositoryIndexer
    from the existing RAG system should be used in production.
    """

    def __init__(
        self,
        repo_root: str,
        chunk_size: int = 750,
        overlap: int = 75,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ):
        self.repo_root = Path(repo_root)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.include_patterns = include_patterns or ["*.py", "*.md"]
        self.exclude_patterns = exclude_patterns or ["deprecated/*", ".venv/*"]

    def _should_include(self, path: Path) -> bool:
        """Check if file should be included."""
        import fnmatch

        rel_path = str(path.relative_to(self.repo_root))

        # Check exclude patterns
        for pattern in self.exclude_patterns:
            if fnmatch.fnmatch(rel_path, pattern):
                return False

        # Check include patterns
        for pattern in self.include_patterns:
            if fnmatch.fnmatch(path.name, pattern):
                return True

        return False

    def _detect_chunk_type(self, path: Path) -> str:
        """Detect chunk type from file path."""
        rel_path = str(path.relative_to(self.repo_root))

        if "test" in rel_path.lower():
            return "test"
        elif path.suffix == ".md":
            return "doc"
        elif path.suffix in (".yaml", ".yml", ".json"):
            return "config"
        elif ".github/agents" in rel_path:
            return "agent"
        else:
            return "code"

    def _chunk_content(
        self,
        content: str,
        file_path: Path,
    ) -> List[Chunk]:
        """Split content into chunks."""
        chunks = []
        lines = content.split("\n")

        # Estimate characters per chunk
        chars_per_token = 4.0
        chunk_chars = int(self.chunk_size * chars_per_token)
        overlap_chars = int(self.overlap * chars_per_token)

        chunk_type = self._detect_chunk_type(file_path)
        rel_path = str(file_path.relative_to(self.repo_root))

        current_chunk = []
        current_size = 0
        line_start = 1

        for i, line in enumerate(lines, 1):
            line_size = len(line) + 1  # +1 for newline

            if current_size + line_size > chunk_chars and current_chunk:
                # Create chunk
                chunk_content = "\n".join(current_chunk)
                chunk_id = f"{rel_path}:{line_start}-{i-1}"

                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    content=chunk_content,
                    file_path=rel_path,
                    line_start=line_start,
                    line_end=i - 1,
                    chunk_type=chunk_type,
                ))

                # Overlap: keep last few lines
                overlap_lines = []
                overlap_size = 0
                for prev_line in reversed(current_chunk):
                    if overlap_size + len(prev_line) > overlap_chars:
                        break
                    overlap_lines.insert(0, prev_line)
                    overlap_size += len(prev_line) + 1

                current_chunk = overlap_lines
                current_size = overlap_size
                line_start = i - len(overlap_lines)

            current_chunk.append(line)
            current_size += line_size

        # Final chunk
        if current_chunk:
            chunk_content = "\n".join(current_chunk)
            chunk_id = f"{rel_path}:{line_start}-{len(lines)}"

            chunks.append(Chunk(
                chunk_id=chunk_id,
                content=chunk_content,
                file_path=rel_path,
                line_start=line_start,
                line_end=len(lines),
                chunk_type=chunk_type,
            ))

        return chunks

    def index_repository(
        self,
        directories: Optional[List[str]] = None,
    ) -> Tuple[List[Chunk], Dict[str, List[str]]]:
        """
        Index repository content.

        Returns:
            Tuple of (chunks, source_files mapping)
        """
        all_chunks = []
        source_files: Dict[str, List[str]] = {}

        # Determine directories to scan
        if directories:
            scan_dirs = [self.repo_root / d for d in directories]
        else:
            scan_dirs = [self.repo_root]

        for scan_dir in scan_dirs:
            if not scan_dir.exists():
                continue

            for file_path in scan_dir.rglob("*"):
                if not file_path.is_file():
                    continue

                if not self._should_include(file_path):
                    continue

                try:
                    content = file_path.read_text(encoding="utf-8")
                except (UnicodeDecodeError, IOError):
                    continue

                chunks = self._chunk_content(content, file_path)
                all_chunks.extend(chunks)

                # Track source files
                rel_path = str(file_path.relative_to(self.repo_root))
                source_files[rel_path] = [c.chunk_id for c in chunks]

        logger.info(
            f"Indexed {len(all_chunks)} chunks from {len(source_files)} files"
        )
        return all_chunks, source_files


# =============================================================================
# RAG System (Unified Interface)
# =============================================================================


class RAGSystem:
    """
    Unified interface for the Phase 1 enhanced RAG system.

    Provides:
    - Automatic persistence with content-hash invalidation
    - Hybrid retrieval with BM25 + semantic search
    - Seamless integration of all components
    - Configuration-driven behavior
    """

    def __init__(self, config: Optional[RAGConfig] = None):
        """
        Initialize the RAG system.

        Args:
            config: System configuration (uses defaults if not provided)
        """
        self.config = config or RAGConfig()

        # Configure logging
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        # Initialize components (lazy loading for optional dependencies)
        self._cache_manager = None
        self._indexer = None
        self._retriever = None

        # State
        self.chunks: List[Chunk] = []
        self.source_files: Dict[str, List[str]] = {}
        self._indexed = False

        logger.info(
            f"RAGSystem initialized (vector_search={self.config.enable_vector_search})"
        )

    @property
    def cache_manager(self):
        """Lazy-load cache manager."""
        if self._cache_manager is None:
            try:
                from .cache_manager import CacheManager, CacheConfig

                cache_config = CacheConfig(
                    cache_dir=self.config.cache_dir,
                    backup_enabled=True,
                )
                self._cache_manager = CacheManager(cache_config)
            except ImportError:
                logger.warning("CacheManager not available, persistence disabled")
                self._cache_manager = None
        return self._cache_manager

    @property
    def indexer(self):
        """Lazy-load indexer."""
        if self._indexer is None:
            self._indexer = SimpleIndexer(
                repo_root=self.config.repo_root,
                chunk_size=self.config.chunk_size_tokens,
                overlap=self.config.overlap_tokens,
                include_patterns=self.config.include_patterns,
                exclude_patterns=self.config.exclude_patterns,
            )
        return self._indexer

    @property
    def retriever(self):
        """Lazy-load retriever."""
        if self._retriever is None:
            try:
                from .enhanced_retriever import (
                    EnhancedHybridRetriever,
                    RetrieverConfig,
                )

                retriever_config = RetrieverConfig(
                    enable_vector_search=self.config.enable_vector_search,
                    bm25_weight=self.config.bm25_weight,
                    vector_weight=self.config.vector_weight,
                    vector_model=self.config.vector_model,
                    top_k_default=self.config.top_k_default,
                    query_cache_size=self.config.query_cache_size,
                )
                self._retriever = EnhancedHybridRetriever(retriever_config)
            except ImportError:
                logger.error("EnhancedHybridRetriever not available")
                raise
        return self._retriever

    def index(self, force_reindex: bool = False) -> int:
        """
        Index the repository.

        Uses cached chunks if available and valid.

        Args:
            force_reindex: Force re-indexing even if cache is valid

        Returns:
            Number of chunks indexed
        """
        start_time = time.time()

        # Check cache
        if self.config.cache_enabled and not force_reindex and self.cache_manager:
            # Get current file list
            current_files = self._get_current_files()

            # Load with validation
            chunks, invalidated = self.cache_manager.load_chunks_with_validation(
                current_files
            )

            if chunks and not invalidated:
                # Cache hit - use cached chunks
                self.chunks = [
                    Chunk.from_dict(c) if isinstance(c, dict) else c
                    for c in chunks
                ]

                # Load embeddings if available
                if self.config.enable_vector_search:
                    self._load_cached_embeddings()

                # Index retriever
                self.retriever.index(self.chunks)

                elapsed = (time.time() - start_time) * 1000
                logger.info(
                    f"Loaded {len(self.chunks)} chunks from cache in {elapsed:.1f}ms"
                )
                self._indexed = True
                return len(self.chunks)

            elif invalidated:
                logger.info(f"{len(invalidated)} files changed, re-indexing")

        # Full indexing
        logger.info("Performing full repository indexing...")

        self.chunks, self.source_files = self.indexer.index_repository(
            directories=self.config.index_directories
        )

        # Index retriever
        self.retriever.index(self.chunks)

        # Save to cache
        if self.config.cache_enabled and self.cache_manager:
            chunk_dicts = [c.to_dict() for c in self.chunks]
            self.cache_manager.save_chunks(chunk_dicts, self.source_files)

            # Save embeddings
            if self.config.enable_vector_search:
                self._save_embeddings()

        elapsed = (time.time() - start_time) * 1000
        logger.info(f"Indexed {len(self.chunks)} chunks in {elapsed:.1f}ms")

        self._indexed = True
        return len(self.chunks)

    def _get_current_files(self) -> Dict[str, Path]:
        """Get current repository files for cache validation."""
        files = {}
        repo_root = Path(self.config.repo_root)

        for pattern in self.config.include_patterns:
            for file_path in repo_root.rglob(pattern.lstrip("*")):
                if file_path.is_file():
                    rel_path = str(file_path.relative_to(repo_root))

                    # Check exclude patterns
                    import fnmatch
                    excluded = False
                    for excl in self.config.exclude_patterns:
                        if fnmatch.fnmatch(rel_path, excl):
                            excluded = True
                            break

                    if not excluded:
                        files[rel_path] = file_path

        return files

    def _load_cached_embeddings(self) -> bool:
        """Load embeddings from cache."""
        if not self.cache_manager:
            return False

        result = self.cache_manager.load_embeddings()
        if result is None:
            return False

        embeddings, chunk_ids = result
        self.retriever.load_cached_embeddings(embeddings, chunk_ids)
        return True

    def _save_embeddings(self) -> bool:
        """Save embeddings to cache."""
        if not self.cache_manager:
            return False

        return self.retriever.save_embeddings(self.cache_manager)

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        chunk_types: Optional[Set[str]] = None,
        file_pattern: Optional[str] = None,
    ) -> List[Any]:
        """
        Search for relevant chunks.

        Args:
            query: Search query
            top_k: Number of results
            chunk_types: Filter by chunk type (code, doc, test, config, agent)
            file_pattern: Regex filter for file paths

        Returns:
            List of RetrievalResult objects
        """
        if not self._indexed:
            self.index()

        return self.retriever.retrieve(
            query=query,
            top_k=top_k or self.config.top_k_default,
            chunk_type_filter=chunk_types,
            file_path_filter=file_pattern,
        )

    def cite(
        self,
        results: List[Any],
        format_type: str = "markdown",
        max_citations: int = 5,
    ) -> str:
        """
        Generate citations from search results.

        Args:
            results: Search results
            format_type: Output format (markdown, text, json)
            max_citations: Maximum number of citations

        Returns:
            Formatted citations string
        """
        citations = []

        for result in results[:max_citations]:
            chunk = result.chunk

            if format_type == "markdown":
                citation = (
                    f"**[{chunk.file_path}:{chunk.line_start}-{chunk.line_end}]** "
                    f"(Score: {result.score:.2f}, Method: {result.retrieval_method})\n"
                    f"```\n{chunk.content[:300]}...\n```\n"
                )
            elif format_type == "text":
                citation = (
                    f"[{chunk.file_path}:{chunk.line_start}-{chunk.line_end}] "
                    f"Score: {result.score:.2f}\n"
                    f"{chunk.content[:200]}...\n\n"
                )
            else:  # json
                citation = {
                    "file_path": chunk.file_path,
                    "line_start": chunk.line_start,
                    "line_end": chunk.line_end,
                    "score": result.score,
                    "method": result.retrieval_method,
                    "snippet": chunk.content[:300],
                }
                citations.append(citation)
                continue

            citations.append(citation)

        if format_type == "json":
            return json.dumps(citations, indent=2)
        else:
            return "\n".join(citations)

    def save(self) -> None:
        """Save current state to cache."""
        if not self.config.cache_enabled or not self.cache_manager:
            logger.warning("Cache disabled, nothing to save")
            return

        if self.chunks:
            chunk_dicts = [c.to_dict() for c in self.chunks]
            self.cache_manager.save_chunks(chunk_dicts, self.source_files)

        if self.config.enable_vector_search:
            self._save_embeddings()

        logger.info("State saved to cache")

    def clear_cache(self) -> None:
        """Clear all cached data."""
        if self.cache_manager:
            self.cache_manager.clear()

        if self._retriever:
            self._retriever.clear_cache()

        self.chunks = []
        self.source_files = {}
        self._indexed = False

        logger.info("Cache cleared")

    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics."""
        stats = {
            "indexed_chunks": len(self.chunks),
            "source_files": len(self.source_files),
            "vector_search_enabled": self.config.enable_vector_search,
            "cache_enabled": self.config.cache_enabled,
        }

        if self._retriever:
            stats["retriever"] = self.retriever.get_statistics()

        if self.cache_manager:
            stats["cache"] = self.cache_manager.get_statistics()

        return stats


# =============================================================================
# CLI Interface
# =============================================================================


def main():
    """CLI entry point for Phase 1 RAG system."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Transformation Portal RAG System (Phase 1)"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Index command
    index_parser = subparsers.add_parser("index", help="Index repository")
    index_parser.add_argument("--repo-root", default=".", help="Repository root")
    index_parser.add_argument("--force", action="store_true", help="Force re-index")
    index_parser.add_argument(
        "--no-vector", action="store_true", help="Disable vector search"
    )

    # Search command
    search_parser = subparsers.add_parser("search", help="Search repository")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument(
        "--top-k", type=int, default=5, help="Number of results"
    )
    search_parser.add_argument(
        "--type", choices=["code", "doc", "test"], help="Filter by type"
    )

    # Stats command
    subparsers.add_parser("stats", help="Show statistics")

    # Clear command
    subparsers.add_parser("clear", help="Clear cache")

    args = parser.parse_args()

    if args.command == "index":
        config = RAGConfig(
            repo_root=args.repo_root,
            enable_vector_search=not args.no_vector,
        )
        rag = RAGSystem(config)
        count = rag.index(force_reindex=args.force)
        print(f"Indexed {count} chunks")

    elif args.command == "search":
        rag = RAGSystem()
        rag.index()

        chunk_types = {args.type} if args.type else None
        results = rag.search(args.query, top_k=args.top_k, chunk_types=chunk_types)

        print(f"\n=== Results for: {args.query} ===\n")
        for i, result in enumerate(results, 1):
            print(f"{i}. [{result.chunk.file_path}:{result.chunk.line_start}]")
            print(f"   Score: {result.score:.4f} ({result.retrieval_method})")
            print(f"   {result.chunk.content[:100]}...")
            print()

    elif args.command == "stats":
        rag = RAGSystem()
        if rag.cache_manager:
            stats = rag.cache_manager.get_statistics()
            print("\n=== Cache Statistics ===")
            for key, value in stats.items():
                print(f"  {key}: {value}")

    elif args.command == "clear":
        rag = RAGSystem()
        rag.clear_cache()
        print("Cache cleared")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
