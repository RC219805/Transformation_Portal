"""
Repository Content Indexer for RAG System

Indexes repository content (docs/, src/, tests/, agent files, changelogs, READMEs)
into chunks with metadata for efficient retrieval.
"""

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from .authority import CATALOG_PATH, DocumentationCatalog
from .config import get_config
from .exceptions import CacheError, IndexingError
from .logger import get_logger

logger = get_logger(__name__)

CACHE_FORMAT_VERSION = 3
CACHE_FORMAT_NAME = "tp.rag.chunks.v3"
CACHE_FILENAME = "chunks.json"
LEGACY_CACHE_FILENAME = "chunks.pkl"


@dataclass
class DocumentChunk:
    """Represents a chunk of documentation or code with metadata."""

    content: str
    file_path: str
    start_line: int
    end_line: int
    chunk_type: str  # 'doc', 'code', 'test', 'config', 'agent'
    language: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    chunk_id: Optional[str] = None

    def __post_init__(self):
        """Generate unique chunk ID if not provided."""
        if self.chunk_id is None:
            # Using SHA-256 for chunk IDs (non-security-critical but modern hash)
            content_hash = hashlib.sha256(f"{self.file_path}:{self.start_line}:{self.content}".encode()).hexdigest()[:8]
            self.chunk_id = f"{self.file_path}:{self.start_line}:{content_hash}"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize chunk to JSON-safe dict."""
        return {
            "content": self.content,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "chunk_type": self.chunk_type,
            "language": self.language,
            "metadata": self.metadata,
            "chunk_id": self.chunk_id,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "DocumentChunk":
        """Deserialize chunk from validated JSON dict."""
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid chunk payload type: {type(payload).__name__}")

        required_keys = {"content", "file_path", "start_line", "end_line", "chunk_type"}
        missing = required_keys - payload.keys()
        if missing:
            raise ValueError(f"Missing required chunk keys: {sorted(missing)}")

        content = payload["content"]
        file_path = payload["file_path"]
        start_line = payload["start_line"]
        end_line = payload["end_line"]
        chunk_type = payload["chunk_type"]
        language = payload.get("language")
        metadata = payload.get("metadata", {})
        chunk_id = payload.get("chunk_id")

        if not isinstance(content, str) or not content:
            raise ValueError("Chunk field 'content' must be a non-empty string")
        if not isinstance(file_path, str) or not file_path:
            raise ValueError("Chunk field 'file_path' must be a non-empty string")
        if not isinstance(start_line, int) or start_line < 1:
            raise ValueError("Chunk field 'start_line' must be a positive integer")
        if not isinstance(end_line, int) or end_line < start_line:
            raise ValueError("Chunk field 'end_line' must be an integer >= start_line")
        if not isinstance(chunk_type, str) or not chunk_type:
            raise ValueError("Chunk field 'chunk_type' must be a non-empty string")
        if language is not None and not isinstance(language, str):
            raise ValueError("Chunk field 'language' must be a string or null")
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise ValueError("Chunk field 'metadata' must be an object")
        if chunk_id is not None and not isinstance(chunk_id, str):
            raise ValueError("Chunk field 'chunk_id' must be a string or null")

        return cls(
            content=content,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            chunk_type=chunk_type,
            language=language,
            metadata=metadata,
            chunk_id=chunk_id,
        )


class RepositoryIndexer:
    """
    Indexes repository content for RAG retrieval.

    Chunking strategy:
    - 500-1000 tokens per chunk with 50-100 token overlap
    - Preserves code structure (functions, classes, docstrings)
    - Maintains file path and line number metadata
    """

    def __init__(
        self,
        repo_root: str,
        chunk_size_tokens: Optional[int] = None,
        overlap_tokens: Optional[int] = None,
        chars_per_token: Optional[float] = None,
        use_cache: Optional[bool] = None,
    ):
        """
        Initialize the indexer.

        Args:
            repo_root: Root directory of the repository
            chunk_size_tokens: Target size for each chunk in tokens (uses config if None)
            overlap_tokens: Overlap between chunks in tokens (uses config if None)
            chars_per_token: Approximate characters per token (uses config if None)
            use_cache: Enable persistent caching (uses config if None)
        """
        self.repo_root = Path(repo_root)
        self.chunks: List[DocumentChunk] = []
        self.catalog = DocumentationCatalog.read(self.repo_root)

        # Load config
        config = get_config()
        indexer_config = config.get_section("indexer")

        # Use config values as defaults
        self.chunk_size_tokens = chunk_size_tokens or indexer_config.get("chunk_size_tokens", 750)
        self.overlap_tokens = overlap_tokens if overlap_tokens is not None else indexer_config.get("overlap_tokens", 75)
        self.chars_per_token = chars_per_token or indexer_config.get("chars_per_token", 4.0)
        self.use_cache = use_cache if use_cache is not None else indexer_config.get("cache_enabled", True)

        # Calculate character-based sizes
        self.chunk_size = int(self.chunk_size_tokens * self.chars_per_token)
        self.overlap = int(self.overlap_tokens * self.chars_per_token)

        # Setup cache directory
        cache_dir = indexer_config.get("cache_dir", ".rag_cache")
        self.cache_dir = self.repo_root / cache_dir
        self.cache_file = self.cache_dir / CACHE_FILENAME
        self.legacy_cache_file = self.cache_dir / LEGACY_CACHE_FILENAME

        logger.debug(
            f"Initialized indexer: chunk_size={self.chunk_size}, " f"overlap={self.overlap}, cache_enabled={self.use_cache}"
        )

    def index_repository(self, force_reindex: bool = False) -> List[DocumentChunk]:
        """
        Index all relevant files in the repository.

        Args:
            force_reindex: Force reindexing even if cache exists

        Returns:
            List of document chunks with metadata
        """
        self.catalog = DocumentationCatalog.read(self.repo_root)
        files = self._collect_files()
        fingerprint = self._source_fingerprint(files) if self.use_cache else None
        if self.use_cache and not force_reindex and fingerprint is not None:
            cached_chunks = self._load_cache(fingerprint)
            if cached_chunks is not None:
                logger.info(f"Loaded {len(cached_chunks)} chunks from cache")
                self.chunks = cached_chunks
                return self.chunks

        logger.info("Indexing repository...")
        self.chunks = []
        indexed_sources = []
        try:
            for file_path, chunk_type in files:
                source_hash = self._index_file(file_path, chunk_type)
                if source_hash is not None:
                    indexed_sources.append((file_path, chunk_type, source_hash))
            logger.info(f"Indexed {len(self.chunks)} chunks from repository")

            consumed_fingerprint = self._fingerprint_sources(indexed_sources)
            final_fingerprint = self._source_fingerprint(self._collect_files())
            if final_fingerprint is None or consumed_fingerprint != final_fingerprint:
                self._downgrade_changed_snapshot(self.chunks)

            # Bind cache identity to the exact bytes chunked, not just matching
            # before/after snapshots: a transient A->B->A edit must not cache B as A.
            if self.use_cache and fingerprint is not None:
                if (
                    len(indexed_sources) == len(files)
                    and fingerprint == consumed_fingerprint
                    and fingerprint == final_fingerprint
                ):
                    self._save_cache(fingerprint)
                else:
                    logger.warning("Repository changed or a source read failed during indexing; cache was not updated")
        except Exception as e:
            logger.error(f"Error during indexing: {e}")
            raise IndexingError(f"Failed to index repository: {e}") from e
        return self.chunks

    @staticmethod
    def _downgrade_changed_snapshot(chunks: List[DocumentChunk]) -> None:
        """Withhold verified authority after observed drift; do not imply a live lock."""
        logger.warning("Repository snapshot changed during indexing; returned documentation authority was withheld")
        for chunk in chunks:
            if chunk.chunk_type not in {"doc", "agent"}:
                continue
            authority = chunk.metadata.get("documentation")
            if isinstance(authority, dict) and authority.get("verification") == "verified":
                chunk.metadata["documentation"] = {
                    **authority,
                    "authority": "unverified",
                    "verification": "changed-during-indexing",
                }

    def _collect_files(self) -> List[Tuple[Path, str]]:
        """Collect a unique, deterministic inventory and prune excluded trees."""
        files = {}
        for rel_path, chunk_type in (
            ("docs", "doc"),
            ("src", "code"),
            ("tests", "test"),
            (".github/agents", "agent"),
            ("examples", "code"),
        ):
            for root, directories, names in os.walk(self.repo_root / rel_path):
                directory = Path(root)
                directories[:] = sorted(name for name in directories if self._should_index(directory / name, directory=True))
                for name in names:
                    file_path = directory / name
                    if file_path.is_file() and self._should_index(file_path):
                        files[file_path] = chunk_type

        for pattern in (
            "README*.md",
            "CHANGELOG*.md",
            "CHANGE_LOG*.md",
            "*_GUIDE.md",
            "*_SUMMARY.md",
            "ARCHITECTURE.md",
            "PERFORMANCE*.md",
            "QUICKSTART*.md",
        ):
            for file_path in self.repo_root.glob(pattern):
                if file_path.is_file() and self._should_index(file_path):
                    files[file_path] = "doc"

        for rel_path in ("AGENTS.md", ".github/copilot-instructions.md"):
            file_path = self.repo_root / rel_path
            if file_path.is_file() and self._should_index(file_path):
                files[file_path] = "agent"
        return sorted(files.items(), key=lambda item: item[0].relative_to(self.repo_root).as_posix())

    def _fingerprint_sources(self, sources: List[Tuple[Path, str, str]], catalog_fingerprint: Optional[str] = None) -> str:
        """Hash an ordered inventory of source digests and effective settings."""
        digest = hashlib.sha256()
        settings = [CACHE_FORMAT_VERSION, self.chunk_size, self.overlap, catalog_fingerprint or self.catalog.fingerprint]
        digest.update(json.dumps(settings, separators=(",", ":")).encode("utf-8"))
        for file_path, chunk_type, source_hash in sources:
            entry = [file_path.relative_to(self.repo_root).as_posix(), chunk_type, source_hash]
            digest.update(json.dumps(entry, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        return digest.hexdigest()

    def _source_fingerprint(self, files: List[Tuple[Path, str]]) -> Optional[str]:
        """Bind cache reuse to source bytes, inventory, and effective chunk settings."""
        catalog = DocumentationCatalog.read(self.repo_root)
        if catalog.fingerprint is None:
            return None
        sources = []
        try:
            for file_path, chunk_type in files:
                source_hash = hashlib.sha256()
                with file_path.open("rb") as source:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        source_hash.update(block)
                sources.append((file_path, chunk_type, source_hash.hexdigest()))
        except OSError as exc:
            logger.warning("Cannot validate repository cache: %s", exc)
            return None
        # Compute with this fresh provenance snapshot without changing the catalog
        # whose metadata was consumed while chunking.
        return self._fingerprint_sources(sources, catalog_fingerprint=catalog.fingerprint)

    def _should_index(self, file_path: Path, directory: bool = False) -> bool:
        """Determine if a file should be indexed."""
        # Skip hidden files, cache, and build artifacts
        skip_dirs = {
            "__pycache__",
            ".git",
            ".pytest_cache",
            "node_modules",
            ".mypy_cache",
            ".tox",
            "venv",
            ".venv",
            "dist",
            "build",
        }

        if file_path.relative_to(self.repo_root).as_posix() == CATALOG_PATH:
            return False
        parts = file_path.relative_to(self.repo_root).parts
        if any(part in skip_dirs for part in parts):
            return False
        if file_path.is_relative_to(self.cache_dir):
            return False
        # Archived profiles are historical evidence, never live agent guidance.
        if parts[:2] == (".github", "agents") and "_archive" in parts:
            return False
        if directory:
            return True

        # Index specific file types
        valid_extensions = {".py", ".md", ".rst", ".txt", ".yaml", ".yml", ".json", ".toml", ".cfg", ".sh", ".bash"}

        return file_path.suffix in valid_extensions

    def _index_file(self, file_path: Path, chunk_type: str) -> Optional[str]:
        """Index one read snapshot and return the digest of those exact bytes."""
        try:
            source_bytes = file_path.read_bytes()
            source_hash = hashlib.sha256(source_bytes).hexdigest()
            # Preserve read_text's universal-newline behavior for existing chunks.
            content = source_bytes.decode("utf-8", errors="ignore").replace("\r\n", "\n").replace("\r", "\n")
        except Exception as e:
            logger.warning(f"Could not read {file_path}: {e}")
            return

        if not content.strip():
            return source_hash

        rel_path = str(file_path.relative_to(self.repo_root))
        language = self._detect_language(file_path)

        first_chunk = len(self.chunks)
        # For Python files, try to chunk by function/class
        if file_path.suffix == ".py" and chunk_type in ("code", "test"):
            self._chunk_python_file(content, rel_path, chunk_type, language)
        else:
            self._chunk_text(content, rel_path, chunk_type, language)
        if chunk_type in {"doc", "agent"}:
            authority = self.catalog.metadata(rel_path, source_hash)
            for chunk in self.chunks[first_chunk:]:
                chunk.metadata["documentation"] = authority.copy()
        return source_hash

    def _chunk_python_file(self, content: str, file_path: str, chunk_type: str, language: str):
        """Chunk Python files by functions and classes when possible."""
        lines = content.split("\n")

        # Try to identify function/class boundaries
        boundaries = self._find_python_boundaries(lines)

        if boundaries:
            # Chunk by logical units (functions/classes)
            for start, end in boundaries:
                chunk_content = "\n".join(lines[start:end])
                if len(chunk_content.strip()) > 50:  # Skip very small chunks
                    metadata = self._extract_python_metadata(chunk_content)
                    self.chunks.append(
                        DocumentChunk(
                            content=chunk_content,
                            file_path=file_path,
                            start_line=start + 1,
                            end_line=end,
                            chunk_type=chunk_type,
                            language=language,
                            metadata=metadata,
                        )
                    )
        else:
            # Fall back to text chunking
            self._chunk_text(content, file_path, chunk_type, language)

    def _find_python_boundaries(self, lines: List[str]) -> List[Tuple[int, int]]:
        """Find function and class boundaries in Python code."""
        boundaries = []
        current_start = None
        indent_stack = []

        for i, line in enumerate(lines):
            stripped = line.lstrip()

            # Detect function or class definition
            if stripped.startswith(("def ", "class ", "async def ")):
                if current_start is not None:
                    boundaries.append((current_start, i))
                current_start = i
                indent_stack = [len(line) - len(stripped)]
            elif current_start is not None:
                # Track indentation to detect end of block
                if stripped and not stripped.startswith("#"):
                    current_indent = len(line) - len(stripped)
                    if current_indent <= indent_stack[0] and i > current_start + 1:
                        boundaries.append((current_start, i))
                        current_start = None
                        indent_stack = []

        # Add final boundary
        if current_start is not None:
            boundaries.append((current_start, len(lines)))

        return boundaries

    def _extract_python_metadata(self, code: str) -> Dict:
        """Extract metadata from Python code chunk."""
        metadata = {}

        # Extract function/class name
        first_line = code.split("\n")[0].strip()
        if first_line.startswith("def ") or first_line.startswith("async def "):
            match = re.match(r"(?:async\s+)?def\s+(\w+)", first_line)
            if match:
                metadata["function_name"] = match.group(1)
                metadata["entity_type"] = "function"
        elif first_line.startswith("class "):
            match = re.match(r"class\s+(\w+)", first_line)
            if match:
                metadata["class_name"] = match.group(1)
                metadata["entity_type"] = "class"

        # Extract docstring if present
        docstring_match = re.search(r'(?:"""|\'\'\')(.*?)(?:"""|\'\'\')', code, re.DOTALL)
        if docstring_match:
            metadata["docstring"] = docstring_match.group(1).strip()[:200]  # First 200 chars

        return metadata

    def _chunk_text(self, content: str, file_path: str, chunk_type: str, language: Optional[str]):
        """Chunk text content with overlap."""
        lines = content.split("\n")

        current_chunk = []
        current_size = 0
        start_line = 0

        for i, line in enumerate(lines):
            line_size = len(line) + 1  # +1 for newline  # noqa: E741

            if current_size + line_size > self.chunk_size and current_chunk:
                # Create chunk
                chunk_content = "\n".join(current_chunk)
                self.chunks.append(
                    DocumentChunk(
                        content=chunk_content,
                        file_path=file_path,
                        start_line=start_line + 1,
                        end_line=i,
                        chunk_type=chunk_type,
                        language=language,
                        metadata=self._extract_text_metadata(chunk_content, file_path),
                    )
                )

                # Start new chunk with overlap
                overlap_lines = self._get_overlap_lines(current_chunk)
                current_chunk = overlap_lines + [line]
                current_size = sum(len(chunk_line) + 1 for chunk_line in current_chunk)
                start_line = i - len(overlap_lines)
            else:
                current_chunk.append(line)
                current_size += line_size

        # Add final chunk
        if current_chunk:
            chunk_content = "\n".join(current_chunk)
            self.chunks.append(
                DocumentChunk(
                    content=chunk_content,
                    file_path=file_path,
                    start_line=start_line + 1,
                    end_line=len(lines),
                    chunk_type=chunk_type,
                    language=language,
                    metadata=self._extract_text_metadata(chunk_content, file_path),
                )
            )

    def _get_overlap_lines(self, lines: List[str]) -> List[str]:
        """Get lines for overlap between chunks."""
        total_size = sum(len(line) + 1 for line in lines)
        if total_size <= self.overlap:
            return lines

        # Take from the end until we reach overlap size
        overlap_lines = []
        size = 0
        for line in reversed(lines):
            line_size = len(line) + 1
            if size + line_size > self.overlap:
                break
            overlap_lines.insert(0, line)
            size += line_size

        return overlap_lines

    def _extract_text_metadata(self, content: str, file_path: str) -> Dict:
        """Extract metadata from text content."""
        metadata = {}

        # Extract title from markdown
        if file_path.endswith(".md"):
            lines = content.split("\n")
            for line in lines[:10]:  # Check first 10 lines
                if line.startswith("# "):
                    metadata["title"] = line[2:].strip()
                    break
                if line.startswith("## "):
                    metadata["section"] = line[3:].strip()
                    break

        # Identify if it's a README or CHANGELOG
        file_lower = file_path.lower()
        if "readme" in file_lower:
            metadata["document_type"] = "readme"
        elif "changelog" in file_lower or "change_log" in file_lower:
            metadata["document_type"] = "changelog"
        elif "guide" in file_lower:
            metadata["document_type"] = "guide"

        return metadata

    def _detect_language(self, file_path: Path) -> Optional[str]:
        """Detect the programming/markup language of a file."""
        extension_map = {
            ".py": "python",
            ".md": "markdown",
            ".rst": "restructuredtext",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".json": "json",
            ".toml": "toml",
            ".sh": "bash",
            ".bash": "bash",
        }
        return extension_map.get(file_path.suffix)

    def _load_cache(self, source_fingerprint: str) -> Optional[List[DocumentChunk]]:
        """
        Load chunks from cache file.

        Returns:
            List of cached chunks or None if cache doesn't exist/is invalid
        """
        if not self.cache_file.exists():
            if self.legacy_cache_file.exists():
                logger.warning(
                    "Ignoring legacy insecure cache format %s; reindexing with JSON cache",
                    self.legacy_cache_file,
                )
            logger.debug("No cache file found")
            return None

        try:
            payload = json.loads(self.cache_file.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Cache root must be an object")
            if payload.get("cache_format") != CACHE_FORMAT_NAME:
                raise ValueError(f"Unexpected cache_format: {payload.get('cache_format')!r}")
            if payload.get("version") != CACHE_FORMAT_VERSION:
                raise ValueError(f"Unsupported cache version: {payload.get('version')!r}")
            if payload.get("source_fingerprint") != source_fingerprint:
                logger.debug("Repository content or chunk settings changed; reindexing")
                return None
            chunks_payload = payload.get("chunks")
            if not isinstance(chunks_payload, list):
                raise ValueError("Cache field 'chunks' must be a list")

            chunks = [DocumentChunk.from_dict(chunk_payload) for chunk_payload in chunks_payload]
            # The JSON cache is an optimization, never authority evidence. Bind
            # cached text to fresh source spans, and derive authority again from
            # the same catalog/source snapshot used to validate this cache hit.
            catalog = DocumentationCatalog.read(self.repo_root)
            if catalog.fingerprint is None:
                return None
            sources, snapshots = [], {}
            for file_path, chunk_type in self._collect_files():
                source_bytes = file_path.read_bytes()
                source_hash = hashlib.sha256(source_bytes).hexdigest()
                source_text = source_bytes.decode("utf-8", errors="ignore").replace("\r\n", "\n").replace("\r", "\n")
                snapshots[file_path.relative_to(self.repo_root).as_posix()] = (
                    chunk_type,
                    source_hash,
                    source_text.split("\n"),
                )
                sources.append((file_path, chunk_type, source_hash))
            if self._fingerprint_sources(sources, catalog_fingerprint=catalog.fingerprint) != source_fingerprint:
                return None
            for chunk in chunks:
                if chunk.file_path not in snapshots:
                    raise ValueError("Cached chunk is outside the current source inventory")
                chunk_type, source_hash, lines = snapshots[chunk.file_path]
                if (
                    chunk.chunk_type != chunk_type
                    or chunk.end_line > len(lines)
                    or chunk.content != "\n".join(lines[chunk.start_line - 1 : chunk.end_line])
                ):
                    raise ValueError("Cached chunk does not match the current source span")
                chunk.metadata.pop("documentation", None)
                if chunk_type in {"doc", "agent"}:
                    chunk.metadata["documentation"] = catalog.metadata(chunk.file_path, source_hash)
            self.catalog = catalog
            # A valid cache hit still needs an end-of-read snapshot check: its
            # source or catalog may change while cached spans are authenticated.
            if self._source_fingerprint(self._collect_files()) != source_fingerprint:
                self._downgrade_changed_snapshot(chunks)
            logger.debug(f"Loaded {len(chunks)} chunks from cache: {self.cache_file}")
            return chunks

        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            # Don't raise - caching is optional
            return None

    def _save_cache(self, source_fingerprint: Optional[str] = None):
        """Save chunks to cache file."""
        tmp_path = self.cache_file.with_name(f".{self.cache_file.name}.{uuid4().hex}.tmp")
        try:
            # Create cache directory if it doesn't exist
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "cache_format": CACHE_FORMAT_NAME,
                "version": CACHE_FORMAT_VERSION,
                "source_fingerprint": source_fingerprint,
                "chunks": [chunk.to_dict() for chunk in self.chunks],
            }
            serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)
            tmp_path.write_text(serialized + "\n", encoding="utf-8")
            tmp_path.replace(self.cache_file)

            logger.debug(f"Saved {len(self.chunks)} chunks to cache: {self.cache_file}")

        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
            # Don't raise - caching is optional
            # raise CacheError(f"Cache saving failed: {e}")
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    # Best effort cleanup only; cache writes are optional.
                    logger.debug("Failed to remove temporary cache file %s", tmp_path)

    def clear_cache(self):
        """Clear the cache file."""
        for cache_path in (self.cache_file, self.legacy_cache_file):
            if cache_path.exists():
                try:
                    cache_path.unlink()
                    logger.info(f"Cleared cache: {cache_path}")
                except Exception as e:
                    logger.warning(f"Failed to clear cache {cache_path}: {e}")
                    raise CacheError(f"Cache clearing failed for {cache_path}: {e}")

    def get_statistics(self) -> Dict:
        """Get indexing statistics."""
        stats = {
            "total_chunks": len(self.chunks),
            "by_type": {},
            "by_language": {},
            "total_chars": sum(len(c.content) for c in self.chunks),
        }

        for chunk in self.chunks:
            stats["by_type"][chunk.chunk_type] = stats["by_type"].get(chunk.chunk_type, 0) + 1
            if chunk.language:
                stats["by_language"][chunk.language] = stats["by_language"].get(chunk.language, 0) + 1

        return stats


def main():
    """CLI for indexing the repository."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Index repository for RAG system")
    parser.add_argument("--repo-root", default=".", help="Repository root directory")
    parser.add_argument("--output", default="index_stats.json", help="Output statistics file")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    indexer = RepositoryIndexer(args.repo_root)
    chunks = indexer.index_repository()
    stats = indexer.get_statistics()

    print(f"Indexed {stats['total_chunks']} chunks")
    print(f"Total characters: {stats['total_chars']:,}")
    print("\nBy type:")
    for chunk_type, count in sorted(stats["by_type"].items()):
        print(f"  {chunk_type}: {count}")
    print("\nBy language:")
    for language, count in sorted(stats["by_language"].items()):
        print(f"  {language}: {count}")

    if args.verbose:
        print("\nSample chunks:")
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"File: {chunk.file_path}")
            print(f"Lines: {chunk.start_line}-{chunk.end_line}")
            print(f"Type: {chunk.chunk_type}")
            print(f"Content preview: {chunk.content[:150]}...")

    # Save statistics
    with open(args.output, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStatistics saved to {args.output}")


if __name__ == "__main__":
    main()
