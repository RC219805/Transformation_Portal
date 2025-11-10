"""
Repository Content Indexer for RAG System

Indexes repository content (docs/, src/, tests/, agent files, changelogs, READMEs)
into chunks with metadata for efficient retrieval.
"""

import hashlib
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import get_config
from .exceptions import CacheError, IndexingError
from .logger import get_logger

logger = get_logger(__name__)


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
            content_hash = hashlib.sha256(
                f"{self.file_path}:{self.start_line}:{self.content}".encode()
            ).hexdigest()[:8]
            self.chunk_id = f"{self.file_path}:{self.start_line}:{content_hash}"


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

        # Load config
        config = get_config()
        indexer_config = config.get_section('indexer')

        # Use config values as defaults
        self.chunk_size_tokens = chunk_size_tokens or indexer_config.get('chunk_size_tokens', 750)
        self.overlap_tokens = overlap_tokens or indexer_config.get('overlap_tokens', 75)
        self.chars_per_token = chars_per_token or indexer_config.get('chars_per_token', 4.0)
        self.use_cache = use_cache if use_cache is not None else indexer_config.get('cache_enabled', True)

        # Calculate character-based sizes
        self.chunk_size = int(self.chunk_size_tokens * self.chars_per_token)
        self.overlap = int(self.overlap_tokens * self.chars_per_token)

        # Setup cache directory
        cache_dir = indexer_config.get('cache_dir', '.rag_cache')
        self.cache_dir = self.repo_root / cache_dir
        self.cache_file = self.cache_dir / 'chunks.pkl'

        logger.debug(
            f"Initialized indexer: chunk_size={self.chunk_size}, "
            f"overlap={self.overlap}, cache_enabled={self.use_cache}"
        )

    def index_repository(self, force_reindex: bool = False) -> List[DocumentChunk]:
        """
        Index all relevant files in the repository.

        Args:
            force_reindex: Force reindexing even if cache exists

        Returns:
            List of document chunks with metadata
        """
        # Try to load from cache if enabled
        if self.use_cache and not force_reindex:
            cached_chunks = self._load_cache()
            if cached_chunks is not None:
                logger.info(f"Loaded {len(cached_chunks)} chunks from cache")
                self.chunks = cached_chunks
                return self.chunks

        logger.info("Indexing repository...")
        self.chunks = []

        try:
            # Index documentation
            self._index_directory('docs', chunk_type='doc')

            # Index source code
            if (self.repo_root / 'src').exists():
                self._index_directory('src', chunk_type='code')

            # Index tests
            if (self.repo_root / 'tests').exists():
                self._index_directory('tests', chunk_type='test')

            # Index agent definitions
            self._index_directory('.github/agents', chunk_type='agent')

            # Index top-level markdown files (READMEs, CHANGELOGs, etc.)
            self._index_top_level_files()

            # Index example code
            if (self.repo_root / 'examples').exists():
                self._index_directory('examples', chunk_type='code')

            logger.info(f"Indexed {len(self.chunks)} chunks from repository")

            # Save to cache if enabled
            if self.use_cache:
                self._save_cache()

        except Exception as e:
            logger.error(f"Error during indexing: {e}")
            raise IndexingError(f"Failed to index repository: {e}")

        return self.chunks

    def _index_directory(self, rel_path: str, chunk_type: str):
        """Index all files in a directory."""
        directory = self.repo_root / rel_path
        if not directory.exists():
            return

        for file_path in directory.rglob('*'):
            if file_path.is_file() and self._should_index(file_path):
                self._index_file(file_path, chunk_type)

    def _index_top_level_files(self):
        """Index important top-level files like README, CHANGELOG, etc."""
        patterns = [
            'README*.md', 'CHANGELOG*.md', 'CHANGE_LOG*.md',
            '*_GUIDE.md', '*_SUMMARY.md', 'ARCHITECTURE.md',
            'PERFORMANCE*.md', 'QUICKSTART*.md'
        ]

        for pattern in patterns:
            for file_path in self.repo_root.glob(pattern):
                if file_path.is_file():
                    self._index_file(file_path, 'doc')

    def _should_index(self, file_path: Path) -> bool:
        """Determine if a file should be indexed."""
        # Skip hidden files, cache, and build artifacts
        skip_dirs = {
            '__pycache__', '.git', '.pytest_cache', 'node_modules',
            '.mypy_cache', '.tox', 'venv', '.venv', 'dist', 'build'
        }

        if any(part in skip_dirs for part in file_path.parts):
            return False

        # Index specific file types
        valid_extensions = {
            '.py', '.md', '.rst', '.txt', '.yaml', '.yml',
            '.json', '.toml', '.cfg', '.sh', '.bash'
        }

        return file_path.suffix in valid_extensions

    def _index_file(self, file_path: Path, chunk_type: str):
        """Index a single file into chunks."""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
        except Exception as e:
            logger.warning(f"Could not read {file_path}: {e}")
            return

        rel_path = str(file_path.relative_to(self.repo_root))
        language = self._detect_language(file_path)

        # For Python files, try to chunk by function/class
        if file_path.suffix == '.py' and chunk_type in ('code', 'test'):
            self._chunk_python_file(content, rel_path, chunk_type, language)
        else:
            self._chunk_text(content, rel_path, chunk_type, language)

    def _chunk_python_file(self, content: str, file_path: str, chunk_type: str, language: str):
        """Chunk Python files by functions and classes when possible."""
        lines = content.split('\n')

        # Try to identify function/class boundaries
        boundaries = self._find_python_boundaries(lines)

        if boundaries:
            # Chunk by logical units (functions/classes)
            for start, end in boundaries:
                chunk_content = '\n'.join(lines[start:end])
                if len(chunk_content.strip()) > 50:  # Skip very small chunks
                    metadata = self._extract_python_metadata(chunk_content)
                    self.chunks.append(DocumentChunk(
                        content=chunk_content,
                        file_path=file_path,
                        start_line=start + 1,
                        end_line=end,
                        chunk_type=chunk_type,
                        language=language,
                        metadata=metadata
                    ))
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
            if stripped.startswith(('def ', 'class ', 'async def ')):
                if current_start is not None:
                    boundaries.append((current_start, i))
                current_start = i
                indent_stack = [len(line) - len(stripped)]
            elif current_start is not None:
                # Track indentation to detect end of block
                if stripped and not stripped.startswith('#'):
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
        first_line = code.split('\n')[0].strip()
        if first_line.startswith('def ') or first_line.startswith('async def '):
            match = re.match(r'(?:async\s+)?def\s+(\w+)', first_line)
            if match:
                metadata['function_name'] = match.group(1)
                metadata['entity_type'] = 'function'
        elif first_line.startswith('class '):
            match = re.match(r'class\s+(\w+)', first_line)
            if match:
                metadata['class_name'] = match.group(1)
                metadata['entity_type'] = 'class'

        # Extract docstring if present
        docstring_match = re.search(r'(?:"""|\'\'\')(.*?)(?:"""|\'\'\')', code, re.DOTALL)
        if docstring_match:
            metadata['docstring'] = docstring_match.group(1).strip()[:200]  # First 200 chars

        return metadata

    def _chunk_text(self, content: str, file_path: str, chunk_type: str, language: Optional[str]):
        """Chunk text content with overlap."""
        lines = content.split('\n')

        current_chunk = []
        current_size = 0
        start_line = 0

        for i, line in enumerate(lines):
            line_size = len(line) + 1  # +1 for newline  # noqa: E741

            if current_size + line_size > self.chunk_size and current_chunk:
                # Create chunk
                chunk_content = '\n'.join(current_chunk)
                self.chunks.append(DocumentChunk(
                    content=chunk_content,
                    file_path=file_path,
                    start_line=start_line + 1,
                    end_line=i,
                    chunk_type=chunk_type,
                    language=language,
                    metadata=self._extract_text_metadata(chunk_content, file_path)
                ))

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
            chunk_content = '\n'.join(current_chunk)
            self.chunks.append(DocumentChunk(
                content=chunk_content,
                file_path=file_path,
                start_line=start_line + 1,
                end_line=len(lines),
                chunk_type=chunk_type,
                language=language,
                metadata=self._extract_text_metadata(chunk_content, file_path)
            ))

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
        if file_path.endswith('.md'):
            lines = content.split('\n')
            for line in lines[:10]:  # Check first 10 lines
                if line.startswith('# '):
                    metadata['title'] = line[2:].strip()
                    break
                if line.startswith('## '):
                    metadata['section'] = line[3:].strip()
                    break

        # Identify if it's a README or CHANGELOG
        file_lower = file_path.lower()
        if 'readme' in file_lower:
            metadata['document_type'] = 'readme'
        elif 'changelog' in file_lower or 'change_log' in file_lower:
            metadata['document_type'] = 'changelog'
        elif 'guide' in file_lower:
            metadata['document_type'] = 'guide'

        return metadata

    def _detect_language(self, file_path: Path) -> Optional[str]:
        """Detect the programming/markup language of a file."""
        extension_map = {
            '.py': 'python',
            '.md': 'markdown',
            '.rst': 'restructuredtext',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.json': 'json',
            '.toml': 'toml',
            '.sh': 'bash',
            '.bash': 'bash',
        }
        return extension_map.get(file_path.suffix)

    def _load_cache(self) -> Optional[List[DocumentChunk]]:
        """
        Load chunks from cache file.

        Returns:
            List of cached chunks or None if cache doesn't exist/is invalid
        """
        if not self.cache_file.exists():
            logger.debug("No cache file found")
            return None

        try:
            with open(self.cache_file, 'rb') as f:
                chunks = pickle.load(f)

            logger.debug(f"Loaded {len(chunks)} chunks from cache: {self.cache_file}")
            return chunks

        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            # Don't raise - caching is optional
            return None

    def _save_cache(self):
        """Save chunks to cache file."""
        try:
            # Create cache directory if it doesn't exist
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.chunks, f, protocol=pickle.HIGHEST_PROTOCOL)

            logger.debug(f"Saved {len(self.chunks)} chunks to cache: {self.cache_file}")

        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
            # Don't raise - caching is optional
            # raise CacheError(f"Cache saving failed: {e}")

    def clear_cache(self):
        """Clear the cache file."""
        if self.cache_file.exists():
            try:
                self.cache_file.unlink()
                logger.info(f"Cleared cache: {self.cache_file}")
            except Exception as e:
                logger.warning(f"Failed to clear cache: {e}")
                raise CacheError(f"Cache clearing failed: {e}")

    def get_statistics(self) -> Dict:
        """Get indexing statistics."""
        stats = {
            'total_chunks': len(self.chunks),
            'by_type': {},
            'by_language': {},
            'total_chars': sum(len(c.content) for c in self.chunks),
        }

        for chunk in self.chunks:
            stats['by_type'][chunk.chunk_type] = stats['by_type'].get(chunk.chunk_type, 0) + 1
            if chunk.language:
                stats['by_language'][chunk.language] = stats['by_language'].get(chunk.language, 0) + 1

        return stats


def main():
    """CLI for indexing the repository."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description='Index repository for RAG system')
    parser.add_argument('--repo-root', default='.', help='Repository root directory')
    parser.add_argument('--output', default='index_stats.json', help='Output statistics file')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    indexer = RepositoryIndexer(args.repo_root)
    chunks = indexer.index_repository()
    stats = indexer.get_statistics()

    print(f"Indexed {stats['total_chunks']} chunks")
    print(f"Total characters: {stats['total_chars']:,}")
    print("\nBy type:")
    for chunk_type, count in sorted(stats['by_type'].items()):
        print(f"  {chunk_type}: {count}")
    print("\nBy language:")
    for language, count in sorted(stats['by_language'].items()):
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
    with open(args.output, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nStatistics saved to {args.output}")


if __name__ == '__main__':
    main()
