"""
Semantic Code Search Engine for RAG System

Provides intelligent code search with:
- Natural language to code translation
- Semantic similarity matching
- Cross-reference mapping
- Usage pattern detection
- API discovery
"""

import ast
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from indexer import DocumentChunk, RepositoryIndexer
from retriever import HybridRetriever


@dataclass
class CodeEntity:
    """Represents a code entity (function, class, method)."""

    name: str
    entity_type: str  # 'function', 'class', 'method'
    file_path: str
    line_number: int
    signature: str
    docstring: Optional[str] = None
    parameters: List[str] = field(default_factory=list)
    return_type: Optional[str] = None
    decorators: List[str] = field(default_factory=list)

    # Relationships
    calls: Set[str] = field(default_factory=set)  # Functions/methods this entity calls
    called_by: Set[str] = field(default_factory=set)  # Entities that call this
    imports: Set[str] = field(default_factory=set)  # Modules imported

    # Usage metadata
    complexity: int = 0  # Cyclomatic complexity
    usage_count: int = 0  # How many times it's referenced


@dataclass
class SemanticSearchResult:
    """Result from semantic code search."""

    entity: CodeEntity
    relevance_score: float
    match_reason: str  # Why this result is relevant
    code_snippet: str
    usage_examples: List[str] = field(default_factory=list)


class CodeParser:
    """Parse Python code to extract entities and relationships."""

    def __init__(self):
        """Initialize the parser."""
        self.entities: Dict[str, CodeEntity] = {}
        self.imports_graph: Dict[str, Set[str]] = defaultdict(set)
        self.call_graph: Dict[str, Set[str]] = defaultdict(set)

    def parse_file(self, file_path: str) -> List[CodeEntity]:
        """
        Parse a Python file and extract all code entities.

        Args:
            file_path: Path to Python file

        Returns:
            List of CodeEntity objects
        """
        entities = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()

            tree = ast.parse(source)

            # Extract imports
            imports = self._extract_imports(tree)

            # Extract classes and functions
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    entity = self._parse_class(node, file_path, imports)
                    entities.append(entity)

                    # Parse methods within class
                    for method_node in node.body:
                        if isinstance(method_node, ast.FunctionDef):
                            method_entity = self._parse_function(
                                method_node,
                                file_path,
                                imports,
                                parent_class=node.name
                            )
                            entities.append(method_entity)

                elif isinstance(node, ast.FunctionDef):
                    # Only top-level functions (not methods)
                    if not any(isinstance(parent, ast.ClassDef)
                              for parent in ast.walk(tree)):
                        entity = self._parse_function(node, file_path, imports)
                        entities.append(entity)

        except Exception as e:
            print(f"Warning: Could not parse {file_path}: {e}")

        return entities

    def _extract_imports(self, tree: ast.AST) -> Set[str]:
        """Extract all imports from AST."""
        imports = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)

        return imports

    def _parse_class(
        self,
        node: ast.ClassDef,
        file_path: str,
        imports: Set[str]
    ) -> CodeEntity:
        """Parse a class definition."""
        docstring = ast.get_docstring(node)

        # Extract decorators
        decorators = [
            self._get_decorator_name(dec)
            for dec in node.decorator_list
        ]

        # Get base classes
        bases = [self._get_name(base) for base in node.bases]
        signature = f"class {node.name}({', '.join(bases)})"

        return CodeEntity(
            name=node.name,
            entity_type='class',
            file_path=file_path,
            line_number=node.lineno,
            signature=signature,
            docstring=docstring,
            decorators=decorators,
            imports=imports.copy()
        )

    def _parse_function(
        self,
        node: ast.FunctionDef,
        file_path: str,
        imports: Set[str],
        parent_class: Optional[str] = None
    ) -> CodeEntity:
        """Parse a function or method definition."""
        docstring = ast.get_docstring(node)

        # Extract parameters
        parameters = [arg.arg for arg in node.args.args]

        # Extract return type
        return_type = None
        if node.returns:
            return_type = self._get_name(node.returns)

        # Extract decorators
        decorators = [
            self._get_decorator_name(dec)
            for dec in node.decorator_list
        ]

        # Build signature
        params_str = ', '.join(parameters)
        signature = f"def {node.name}({params_str})"
        if return_type:
            signature += f" -> {return_type}"

        # Extract function calls
        calls = self._extract_calls(node)

        # Calculate complexity
        complexity = self._calculate_complexity(node)

        entity_type = 'method' if parent_class else 'function'

        return CodeEntity(
            name=node.name,
            entity_type=entity_type,
            file_path=file_path,
            line_number=node.lineno,
            signature=signature,
            docstring=docstring,
            parameters=parameters,
            return_type=return_type,
            decorators=decorators,
            calls=calls,
            imports=imports.copy(),
            complexity=complexity
        )

    def _extract_calls(self, node: ast.AST) -> Set[str]:
        """Extract all function/method calls from a node."""
        calls = set()

        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                call_name = self._get_name(child.func)
                if call_name:
                    calls.add(call_name)

        return calls

    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        return complexity

    def _get_decorator_name(self, node: ast.AST) -> str:
        """Get decorator name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Call):
            return self._get_name(node.func)
        return str(node)

    def _get_name(self, node: ast.AST) -> Optional[str]:
        """Get name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            value = self._get_name(node.value)
            return f"{value}.{node.attr}" if value else node.attr
        return None


class SemanticCodeSearch:
    """
    Semantic code search engine.

    Features:
    - Natural language to code translation
    - Cross-reference mapping
    - Usage pattern detection
    - API discovery
    """

    def __init__(self, repo_root: str):
        """
        Initialize semantic search.

        Args:
            repo_root: Repository root directory
        """
        self.repo_root = Path(repo_root)
        self.parser = CodeParser()
        self.entities: Dict[str, CodeEntity] = {}
        self.entity_index: Dict[str, List[CodeEntity]] = defaultdict(list)

        # RAG components
        self.indexer = RepositoryIndexer(repo_root)
        self.retriever = HybridRetriever()

    def index_codebase(self):
        """Index all code entities in the repository."""
        print("Indexing codebase for semantic search...")

        # Parse all Python files
        for py_file in self.repo_root.rglob('*.py'):
            if self._should_index(py_file):
                entities = self.parser.parse_file(str(py_file))
                for entity in entities:
                    self.entities[f"{entity.file_path}:{entity.name}"] = entity
                    self.entity_index[entity.name.lower()].append(entity)

        # Build call graph
        self._build_call_graph()

        # Index for RAG retrieval
        chunks = self.indexer.index_repository()
        self.retriever.index(chunks)

        print(f"Indexed {len(self.entities)} code entities")

    def search(
        self,
        query: str,
        entity_type: Optional[str] = None,
        top_k: int = 10
    ) -> List[SemanticSearchResult]:
        """
        Semantic search for code entities.

        Args:
            query: Natural language query or code pattern
            entity_type: Filter by type ('function', 'class', 'method')
            top_k: Number of results

        Returns:
            List of search results
        """
        results = []

        # Step 1: Extract intent from query
        intent = self._analyze_query_intent(query)

        # Step 2: Search by different strategies
        name_matches = self._search_by_name(query, entity_type)
        semantic_matches = self._search_by_semantics(query, entity_type)
        usage_matches = self._search_by_usage_pattern(query, entity_type)

        # Step 3: Combine and rank results
        all_matches = {}

        for entity, score, reason in name_matches:
            key = f"{entity.file_path}:{entity.name}"
            if key not in all_matches or all_matches[key][1] < score:
                all_matches[key] = (entity, score, reason)

        for entity, score, reason in semantic_matches:
            key = f"{entity.file_path}:{entity.name}"
            current_score = all_matches.get(key, (None, 0, ""))[1]
            all_matches[key] = (entity, current_score + score * 0.5, reason)

        for entity, score, reason in usage_matches:
            key = f"{entity.file_path}:{entity.name}"
            current_score = all_matches.get(key, (None, 0, ""))[1]
            all_matches[key] = (entity, current_score + score * 0.3, reason)

        # Step 4: Create results with usage examples
        for entity, score, reason in sorted(
            all_matches.values(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]:
            code_snippet = self._get_code_snippet(entity)
            usage_examples = self._find_usage_examples(entity, max_examples=3)

            results.append(SemanticSearchResult(
                entity=entity,
                relevance_score=score,
                match_reason=reason,
                code_snippet=code_snippet,
                usage_examples=usage_examples
            ))

        return results

    def find_similar_code(
        self,
        code_snippet: str,
        top_k: int = 5
    ) -> List[SemanticSearchResult]:
        """
        Find similar code patterns in the repository.

        Args:
            code_snippet: Code to find similar patterns for
            top_k: Number of results

        Returns:
            List of similar code entities
        """
        # Extract key patterns from snippet
        patterns = self._extract_code_patterns(code_snippet)

        # Search for entities with similar patterns
        results = []
        for entity in self.entities.values():
            similarity = self._calculate_pattern_similarity(patterns, entity)
            if similarity > 0.3:  # Threshold
                results.append((entity, similarity, "Similar code pattern"))

        # Sort and create results
        results.sort(key=lambda x: x[1], reverse=True)

        return [
            SemanticSearchResult(
                entity=entity,
                relevance_score=score,
                match_reason=reason,
                code_snippet=self._get_code_snippet(entity),
                usage_examples=self._find_usage_examples(entity, max_examples=2)
            )
            for entity, score, reason in results[:top_k]
        ]

    def discover_api(
        self,
        task_description: str
    ) -> Dict[str, List[CodeEntity]]:
        """
        Discover relevant APIs for a task.

        Args:
            task_description: What you want to do

        Returns:
            Dictionary of API categories with relevant entities
        """
        # Search for relevant entities
        results = self.search(task_description, top_k=20)

        # Categorize by type and purpose
        api_map = {
            'core_functions': [],
            'utilities': [],
            'processors': [],
            'models': [],
            'configuration': []
        }

        for result in results:
            entity = result.entity

            if 'processor' in entity.file_path.lower():
                api_map['processors'].append(entity)
            elif 'model' in entity.file_path.lower():
                api_map['models'].append(entity)
            elif 'util' in entity.file_path.lower() or 'helper' in entity.name.lower():
                api_map['utilities'].append(entity)
            elif 'config' in entity.file_path.lower():
                api_map['configuration'].append(entity)
            else:
                api_map['core_functions'].append(entity)

        return {k: v for k, v in api_map.items() if v}

    def _analyze_query_intent(self, query: str) -> Dict[str, any]:
        """Analyze query to determine search intent."""
        query_lower = query.lower()

        intent = {
            'action': None,  # 'find', 'how_to', 'example', etc.
            'keywords': [],
            'entity_type': None
        }

        # Detect action verbs
        if any(word in query_lower for word in ['how to', 'how do i', 'way to']):
            intent['action'] = 'how_to'
        elif any(word in query_lower for word in ['find', 'search', 'locate']):
            intent['action'] = 'find'
        elif any(word in query_lower for word in ['example', 'show me', 'demonstrate']):
            intent['action'] = 'example'

        # Extract keywords
        keywords = re.findall(r'\b[a-z_][a-z0-9_]{2,}\b', query_lower)
        intent['keywords'] = keywords

        # Detect entity type
        if 'class' in query_lower:
            intent['entity_type'] = 'class'
        elif 'function' in query_lower or 'method' in query_lower:
            intent['entity_type'] = 'function'

        return intent

    def _search_by_name(
        self,
        query: str,
        entity_type: Optional[str]
    ) -> List[Tuple[CodeEntity, float, str]]:
        """Search by entity name."""
        results = []
        query_lower = query.lower()
        keywords = re.findall(r'\b[a-z_][a-z0-9_]{2,}\b', query_lower)

        for name, entities in self.entity_index.items():
            for entity in entities:
                if entity_type and entity.entity_type != entity_type:
                    continue

                # Calculate name similarity
                score = 0.0

                # Exact match
                if name == query_lower:
                    score = 10.0
                # Contains query
                elif query_lower in name:
                    score = 5.0
                # Keyword matches
                else:
                    keyword_matches = sum(1 for kw in keywords if kw in name)
                    score = keyword_matches * 2.0

                if score > 0:
                    results.append((
                        entity,
                        score,
                        f"Name match: {entity.name}"
                    ))

        return results

    def _search_by_semantics(
        self,
        query: str,
        entity_type: Optional[str]
    ) -> List[Tuple[CodeEntity, float, str]]:
        """Search using RAG semantic retrieval."""
        # Use hybrid retriever for semantic search
        chunk_filter = None
        if entity_type:
            chunk_filter = ['code']  # Only search code chunks

        retrieval_results = self.retriever.retrieve(
            query,
            top_k=20,
            chunk_type_filter=chunk_filter
        )

        results = []
        for retrieval_result in retrieval_results:
            # Find entities in this file
            file_path = retrieval_result.file_path
            for entity in self.entities.values():
                if entity.file_path == file_path:
                    if not entity_type or entity.entity_type == entity_type:
                        # Check if entity is in the retrieved chunk
                        if (entity.line_number >= retrieval_result.start_line and
                            entity.line_number <= retrieval_result.end_line):
                            results.append((
                                entity,
                                retrieval_result.score * 0.5,
                                f"Semantic match in {file_path}"
                            ))

        return results

    def _search_by_usage_pattern(
        self,
        query: str,
        entity_type: Optional[str]
    ) -> List[Tuple[CodeEntity, float, str]]:
        """Search based on usage patterns."""
        results = []
        query_lower = query.lower()

        # Look for common usage patterns in query
        patterns = {
            'depth': ['depth', 'estimate', 'map'],
            'color': ['color', 'grade', 'lut', 'tone'],
            'material': ['material', 'surface', 'texture'],
            'enhance': ['enhance', 'improve', 'optimize'],
            'batch': ['batch', 'process', 'multiple']
        }

        for pattern_name, pattern_keywords in patterns.items():
            if any(kw in query_lower for kw in pattern_keywords):
                # Find entities related to this pattern
                for entity in self.entities.values():
                    if entity_type and entity.entity_type != entity_type:
                        continue

                    # Check if entity is related to pattern
                    entity_text = f"{entity.name} {entity.file_path} {entity.docstring or ''}".lower()
                    match_count = sum(1 for kw in pattern_keywords if kw in entity_text)

                    if match_count > 0:
                        score = match_count * 1.5
                        results.append((
                            entity,
                            score,
                            f"Usage pattern: {pattern_name}"
                        ))

        return results

    def _build_call_graph(self):
        """Build call graph between entities."""
        for entity in self.entities.values():
            for called_name in entity.calls:
                # Find entities with this name
                if called_name in self.entity_index:
                    for called_entity in self.entity_index[called_name]:
                        called_entity.called_by.add(entity.name)
                        entity.usage_count += 1

    def _get_code_snippet(self, entity: CodeEntity) -> str:
        """Get code snippet for an entity."""
        try:
            with open(entity.file_path, 'r') as f:
                lines = f.readlines()

            # Get 10 lines starting from entity
            start = max(0, entity.line_number - 1)
            end = min(len(lines), start + 10)

            return ''.join(lines[start:end])
        except Exception:
            return entity.signature

    def _find_usage_examples(
        self,
        entity: CodeEntity,
        max_examples: int = 3
    ) -> List[str]:
        """Find usage examples for an entity."""
        examples = []

        # Search in test files first
        test_chunks = self.retriever.retrieve(
            entity.name,
            top_k=10,
            chunk_type_filter=['test']
        )

        for chunk in test_chunks[:max_examples]:
            if entity.name in chunk.content:
                examples.append(chunk.content[:300])

        return examples

    def _extract_code_patterns(self, code: str) -> Dict[str, any]:
        """Extract patterns from code snippet."""
        patterns = {
            'imports': set(),
            'function_calls': set(),
            'keywords': set()
        }

        try:
            tree = ast.parse(code)

            # Extract patterns
            patterns['imports'] = self.parser._extract_imports(tree)

            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    call_name = self.parser._get_name(node.func)
                    if call_name:
                        patterns['function_calls'].add(call_name)
        except Exception:
            pass

        # Extract keywords
        keywords = re.findall(r'\b[a-z_][a-z0-9_]{3,}\b', code.lower())
        patterns['keywords'] = set(keywords)

        return patterns

    def _calculate_pattern_similarity(
        self,
        patterns: Dict[str, any],
        entity: CodeEntity
    ) -> float:
        """Calculate similarity between patterns and entity."""
        similarity = 0.0

        # Compare imports
        if patterns['imports']:
            common_imports = patterns['imports'].intersection(entity.imports)
            similarity += len(common_imports) / len(patterns['imports']) * 0.3

        # Compare function calls
        if patterns['function_calls']:
            common_calls = patterns['function_calls'].intersection(entity.calls)
            similarity += len(common_calls) / len(patterns['function_calls']) * 0.4

        # Compare keywords
        if patterns['keywords']:
            entity_text = f"{entity.name} {entity.docstring or ''}".lower()
            keyword_matches = sum(1 for kw in patterns['keywords'] if kw in entity_text)
            similarity += (keyword_matches / len(patterns['keywords'])) * 0.3

        return similarity

    def _should_index(self, file_path: Path) -> bool:
        """Check if file should be indexed."""
        # Skip test files, cache, build artifacts
        skip_dirs = {'__pycache__', '.pytest_cache', 'build', 'dist', '.git'}

        if any(part in skip_dirs for part in file_path.parts):
            return False

        # Skip __init__.py and setup.py
        if file_path.name in {'__init__.py', 'setup.py'}:
            return False

        return True


def main():
    """CLI for semantic code search."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description='Semantic Code Search')
    parser.add_argument('--repo-root', default='.', help='Repository root')
    parser.add_argument('--query', required=True, help='Search query')
    parser.add_argument('--type', choices=['function', 'class', 'method'],
                       help='Filter by entity type')
    parser.add_argument('--top-k', type=int, default=10, help='Number of results')
    parser.add_argument('--discover-api', action='store_true',
                       help='Discover API for task')
    parser.add_argument('--json', action='store_true', help='Output as JSON')

    args = parser.parse_args()

    # Initialize and index
    print("Initializing semantic search...")
    search = SemanticCodeSearch(args.repo_root)
    search.index_codebase()

    if args.discover_api:
        # API discovery mode
        print(f"\nDiscovering API for: {args.query}")
        api_map = search.discover_api(args.query)

        for category, entities in api_map.items():
            print(f"\n{category.replace('_', ' ').title()}:")
            for entity in entities:
                print(f"  - {entity.name} ({entity.file_path}:{entity.line_number})")
                if entity.docstring:
                    print(f"    {entity.docstring.split(chr(10))[0]}")
    else:
        # Normal search
        print(f"\nSearching for: {args.query}")
        results = search.search(args.query, entity_type=args.type, top_k=args.top_k)

        if args.json:
            # JSON output
            output = []
            for result in results:
                output.append({
                    'name': result.entity.name,
                    'type': result.entity.entity_type,
                    'file_path': result.entity.file_path,
                    'line_number': result.entity.line_number,
                    'signature': result.entity.signature,
                    'relevance_score': result.relevance_score,
                    'match_reason': result.match_reason,
                    'docstring': result.entity.docstring
                })
            print(json.dumps(output, indent=2))
        else:
            # Human-readable output
            print(f"\nFound {len(results)} results:")
            print("=" * 80)

            for i, result in enumerate(results, 1):
                print(f"\n[{i}] {result.entity.name} ({result.entity.entity_type})")
                print(f"    {result.entity.file_path}:{result.entity.line_number}")
                print(f"    Score: {result.relevance_score:.2f}")
                print(f"    Reason: {result.match_reason}")
                print(f"    Signature: {result.entity.signature}")

                if result.entity.docstring:
                    print(f"    Description: {result.entity.docstring.split(chr(10))[0]}")

                if result.usage_examples:
                    print(f"    Usage examples: {len(result.usage_examples)} found")


if __name__ == '__main__':
    main()
