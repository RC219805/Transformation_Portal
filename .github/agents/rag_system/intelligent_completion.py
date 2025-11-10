"""
Intelligent Code Completion System

Provides context-aware code completions based on:
- Repository patterns and conventions
- Similar code in the codebase
- Function signatures and usage patterns
- Import suggestions
- Parameter completion with type hints
"""

import ast
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from semantic_search import CodeEntity, CodeParser, SemanticCodeSearch


@dataclass
class CompletionSuggestion:
    """A code completion suggestion."""

    text: str
    completion_type: str  # 'import', 'function', 'parameter', 'snippet'
    confidence: float  # 0.0-1.0
    context: str  # Why this suggestion
    source_file: Optional[str] = None  # Where this pattern comes from
    usage_count: int = 0  # How many times it's used in repo
    example: Optional[str] = None  # Usage example


class PatternExtractor:
    """Extract common code patterns from repository."""

    def __init__(self):
        """Initialize pattern extractor."""
        self.import_patterns: Dict[str, Counter] = defaultdict(Counter)
        self.function_call_patterns: Dict[str, List[Tuple[str, List[str]]]] = defaultdict(list)
        self.parameter_patterns: Dict[str, Counter] = defaultdict(Counter)
        self.snippet_patterns: List[Tuple[str, str, int]] = []  # (pattern, code, count)

    def analyze_repository(self, search_engine: SemanticCodeSearch):
        """
        Analyze repository to extract patterns.

        Args:
            search_engine: Semantic search engine with indexed codebase
        """
        print("Extracting code patterns...")

        for entity in search_engine.entities.values():
            # Extract import patterns
            for imp in entity.imports:
                file_type = self._get_file_category(entity.file_path)
                self.import_patterns[file_type][imp] += 1

            # Extract function call patterns
            if entity.entity_type in ('function', 'method'):
                for call in entity.calls:
                    self.function_call_patterns[entity.name].append(
                        (call, entity.parameters)
                    )

            # Extract parameter patterns
            if entity.parameters:
                param_tuple = tuple(entity.parameters)
                self.parameter_patterns[entity.name][param_tuple] += 1

        # Extract common code snippets
        self._extract_snippet_patterns(search_engine)

        print(f"Extracted {len(self.import_patterns)} import patterns")
        print(f"Extracted {len(self.function_call_patterns)} call patterns")

    def _extract_snippet_patterns(self, search_engine: SemanticCodeSearch):
        """Extract common code snippets (e.g., error handling, logging)."""
        # Common patterns to look for
        pattern_signatures = {
            'try_except': r'try:.*?except.*?:',
            'context_manager': r'with\s+\w+\(',
            'list_comprehension': r'\[.*?for\s+.*?in\s+.*?\]',
            'logging': r'(?:logger|logging)\.',
            'type_check': r'isinstance\(',
        }

        snippet_counts = defaultdict(int)

        # Scan through code to find patterns
        for entity in search_engine.entities.values():
            try:
                with open(entity.file_path, 'r') as f:
                    content = f.read()

                for pattern_name, pattern_regex in pattern_signatures.items():
                    matches = re.findall(pattern_regex, content, re.DOTALL)
                    for match in matches:
                        # Normalize and store
                        normalized = self._normalize_snippet(match)
                        if normalized:
                            snippet_counts[(pattern_name, normalized)] += 1
            except Exception:
                pass

        # Store top patterns
        for (pattern_name, code), count in sorted(
            snippet_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:50]:
            self.snippet_patterns.append((pattern_name, code, count))

    def _normalize_snippet(self, code: str) -> Optional[str]:
        """Normalize a code snippet."""
        # Remove extra whitespace
        code = re.sub(r'\s+', ' ', code.strip())

        # Minimum length
        if len(code) < 10:
            return None

        return code[:200]  # Limit length

    def _get_file_category(self, file_path: str) -> str:
        """Categorize file by purpose."""
        path_lower = file_path.lower()

        if '/test' in path_lower or 'test_' in path_lower:
            return 'test'
        elif '/pipeline' in path_lower or 'pipeline' in path_lower:
            return 'pipeline'
        elif '/processor' in path_lower or 'processor' in path_lower:
            return 'processor'
        elif '/util' in path_lower or 'helper' in path_lower:
            return 'utility'
        elif '/model' in path_lower:
            return 'model'
        else:
            return 'core'


class IntelligentCompletion:
    """
    Intelligent code completion engine.

    Features:
    - Context-aware suggestions
    - Import completion based on file type
    - Function parameter suggestions
    - Common snippet completion
    """

    def __init__(self, semantic_search: SemanticCodeSearch):
        """
        Initialize completion engine.

        Args:
            semantic_search: Semantic search engine
        """
        self.search = semantic_search
        self.patterns = PatternExtractor()
        self.patterns.analyze_repository(semantic_search)

    def suggest_imports(
        self,
        partial_import: str,
        file_type: str = 'core',
        top_k: int = 10
    ) -> List[CompletionSuggestion]:
        """
        Suggest imports based on partial text and file type.

        Args:
            partial_import: Partial import string (e.g., "from PIL")
            file_type: Type of file (pipeline, test, processor, etc.)
            top_k: Number of suggestions

        Returns:
            List of import suggestions
        """
        suggestions = []

        # Get common imports for this file type
        if file_type in self.patterns.import_patterns:
            common_imports = self.patterns.import_patterns[file_type]

            for imp, count in common_imports.most_common(top_k * 2):
                # Check if it matches partial
                if not partial_import or partial_import.lower() in imp.lower():
                    confidence = min(1.0, count / 10.0)  # Normalize

                    suggestions.append(CompletionSuggestion(
                        text=f"import {imp}",
                        completion_type='import',
                        confidence=confidence,
                        context=f"Common in {file_type} files",
                        usage_count=count
                    ))

        # Sort by confidence and usage
        suggestions.sort(key=lambda x: (x.confidence, x.usage_count), reverse=True)

        return suggestions[:top_k]

    def suggest_function_calls(
        self,
        context: str,
        cursor_position: int,
        top_k: int = 5
    ) -> List[CompletionSuggestion]:
        """
        Suggest function calls based on context.

        Args:
            context: Code context around cursor
            cursor_position: Position in context
            top_k: Number of suggestions

        Returns:
            List of function call suggestions
        """
        suggestions = []

        # Extract what's being typed
        before_cursor = context[:cursor_position]

        # Check if we're in a function call position
        # Look for patterns like "result = " or "self."
        call_match = re.search(r'([\w.]+)\s*=\s*$', before_cursor)
        if call_match:
            # User is assigning, suggest functions that return values
            relevant_functions = self._find_return_value_functions()

            for entity in relevant_functions[:top_k]:
                example = self._create_function_call_example(entity)

                suggestions.append(CompletionSuggestion(
                    text=example,
                    completion_type='function',
                    confidence=0.7,
                    context=f"Function that returns {entity.return_type or 'value'}",
                    source_file=entity.file_path,
                    example=self._find_usage_example(entity)
                ))

        # Check for method call pattern (after a dot)
        method_match = re.search(r'(\w+)\.$', before_cursor)
        if method_match:
            obj_name = method_match.group(1)

            # Try to infer object type and suggest methods
            suggestions.extend(
                self._suggest_methods_for_object(obj_name, top_k)
            )

        return suggestions

    def suggest_parameters(
        self,
        function_name: str,
        existing_params: List[str],
        top_k: int = 5
    ) -> List[CompletionSuggestion]:
        """
        Suggest parameters for a function call.

        Args:
            function_name: Name of function being called
            existing_params: Parameters already provided
            top_k: Number of suggestions

        Returns:
            List of parameter suggestions
        """
        suggestions = []

        # Find entities with this function name
        if function_name.lower() in self.search.entity_index:
            entities = self.search.entity_index[function_name.lower()]

            for entity in entities:
                # Get expected parameters
                expected_params = entity.parameters

                # Find missing parameters
                missing_params = [
                    p for p in expected_params
                    if p not in existing_params and p != 'self'
                ]

                for param in missing_params:
                    # Try to infer type and suggest value
                    param_suggestion = self._infer_parameter_value(param, entity)

                    suggestions.append(CompletionSuggestion(
                        text=param_suggestion,
                        completion_type='parameter',
                        confidence=0.8,
                        context=f"Parameter for {function_name}",
                        source_file=entity.file_path,
                        example=f"{function_name}(..., {param_suggestion}, ...)"
                    ))

        return suggestions[:top_k]

    def suggest_snippets(
        self,
        context: str,
        intent: str,
        top_k: int = 3
    ) -> List[CompletionSuggestion]:
        """
        Suggest code snippets based on intent.

        Args:
            context: Current code context
            intent: What user wants to do (e.g., "error handling", "logging")
            top_k: Number of suggestions

        Returns:
            List of snippet suggestions
        """
        suggestions = []

        # Match intent to pattern types
        intent_lower = intent.lower()

        for pattern_name, code, count in self.patterns.snippet_patterns:
            # Check if pattern matches intent
            relevance = 0.0

            if pattern_name in intent_lower or intent_lower in pattern_name:
                relevance = 1.0
            elif any(word in pattern_name for word in intent_lower.split()):
                relevance = 0.6

            if relevance > 0:
                confidence = min(1.0, (count / 20.0) * relevance)

                suggestions.append(CompletionSuggestion(
                    text=code,
                    completion_type='snippet',
                    confidence=confidence,
                    context=f"Common {pattern_name} pattern",
                    usage_count=count
                ))

        # Sort and return top
        suggestions.sort(key=lambda x: x.confidence, reverse=True)
        return suggestions[:top_k]

    def complete_pipeline_workflow(
        self,
        pipeline_type: str,
        current_step: str
    ) -> List[CompletionSuggestion]:
        """
        Suggest next steps in a pipeline workflow.

        Args:
            pipeline_type: Type of pipeline (depth, material, color, etc.)
            current_step: Current processing step

        Returns:
            List of next step suggestions
        """
        # Define common pipeline workflows
        workflows = {
            'depth': [
                'load_image',
                'estimate_depth',
                'apply_depth_effects',
                'apply_tone_mapping',
                'save_result'
            ],
            'material': [
                'load_image',
                'detect_materials',
                'enhance_surfaces',
                'apply_color_grade',
                'save_result'
            ],
            'color': [
                'load_image',
                'apply_lut',
                'adjust_exposure',
                'adjust_saturation',
                'save_result'
            ],
            'video': [
                'load_video',
                'build_filter_graph',
                'apply_filters',
                'encode_output',
                'save_result'
            ]
        }

        suggestions = []

        if pipeline_type in workflows:
            workflow = workflows[pipeline_type]

            # Find current position
            try:
                current_idx = workflow.index(current_step)
                next_steps = workflow[current_idx + 1:]

                for i, step in enumerate(next_steps[:3]):
                    # Find entities that implement this step
                    entities = self.search.search(step, top_k=1)

                    if entities:
                        entity = entities[0].entity
                        example = self._create_function_call_example(entity)

                        suggestions.append(CompletionSuggestion(
                            text=example,
                            completion_type='function',
                            confidence=1.0 - (i * 0.2),
                            context=f"Next step in {pipeline_type} pipeline",
                            source_file=entity.file_path,
                            example=self._find_usage_example(entity)
                        ))
            except ValueError:
                # Current step not found, suggest first step
                pass

        return suggestions

    def _find_return_value_functions(self) -> List[CodeEntity]:
        """Find functions that return values."""
        return [
            entity for entity in self.search.entities.values()
            if entity.return_type and entity.return_type != 'None'
        ][:20]

    def _suggest_methods_for_object(
        self,
        obj_name: str,
        top_k: int
    ) -> List[CompletionSuggestion]:
        """Suggest methods for an object."""
        suggestions = []

        # Try to infer object type from name
        type_hints = {
            'image': ['convert', 'resize', 'save', 'crop'],
            'depth': ['estimate', 'apply', 'normalize'],
            'model': ['load', 'predict', 'evaluate'],
            'pipeline': ['process', 'run', 'execute'],
            'result': ['save', 'show', 'export']
        }

        for type_name, methods in type_hints.items():
            if type_name in obj_name.lower():
                for method in methods:
                    suggestions.append(CompletionSuggestion(
                        text=f"{method}()",
                        completion_type='function',
                        confidence=0.6,
                        context=f"Common method for {type_name} objects"
                    ))

        return suggestions[:top_k]

    def _create_function_call_example(self, entity: CodeEntity) -> str:
        """Create a function call example with parameters."""
        if not entity.parameters or entity.parameters == ['self']:
            return f"{entity.name}()"

        # Filter out 'self'
        params = [p for p in entity.parameters if p != 'self']

        # Create placeholder parameters
        param_str = ', '.join(f"{p}=..." for p in params)
        return f"{entity.name}({param_str})"

    def _infer_parameter_value(
        self,
        param_name: str,
        entity: CodeEntity
    ) -> str:
        """Infer appropriate value for a parameter."""
        # Common parameter name patterns
        if 'path' in param_name or 'file' in param_name:
            return f'{param_name}="path/to/file"'
        elif 'size' in param_name or 'width' in param_name or 'height' in param_name:
            return f'{param_name}=1024'
        elif 'strength' in param_name or 'intensity' in param_name or 'alpha' in param_name:
            return f'{param_name}=0.7'
        elif 'enable' in param_name or 'use' in param_name:
            return f'{param_name}=True'
        elif 'count' in param_name or 'num' in param_name:
            return f'{param_name}=10'
        else:
            return f'{param_name}=None'

    def _find_usage_example(self, entity: CodeEntity) -> Optional[str]:
        """Find a real usage example from the codebase."""
        # Search for test files that use this entity
        examples = self.search.retriever.retrieve(
            entity.name,
            top_k=5,
            chunk_type_filter=['test']
        )

        for example in examples:
            if entity.name in example.content:
                # Extract the line with the function call
                lines = example.content.split('\n')
                for line in lines:
                    if entity.name in line and '(' in line:
                        return line.strip()

        return None


def main():
    """CLI for intelligent completion."""
    import argparse

    parser = argparse.ArgumentParser(description='Intelligent Code Completion')
    parser.add_argument('--repo-root', default='.', help='Repository root')
    parser.add_argument('--mode', required=True,
                       choices=['import', 'function', 'parameter', 'snippet', 'pipeline'],
                       help='Completion mode')
    parser.add_argument('--context', help='Code context')
    parser.add_argument('--function', help='Function name (for parameter mode)')
    parser.add_argument('--params', nargs='*', default=[], help='Existing parameters')
    parser.add_argument('--intent', help='Intent (for snippet mode)')
    parser.add_argument('--pipeline-type', help='Pipeline type')
    parser.add_argument('--current-step', help='Current step in pipeline')
    parser.add_argument('--top-k', type=int, default=5, help='Number of suggestions')

    args = parser.parse_args()

    # Initialize
    print("Initializing intelligent completion...")
    search = SemanticCodeSearch(args.repo_root)
    search.index_codebase()

    completion = IntelligentCompletion(search)

    # Generate suggestions based on mode
    suggestions = []

    if args.mode == 'import':
        suggestions = completion.suggest_imports(
            args.context or '',
            top_k=args.top_k
        )

    elif args.mode == 'function':
        if not args.context:
            print("Error: --context required for function mode")
            return

        suggestions = completion.suggest_function_calls(
            args.context,
            len(args.context),
            top_k=args.top_k
        )

    elif args.mode == 'parameter':
        if not args.function:
            print("Error: --function required for parameter mode")
            return

        suggestions = completion.suggest_parameters(
            args.function,
            args.params,
            top_k=args.top_k
        )

    elif args.mode == 'snippet':
        suggestions = completion.suggest_snippets(
            args.context or '',
            args.intent or '',
            top_k=args.top_k
        )

    elif args.mode == 'pipeline':
        if not args.pipeline_type or not args.current_step:
            print("Error: --pipeline-type and --current-step required")
            return

        suggestions = completion.complete_pipeline_workflow(
            args.pipeline_type,
            args.current_step
        )

    # Display suggestions
    print(f"\nFound {len(suggestions)} suggestions:")
    print("=" * 80)

    for i, suggestion in enumerate(suggestions, 1):
        print(f"\n[{i}] {suggestion.text}")
        print(f"    Type: {suggestion.completion_type}")
        print(f"    Confidence: {suggestion.confidence:.0%}")
        print(f"    Context: {suggestion.context}")

        if suggestion.source_file:
            print(f"    Source: {suggestion.source_file}")

        if suggestion.usage_count > 0:
            print(f"    Used {suggestion.usage_count} times in repository")

        if suggestion.example:
            print(f"    Example: {suggestion.example}")


if __name__ == '__main__':
    main()
