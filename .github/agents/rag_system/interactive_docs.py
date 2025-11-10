"""
Interactive Documentation System

Automatically generates and maintains documentation:
- API reference from docstrings
- Usage examples from tests
- Tutorial generation
- FAQ from common issues
- Architecture diagrams from code
"""

import ast
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from semantic_search import CodeEntity, SemanticCodeSearch


@dataclass
class APIDocumentation:
    """API documentation for a code entity."""

    name: str
    entity_type: str
    signature: str
    docstring: str
    parameters: List[Tuple[str, Optional[str], Optional[str]]]  # (name, type, description)
    return_type: Optional[str]
    return_description: Optional[str]
    raises: List[Tuple[str, str]]  # (exception_type, description)
    examples: List[str]
    source_file: str
    line_number: int
    related_functions: List[str] = field(default_factory=list)


@dataclass
class Tutorial:
    """Tutorial document."""

    title: str
    description: str
    difficulty: str  # 'beginner', 'intermediate', 'advanced'
    estimated_time_minutes: int
    prerequisites: List[str]
    steps: List[Tuple[str, str]]  # (step_title, content)
    code_examples: List[str]
    related_apis: List[str]


@dataclass
class FAQItem:
    """FAQ item."""

    question: str
    answer: str
    category: str
    code_example: Optional[str] = None
    related_docs: List[str] = field(default_factory=list)
    popularity: int = 0  # How often this is accessed


class InteractiveDocumentationSystem:
    """
    Interactive documentation generation and maintenance system.

    Features:
    - Auto-generate API docs from code
    - Extract examples from tests
    - Create tutorials from common workflows
    - Build FAQ from issues/errors
    - Generate architecture diagrams
    """

    def __init__(self, search_engine: SemanticCodeSearch):
        """
        Initialize documentation system.

        Args:
            search_engine: Semantic search engine
        """
        self.search = search_engine
        self.api_docs: Dict[str, APIDocumentation] = {}
        self.tutorials: List[Tutorial] = []
        self.faq: List[FAQItem] = []

    def generate_api_documentation(
        self,
        module_name: Optional[str] = None
    ) -> List[APIDocumentation]:
        """
        Generate API documentation from code.

        Args:
            module_name: Optional module to document (None = all)

        Returns:
            List of API documentation
        """
        print("Generating API documentation...")

        docs = []

        for entity in self.search.entities.values():
            # Filter by module if specified
            if module_name and module_name not in entity.file_path:
                continue

            # Parse docstring
            parsed = self._parse_docstring(entity.docstring or "")

            # Find examples from tests
            examples = self._find_code_examples(entity.name)

            # Find related functions
            related = self._find_related_functions(entity)

            api_doc = APIDocumentation(
                name=entity.name,
                entity_type=entity.entity_type,
                signature=entity.signature,
                docstring=parsed['description'],
                parameters=parsed['parameters'],
                return_type=entity.return_type,
                return_description=parsed['returns'],
                raises=parsed['raises'],
                examples=examples,
                source_file=entity.file_path,
                line_number=entity.line_number,
                related_functions=related
            )

            docs.append(api_doc)
            self.api_docs[entity.name] = api_doc

        print(f"Generated documentation for {len(docs)} entities")
        return docs

    def generate_tutorials(
        self,
        workflow_patterns: Optional[List[str]] = None
    ) -> List[Tutorial]:
        """
        Generate tutorials from common workflows.

        Args:
            workflow_patterns: Optional workflow patterns to document

        Returns:
            List of tutorials
        """
        # Default workflow patterns
        if not workflow_patterns:
            workflow_patterns = [
                'depth_pipeline',
                'material_response',
                'video_grading',
                'batch_processing',
                'custom_lut'
            ]

        tutorials = []

        for pattern in workflow_patterns:
            tutorial = self._create_tutorial_for_workflow(pattern)
            if tutorial:
                tutorials.append(tutorial)

        self.tutorials = tutorials
        return tutorials

    def generate_faq(
        self,
        common_issues: Optional[List[str]] = None
    ) -> List[FAQItem]:
        """
        Generate FAQ from common issues.

        Args:
            common_issues: Optional list of common issues

        Returns:
            List of FAQ items
        """
        faq_items = []

        # Common questions from code patterns
        common_questions = [
            {
                'question': 'How do I process a batch of images?',
                'category': 'batch_processing',
                'search_terms': ['batch', 'process', 'multiple', 'images']
            },
            {
                'question': 'How do I add a custom LUT preset?',
                'category': 'color_grading',
                'search_terms': ['lut', 'preset', 'color', 'grade']
            },
            {
                'question': 'How do I optimize GPU performance?',
                'category': 'performance',
                'search_terms': ['gpu', 'cuda', 'mps', 'performance']
            },
            {
                'question': 'How do I preserve metadata?',
                'category': 'metadata',
                'search_terms': ['metadata', 'iptc', 'xmp', 'preserve']
            },
        ]

        for q_data in common_questions:
            # Search for relevant code
            query = ' '.join(q_data['search_terms'])
            results = self.search.search(query, top_k=3)

            # Build answer from results
            answer = self._build_faq_answer(q_data['question'], results)

            # Find code example
            code_example = results[0].code_snippet if results else None

            faq_items.append(FAQItem(
                question=q_data['question'],
                answer=answer,
                category=q_data['category'],
                code_example=code_example,
                related_docs=[r.entity.file_path for r in results]
            ))

        self.faq = faq_items
        return faq_items

    def export_markdown_documentation(
        self,
        output_dir: str
    ):
        """
        Export documentation as Markdown.

        Args:
            output_dir: Output directory for docs
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Export API reference
        self._export_api_reference(output_path / 'api')

        # Export tutorials
        self._export_tutorials(output_path / 'tutorials')

        # Export FAQ
        self._export_faq(output_path / 'faq.md')

        print(f"Documentation exported to {output_path}")

    def _parse_docstring(self, docstring: str) -> Dict:
        """Parse Google-style docstring."""
        parsed = {
            'description': '',
            'parameters': [],
            'returns': None,
            'raises': []
        }

        if not docstring:
            return parsed

        lines = docstring.split('\n')
        current_section = 'description'
        description_lines = []
        param_lines = []
        return_lines = []
        raise_lines = []

        for line in lines:
            line = line.strip()

            if line.startswith('Args:'):
                current_section = 'args'
                continue
            elif line.startswith('Returns:'):
                current_section = 'returns'
                continue
            elif line.startswith('Raises:'):
                current_section = 'raises'
                continue

            if current_section == 'description':
                description_lines.append(line)
            elif current_section == 'args':
                param_lines.append(line)
            elif current_section == 'returns':
                return_lines.append(line)
            elif current_section == 'raises':
                raise_lines.append(line)

        parsed['description'] = '\n'.join(description_lines).strip()

        # Parse parameters
        for line in param_lines:
            if ':' in line:
                match = re.match(r'(\w+)\s*(?:\(([^)]+)\))?\s*:\s*(.*)', line)
                if match:
                    param_name, param_type, param_desc = match.groups()
                    parsed['parameters'].append((param_name, param_type, param_desc))

        # Parse returns
        if return_lines:
            parsed['returns'] = ' '.join(return_lines).strip()

        # Parse raises
        for line in raise_lines:
            if ':' in line:
                match = re.match(r'(\w+)\s*:\s*(.*)', line)
                if match:
                    exception_type, desc = match.groups()
                    parsed['raises'].append((exception_type, desc))

        return parsed

    def _find_code_examples(self, entity_name: str) -> List[str]:
        """Find code examples from tests."""
        examples = []

        # Search in test files
        results = self.search.retriever.retrieve(
            entity_name,
            top_k=10,
            chunk_type_filter=['test']
        )

        for result in results:
            if entity_name in result.content:
                # Extract usage lines
                lines = result.content.split('\n')
                for i, line in enumerate(lines):
                    if entity_name in line and '(' in line:
                        # Get context (3 lines before and after)
                        start = max(0, i - 3)
                        end = min(len(lines), i + 4)
                        example = '\n'.join(lines[start:end])
                        examples.append(example)
                        break

        return examples[:3]  # Top 3 examples

    def _find_related_functions(self, entity: CodeEntity) -> List[str]:
        """Find related functions based on calls and usage."""
        related = []

        # Functions this entity calls
        for called in entity.calls:
            if called in self.search.entity_index:
                related.append(called)

        # Functions that call this entity
        for other_entity in self.search.entities.values():
            if entity.name in other_entity.calls:
                related.append(other_entity.name)

        return list(set(related))[:5]  # Top 5 unique

    def _create_tutorial_for_workflow(
        self,
        workflow_pattern: str
    ) -> Optional[Tutorial]:
        """Create tutorial for a workflow pattern."""
        # Search for entities related to this workflow
        results = self.search.search(workflow_pattern, top_k=10)

        if not results:
            return None

        # Determine difficulty based on complexity
        avg_complexity = sum(
            r.entity.complexity for r in results if r.entity.complexity > 0
        ) / len(results)

        if avg_complexity > 15:
            difficulty = 'advanced'
            estimated_time = 45
        elif avg_complexity > 8:
            difficulty = 'intermediate'
            estimated_time = 30
        else:
            difficulty = 'beginner'
            estimated_time = 15

        # Build tutorial steps
        steps = []

        # Step 1: Setup
        steps.append((
            "Setup and Imports",
            f"First, import the necessary modules for {workflow_pattern}."
        ))

        # Step 2-N: Based on top results
        for i, result in enumerate(results[:3], 2):
            steps.append((
                f"Step {i}: Use {result.entity.name}",
                result.entity.docstring or f"Apply {result.entity.name} to process your data."
            ))

        # Collect code examples
        code_examples = [r.code_snippet for r in results[:3]]

        # Related APIs
        related_apis = [r.entity.name for r in results]

        return Tutorial(
            title=f"{workflow_pattern.replace('_', ' ').title()} Tutorial",
            description=f"Learn how to use the {workflow_pattern} workflow.",
            difficulty=difficulty,
            estimated_time_minutes=estimated_time,
            prerequisites=['Python 3.10+', 'Basic image processing knowledge'],
            steps=steps,
            code_examples=code_examples,
            related_apis=related_apis
        )

    def _build_faq_answer(
        self,
        question: str,
        search_results: List
    ) -> str:
        """Build FAQ answer from search results."""
        if not search_results:
            return "No specific answer found. Please check the documentation or ask for help."

        # Build answer from top result
        top_result = search_results[0]

        answer = f"To {question.lower()[7:]}, you can use `{top_result.entity.name}`.\n\n"

        if top_result.entity.docstring:
            answer += top_result.entity.docstring.split('\n')[0] + '\n\n'

        answer += f"See {top_result.entity.file_path}:{top_result.entity.line_number} for implementation details."

        return answer

    def _export_api_reference(self, output_dir: Path):
        """Export API reference as Markdown."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Group by module
        by_module = defaultdict(list)
        for doc in self.api_docs.values():
            module = self._get_module_name(doc.source_file)
            by_module[module].append(doc)

        # Create index
        with open(output_dir / 'README.md', 'w') as f:
            f.write("# API Reference\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            f.write("## Modules\n\n")

            for module in sorted(by_module.keys()):
                f.write(f"- [{module}]({module}.md)\n")

        # Create module docs
        for module, docs in by_module.items():
            with open(output_dir / f"{module}.md", 'w') as f:
                f.write(f"# {module}\n\n")

                for doc in sorted(docs, key=lambda x: x.name):
                    f.write(f"## {doc.name}\n\n")
                    f.write(f"**Type:** {doc.entity_type}\n\n")
                    f.write(f"**Signature:**\n```python\n{doc.signature}\n```\n\n")

                    if doc.docstring:
                        f.write(f"**Description:**\n{doc.docstring}\n\n")

                    if doc.parameters:
                        f.write("**Parameters:**\n")
                        for param_name, param_type, param_desc in doc.parameters:
                            type_str = f" ({param_type})" if param_type else ""
                            f.write(f"- `{param_name}`{type_str}: {param_desc}\n")
                        f.write("\n")

                    if doc.return_description:
                        f.write(f"**Returns:** {doc.return_description}\n\n")

                    if doc.examples:
                        f.write("**Examples:**\n")
                        for example in doc.examples:
                            f.write(f"```python\n{example}\n```\n\n")

                    if doc.related_functions:
                        f.write("**Related Functions:** ")
                        f.write(", ".join(f"`{fn}`" for fn in doc.related_functions))
                        f.write("\n\n")

                    f.write(f"**Source:** `{doc.source_file}:{doc.line_number}`\n\n")
                    f.write("---\n\n")

    def _export_tutorials(self, output_dir: Path):
        """Export tutorials as Markdown."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create index
        with open(output_dir / 'README.md', 'w') as f:
            f.write("# Tutorials\n\n")

            # Group by difficulty
            by_difficulty = defaultdict(list)
            for tutorial in self.tutorials:
                by_difficulty[tutorial.difficulty].append(tutorial)

            for difficulty in ['beginner', 'intermediate', 'advanced']:
                if difficulty in by_difficulty:
                    f.write(f"## {difficulty.title()}\n\n")
                    for tutorial in by_difficulty[difficulty]:
                        filename = tutorial.title.lower().replace(' ', '_') + '.md'
                        f.write(f"- [{tutorial.title}]({filename}) ")
                        f.write(f"({tutorial.estimated_time_minutes} min)\n")
                    f.write("\n")

        # Create individual tutorials
        for tutorial in self.tutorials:
            filename = tutorial.title.lower().replace(' ', '_') + '.md'

            with open(output_dir / filename, 'w') as f:
                f.write(f"# {tutorial.title}\n\n")
                f.write(f"**Difficulty:** {tutorial.difficulty}\n")
                f.write(f"**Estimated Time:** {tutorial.estimated_time_minutes} minutes\n\n")

                f.write(f"## Description\n\n{tutorial.description}\n\n")

                if tutorial.prerequisites:
                    f.write("## Prerequisites\n\n")
                    for prereq in tutorial.prerequisites:
                        f.write(f"- {prereq}\n")
                    f.write("\n")

                f.write("## Steps\n\n")
                for step_title, step_content in tutorial.steps:
                    f.write(f"### {step_title}\n\n{step_content}\n\n")

                if tutorial.code_examples:
                    f.write("## Code Examples\n\n")
                    for example in tutorial.code_examples:
                        f.write(f"```python\n{example}\n```\n\n")

                if tutorial.related_apis:
                    f.write("## Related APIs\n\n")
                    for api in tutorial.related_apis:
                        f.write(f"- `{api}`\n")

    def _export_faq(self, output_file: Path):
        """Export FAQ as Markdown."""
        with open(output_file, 'w') as f:
            f.write("# Frequently Asked Questions\n\n")

            # Group by category
            by_category = defaultdict(list)
            for item in self.faq:
                by_category[item.category].append(item)

            for category in sorted(by_category.keys()):
                f.write(f"## {category.replace('_', ' ').title()}\n\n")

                for item in by_category[category]:
                    f.write(f"### {item.question}\n\n")
                    f.write(f"{item.answer}\n\n")

                    if item.code_example:
                        f.write("**Example:**\n")
                        f.write(f"```python\n{item.code_example}\n```\n\n")

                    if item.related_docs:
                        f.write("**Related Documentation:**\n")
                        for doc in item.related_docs[:3]:
                            f.write(f"- `{doc}`\n")
                        f.write("\n")

    def _get_module_name(self, file_path: str) -> str:
        """Get module name from file path."""
        path = Path(file_path)

        # Remove extension
        name = path.stem

        # Get parent if in a package
        if path.parent.name not in {'.', 'src', 'tests'}:
            name = f"{path.parent.name}_{name}"

        return name


def main():
    """CLI for documentation generation."""
    import argparse

    parser = argparse.ArgumentParser(description='Interactive Documentation System')
    parser.add_argument('--repo-root', default='.', help='Repository root')
    parser.add_argument('--output', default='docs/generated', help='Output directory')
    parser.add_argument('--module', help='Specific module to document')

    args = parser.parse_args()

    # Initialize
    print("Initializing documentation system...")
    from semantic_search import SemanticCodeSearch

    search = SemanticCodeSearch(args.repo_root)
    search.index_codebase()

    doc_system = InteractiveDocumentationSystem(search)

    # Generate all documentation
    print("\nGenerating API documentation...")
    api_docs = doc_system.generate_api_documentation(args.module)

    print("\nGenerating tutorials...")
    tutorials = doc_system.generate_tutorials()

    print("\nGenerating FAQ...")
    faq = doc_system.generate_faq()

    # Export
    print(f"\nExporting documentation to {args.output}...")
    doc_system.export_markdown_documentation(args.output)

    print("\nDocumentation generation complete!")
    print(f"  API docs: {len(api_docs)} entities")
    print(f"  Tutorials: {len(tutorials)}")
    print(f"  FAQ items: {len(faq)}")


if __name__ == '__main__':
    main()
