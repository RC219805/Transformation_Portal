"""Tests for custom agent configuration files.

Validates that custom agent markdown files follow the correct format
and contain expected sections.
"""

import re
from pathlib import Path


def test_agent_file_exists():
    """Verify the custom agent file exists."""
    agent_file = Path(__file__).parents[1] / ".github" / "agents" / "transformation-portal-specialist.md"
    assert agent_file.exists(), f"Agent file not found: {agent_file}"


def test_agent_has_frontmatter():
    """Verify the agent file has proper YAML frontmatter."""
    agent_file = Path(__file__).parents[1] / ".github" / "agents" / "transformation-portal-specialist.md"
    content = agent_file.read_text()
    
    # Check for frontmatter delimiters
    assert content.startswith("---\n"), "Agent file must start with YAML frontmatter (---)"
    
    # Check for closing delimiter
    lines = content.split("\n")
    closing_delimiter_found = False
    for i, line in enumerate(lines[1:], start=1):
        if line == "---":
            closing_delimiter_found = True
            break
    
    assert closing_delimiter_found, "Agent file must have closing frontmatter delimiter (---)"


def test_agent_frontmatter_has_name():
    """Verify the agent frontmatter includes a name field."""
    agent_file = Path(__file__).parents[1] / ".github" / "agents" / "transformation-portal-specialist.md"
    content = agent_file.read_text()
    
    # Extract frontmatter
    lines = content.split("\n")
    frontmatter_lines = []
    in_frontmatter = False
    
    for line in lines:
        if line == "---":
            if not in_frontmatter:
                in_frontmatter = True
                continue
            else:
                break
        if in_frontmatter:
            frontmatter_lines.append(line)
    
    frontmatter = "\n".join(frontmatter_lines)
    assert re.search(r"^name:\s*.+", frontmatter, re.MULTILINE), "Frontmatter must include 'name:' field"


def test_agent_frontmatter_has_description():
    """Verify the agent frontmatter includes a description field."""
    agent_file = Path(__file__).parents[1] / ".github" / "agents" / "transformation-portal-specialist.md"
    content = agent_file.read_text()
    
    # Extract frontmatter
    lines = content.split("\n")
    frontmatter_lines = []
    in_frontmatter = False
    
    for line in lines:
        if line == "---":
            if not in_frontmatter:
                in_frontmatter = True
                continue
            else:
                break
        if in_frontmatter:
            frontmatter_lines.append(line)
    
    frontmatter = "\n".join(frontmatter_lines)
    assert re.search(r"^description:\s*.+", frontmatter, re.MULTILINE), "Frontmatter must include 'description:' field"


def test_agent_has_title():
    """Verify the agent file has a markdown title."""
    agent_file = Path(__file__).parents[1] / ".github" / "agents" / "transformation-portal-specialist.md"
    content = agent_file.read_text()
    
    # Check for H1 title after frontmatter
    assert re.search(r"\n# .+\n", content), "Agent file must have a markdown H1 title"


def test_agent_has_expertise_sections():
    """Verify the agent file describes core expertise areas."""
    agent_file = Path(__file__).parents[1] / ".github" / "agents" / "transformation-portal-specialist.md"
    content = agent_file.read_text()
    
    # Check for key expertise sections
    assert "expertise" in content.lower(), "Agent should describe its expertise"
    assert "pipeline" in content.lower(), "Agent should mention pipelines"
    assert "depth" in content.lower() or "image" in content.lower(), "Agent should mention image/depth processing"


def test_agent_mentions_key_technologies():
    """Verify the agent mentions key repository technologies."""
    agent_file = Path(__file__).parents[1] / ".github" / "agents" / "transformation-portal-specialist.md"
    content = content_lower = agent_file.read_text().lower()
    
    # Key technologies that should be mentioned
    key_techs = [
        ("pytorch" or "torch"),
        ("ffmpeg"),
        ("numpy"),
        ("pillow" or "pil"),
    ]
    
    mentioned_count = sum(1 for tech in key_techs if tech in content_lower)
    assert mentioned_count >= 3, f"Agent should mention at least 3 key technologies, found {mentioned_count}"


def test_agent_provides_examples():
    """Verify the agent file includes code examples."""
    agent_file = Path(__file__).parents[1] / ".github" / "agents" / "transformation-portal-specialist.md"
    content = agent_file.read_text()
    
    # Check for code blocks
    code_blocks = re.findall(r"```[\s\S]*?```", content)
    assert len(code_blocks) >= 3, f"Agent should include at least 3 code examples, found {len(code_blocks)}"


def test_agent_has_troubleshooting_section():
    """Verify the agent file includes troubleshooting guidance."""
    agent_file = Path(__file__).parents[1] / ".github" / "agents" / "transformation-portal-specialist.md"
    content = agent_file.read_text()
    
    assert "troubleshoot" in content.lower() or "issue" in content.lower(), \
        "Agent should include troubleshooting or issue guidance"


def test_agent_readme_exists():
    """Verify the agent README file exists."""
    readme_file = Path(__file__).parents[1] / ".github" / "agents" / "README.md"
    assert readme_file.exists(), f"Agent README not found: {readme_file}"


def test_agent_readme_references_specialist():
    """Verify the agent README references the specialist agent."""
    readme_file = Path(__file__).parents[1] / ".github" / "agents" / "README.md"
    content = readme_file.read_text()
    
    assert "transformation-portal-specialist" in content.lower() or "specialist" in content.lower(), \
        "README should reference the specialist agent"


def test_custom_agent_guide_exists():
    """Verify the custom agent guide exists in docs."""
    guide_file = Path(__file__).parents[1] / "docs" / "CUSTOM_AGENT_GUIDE.md"
    assert guide_file.exists(), f"Custom agent guide not found: {guide_file}"


def test_custom_agent_guide_has_usage_examples():
    """Verify the custom agent guide includes usage examples."""
    guide_file = Path(__file__).parents[1] / "docs" / "CUSTOM_AGENT_GUIDE.md"
    content = guide_file.read_text()
    
    # Should have example prompts using @ notation
    assert "@transformation-portal-specialist" in content, \
        "Guide should include example prompts using @ notation"
    
    # Should have multiple examples
    example_count = content.count("@transformation-portal-specialist")
    assert example_count >= 5, f"Guide should include at least 5 usage examples, found {example_count}"


def test_agent_file_not_too_large():
    """Verify the agent file is not excessively large."""
    agent_file = Path(__file__).parents[1] / ".github" / "agents" / "transformation-portal-specialist.md"
    size_bytes = agent_file.stat().st_size
    
    # Agent files should typically be under 50KB
    max_size = 50 * 1024  # 50KB
    assert size_bytes < max_size, \
        f"Agent file is {size_bytes} bytes, should be under {max_size} bytes for performance"


def test_agent_file_has_reasonable_line_length():
    """Verify the agent file lines are not excessively long."""
    agent_file = Path(__file__).parents[1] / ".github" / "agents" / "transformation-portal-specialist.md"
    lines = agent_file.read_text().split("\n")
    
    # Check that most lines are under 200 characters (markdown can be longer than code)
    long_lines = [i for i, line in enumerate(lines, 1) if len(line) > 200 and not line.strip().startswith("http")]
    
    # Allow up to 10% of lines to be long (for tables, etc.)
    assert len(long_lines) < len(lines) * 0.1, \
        f"Too many long lines ({len(long_lines)}), check formatting"
