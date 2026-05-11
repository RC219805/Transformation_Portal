# RAG System Quick Start Guide

Quick reference for using the RAG-enhanced Transformation Portal Specialist agent.

Current documentation baseline: repo-wide refresh audit dated May 11, 2026,
building on `main` through PR #1721. This is support material for repository
retrieval; live role boundaries are defined by `.github/agents/README.md`,
`.github/copilot-instructions.md`, and `docs/architecture/agent_governance.md`.

## What is RAG?

**Retrieval-Augmented Generation (RAG)** enhances the agent by:
- Searching the repository for relevant code/docs before responding
- Citing actual examples with file paths and line numbers
- Providing confidence scores for recommendations
- Using structured JSON responses for automation

## 🚀 Quick Examples

### Index the Repository (One-Time Setup)

```bash
cd /path/to/Transformation_Portal

# Index all content
python .github/agents/rag_system/indexer.py --repo-root . --verbose

# Output:
# Indexed 1838 chunks
# Total characters: 1,901,957
# By type: agent: 81, code: 638, doc: 421, test: 698
```

### Search for Code Examples

```bash
# Find depth pipeline examples
python .github/agents/rag_system/retriever.py \
    --repo-root . \
    --query "depth pipeline atmospheric effects" \
    --top-k 3 \
    --type code doc

# Output: Top 3 results with file paths, scores, and previews
```

### Generate Citations

```bash
# Get markdown citations for material response
python .github/agents/rag_system/citation.py \
    --repo-root . \
    --query "material response enhancement" \
    --max-citations 3 \
    --format markdown

# Output: Formatted citations with confidence scores
```

### Create Workflow Templates

```bash
# Feature implementation template
python .github/agents/rag_system/templates.py \
    --type feature \
    --description "Add sunset LUT preset" \
    --with-examples > /tmp/feature_template.md

# Bug triage template
python .github/agents/rag_system/templates.py \
    --type bug \
    --description "ImportError: No module named torch" \
    --context "Python 3.11, Ubuntu 22.04"

# CI workflow change template
python .github/agents/rag_system/templates.py \
    --type ci \
    --description "build.yml Add Python 3.12 to matrix" \
    --reason "Ensure compatibility"
```

## 📋 Using Templates in Agent Conversations

### Feature Implementation

**Your prompt to the agent**:
```
@transformation-portal-specialist

Use the feature implementation template to help me add a new
atmospheric fog effect to the depth pipeline. The effect should
apply haze based on depth distance.
```

**Expected agent response structure**:
```json
{
  "summary": "Add depth-based fog effect to atmospheric processor",
  "files": [
    {
      "path": "depth_pipeline/processors/atmospheric.py",
      "patch": "Add fog_density parameter and blend_fog() function",
      "description": "Implements depth-proportional fog blending"
    }
  ],
  "tests": ["tests/test_atmospheric_processor.py"],
  "explanation": "Fog is applied by blending a configurable fog color...",
  "confidence": 0.85,
  "citations": [
    {
      "file_path": "src/transformation_portal/depth/processors/atmospheric_effects.py",
      "snippet": "class DepthFog: ...",
      "relevance": "Existing fog implementation pattern"
    }
  ]
}
```

### Bug Triage

**Your prompt**:
```
@transformation-portal-specialist

Help me debug this error:
ImportError: cannot import name 'DepthEstimator' from 'depth_tools'

The error happens when running: python lux_render_pipeline.py
```

**Expected response includes**:
- Error classification and severity
- Root cause analysis with file citations
- Minimal reproduction steps
- Fix strategy with patches
- Testing recommendations

### CI Workflow Changes

**Your prompt**:
```
@transformation-portal-specialist

I need to modify the build.yml workflow to add Python 3.12 to the test
matrix. What changes are needed?
```

**Expected response includes**:
- Current workflow analysis
- Proposed YAML changes (with diff)
- Testing strategy (how to test the change)
- Impact assessment (build time, compatibility)

## 🔍 Understanding Citations

Citations look like this:

```
[File: depth_pipeline/processors/atmospheric.py:45-60] (Confidence: 90%)
Relevance: Function: apply_haze | Has documentation | Similar pattern
```
```python
def apply_haze(image, depth_map, intensity=0.3):
    """Apply depth-based atmospheric haze."""
    fog_color = np.array([200, 200, 220])  # Light blue-gray
    depth_normalized = depth_map / depth_map.max()
    alpha = depth_normalized * intensity
    fogged = image * (1 - alpha[..., None]) + fog_color * alpha[..., None]
    return fogged.astype(image.dtype)
```

**What this tells you**:
- **File & lines**: Exact location in repository
- **Confidence**: How relevant (90% = very relevant)
- **Relevance**: Why it was cited (function name, has docs, similar pattern)
- **Snippet**: Actual code you can reference

## 📊 Confidence Scores

| Score | Meaning | Action |
|-------|---------|--------|
| 0.9-1.0 | Very high confidence | Implement as suggested |
| 0.7-0.9 | High confidence | Review and implement |
| 0.5-0.7 | Moderate confidence | Verify before implementing |
| 0.3-0.5 | Low confidence | Use as starting point, needs revision |
| 0.0-0.3 | Very low confidence | Consider alternative approach |

## 🎯 Best Practices

### When Asking the Agent

✅ **Do:**
- Be specific about the component/pipeline involved
- Provide error messages and stack traces
- Mention relevant files or functions you've looked at
- Ask for citations to verify recommendations

❌ **Don't:**
- Ask vague questions without context
- Ignore confidence scores
- Skip verifying cited examples
- Assume agent knows recent uncommitted changes

### When Implementing Suggestions

✅ **Do:**
- Check cited files to understand patterns
- Run tests after implementing changes
- Verify the suggested approach fits your use case
- Compare confidence scores for alternative approaches

❌ **Don't:**
- Blindly copy-paste without understanding
- Skip testing the implemented changes
- Ignore low confidence warnings
- Modify files without checking citations first

## 🛠️ Troubleshooting

### "No results found" or Low Relevance

**Problem**: Query returns no results or low-quality matches

**Solution**:
```bash
# Try broader query terms
python .github/agents/rag_system/retriever.py \
    --query "depth processing" \
    --top-k 10  # Get more results

# Filter by specific types
python .github/agents/rag_system/retriever.py \
    --query "pipeline" \
    --type code  # Only search code, not docs
```

### Index Seems Outdated

**Problem**: Recent changes aren't showing up in retrieval

**Solution**:
```bash
# Re-index the repository
python .github/agents/rag_system/indexer.py --repo-root . --verbose

# Verify new content is indexed
grep -r "your_new_function" .github/agents/rag_system/ || \
    echo "May need to commit changes first"
```

### Response Schema Validation Fails

**Problem**: JSON response doesn't match expected schema

**Solution**:
```bash
# Validate a response file
python .github/agents/rag_system/templates.py --validate response.json

# Required fields:
# - summary (string)
# - files (list of {path, patch, description})
# - tests (list of strings)
# - explanation (string)
# - confidence (float 0.0-1.0, optional)
# - citations (list of dicts, optional)
```

## 📚 Further Reading

- **Full Documentation**: `.github/agents/rag_system/README.md`
- **Agent Definition**: `.github/agents/transformation-portal-specialist.md`
- **Test Suite**: `tests/test_rag_system.py` (24 tests, examples of usage)
- **Architecture**: See RAG system README for component details

## 💡 Tips

1. **Use filtering**: Narrow searches with `--type` and `--file` filters
2. **Check citations**: Always review cited code before adapting patterns
3. **Validate schemas**: Use template validation for JSON responses
4. **Profile queries**: More specific queries get better results
5. **Iterate**: Start broad, then refine with filters

## 🆘 Need Help?

If you're stuck:
1. Check test files for usage examples
2. Run components with `--help` flag
3. Review `.github/agents/rag_system/README.md`
4. Ask the agent with specific error messages and context

---

**Remember**: RAG enhances the agent with repository knowledge, but you should always verify suggestions by:
- Reading cited code
- Running tests
- Checking confidence scores
- Understanding the implementation (not just copying)
