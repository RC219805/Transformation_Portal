# RAG System Quick Start Guide

Quick reference for using the RAG-enhanced Transformation Portal Specialist agent.

Current documentation navigation follows `docs/governance/DOCUMENTATION_MAP.md`.
This is support material for repository retrieval; live role boundaries are defined by `.github/agents/README.md`,
`.github/copilot-instructions.md`, and `docs/architecture/agent_governance.md`.

## What is RAG?

**Retrieval-Augmented Generation (RAG)** enhances the agent by:
- Searching the repository for relevant code/docs before responding
- Citing actual examples with file paths and line numbers
- Reporting retrieval relevance separately from source authority
- Using structured JSON responses for automation

## 🚀 Quick Examples

### Index the Repository

```bash
cd /path/to/Transformation_Portal

# Index current source and guidance; validate cached content before reuse
PYTHONPATH=.github/agents ./.venv/bin/python -m rag_system.indexer --repo-root . --verbose

# Output:
# Indexed 1838 chunks
# Total characters: 1,901,957
# By type: agent: 81, code: 638, doc: 421, test: 698
```

### Search for Code Examples

```bash
# Find depth pipeline examples
PYTHONPATH=.github/agents ./.venv/bin/python -m rag_system.retriever \
    --repo-root . \
    --query "depth pipeline atmospheric effects" \
    --top-k 3 \
    --type code doc

# Output: Top 3 results with file paths, scores, and previews
```

### Generate Citations

```bash
# Get markdown citations for material response
PYTHONPATH=.github/agents ./.venv/bin/python -m rag_system.citation \
    --repo-root . \
    --query "material response enhancement" \
    --max-citations 3 \
    --format markdown

# Output: Formatted citations with relevance and authority notes
```

### Create Workflow Templates

```bash
# Feature implementation template
PYTHONPATH=.github/agents ./.venv/bin/python -m rag_system.templates \
    --type feature \
    --description "Add sunset LUT preset" \
    --with-examples > /tmp/feature_template.md

# Bug triage template
PYTHONPATH=.github/agents ./.venv/bin/python -m rag_system.templates \
    --type bug \
    --description "ImportError: No module named torch" \
    --context "Python 3.11, Ubuntu 22.04"

# CI workflow change template
PYTHONPATH=.github/agents ./.venv/bin/python -m rag_system.templates \
    --type ci \
    --description "build.yml Add Python 3.12 to matrix" \
    --context "Ensure compatibility"
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
      "path": "src/transformation_portal/depth/processors/atmospheric_effects.py",
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

The error happens when running: ./.venv/bin/lux-depth-v3 --help
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
@transformation-portal-architect

Review a proposed build.yml matrix change against current runtime and
dependency policy. What enforcement updates would be required?
```

**Expected response includes**:
- Current workflow analysis
- Proposed YAML changes (with diff)
- Testing strategy (how to test the change)
- Impact assessment (build time, compatibility)

## 🔍 Understanding Citations

Citations look like this:

```
[File: src/transformation_portal/depth/processors/atmospheric_effects.py:45-60] (Relevance score (not correctness or authority): 90%)
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
- **Relevance score**: Retrieval match only; not correctness or source authority
- **Relevance**: Why it was cited (function name, has docs, similar pattern)
- **Snippet**: Actual code you can reference

## Documentation authority and historical retrieval

`retrieval_mode="operator"` (the default) keeps matching evidence available and
ranks verified maintained guidance first. `retrieval_mode="historical"` selects
cataloged historical/archive-only documents; `retrieval_mode="all"` keeps ordinary
relevance ordering across the corpus. The retrieval, reranking, citation, and
combined `rag_system.cli search`/`cite` CLIs expose the same `--mode` choices.

```bash
PYTHONPATH=.github/agents ./.venv/bin/python -m rag_system.retriever \
    --repo-root . --query "depth pipeline" --mode historical
PYTHONPATH=.github/agents ./.venv/bin/python -m rag_system.cli search \
    "depth pipeline" --repo-root . --mode all
```

Authority comes from `docs/governance/documentation_catalog.json`, matching
source/document hashes, and explicit `source-reviewed` status. Missing catalog,
missing entries, stale hashes, incomplete reviews, and unresolved successors
remain unverified. These results remain searchable; their scores never upgrade
them into current guidance. Citations display authority and successor pointers.
Even verified maintained documentation is not production or photographic
acceptance evidence. Historical mode covers cataloged history; `all` also finds
unclassified material.

## 📊 Relevance Scores

Scores are rank-and-match heuristics, not calibrated probabilities of
correctness, currency, or authorization. The JSON `confidence` field remains
for compatibility and has this same limited meaning.

| Score | Meaning | Action |
|-------|---------|--------|
| 0.7–1.0 | Stronger retrieval match | Check source authority, exact current bytes, and relevant contracts |
| 0.3–0.7 | Partial retrieval match | Inspect surrounding context and current successor |
| 0.0–0.3 | Weaker retrieval match | Refine the query and verify any usable evidence |

## 🎯 Best Practices

### When Asking the Agent

✅ **Do:**
- Be specific about the component/pipeline involved
- Provide error messages and stack traces
- Mention relevant files or functions you've looked at
- Ask for citations to verify recommendations

❌ **Don't:**
- Ask vague questions without context
- Ignore source authority and retrieval relevance
- Skip verifying cited examples
- Assume agent knows recent uncommitted changes

### When Implementing Suggestions

✅ **Do:**
- Check cited files to understand patterns
- Run tests after implementing changes
- Verify the suggested approach fits your use case
- Compare cited implementation evidence for alternative approaches

❌ **Don't:**
- Blindly copy-paste without understanding
- Skip testing the implemented changes
- Treat a strong retrieval match as proof of correctness
- Modify files without checking citations first

## 🛠️ Troubleshooting

### "No results found" or Low Relevance

**Problem**: Query returns no results or low-quality matches

**Solution**:
```bash
# Try broader query terms
PYTHONPATH=.github/agents ./.venv/bin/python -m rag_system.retriever \
    --query "depth processing" \
    --top-k 10  # Get more results

# Filter by specific types
PYTHONPATH=.github/agents ./.venv/bin/python -m rag_system.retriever \
    --query "pipeline" \
    --type code  # Only search code, not docs
```

### Index Seems Outdated

**Problem**: Recent changes aren't showing up in retrieval

**Solution**:
```bash
# Re-index the repository
PYTHONPATH=.github/agents ./.venv/bin/python -m rag_system.indexer --repo-root . --verbose

# Verify new content is indexed
grep -r "your_new_function" .github/agents/rag_system/ || \
    echo "May need to commit changes first"
```

### Response Schema Validation Fails

**Problem**: JSON response doesn't match expected schema

**Solution**:
```bash
# Validate a response file
PYTHONPATH=.github/agents ./.venv/bin/python -m rag_system.templates \
    --type feature --description "Validate a feature response" \
    --validate response.json

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
- Checking retrieval relevance and source authority
- Understanding the implementation (not just copying)
