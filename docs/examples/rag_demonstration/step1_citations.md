## Citations

### [1] .github/agents/rag_system/templates.py:339-404
**Confidence**: 99%
**Relevance**: Text match

```
            examples: List of example dicts with 'input' and 'output' keys

        Returns:
            Template with examples appended
        """
        if not examples:
            return template

        examples_section = "\n\n## Few-Shot Examples\n\n"

...
```

### [2] .github/agents/RAG_INTEGRATION_GUIDE.md:98-200
**Confidence**: 84%
**Relevance**: Type: guide | Text match

```

# 2. Setup retrieval
retriever = HybridRetriever()
retriever.index(chunks)

# 3. Search for content
results = retriever.retrieve("How to add a new LUT preset?", top_k=10)
print(f"Found {len(results)} results")

# 4. Rerank for better precision
...
```

### [3] .github/agents/rag_system/templates/feature_implementation.md:566-666
**Confidence**: 73%
**Relevance**: Text match

```
  "confidence": 0.85,
  "citations": [
    {
      "file_path": "depth_pipeline/processors/zone_mapper.py",
      "snippet": "def apply_zone_processing(depth_map, zones): ...",
      "relevance": "Shows similar depth-based effect modulation pattern"
    }
  ]
}
```
...
```
