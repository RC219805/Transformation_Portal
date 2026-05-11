# RAG System Enhancements Guide

Comprehensive guide for the enhanced RAG system with artifact classification and knowledge integration capabilities.

Current documentation baseline: repo-wide refresh audit dated May 11, 2026,
building on `main` through PR #1721. This is RAG support material, not a live
role-boundary document. Live agent authority remains in `.github/agents/README.md`,
the profile files, `.github/copilot-instructions.md`, and
`docs/architecture/agent_governance.md`.

## Overview

The RAG system has been enhanced with two powerful new components:

1. **Artifact Classifier** - Automatically classifies and organizes image processing artifacts
2. **Knowledge Integration Engine** - Analyzes patterns, tracks KPIs, and provides recommendations

## New Capabilities

### 1. Artifact Classification & Organization

**What it does**:
- Automatically classifies image processing artifacts (depth maps, color grades, HDR outputs, metrics, logs, etc.)
- Extracts metadata from filenames and file contents
- Organizes artifacts hierarchically with parent/child/related relationships
- Tags artifacts for efficient retrieval
- Tracks transformation chains (original → processed → enhanced)
- Provides statistics and exports to JSON

**Supported Artifact Types**:
- Analysis
- Depth Map
- Color Grade
- HDR Output
- Metric
- Log
- Profile
- Render
- Material Response
- LUT Application
- Comparison

**Supported Pipeline Types**:
- Depth Pipeline
- Lux Render
- Material Response
- Video Grader
- TIFF Processor
- HDR Production
- AGX Filmic

### 2. Knowledge Integration Engine

**What it does**:
- Analyzes patterns in processing pipelines (success rates, failure modes, performance trends)
- Tracks quality evolution over time
- Identifies optimal parameters for each pipeline
- Generates recommendations for improvements
- Provides natural language query interface
- Tracks KPIs (success rate, processing time, quality score)
- Exports knowledge base for persistence

**Analysis Capabilities**:
- Success rate tracking per pipeline
- Processing time statistics (average, median, P95)
- Failure mode detection and categorization
- Trend analysis (improving, degrading, stable)
- Common and optimal parameter identification
- Quality score tracking

**Recommendation Types**:
- **Regression**: Low success rate or quality degradation
- **Optimization**: Performance degradation detected
- **Missing Test**: Recurring errors without test coverage
- **Undocumented Feature**: Features used but not documented

## Usage Examples

### Artifact Classification

#### Basic Classification

```bash
# Classify all artifacts in a directory
python .github/agents/rag_system/classifier.py \
    --input-dir processed_images/ \
    --output artifacts.json \
    --verbose
```

**Output**:
```
Scanning processed_images/...
Classified 150 artifacts

By type:
  depth_map: 45
  color_grade: 38
  hdr_output: 25
  metric: 20
  render: 22

By pipeline:
  depth_pipeline: 45
  lux_render: 28
  material_response: 35
  video_grader: 42

Success rate: 94.7%
Average processing time: 3.45s
Artifacts with errors: 8

Exported to artifacts.json
```

#### Search by Tags

```bash
# Find all successful 4K depth maps
python .github/agents/rag_system/classifier.py \
    --input-dir processed_images/ \
    --tags depth_map 4k_plus success \
    --require-all-tags
```

**Use Cases**:
- Find all failed processing runs: `--tags failure`
- Find all high-resolution outputs: `--tags 4k_plus`
- Find all GPU-accelerated runs: `--tags gpu`
- Find all errors of a specific type: `--tags error_type:ValueError`

### Knowledge Integration

#### Analyze Pipeline Performance

```bash
# Analyze depth pipeline performance over last 30 days
python .github/agents/rag_system/knowledge_engine.py \
    --feedback-file feedback.json \
    --analyze-pipeline depth_pipeline \
    --days 30
```

**Output**:
```
Pattern Analysis for depth_pipeline
  Total runs: 156
  Success rate: 94.2%
  Avg time: 2.34s
  Median time: 2.21s
  P95 time: 3.85s
  Time trend: improving
  Quality trend: stable

  Failure modes:
    ValueError: 5
    RuntimeError: 3
    MemoryError: 1
```

#### Generate Recommendations

```bash
# Get improvement recommendations
python .github/agents/rag_system/knowledge_engine.py \
    --feedback-file feedback.json \
    --recommendations
```

**Output**:
```
Generated 4 recommendations

1. [HIGH] Low success rate for material_response
   Success rate is 82.5%, which is below the 90% threshold.
   Action: Review recent changes and error patterns. Consider adding more error handling.
   Confidence: 90%

2. [MEDIUM] Performance degradation in lux_render
   Processing time has increased. Current average: 8.45s
   Action: Profile the pipeline to identify bottlenecks. Consider optimizing slow operations.
   Confidence: 80%

3. [MEDIUM] Recurring ValueError in depth_pipeline
   This error has occurred 5 times.
   Action: Add test coverage for ValueError. Implement better error handling.
   Confidence: 90%

4. [HIGH] Quality degradation in video_grader
   Output quality has decreased compared to previous period.
   Action: Review recent parameter changes. Consider reverting to previous settings.
   Confidence: 85%
```

#### Natural Language Queries

```bash
# Ask questions about your pipelines
python .github/agents/rag_system/knowledge_engine.py \
    --feedback-file feedback.json \
    --query "What is the success rate for depth_pipeline?"
```

**Supported Queries**:
- "What is the success rate for [pipeline]?"
- "How is the performance of [pipeline]?"
- "What errors occurred in [pipeline]?"
- "What improvements do you recommend?"
- "What is the average processing time?"

#### Export Knowledge Base

```bash
# Export all patterns, recommendations, and KPIs
python .github/agents/rag_system/knowledge_engine.py \
    --feedback-file feedback.json \
    --export knowledge_base.json
```

## Integration with Existing Pipelines

### Python API

#### Artifact Classification

```python
from rag_system import ArtifactClassifier

# Initialize classifier
classifier = ArtifactClassifier()

# Classify an artifact
artifact = classifier.add_artifact(
    file_path="output/depth_map_2024-11-05.png",
    content=None,  # Optional: file contents for content-based classification
)

print(f"Type: {artifact.artifact_type.value}")
print(f"Pipeline: {artifact.metadata.pipeline.value}")
print(f"Tags: {artifact.tags}")

# Search by tags
results = classifier.search_by_tags({'depth_map', 'success'}, require_all=True)
print(f"Found {len(results)} matching artifacts")

# Get transformation chain
chain = classifier.get_transformation_chain(artifact.artifact_id)
print(f"Transformation chain has {len(chain)} steps")

# Export
classifier.export_to_json("artifacts.json")
```

#### Knowledge Integration

```python
from rag_system import KnowledgeIntegrationEngine

# Initialize engine
engine = KnowledgeIntegrationEngine()

# Add feedback
engine.add_feedback(
    pipeline='depth_pipeline',
    artifact_id='img001',
    success=True,
    processing_time=2.5,
    parameters={'quality': 'high', 'denoise': 0.5},
    quality_score=0.92,
)

# Analyze patterns
analysis = engine.analyze_patterns('depth_pipeline', days=30)
print(f"Success rate: {analysis.success_rate:.1%}")
print(f"Avg time: {analysis.avg_processing_time:.2f}s")
print(f"Trend: {analysis.time_trend}")

# Generate recommendations
recommendations = engine.generate_recommendations()
for rec in recommendations:
    print(f"[{rec.severity.upper()}] {rec.title}")
    print(f"  Action: {rec.suggested_action}")

# Natural language query
response = engine.query_natural_language("What is the success rate?")
print(response)

# Get KPI summary
kpi_summary = engine.get_kpi_summary(pipeline='depth_pipeline', days=7)
for kpi_key, kpi_data in kpi_summary.items():
    print(f"{kpi_key}: current={kpi_data['current']:.2f}, avg={kpi_data['average']:.2f}")

# Export
engine.export_knowledge_base("knowledge_base.json")
```

### Feedback Data Format

The knowledge engine expects feedback data in this format:

```json
{
  "records": [
    {
      "pipeline": "depth_pipeline",
      "artifact_id": "img001",
      "success": true,
      "processing_time": 2.5,
      "parameters": {
        "quality": "high",
        "denoise": 0.5
      },
      "quality_score": 0.92,
      "error_message": null
    }
  ]
}
```

**Required Fields**:
- `pipeline`: Pipeline name (string)
- `artifact_id`: Unique artifact identifier (string)
- `success`: Whether processing succeeded (boolean)
- `processing_time`: Processing time in seconds (float)
- `parameters`: Pipeline parameters used (object)

**Optional Fields**:
- `error_message`: Error message if failed (string or null)
- `quality_score`: Quality score 0-1 (float or null)
- `user_feedback`: User feedback text (string or null)

## Performance Characteristics

### Artifact Classifier

| Operation | Complexity | Typical Time | Memory |
|-----------|-----------|--------------|--------|
| Classify single artifact | O(1) | <1ms | Minimal |
| Classify N artifacts | O(N) | ~1ms per artifact | ~1KB per artifact |
| Search by tags | O(N) | <10ms for 1000 artifacts | Minimal |
| Get transformation chain | O(depth) | <5ms | Minimal |
| Export to JSON | O(N) | ~50ms for 1000 artifacts | 2x artifact size |

### Knowledge Integration Engine

| Operation | Complexity | Typical Time | Memory |
|-----------|-----------|--------------|--------|
| Add feedback | O(1) | <1ms | ~1KB per record |
| Analyze patterns | O(N) | ~10ms for 1000 records | ~100KB (cached) |
| Generate recommendations | O(P*N) | ~50ms for 10 pipelines | Minimal |
| Natural language query | O(N) | ~20ms for 1000 records | Minimal |
| Get KPI summary | O(N) | ~5ms for 1000 records | ~50KB |
| Export knowledge base | O(N+P) | ~100ms | 3x data size |

Where:
- N = number of artifacts/feedback records
- P = number of pipelines
- Cached operations reuse previous computations

## Best Practices

### Artifact Classification

1. **Tag Consistently**: Use consistent tag names across your organization
2. **Track Lineage**: Always link child artifacts to parent artifacts
3. **Export Regularly**: Export classification data for backup and analysis
4. **Search Efficiently**: Use tag-based search instead of scanning all artifacts
5. **Update Metadata**: Add metadata as it becomes available (quality scores, etc.)

### Knowledge Integration

1. **Collect Feedback Continuously**: Add feedback for every processing run
2. **Analyze Regularly**: Run pattern analysis weekly or monthly
3. **Act on Recommendations**: Address high-severity recommendations promptly
4. **Track Trends**: Monitor trend analysis to catch degradations early
5. **Export Knowledge Base**: Back up knowledge base regularly
6. **Use Optimal Parameters**: Apply optimal parameters identified by the engine

## Troubleshooting

### Classifier Issues

**Problem**: Artifacts classified as "unknown"

**Solution**:
- Check if filename matches any patterns in `PATTERNS` dictionary
- Provide file contents for content-based classification
- Add custom patterns for your organization's naming conventions

**Problem**: Tags not matching expected values

**Solution**:
- Verify metadata extraction is working correctly
- Check if resolution/color space/AI model info is in filename
- Manually add tags if needed

### Knowledge Engine Issues

**Problem**: No patterns found

**Solution**:
- Ensure feedback records are in correct format
- Check that `pipeline` field matches expected pipeline names
- Verify date range covers your feedback records

**Problem**: Recommendations not generated

**Solution**:
- Need at least 10 feedback records per pipeline
- Check if success rate/performance meets threshold for recommendations
- Ensure feedback includes error messages for recurring error detection

## Advanced Usage

### Custom Artifact Types

```python
# Add custom artifact type
class CustomArtifactType(Enum):
    MY_CUSTOM_TYPE = "my_custom_type"

# Add custom pattern
classifier.PATTERNS[CustomArtifactType.MY_CUSTOM_TYPE] = [
    r'custom_output', r'my_.*\.custom'
]
```

### Custom Recommendation Logic

```python
# Extend recommendation generation
class CustomKnowledgeEngine(KnowledgeIntegrationEngine):
    def generate_recommendations(self, pipeline=None):
        recommendations = super().generate_recommendations(pipeline)

        # Add custom recommendation logic
        # ...

        return recommendations
```

### Visualization

```python
import matplotlib.pyplot as plt

# Get KPI data
kpi_summary = engine.get_kpi_summary(pipeline='depth_pipeline', days=30)

# Plot success rate over time
success_data = kpi_summary['depth_pipeline:success_rate']
timestamps = [ts for ts, _ in success_data['data_points']]
values = [val for _, val in success_data['data_points']]

plt.plot(timestamps, values)
plt.title('Success Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Success Rate')
plt.show()
```

## Future Enhancements

Potential future improvements:

1. **Machine Learning Classification**: Use ML models for more accurate artifact classification
2. **Predictive Analytics**: Predict failures before they occur based on patterns
3. **Real-time Monitoring**: Live dashboards with WebSocket updates
4. **Alert System**: Automatic alerts when KPIs drop below thresholds
5. **A/B Testing**: Compare pipeline configurations scientifically
6. **Cost Optimization**: Track and optimize processing costs (GPU time, storage)
7. **Multi-tenant Support**: Separate classification and knowledge bases per team/project

## References

- **Main RAG README**: `.github/agents/rag_system/README.md`
- **Classifier Source**: `.github/agents/rag_system/classifier.py`
- **Knowledge Engine Source**: `.github/agents/rag_system/knowledge_engine.py`
- **Tests**: `tests/test_rag_classifier.py`, `tests/test_rag_knowledge_engine.py`
- **Implementation Summary**: `.github/agents/RAG_IMPLEMENTATION_SUMMARY.md`

## Support

For questions or issues:
1. Check the test files for usage examples
2. Review the source code docstrings
3. Run CLI tools with `--help` flag
4. Consult the main RAG system README

## License

Same as parent repository.
