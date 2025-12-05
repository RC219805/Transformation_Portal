# Utils Directory - Processing Utilities

**Status:** ✅ Production Ready  
**Version:** 1.0.0

Core utilities for image processing pipelines.

## Modules

### `adaptive_tone_mapping.py`
Intelligent tone mapping with automatic parameter selection.

```python
from utils.adaptive_tone_mapping import AdaptiveToneMapper

mapper = AdaptiveToneMapper()
tone_mapped, metadata = mapper.apply_adaptive_tone_mapping(hdr_image)
```

### `alpha_compositor.py`
Advanced alpha channel handling with multiple compositing modes.

```python
from utils.alpha_compositor import AlphaCompositor

compositor = AlphaCompositor()
result = compositor.composite(image_rgba, mode='flatten-white')
```

### `enhanced_reporter.py`
Comprehensive processing reports with embedded visualizations.

```python
from utils.enhanced_reporter import ProcessingReport

reporter = ProcessingReport(output_dir, "Project Name")
reporter.add_result(...)
reporter.finalize()
```

## Documentation

Full documentation: `docs/PHASE1_ENHANCEMENTS.md`
