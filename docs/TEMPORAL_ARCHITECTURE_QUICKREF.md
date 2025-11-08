# Temporal Architecture Quick Reference

**Transformation Portal - Quick Start Guide**

---

## Plugin System

### Create a Plugin

```python
from transformation_portal.plugins import plugin, DepthModelPlugin, PluginType

@plugin(
    name="my_model",
    plugin_type=PluginType.DEPTH_MODEL,
    version="1.0.0"
)
class MyDepthModel(DepthModelPlugin):
    def _create_metadata(self):
        return self._decorator_metadata
    
    def initialize(self, config=None):
        self._initialized = True
    
    def estimate_depth(self, image):
        return process(image)
```

### Use a Plugin

```python
from transformation_portal.plugins import get_global_registry

registry = get_global_registry()
model = registry.get_plugin('depth_model', 'my_model', initialize=True)
depth = model.estimate_depth(image)
```

---

## Deprecation Framework

### Mark as Deprecated

```python
from transformation_portal.compat import deprecated

@deprecated(replacement="new_function", removal_version="2.0.0")
def old_function():
    return new_function()
```

### Version Checking

```python
from transformation_portal.compat import require_version

@require_version(min_version="0.1.0", max_version="2.0.0")
def my_function():
    pass
```

---

## Real-Time Progress

### Progress Bar

```python
from transformation_portal.streaming import ProgressBar

with ProgressBar(total=100, description="Processing") as pbar:
    for i in range(100):
        process_item(i)
        pbar.update(1)
```

### Checkpoint/Resume

```python
from transformation_portal.streaming import CheckpointManager

mgr = CheckpointManager("my_job")

# Resume from checkpoint
state = mgr.get_latest()
start = state.state.get('index', 0) if state else 0

for i in range(start, 100):
    process(i)
    
    if i % 10 == 0:
        checkpoint = mgr.create_checkpoint(
            progress=i,
            state={'index': i}
        )
        mgr.save(checkpoint)
```

---

## Event Tracking

### Track Events

```python
from transformation_portal.events import event

@event("image.processed")
def process_image(path):
    return enhance(path)
```

### Query Events

```python
from transformation_portal.events import get_global_store

store = get_global_store()
recent = store.get_events_by_type("image.processed", limit=10)

for evt in recent:
    print(f"{evt.type} at {evt.timestamp}")
```

---

## Docker Deployment

### Build and Run

```bash
# CPU service
docker-compose up transformation-portal-cpu

# GPU service  
docker-compose up transformation-portal-gpu

# Batch worker
docker-compose run transformation-portal-worker
```

---

## Documentation

- **Plugin Development**: `docs/PLUGIN_DEVELOPMENT.md`
- **Migration Guide**: `MIGRATION_GUIDE.md`
- **Deprecation Policy**: `DEPRECATION_POLICY.md`
- **Architecture**: `docs/ARCHITECTURE_PHILOSOPHY.md`
- **Summary**: `TEMPORAL_ARCHITECTURE_SUMMARY.md`

---

## Key Principles

1. **Backwards-Compatible**: Old code keeps working
2. **Real-Time**: Immediate feedback on operations
3. **Extensible**: Add features via plugins
4. **Observable**: Complete event history

---

**Questions?** See full documentation or open an issue on GitHub.
