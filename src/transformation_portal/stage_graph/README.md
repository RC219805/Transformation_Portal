# Stage Graph Architecture

**Status:** Target executor infrastructure; not active for production Lux or
Spatial execution.

`StageGraph` provides trusted in-process dependency ordering and stage
execution primitives. ADR-051 designates it, together with
`core.cas_dag_executor.CASDAGExecutor`, as the target executor only after a
separate production vertical-slice gate. The current Lux orchestrator and
imperative Spatial APIs remain the live rollback paths.

## Canonical plan boundary

`tp.execution.plan.v1` is the external semantic plan contract. Its static
registry is exposed from `transformation_portal.stage_graph.registry`:

```python
from transformation_portal.stage_graph.registry import (
    get_stage_definition,
    stage_registry_identifiers,
)

print(stage_registry_identifiers())
definition = get_stage_definition("tp.stage.lux.depth.v1")
```

Registry definitions contain identifiers, configuration-schema identifiers,
logical output kinds, and bounded resource declarations only. They contain no
classes, callables, module paths, commands, or plugin hooks. Parsing or
validating a plan does not construct a `StageGraph`.

Do not deserialize Python classes or modules from a plan. Do not route a
canonical plan into `StageGraph` or `CASDAGExecutor` until ADR-051 Phase C has
passed for that exact execution slice.

## Trusted in-process API

The existing graph API remains available for trusted tests and internal
experiments:

```python
from transformation_portal.stage_graph import GraphBuilder, StageContext

graph = GraphBuilder("trusted_internal_pipeline").build()
context = StageContext(cache_enabled=False)
execution = graph.execute(context, parallel=False)
```

This API is not an untrusted plan parser and is not the current production Lux
executor. Its stage-local cache is also not production cache authority under
ADR-051.

## Validation

```bash
./.venv/bin/pytest tests/stage_graph -q
./.venv/bin/pytest tests/core/test_execution_plan.py \
  tests/lux_depth_v3/test_execution_plan_adapter.py -q
```

See [ADR-051](../../../docs/architecture/ADR-051-execution-artifact-authority-designation.md)
and the [execution plan v1 contract](../../../docs/reference/EXECUTION_PLAN_V1.md).
