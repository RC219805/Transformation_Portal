# Process Orchestrator User Guide

## Overview

The Process Orchestrator manages batch processing with fault tolerance, resource management, and graceful shutdown. It ensures that one failed image doesn't stop an entire batch, and provides real-time progress tracking.

## Key Features

- **Fault Isolation**: Each image processes in a separate subprocess - one failure doesn't crash the batch
- **Resource Management**: Memory budgets and device assignment per task
- **Graceful Shutdown**: CTRL+C stops gracefully, finishing current images
- **Progress Tracking**: Real-time statistics on completed/failed/active tasks
- **Priority Queue**: Process important images first

## Basic Usage

### Automatic (Default)

The orchestrator is enabled by default in Phase 1:

```bash
python3 -m lux_depth_v2.cli \
  --input-dir input_images/ \
  --output-dir output/ \
  --preset photo_realistic
```

### Disable Orchestrator (Legacy Mode)

For compatibility with pre-Phase 1 behavior:

```bash
python3 -m lux_depth_v2.cli \
  --input-dir input_images/ \
  --output-dir output/ \
  --disable-orchestrator
```

## Configuration

### Max Workers

Control concurrent processing (default: 1 for GPU safety):

```bash
# Sequential processing (safest for GPU)
python3 -m lux_depth_v2.cli \
  --input-dir input_images/ \
  --output-dir output/

# Note: max_workers currently fixed at 1 in Phase 1
# Parallel processing planned for Phase 2
```

### Memory Budget

Set memory limit per task (in GB):

```bash
# Limit each task to 32GB
python3 -m lux_depth_v2.cli \
  --input-dir input_images/ \
  --output-dir output/ \
  --memory-budget 32.0
```

The orchestrator will:
- Monitor memory usage per task
- Terminate tasks exceeding budget
- Log budget violations

### Device Assignment

Assign processing device:

```bash
# Auto-detect best device (MPS > CUDA > CPU)
python3 -m lux_depth_v2.cli --device auto

# Force CPU (useful for memory-constrained systems)
python3 -m lux_depth_v2.cli --device cpu

# Use CUDA (NVIDIA GPU)
python3 -m lux_depth_v2.cli --device cuda

# Use MPS (Apple Silicon)
python3 -m lux_depth_v2.cli --device mps
```

## Fault Isolation

### How It Works

Each image is processed in a separate subprocess:

```
Main Process (Orchestrator)
├── Worker Process 1: image_001.tif
│   └── Exit code 0 (success)
├── Worker Process 2: image_002.tif
│   └── Exit code 1 (failed) ← Isolated failure
├── Worker Process 3: image_003.tif
│   └── Exit code 0 (success)
└── ...
```

**Benefits**:
- One corrupted image doesn't crash the batch
- Memory leaks are contained per-image
- GPU errors don't propagate
- Clean environment per image

### Example: Batch with Failure

```bash
# Processing 6 images, #3 fails
Input:
  image_001.tif ✅ Success
  image_002.tif ✅ Success
  image_003.tif ❌ Failed (OOM)
  image_004.tif ✅ Success
  image_005.tif ✅ Success
  image_006.tif ✅ Success

Result: 5/6 complete (83% success rate)
# Without orchestrator: entire batch would have crashed
```

## Graceful Shutdown

### CTRL+C Handling

Press CTRL+C to stop gracefully:

```bash
$ python3 -m lux_depth_v2.cli --input-dir input_images/ --output-dir output/

Processing: image_003.tif...
^C  # <- User presses CTRL+C

[INFO] Received signal 2, initiating graceful shutdown...
[INFO] Waiting for active workers to complete...
[INFO] Worker completed | task_id=image_003
[INFO] Shutdown complete
```

**Behavior**:
- Current image finishes processing
- Queued images are cancelled
- Checkpoints saved for resume
- Clean exit (no corruption)

### Timeout

Configure shutdown timeout:

```python
from lux_depth_v2.orchestrator import ProcessOrchestrator

orch = ProcessOrchestrator()
# ... submit tasks ...

# Graceful shutdown with 60s timeout
orch.shutdown(graceful=True, timeout=60.0)

# Force shutdown immediately
orch.shutdown(graceful=False)
```

## Progress Tracking

### Real-time Statistics

The orchestrator tracks:
- **Total tasks**: Images submitted to queue
- **Completed**: Successfully finished
- **Failed**: Errors during processing
- **Cancelled**: Stopped by user/shutdown
- **Active**: Currently processing
- **Queued**: Waiting to start

### Example Output

```bash
$ python3 -m lux_depth_v2.cli --input-dir input_images/ --output-dir output/

[INFO] ProcessOrchestrator initialized | workers=1 memory_budget=None device=auto
[INFO] Processing batch | total_tasks=10 workers=1
[INFO] Worker started | task_id=image_001 pid=12345 input=image_001.tif
[INFO] Worker completed | task_id=image_001
[INFO] Worker started | task_id=image_002 pid=12346 input=image_002.tif
...
[INFO] Batch processing complete | total=10 completed=9 failed=1 cancelled=0 elapsed=450.2s
```

### Progress API

Get progress programmatically:

```python
from lux_depth_v2.orchestrator import ProcessOrchestrator

orch = ProcessOrchestrator()
# ... submit tasks ...

progress = orch.get_progress()
print(f"Progress: {progress['completed']}/{progress['total_tasks']}")
print(f"Success rate: {progress['success_rate']*100:.1f}%")
print(f"Active workers: {progress['active']}")
print(f"Queue depth: {progress['queued']}")
```

## Advanced Usage

### Custom Progress Callback

Receive notifications on task completion:

```python
from lux_depth_v2.orchestrator import ProcessOrchestrator, TaskConfig, TaskResult
from pathlib import Path

def on_task_complete(result: TaskResult):
    if result.status == "success":
        print(f"✅ {result.input_path.name} complete in {result.elapsed_time:.1f}s")
    else:
        print(f"❌ {result.input_path.name} failed: {result.error}")

orch = ProcessOrchestrator()

tasks = [
    TaskConfig(
        task_id=f"task_{i}",
        input_path=Path(f"input_{i}.tif"),
        output_dir=Path("output"),
    )
    for i in range(10)
]

results = orch.process_batch(tasks, progress_callback=on_task_complete)
```

### Priority Processing

Process important images first:

```python
from lux_depth_v2.orchestrator import ProcessOrchestrator, TaskConfig

orch = ProcessOrchestrator()

# High priority (processed first)
high_priority = TaskConfig(task_id="hero", input_path=Path("hero.tif"), ...)
orch.submit_task(high_priority, priority=0)

# Normal priority
normal = TaskConfig(task_id="normal", input_path=Path("normal.tif"), ...)
orch.submit_task(normal, priority=10)

# Low priority (processed last)
low_priority = TaskConfig(task_id="test", input_path=Path("test.tif"), ...)
orch.submit_task(low_priority, priority=100)

# Process queue (respects priority)
results = orch.process_batch([])  # Tasks already submitted
```

**Priority Rules**:
- Lower number = higher priority
- Default priority: 0
- Negative priorities allowed

### Manual Worker Management

For advanced use cases:

```python
from lux_depth_v2.orchestrator import ProcessOrchestrator, TaskConfig

orch = ProcessOrchestrator(max_workers=1)

# Submit tasks
for i in range(100):
    task = TaskConfig(task_id=f"img_{i}", ...)
    orch.submit_task(task)

# Process in chunks with monitoring
while not orch.task_queue.empty():
    # Process next batch of 10
    for _ in range(10):
        if orch.task_queue.empty():
            break
        # Orchestrator will process tasks
    
    # Check progress
    progress = orch.get_progress()
    print(f"Progress: {progress['completed']}/{progress['total_tasks']}")
    
    # Optional: pause for resource monitoring
    time.sleep(1)
```

## Integration with Other Phase 1 Components

### With Checkpoint System

Orchestrator + Checkpoints = Resumable batch processing:

```bash
# Start batch
python3 -m lux_depth_v2.cli \
  --input-dir input_images/ \
  --output-dir output/ \
  --checkpoint-dir .checkpoints/batch1

# Interrupt with CTRL+C
^C

# Resume later (exact same command)
python3 -m lux_depth_v2.cli \
  --input-dir input_images/ \
  --output-dir output/ \
  --checkpoint-dir .checkpoints/batch1
```

### With Resource Monitor

Orchestrator checks resources before each task:

```bash
python3 -m lux_depth_v2.cli \
  --input-dir input_images/ \
  --output-dir output/ \
  --memory-budget 48.0  # 48GB limit
```

If resources insufficient:
- Task queued until resources available
- Alert logged
- Graceful degradation (fallback to CPU)

### With Error Recovery

Orchestrator + Error Recovery = Automatic retry:

```bash
python3 -m lux_depth_v2.cli \
  --input-dir input_images/ \
  --output-dir output/ \
  --max-retries 3
```

Failure sequence:
1. Task fails (e.g., OOM)
2. Orchestrator isolates failure
3. Error recovery analyzes error
4. Retry with fallback config (MPS→CPU)
5. Success or mark as failed after 3 attempts

## Performance Considerations

### Overhead

- **Subprocess creation**: ~50-100ms per image
- **IPC overhead**: ~10-20ms per message
- **Total overhead**: <2% of processing time

### Memory

- **Orchestrator**: ~5MB (task queue + worker tracking)
- **Per worker**: Normal pipeline memory + ~2MB overhead

### Throughput

With `max_workers=1` (Phase 1):
- **Sequential processing**: Same as non-orchestrated
- **Benefit**: Fault tolerance, not parallelism
- **Future**: Phase 2 will enable parallel workers

## Troubleshooting

### Issue: Workers not starting

**Symptoms**: Tasks queued but not processing

**Causes**:
1. Resource constraints (memory/disk)
2. Device unavailable (GPU not found)
3. Input files locked/inaccessible

**Solutions**:
```bash
# Check resources
python3 -m lux_depth_v2.cli --input test.tif --output-dir /tmp --pre-flight-check

# Try CPU device
python3 -m lux_depth_v2.cli --input test.tif --output-dir /tmp --device cpu

# Check logs
python3 -m lux_depth_v2.cli ... 2>&1 | grep -i "worker\|error"
```

### Issue: Workers hanging

**Symptoms**: Active worker never completes

**Causes**:
1. Deadlock in pipeline code
2. GPU hang
3. I/O blocking

**Solutions**:
```bash
# Identify hanging process
ps aux | grep python | grep lux_depth

# Force kill (if needed)
kill -9 <PID>

# Restart with timeout monitoring
# (Future: Phase 2 will add task timeouts)
```

### Issue: High failure rate

**Symptoms**: Most images failing

**Causes**:
1. Invalid input files
2. Insufficient resources
3. Configuration issues

**Solutions**:
```bash
# Enable pre-flight checks
python3 -m lux_depth_v2.cli --input-dir input/ --output-dir output/ --pre-flight-check

# Check first image manually
python3 -m lux_depth_v2.cli --input input/test.tif --output-dir /tmp --device cpu

# Review failure logs
grep -i "error\|failed" output/processing.log
```

### Issue: Slow graceful shutdown

**Symptoms**: CTRL+C takes long time to stop

**Causes**:
1. Large image still processing
2. Cleanup operations slow
3. Network I/O blocking

**Solutions**:
```bash
# Force shutdown (CTRL+C twice)
^C  # First: request graceful shutdown
^C  # Second: force immediate shutdown

# Or kill directly
killall -9 python
```

## Best Practices

### 1. Always Use Orchestrator for Batch Processing

```bash
# Good: Orchestrator handles failures gracefully
python3 -m lux_depth_v2.cli --input-dir batch/ --output-dir output/

# Avoid: Legacy mode has no fault tolerance
python3 -m lux_depth_v2.cli --input-dir batch/ --output-dir output/ --disable-orchestrator
```

### 2. Combine with Checkpoints

```bash
# Best: Orchestrator + Checkpoints = Resumable + Fault-tolerant
python3 -m lux_depth_v2.cli \
  --input-dir batch/ \
  --output-dir output/ \
  --checkpoint-dir .checkpoints/batch1 \
  --max-retries 3
```

### 3. Monitor Large Batches

For batches >100 images:

```bash
# Use tail to monitor progress in real-time
python3 -m lux_depth_v2.cli --input-dir batch/ --output-dir output/ 2>&1 | \
  tee processing.log | \
  grep -E "(completed|failed|Batch processing)"
```

### 4. Set Memory Budgets on Constrained Systems

```bash
# For 64GB system, leave 16GB for OS
python3 -m lux_depth_v2.cli \
  --input-dir batch/ \
  --output-dir output/ \
  --memory-budget 48.0
```

### 5. Use Pre-flight Checks

```bash
# Catch issues before processing starts
python3 -m lux_depth_v2.cli \
  --input-dir batch/ \
  --output-dir output/ \
  --pre-flight-check
```

## FAQ

**Q: Does orchestrator make processing faster?**  
A: No (Phase 1 is sequential). It makes processing **more reliable** by isolating failures. Phase 2 will add parallel processing for speed.

**Q: Can I run multiple orchestrators simultaneously?**  
A: Not recommended - they may compete for GPU. Use one orchestrator with higher `max_workers` instead (Phase 2).

**Q: What happens if I kill the main process?**  
A: Worker processes become orphaned and may continue running. Use graceful shutdown (CTRL+C) instead.

**Q: Can I pause and resume orchestrator?**  
A: Not directly, but you can:
1. CTRL+C to stop gracefully
2. Checkpoints are saved
3. Restart with same command to resume

**Q: Does orchestrator work with service mode?**  
A: Orchestrator is for batch (CLI) mode. Service mode has its own concurrency handling.

---

For more information, see:
- [PHASE1_IMPLEMENTATION.md](../lux_depth_v2/PHASE1_IMPLEMENTATION.md)
- [CHECKPOINT_GUIDE.md](./CHECKPOINT_GUIDE.md)
