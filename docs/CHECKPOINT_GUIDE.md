# Checkpoint System User Guide

## Overview

The Checkpoint System provides automatic progress saving and resume capability for the Lux Depth V2 pipeline. If processing is interrupted (crash, power loss, manual stop), you can resume from the last successful stage instead of starting over.

## How It Works

### Processing Stages

The pipeline processes images through these stages:

1. **INIT** - Initialization and validation
2. **DEPTH_LOAD** - Load depth maps and zone masks
3. **MATERIAL_SEGMENTATION** - Detect materials (wood, metal, glass, etc.)
4. **POST_PROCESSING** - Apply depth-aware enhancements
5. **UPSCALING** - 2x or 4x resolution enhancement
6. **EXPORT** - Save final outputs

After each stage completes successfully, a checkpoint is saved. If processing fails or is interrupted, you can resume from the last checkpoint.

### Checkpoint Storage

Checkpoints are stored as JSON files in the checkpoint directory (default: `.checkpoints/`):

```
.checkpoints/
├── image_001.json
├── image_002.json
└── image_003.json
```

Each checkpoint file contains:
- Task ID and input/output paths
- Current stage and completion status
- Timing information
- Error messages (if any)
- Retry count

## Basic Usage

### Enable Checkpointing (Default)

Checkpointing is enabled by default:

```bash
python3 -m lux_depth_v2.cli \
  --input-dir input_images/ \
  --output-dir output/ \
  --preset photo_realistic
```

Checkpoints will be saved to `.checkpoints/` in the current directory.

### Custom Checkpoint Directory

Specify a custom checkpoint directory:

```bash
python3 -m lux_depth_v2.cli \
  --input-dir input_images/ \
  --output-dir output/ \
  --checkpoint-dir /path/to/checkpoints
```

### Resume Processing

To resume interrupted processing, simply **run the same command again**:

```bash
# Original command that was interrupted
python3 -m lux_depth_v2.cli \
  --input-dir input_images/ \
  --output-dir output/ \
  --checkpoint-dir .checkpoints/batch1

# Resume by running the exact same command
python3 -m lux_depth_v2.cli \
  --input-dir input_images/ \
  --output-dir output/ \
  --checkpoint-dir .checkpoints/batch1
```

The system will:
1. Detect existing checkpoints
2. Skip completed images
3. Resume incomplete images from last successful stage

## Advanced Usage

### Check Checkpoint Status

Use Python to inspect checkpoints:

```python
from lux_depth_v2.checkpoint import CheckpointManager

manager = CheckpointManager(checkpoint_dir=".checkpoints")

# List all checkpoints
checkpoints = manager.list_checkpoints()

for cp in checkpoints:
    print(f"Task: {cp.task_id}")
    print(f"  Status: {'✅ Complete' if cp.success else '⏸️  Incomplete'}")
    print(f"  Current Stage: {cp.current_stage.value}")
    print(f"  Retry Count: {cp.retry_count}/{cp.max_retries}")
```

### Check Statistics

```python
from lux_depth_v2.checkpoint import CheckpointManager

manager = CheckpointManager(checkpoint_dir=".checkpoints")
stats = manager.get_statistics()

print(f"Total: {stats['total']}")
print(f"Completed: {stats['completed']}")
print(f"Failed: {stats['failed']}")
print(f"In Progress: {stats['in_progress']}")
```

### Manual Checkpoint Cleanup

Remove old checkpoints to save disk space:

```bash
python3 -c "
from lux_depth_v2.checkpoint import CheckpointManager
manager = CheckpointManager('.checkpoints')
manager.cleanup(older_than_days=7, completed_only=True)
print('Cleanup complete')
"
```

Options:
- `older_than_days`: Delete checkpoints older than N days (default: 7)
- `completed_only`: If True, only delete successful tasks (default: True)

### Delete Single Checkpoint

```python
from lux_depth_v2.checkpoint import CheckpointManager

manager = CheckpointManager(".checkpoints")
manager.delete_checkpoint("image_001")
```

## Checkpoint File Format

Checkpoint files are JSON with this structure:

```json
{
  "task_id": "image_001",
  "input_path": "/path/to/input.tif",
  "output_dir": "/path/to/output",
  "depth_path": "/path/to/depth/image_001.tif",
  "preset": "photo_realistic",
  "device": "auto",
  "upscale": 4,
  "current_stage": "post_processing",
  "stages": {
    "init": {
      "stage": "init",
      "status": "success",
      "timestamp": 1733684400.0,
      "elapsed_time": 0.1
    },
    "depth_load": {
      "stage": "depth_load",
      "status": "success",
      "timestamp": 1733684402.5,
      "elapsed_time": 2.4
    },
    "material_segmentation": {
      "stage": "material_segmentation",
      "status": "success",
      "timestamp": 1733684410.0,
      "elapsed_time": 7.5
    }
  },
  "started_at": 1733684400.0,
  "completed_at": null,
  "success": false,
  "retry_count": 0,
  "max_retries": 3
}
```

## Resume Behavior

### What Gets Resumed

- **Incomplete stages**: Processing continues from the last successful stage
- **Failed tasks**: Retried up to `max_retries` times (default: 3)
- **Completed images**: Skipped entirely (unless `--overwrite` is used)

### What Gets Skipped

- **Already successful**: Tasks marked as `success: true`
- **Max retries exceeded**: Tasks that failed >= 3 times
- **Corrupted checkpoints**: Invalid JSON files are skipped

### Resume Examples

#### Example 1: Batch Interrupted During Upscaling

```
Initial batch (interrupted at image 3 during upscaling):
  ✅ image_001: COMPLETE
  ✅ image_002: COMPLETE  
  ⏸️  image_003: POST_PROCESSING complete, UPSCALING interrupted
  ⏹️  image_004: Not started
  ⏹️  image_005: Not started

After resume:
  ⏭️  image_001: Skipped (already complete)
  ⏭️  image_002: Skipped (already complete)
  ▶️  image_003: Resume from UPSCALING stage
  ▶️  image_004: Start from INIT
  ▶️  image_005: Start from INIT
```

#### Example 2: Failed Task with Retry

```
Image processing failed during MATERIAL_SEGMENTATION:
  Attempt 1: ❌ Failed (OOM error)
  Checkpoint saved with retry_count=1

Resume #1 (automatic fallback to CPU):
  Attempt 2: ✅ Success (CPU device)
  
Final result: SUCCESS after 1 retry
```

## Best Practices

### 1. Use Descriptive Checkpoint Directories

```bash
# Good: Specific to this batch
--checkpoint-dir .checkpoints/750_picacho_production

# Bad: Generic name might conflict
--checkpoint-dir .checkpoints/temp
```

### 2. Regular Cleanup

Set up a cron job or scheduled task:

```bash
# Daily cleanup of completed checkpoints older than 7 days
0 2 * * * cd /path/to/project && python3 -c "from lux_depth_v2.checkpoint import CheckpointManager; CheckpointManager('.checkpoints').cleanup(7, True)"
```

### 3. Monitor Checkpoint Growth

```bash
# Check checkpoint directory size
du -sh .checkpoints/

# Count checkpoints
ls -1 .checkpoints/*.json | wc -l
```

### 4. Backup Important Checkpoints

Before major changes or cleanup:

```bash
# Backup checkpoint directory
tar -czf checkpoints_backup_$(date +%Y%m%d).tar.gz .checkpoints/
```

### 5. Use Different Checkpoint Dirs for Different Batches

```bash
# Production batch
python3 -m lux_depth_v2.cli \
  --input-dir production/ \
  --checkpoint-dir .checkpoints/production

# Test batch
python3 -m lux_depth_v2.cli \
  --input-dir test/ \
  --checkpoint-dir .checkpoints/test
```

## Troubleshooting

### Issue: Resume not working

**Symptoms**: Processing starts from scratch despite checkpoints existing

**Causes**:
1. Different checkpoint directory specified
2. Checkpoint files corrupted
3. Input path changed

**Solutions**:
```bash
# Verify checkpoint directory exists and has files
ls -la .checkpoints/

# Check specific checkpoint
cat .checkpoints/<task_id>.json | python3 -m json.tool

# Enable debug logging
python3 -m lux_depth_v2.cli --input test.tif --output-dir /tmp 2>&1 | grep -i checkpoint
```

### Issue: Checkpoint corruption

**Symptoms**: Error loading checkpoint JSON

**Solutions**:
```bash
# Validate all checkpoints
for f in .checkpoints/*.json; do
  echo "Checking $f"
  python3 -m json.tool "$f" > /dev/null && echo "  ✅ Valid" || echo "  ❌ Invalid"
done

# Delete corrupted checkpoint (will restart from scratch)
rm .checkpoints/corrupted_task_id.json
```

### Issue: Disk space from checkpoints

**Symptoms**: `.checkpoints/` directory growing too large

**Solutions**:
```bash
# Aggressive cleanup (all checkpoints > 1 day)
python3 -c "from lux_depth_v2.checkpoint import CheckpointManager; CheckpointManager('.checkpoints').cleanup(1, False)"

# Delete entire checkpoint directory (will lose resume capability)
rm -rf .checkpoints/
```

### Issue: Resume behavior not as expected

**Debug**:
```python
from lux_depth_v2.checkpoint import CheckpointManager

manager = CheckpointManager(".checkpoints")
cp = manager.load_checkpoint("task_id")

if cp:
    print(f"Can resume: {cp.can_resume()}")
    print(f"Last successful stage: {cp.get_last_successful_stage()}")
    print(f"Next stage: {cp.get_next_stage()}")
    print(f"Retry count: {cp.retry_count}/{cp.max_retries}")
else:
    print("No checkpoint found")
```

## Performance Impact

### Storage

- **Size per checkpoint**: ~2-5 KB (JSON)
- **1000 images**: ~2-5 MB total
- **10000 images**: ~20-50 MB total

### Processing Overhead

- **Save checkpoint**: ~5-10ms per stage
- **Load checkpoint**: ~2-5ms per task
- **Total overhead**: <1% of processing time

### Recommendation

**Keep checkpoints enabled** - the overhead is negligible and resume capability is invaluable for long batches or unstable environments.

## Integration with Error Recovery

Checkpoints work seamlessly with the error recovery system:

```
Processing Flow with Checkpoints + Error Recovery:

1. Start processing image
2. Save checkpoint at each stage
3. If error occurs:
   a. Checkpoint saved with error details
   b. Error recovery classifies error
   c. Retry with fallback config if appropriate
   d. Update checkpoint with retry count
4. If success:
   a. Save final checkpoint with success=true
5. If max retries exceeded:
   a. Save final checkpoint with failure details
   b. Move to next image
```

## FAQ

**Q: Do checkpoints slow down processing?**  
A: No, overhead is <1% of total processing time (~5-10ms per stage).

**Q: Can I disable checkpointing?**  
A: Yes, but not recommended. To disable, use a legacy workflow or manually delete checkpoints after each run.

**Q: How long are checkpoints kept?**  
A: Forever unless you run manual cleanup. Recommended: cleanup completed checkpoints after 7 days.

**Q: Can I move checkpoint files?**  
A: Yes, but update paths in JSON files or they won't match inputs.

**Q: Do checkpoints work with --overwrite?**  
A: Yes, but `--overwrite` will reprocess all images. Checkpoints are primarily for resume, not for intentional reprocessing.

**Q: Can multiple processes share checkpoints?**  
A: Not recommended - each process should have its own checkpoint directory to avoid conflicts.

---

For more information, see [PHASE1_IMPLEMENTATION.md](../lux_depth_v2/PHASE1_IMPLEMENTATION.md)
