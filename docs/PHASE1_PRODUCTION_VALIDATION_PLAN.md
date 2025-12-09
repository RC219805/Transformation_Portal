# Phase 1 Production Validation Plan

**Date**: 2025-12-08  
**Version**: 1.0  
**Owner**: Transformation Portal Architect  
**Status**: Ready for Execution

---

## Executive Summary

This validation plan confirms Phase 1 stability architecture meets production requirements through comprehensive testing of fault isolation, resume capability, checkpoint management, resource monitoring, and error recovery.

**Success Criteria**: 6/6 test scenarios pass with <5% performance overhead.

---

## 1. Test Scenarios

### Scenario 1: Normal Batch Completion
**Objective**: Verify clean batch processing with no failures

**Setup**:
- 6 test images (mixed interior/exterior)
- Standard preset (`photo_realistic`)
- Clean output directory
- No pre-existing checkpoints

**Execution**:
```bash
lux-depth-v2 \
  --input-dir test_data/validation_batch_6/ \
  --output-dir output_validation_scenario1/ \
  --depth-dir test_data/validation_batch_6_depth/ \
  --preset photo_realistic \
  --enable-orchestrator \
  --orchestrator-max-workers 1
```

**Expected Results**:
- ✅ 6/6 images complete successfully
- ✅ All stages checkpoint correctly (6 stages × 6 images = 36 checkpoints)
- ✅ Final report shows 0 failures
- ✅ Checkpoints auto-cleaned after success
- ✅ Processing time: 240-360 seconds (40-60s/image)

**Validation Checks**:
```bash
# Check all outputs exist
ls output_validation_scenario1/*.tif | wc -l  # Should be 6

# Check report
cat output_validation_scenario1/*_report.json | jq '.summary.success_count'  # Should be 6
cat output_validation_scenario1/*_report.json | jq '.summary.failure_count'  # Should be 0

# Check checkpoints cleaned
ls output_validation_scenario1/.checkpoints/ | wc -l  # Should be 0 (auto-cleaned)
```

**Success Criteria**: 6/6 outputs, 0 failures, checkpoints cleaned

---

### Scenario 2: Interrupted Batch with Resume
**Objective**: Verify resume capability from checkpoint

**Setup**:
- Same 6 test images
- Interrupt processing after 2nd image completes
- Resume from checkpoint

**Execution**:
```bash
# Start batch
lux-depth-v2 \
  --input-dir test_data/validation_batch_6/ \
  --output-dir output_validation_scenario2/ \
  --depth-dir test_data/validation_batch_6_depth/ \
  --preset photo_realistic \
  --enable-orchestrator \
  --orchestrator-max-workers 1 &

PID=$!

# Wait for 2nd image to complete (~80-120 seconds)
sleep 100

# Interrupt (SIGINT = Ctrl+C)
kill -INT $PID
wait $PID

# Wait for graceful shutdown
sleep 5

# Verify checkpoints exist
ls output_validation_scenario2/.checkpoints/*.json | wc -l  # Should be 2-3

# Resume from checkpoint
lux-depth-v2 \
  --input-dir test_data/validation_batch_6/ \
  --output-dir output_validation_scenario2/ \
  --depth-dir test_data/validation_batch_6_depth/ \
  --preset photo_realistic \
  --enable-orchestrator \
  --orchestrator-max-workers 1 \
  --resume
```

**Expected Results**:
- ✅ Initial run processes 2/6 images before interrupt
- ✅ Checkpoints saved for completed images
- ✅ Graceful shutdown (no corruption)
- ✅ Resume skips completed images (2 already done)
- ✅ Resume completes remaining 4 images
- ✅ Total: 6/6 images complete
- ✅ Resume latency <5 seconds

**Validation Checks**:
```bash
# Check resume detected completed work
grep "Resuming from checkpoint" logs/lux_depth_v2.log  # Should appear

# Check final outputs
ls output_validation_scenario2/*.tif | wc -l  # Should be 6

# Check no duplicate processing
grep "Skipping completed" logs/lux_depth_v2.log | wc -l  # Should be 2

# Check total time (should be ~6 images worth, not 8)
cat output_validation_scenario2/*_report.json | jq '.summary.total_time'
```

**Success Criteria**: 6/6 outputs, resume works, no duplicate processing

---

### Scenario 3: Single Image Failure (Fault Isolation)
**Objective**: Verify one failure doesn't kill batch

**Setup**:
- 6 test images
- Inject failure in 3rd image (corrupt file or missing depth map)
- Verify batch continues

**Execution**:
```bash
# Create test data with one corrupt image
cp -r test_data/validation_batch_6/ test_data/validation_batch_6_corrupt/
# Corrupt the 3rd image
dd if=/dev/zero of=test_data/validation_batch_6_corrupt/image_003.tif bs=1024 count=1

# Run batch
lux-depth-v2 \
  --input-dir test_data/validation_batch_6_corrupt/ \
  --output-dir output_validation_scenario3/ \
  --depth-dir test_data/validation_batch_6_depth/ \
  --preset photo_realistic \
  --enable-orchestrator \
  --orchestrator-max-workers 1
```

**Expected Results**:
- ✅ Images 1-2 complete successfully
- ✅ Image 3 fails (subprocess isolated)
- ✅ Batch continues to images 4-6
- ✅ Final: 5/6 success, 1/6 failure
- ✅ No batch-level crash
- ✅ Failed task logged with error details

**Validation Checks**:
```bash
# Check outputs (5 successful)
ls output_validation_scenario3/*.tif | wc -l  # Should be 5

# Check report
cat output_validation_scenario3/*_report.json | jq '.summary.success_count'  # Should be 5
cat output_validation_scenario3/*_report.json | jq '.summary.failure_count'  # Should be 1

# Check failure logged
grep "Task failed" logs/lux_depth_v2.log | grep "image_003"  # Should exist

# Check no crash
grep "FATAL" logs/lux_depth_v2.log  # Should be empty
```

**Success Criteria**: 5/6 success, 1/6 failure, batch completes, no crash

---

### Scenario 4: Resource Constraint (Preflight Catches)
**Objective**: Verify preflight validation catches resource issues

**Setup**:
- Simulate low disk space (via config override)
- Or simulate low memory threshold
- Verify preflight fails gracefully

**Execution**:
```bash
# Test 1: Low disk space threshold
lux-depth-v2 \
  --input-dir test_data/validation_batch_6/ \
  --output-dir output_validation_scenario4/ \
  --depth-dir test_data/validation_batch_6_depth/ \
  --preset photo_realistic \
  --enable-orchestrator \
  --orchestrator-max-workers 1 \
  --orchestrator-disk-threshold-gb 999999  # Impossibly high
```

**Expected Results**:
- ✅ Preflight validation fails
- ✅ Clear error message about disk space
- ✅ No processing attempted
- ✅ Clean exit (no corruption)

**Validation Checks**:
```bash
# Check preflight failed
grep "Preflight validation failed" logs/lux_depth_v2.log  # Should exist

# Check disk space error
grep "disk space" logs/lux_depth_v2.log  # Should exist

# Check no outputs (should not have started)
ls output_validation_scenario4/*.tif 2>/dev/null | wc -l  # Should be 0

# Check exit code
echo $?  # Should be non-zero (failure)
```

**Success Criteria**: Preflight fails gracefully, no processing, clear error

---

### Scenario 5: Error Recovery with Fallback
**Objective**: Verify fallback strategies work (e.g., MPS→CPU)

**Setup**:
- Force MPS OOM error (via artificially high memory usage)
- Or use unsupported device
- Verify fallback to CPU

**Execution**:
```bash
# Simulate MPS unavailable (device override)
lux-depth-v2 \
  --input-dir test_data/validation_batch_6/ \
  --output-dir output_validation_scenario5/ \
  --depth-dir test_data/validation_batch_6_depth/ \
  --preset photo_realistic \
  --device mps_unavailable \
  --enable-orchestrator \
  --orchestrator-max-workers 1
```

**Expected Results**:
- ✅ First attempt fails (device unavailable)
- ✅ Error recovery triggers
- ✅ Fallback to CPU
- ✅ Retry succeeds
- ✅ All images complete (slower on CPU)

**Validation Checks**:
```bash
# Check fallback triggered
grep "Fallback strategy" logs/lux_depth_v2.log  # Should exist

# Check device change
grep "device.*cpu" logs/lux_depth_v2.log  # Should exist after fallback

# Check final success
cat output_validation_scenario5/*_report.json | jq '.summary.success_count'  # Should be 6

# Check retry count
cat output_validation_scenario5/*_report.json | jq '.tasks[0].retry_count'  # Should be 1
```

**Success Criteria**: Fallback works, retry succeeds, 6/6 completion

---

### Scenario 6: Checkpoint Retention Policy
**Objective**: Verify checkpoint cleanup follows retention policy

**Setup**:
- Run batch with checkpoint retention enabled
- Verify old checkpoints cleaned
- Verify failed task checkpoints retained

**Execution**:
```bash
# Run with checkpoint retention (retain failed only)
lux-depth-v2 \
  --input-dir test_data/validation_batch_6_corrupt/ \
  --output-dir output_validation_scenario6/ \
  --depth-dir test_data/validation_batch_6_depth/ \
  --preset photo_realistic \
  --enable-orchestrator \
  --orchestrator-max-workers 1 \
  --checkpoint-retention failed  # Only keep failed task checkpoints
```

**Expected Results**:
- ✅ Successful tasks checkpoints cleaned
- ✅ Failed task checkpoints retained (image_003)
- ✅ Checkpoint directory exists
- ✅ Only 1 checkpoint file (failed task)

**Validation Checks**:
```bash
# Check checkpoint directory
ls output_validation_scenario6/.checkpoints/ | wc -l  # Should be 1

# Check retained checkpoint is for failed task
cat output_validation_scenario6/.checkpoints/*.json | jq '.task_id'  # Should be "image_003"

# Check retention policy logged
grep "Cleaning successful checkpoints" logs/lux_depth_v2.log  # Should exist
```

**Success Criteria**: Failed checkpoint retained, successful cleaned

---

## 2. Performance Validation

### Objective
Verify performance overhead remains <5% with orchestrator enabled.

### Methodology
Compare processing time with and without orchestrator for identical batch.

**Baseline (No Orchestrator)**:
```bash
lux-depth-v2 \
  --input-dir test_data/validation_batch_6/ \
  --output-dir output_baseline/ \
  --depth-dir test_data/validation_batch_6_depth/ \
  --preset photo_realistic
  # No --enable-orchestrator

# Record total time
BASELINE_TIME=$(cat output_baseline/*_report.json | jq '.summary.total_time')
```

**With Orchestrator**:
```bash
lux-depth-v2 \
  --input-dir test_data/validation_batch_6/ \
  --output-dir output_with_orchestrator/ \
  --depth-dir test_data/validation_batch_6_depth/ \
  --preset photo_realistic \
  --enable-orchestrator \
  --orchestrator-max-workers 1

# Record total time
ORCHESTRATOR_TIME=$(cat output_with_orchestrator/*_report.json | jq '.summary.total_time')
```

**Calculate Overhead**:
```bash
OVERHEAD=$(echo "scale=2; (($ORCHESTRATOR_TIME - $BASELINE_TIME) / $BASELINE_TIME) * 100" | bc)
echo "Overhead: ${OVERHEAD}%"
```

**Success Criteria**: Overhead <5%

---

## 3. Test Data Preparation

### Required Test Images
Create validation dataset with diverse scenes:

```bash
mkdir -p test_data/validation_batch_6/
mkdir -p test_data/validation_batch_6_depth/

# Copy or symlink diverse test images
cp data/750_Picacho/GreatRoom.tif test_data/validation_batch_6/image_001.tif
cp data/750_Picacho/Kitchen.tif test_data/validation_batch_6/image_002.tif
cp data/750_Picacho/PrimaryBedroom.tif test_data/validation_batch_6/image_003.tif
cp data/750_Picacho/PrimaryBathroom.tif test_data/validation_batch_6/image_004.tif
cp data/750_Picacho/Pool.tif test_data/validation_batch_6/image_005.tif
cp data/750_Picacho/Aerial.tif test_data/validation_batch_6/image_006.tif

# Copy depth maps
cp output_750_Picacho_Depth_Maps/GreatRoom_depth.tif test_data/validation_batch_6_depth/image_001_depth.tif
cp output_750_Picacho_Depth_Maps/Kitchen_depth.tif test_data/validation_batch_6_depth/image_002_depth.tif
cp output_750_Picacho_Depth_Maps/PrimaryBedroom_depth.tif test_data/validation_batch_6_depth/image_003_depth.tif
cp output_750_Picacho_Depth_Maps/PrimaryBathroom_depth.tif test_data/validation_batch_6_depth/image_004_depth.tif
cp output_750_Picacho_Depth_Maps/Pool_depth.tif test_data/validation_batch_6_depth/image_005_depth.tif
cp output_750_Picacho_Depth_Maps/Aerial_depth.tif test_data/validation_batch_6_depth/image_006_depth.tif
```

### Test Data Characteristics
- **Image 1-4**: Interior scenes (living, kitchen, bedrooms)
- **Image 5**: Exterior pool (water, glass challenges)
- **Image 6**: Aerial (different depth distribution)
- **Resolution**: Mixed (2000-4000px)
- **Format**: 16-bit TIFF

---

## 4. Validation Script

Create automated validation script:

```bash
#!/bin/bash
# File: scripts/validate_phase1.sh

set -e

echo "=========================================="
echo "Phase 1 Production Validation"
echo "=========================================="
echo ""

# Setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Create test data
echo "📦 Preparing test data..."
bash scripts/prepare_validation_data.sh

# Scenario 1: Normal completion
echo ""
echo "🧪 Scenario 1: Normal Batch Completion"
bash scripts/run_scenario1.sh
if [ $? -eq 0 ]; then
  echo "✅ Scenario 1 PASSED"
else
  echo "❌ Scenario 1 FAILED"
  exit 1
fi

# Scenario 2: Resume
echo ""
echo "🧪 Scenario 2: Interrupted Batch with Resume"
bash scripts/run_scenario2.sh
if [ $? -eq 0 ]; then
  echo "✅ Scenario 2 PASSED"
else
  echo "❌ Scenario 2 FAILED"
  exit 1
fi

# Scenario 3: Fault isolation
echo ""
echo "🧪 Scenario 3: Single Image Failure (Fault Isolation)"
bash scripts/run_scenario3.sh
if [ $? -eq 0 ]; then
  echo "✅ Scenario 3 PASSED"
else
  echo "❌ Scenario 3 FAILED"
  exit 1
fi

# Scenario 4: Preflight
echo ""
echo "🧪 Scenario 4: Resource Constraint (Preflight Catches)"
bash scripts/run_scenario4.sh
if [ $? -eq 0 ]; then
  echo "✅ Scenario 4 PASSED"
else
  echo "❌ Scenario 4 FAILED"
  exit 1
fi

# Scenario 5: Error recovery
echo ""
echo "🧪 Scenario 5: Error Recovery with Fallback"
bash scripts/run_scenario5.sh
if [ $? -eq 0 ]; then
  echo "✅ Scenario 5 PASSED"
else
  echo "❌ Scenario 5 FAILED"
  exit 1
fi

# Scenario 6: Checkpoint retention
echo ""
echo "🧪 Scenario 6: Checkpoint Retention Policy"
bash scripts/run_scenario6.sh
if [ $? -eq 0 ]; then
  echo "✅ Scenario 6 PASSED"
else
  echo "❌ Scenario 6 FAILED"
  exit 1
fi

# Performance validation
echo ""
echo "📊 Performance Validation"
bash scripts/run_performance_test.sh
if [ $? -eq 0 ]; then
  echo "✅ Performance validation PASSED (<5% overhead)"
else
  echo "❌ Performance validation FAILED (>5% overhead)"
  exit 1
fi

echo ""
echo "=========================================="
echo "✅ All validation scenarios PASSED"
echo "=========================================="
echo ""
echo "Summary:"
echo "  - Normal completion: ✅"
echo "  - Resume capability: ✅"
echo "  - Fault isolation: ✅"
echo "  - Preflight validation: ✅"
echo "  - Error recovery: ✅"
echo "  - Checkpoint retention: ✅"
echo "  - Performance overhead: <5% ✅"
echo ""
echo "Phase 1 is PRODUCTION-READY ✅"
```

---

## 5. Expected Outcomes

### Success Criteria Summary

| Scenario | Success Criteria | Target |
|----------|------------------|--------|
| **1. Normal Completion** | 6/6 images complete | 100% |
| **2. Resume** | Resume works, no duplicate processing | ✅ |
| **3. Fault Isolation** | 5/6 success, 1/6 failure, no crash | ✅ |
| **4. Preflight** | Validation fails gracefully | ✅ |
| **5. Error Recovery** | Fallback works, retry succeeds | ✅ |
| **6. Checkpoint Retention** | Failed retained, successful cleaned | ✅ |
| **Performance** | Overhead <5% | <5% |

### Acceptance Criteria
- ✅ All 6 scenarios pass
- ✅ Performance overhead <5%
- ✅ No critical bugs found
- ✅ Error messages are clear and actionable
- ✅ Documentation matches behavior

---

## 6. Risk Assessment

### Low Risk
- Normal completion (well tested)
- Checkpoint creation (well tested)

### Medium Risk
- Resume logic (complex state management)
- Error recovery fallbacks (many code paths)

### High Risk
- Interrupt handling (signal handling edge cases)
- Checkpoint corruption (power failure scenarios)

### Mitigation
- **Resume logic**: Extensive unit tests + integration tests
- **Error recovery**: Fallback decision tree tests
- **Interrupt handling**: Graceful shutdown tests with SIGINT/SIGTERM
- **Checkpoint corruption**: Checksums + validation on load

---

## 7. Validation Timeline

### Day 1: Test Data Preparation
- Create validation dataset (6 images)
- Generate depth maps
- Create corrupt test data

### Day 2: Scenario Execution (1-3)
- Run normal completion test
- Run resume test
- Run fault isolation test

### Day 3: Scenario Execution (4-6)
- Run preflight validation test
- Run error recovery test
- Run checkpoint retention test

### Day 4: Performance Validation
- Baseline run (no orchestrator)
- Orchestrator run
- Calculate overhead

### Day 5: Analysis & Reporting
- Analyze results
- Document findings
- Create validation report

---

## 8. Reporting

### Validation Report Format

```markdown
# Phase 1 Production Validation Report

**Date**: [Execution Date]
**Executor**: [Name]
**System**: [Hardware/OS]

## Summary
- Total Scenarios: 6
- Passed: X/6
- Failed: Y/6
- Performance Overhead: Z%

## Detailed Results

### Scenario 1: Normal Completion
- Status: PASS/FAIL
- Success Rate: X/6
- Time: Xs
- Notes: [Any observations]

[... repeat for all scenarios ...]

## Performance Analysis
- Baseline Time: Xs
- Orchestrator Time: Xs
- Overhead: X% (Target: <5%)

## Issues Found
[List any issues discovered]

## Recommendations
[Any recommendations for improvement]

## Conclusion
Phase 1 is PRODUCTION-READY / NEEDS WORK
```

---

## 9. Post-Validation Actions

### If All Tests Pass (6/6)
1. ✅ Mark Phase 1 as **PRODUCTION-READY**
2. ✅ Create validation report
3. ✅ Update documentation with validation results
4. ✅ Proceed to Phase 1.1 (instrumentation)

### If Tests Fail (< 6/6)
1. ❌ Document failures in detail
2. ❌ Create bug tickets for each failure
3. ❌ Fix critical issues
4. ❌ Re-run validation
5. ❌ Do not proceed to Phase 1.1 until all tests pass

---

## 10. Conclusion

This validation plan ensures Phase 1 stability architecture meets production requirements through comprehensive testing of all critical features: fault isolation, resume capability, checkpoint management, resource monitoring, and error recovery.

**Execution Time**: 5 days  
**Required Resources**: Test images, depth maps, ~20GB disk space  
**Expected Outcome**: 6/6 scenarios pass, <5% overhead, PRODUCTION-READY ✅

---

**Author**: Transformation Portal Architect  
**Date**: 2025-12-08  
**Version**: 1.0
