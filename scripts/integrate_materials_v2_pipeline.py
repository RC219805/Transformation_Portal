#!/usr/bin/env python3
"""Integration helper for Materials v2 into pipeline.py.

This script provides guidance and code snippets for integrating
Materials v2 engine into the main processing pipeline.
"""

INTEGRATION_GUIDE = """
================================================================================
Materials v2 Pipeline Integration Guide
================================================================================

FILE: lux_depth_v2/pipeline.py

STEP 1: Add Import (near top of file)
--------------------------------------
from .materials_v2 import MaterialsV2Engine, MaterialsV2Config

STEP 2: Initialize Engine in __init__ (after existing init code)
-----------------------------------------------------------------
def __init__(self, cfg: PipelineConfig):
    # ... existing initialization ...
    
    # Initialize Materials v2 if enabled
    self.materials_engine = None
    if cfg.materials_v2 and cfg.materials_v2.enabled:
        logger.info("Initializing Materials v2 engine...")
        self.materials_engine = MaterialsV2Engine(
            config=cfg.materials_v2,
            device=self.device,
        )
        logger.info(f"Materials v2 enabled: confidence_threshold={cfg.materials_v2.confidence.confidence_threshold}")

STEP 3: Add Processing Stage (in _process_one, after depth processing, before upscaling)
-----------------------------------------------------------------------------------------
def _process_one(self, input_path: Path, depth_path: Optional[Path], output_dir: Path):
    # ... existing code: load image, depth processing ...
    
    # Apply depth-aware processing
    img = self._apply_depth_aware_processing(img, depth_map, zones)
    
    # === Materials v2 Stage ===
    if self.materials_engine:
        import time
        
        logger.info("Applying Materials v2 enhancements...")
        materials_start = time.time()
        
        try:
            # Process with VRAM lifecycle management
            with self.materials_engine.vram_manager.context_manager():
                img = self.materials_engine.process(
                    image=img,
                    depth_map=depth_map,
                    zones=zones,
                )
            
            materials_time = time.time() - materials_start
            logger.info(f"Materials v2 complete: {materials_time:.2f}s")
            
        except Exception as e:
            logger.warning(f"Materials v2 failed: {e}")
            logger.info("Continuing without material enhancement")
            # img unchanged (use depth-processed version)
    # === End Materials v2 Stage ===
    
    # Continue with upscaling
    img_up = self._upscale(img)
    # ... rest of processing ...

STEP 4: Verification Commands
------------------------------
# Test single image
python3 -m lux_depth_v2.cli \\
  --input input_images/750_Picacho/Optimized_TIFFs/750Picacho_Pool_Ultimate.tif \\
  --output-dir output_Materials_V2_Integration_Test \\
  --materials-v2 \\
  --confidence-threshold 0.6 \\
  --cache-masks

# Check for Materials v2 logging
# Should see: "Initializing Materials v2 engine..."
#             "Applying Materials v2 enhancements..."
#             "Materials v2 complete: X.XXs"

# Verify cache files created
ls -la .materials_v2_cache/

STEP 5: Integration Checklist
------------------------------
[ ] Import MaterialsV2Engine added
[ ] materials_engine initialized in __init__
[ ] Processing stage added in _process_one
[ ] VRAM context manager used
[ ] Error handling with graceful degradation
[ ] Timing logged
[ ] Single image test passes
[ ] Cache files generated
[ ] Materials v2 logging visible
[ ] Output quality acceptable

================================================================================
ALTERNATIVE: Automated Integration
================================================================================

If you prefer automated integration, run:

python3 scripts/integrate_materials_v2_pipeline.py --auto

This will:
1. Backup pipeline.py
2. Apply integration patches
3. Verify syntax
4. Run test

================================================================================
"""

def print_guide():
    """Print integration guide."""
    print(INTEGRATION_GUIDE)


def find_integration_points(pipeline_file: str = "lux_depth_v2/pipeline.py"):
    """Analyze pipeline.py and identify integration points."""
    
    try:
        with open(pipeline_file) as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: {pipeline_file} not found")
        return
    
    print("\nAnalyzing pipeline.py...")
    print("=" * 60)
    
    # Find import section
    import_section_end = 0
    for i, line in enumerate(lines):
        if line.strip() and not line.startswith(('from', 'import', '#')):
            import_section_end = i
            break
    
    print(f"\n✓ Import section ends at line {import_section_end}")
    print(f"  Add Materials v2 import here")
    
    # Find __init__ method
    init_line = None
    for i, line in enumerate(lines):
        if "def __init__" in line and "PipelineConfig" in line:
            init_line = i
            break
    
    if init_line:
        print(f"\n✓ __init__ method found at line {init_line}")
        print(f"  Add materials_engine initialization here")
    
    # Find _process_one method
    process_one_line = None
    for i, line in enumerate(lines):
        if "def _process_one" in line:
            process_one_line = i
            break
    
    if process_one_line:
        print(f"\n✓ _process_one method found at line {process_one_line}")
        print(f"  Add Materials v2 processing stage here")
        
        # Find upscaling call
        for i in range(process_one_line, min(process_one_line + 200, len(lines))):
            if "_upscale" in lines[i]:
                print(f"\n✓ Upscaling call at line {i}")
                print(f"  Materials v2 should be called BEFORE this line")
                break
    
    print("\n" + "=" * 60)
    print("\nRecommendation: Add Materials v2 stage between depth")
    print("processing and upscaling for optimal quality.")
    print("\n")


def create_backup(pipeline_file: str = "lux_depth_v2/pipeline.py"):
    """Create backup of pipeline.py."""
    import shutil
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = f"{pipeline_file}.backup_{timestamp}"
    
    try:
        shutil.copy2(pipeline_file, backup_file)
        print(f"✓ Backup created: {backup_file}")
        return backup_file
    except Exception as e:
        print(f"✗ Backup failed: {e}")
        return None


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--analyze":
        find_integration_points()
    elif len(sys.argv) > 1 and sys.argv[1] == "--backup":
        create_backup()
    else:
        print_guide()
        
        if input("\nAnalyze pipeline.py for integration points? (y/n): ").lower() == 'y':
            print()
            find_integration_points()
        
        if input("\nCreate backup of pipeline.py? (y/n): ").lower() == 'y':
            print()
            create_backup()
