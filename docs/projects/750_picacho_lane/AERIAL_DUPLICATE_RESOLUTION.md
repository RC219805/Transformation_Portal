# Aerial View Duplicate Issue - Resolution Plan

## Problem
Two aerial view files with different sizes that normalize to the same base name:
- `750Picacho_Aerial.exr` (86M) → `750Picacho_Aerial`
- `2-750Picacho_Aerial-2.exr` (142M) → `750Picacho_Aerial`

This causes **output file conflicts** where one overwrites the other.

## Processed Outputs Show Differences
- `750Picacho_Aerial_MaxQuality.jpg` (4.8M)
- `2-750Picacho_Aerial-2_MaxQuality.jpg` (15M)

The 3x size difference suggests these are genuinely different views.

## Recommended Solutions

### Option 1: Rename for Clarity (RECOMMENDED)
Rename the files to reflect distinct camera angles:
```bash
# In source directory
mv "2-750Picacho_Aerial-2.exr" "750Picacho_Aerial_Wide.exr"
# Keep "750Picacho_Aerial.exr" as is (or rename to "750Picacho_Aerial_Standard.exr")
```

### Option 2: Keep Both with Suffixes
Process both but use different naming patterns:
```bash
# Process as two distinct views
750Picacho_Aerial_v1.exr
750Picacho_Aerial_v2.exr
```

### Option 3: Choose the Best
If one is a revision/replacement:
- Determine which is the final approved version
- Archive or delete the superseded file
- Keep only one aerial view

## Impact on Pipeline

**Current State:**
- Pipeline processes both files
- Later file overwrites earlier file's outputs
- Loss of one aerial view in final deliverables

**After Fix:**
- Each aerial view gets unique outputs
- All views preserved in final deliverables
- No file conflicts

## Next Steps

1. **Inspect both processed JPEGs** to determine if they're different angles or revisions
2. **Consult with stakeholder** (if needed) about which view(s) to keep
3. **Rename source files** according to chosen option
4. **Re-run pipeline** to generate complete deliverables

## Files Affected

### Source Files
- `/Users/rc/Desktop/Cache/750_LightFiction_Final_Views/16-Bit_EXRs/750Picacho_Aerial.exr`
- `/Users/rc/Desktop/Cache/750_LightFiction_Final_Views/16-Bit_EXRs/2-750Picacho_Aerial-2.exr`

### Output Directories (all contain duplicates)
- `Maximum_Quality_Final/`
- `Phase3_Refined/`
- `Processed_Output/Master_TIFFs/`
- `Processed_Output/Web_JPEGs/`
- `TIFFs/16-Bit_TIFFs/`

