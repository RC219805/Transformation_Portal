# 750 Picacho Lane - Professional Refinement Plan

**Date**: November 17, 2025  
**Client Direction**: Premium luxury real estate photography refinement  
**Goal**: Museum-quality hero shots for campaign

---

## Image 1: Front Exterior / Arrival (HERO SHOT)
**File**: `V2_750Picacho_Aerial.tiff` (396 MB)  
**Priority**: HIGHEST - Primary campaign visual

### Refinements Required:

**Exposure & Contrast**:
- Drop highlights in sky (-0.15 to -0.20 stops) → Recover color and gradation
- Add contrast and clarity to stone/stucco (+0.20 contrast, +0.35 clarity)
- Ensure crisp architectural form reading

**Color Balance**:
- Cool shadows (-200K tint shift) → Separate warm interior from exterior
- Desaturate greenery (-15% saturation) → Reduce competition with house
- Preserve warm interior lighting glow

**Cars & Foreground**:
- Reduce car contrast/saturation (-10-15%) → Lifestyle props, not subjects
- Darken cobblestone drive (-0.10 stops) → Keep focus on illuminated façade

**Composition**:
- Add subtle vignette centered on main entry volume (-0.15 edge darkening)
- Lift ocean view haze → Add clarity to distant horizon (+0.25 clarity selective)
- Emphasize coastal setting

**Technical Parameters**:
```python
settings = {
    'exposure': 0.05,
    'highlights': -0.18,
    'contrast': 1.20,
    'clarity': 0.35,
    'white_balance_temp': 5200,
    'shadow_tint': -5,  # Cooler
    'vibrance': 0.15,
    'saturation': 0.95,  # Slight reduction
    'vignette': -0.15,
    'selective_clarity': {
        'ocean_horizon': 0.25,
        'architecture': 0.35,
        'cars': -0.10
    }
}
```

---

## Image 2: Great Room / Double-Height Living
**File**: `V2_750Picacho_GreatRoom.tiff` (69 MB)  
**Goal**: Reduce blown-out feel, increase texture richness

### Refinements Required:

**Exposure**:
- Pull highlights down (-0.20 to -0.25) → Avoid clipping on windows/walls
- Raise midtone contrast (+0.22) → Define stone, rug, furniture textures

**Color & Warmth**:
- Reduce overall warmth (-300K) → Avoid yellow cast on ceiling/walls
- Keep interior lighting warm (preserve 2800-3000K zones)
- Neutralize daylight whites (5800K window light)

**Local Adjustments**:
- Add clarity to stone walls (+0.40)
- Add clarity to rug pattern (+0.35)
- Darken ceiling around chandelier (-0.12 stops selective)
- Make fixture stand out

**Staging Refinement** (if re-rendering):
- Remove 1-2 tabletop accessories on coffee tables/console
- Ensure verticals perfectly straight (architectural editorial look)

**Technical Parameters**:
```python
settings = {
    'exposure': 0.08,
    'highlights': -0.22,
    'shadows': 0.12,
    'midtone_contrast': 0.22,
    'clarity': 0.35,
    'white_balance_temp': 5400,  # Warmer for interiors
    'vibrance': 0.18,
    'local_adjustments': {
        'stone_walls': {'clarity': 0.40},
        'rug': {'clarity': 0.35},
        'ceiling_chandelier': {'exposure': -0.12}
    }
}
```

---

## Image 3: Kitchen + Casual Living
**File**: `V2_750Picacho_Kitchen.tiff` (116 MB)  
**Goal**: Clean, bright, calm - reduce busyness

### Refinements Required:

**Tone & Contrast**:
- Add selective contrast to cabinetry/island (+0.18)
- Deepen shadows beneath island/furniture (-0.08) → Anchor them
- Maintain brightness without harshness

**Color Adjustments**:
- Desaturate wood tones (-12%) → Avoid orange, aim for European oak
- Neutralize sofa/rug color cast → Soft calm neutrals
- Overall cooler tone (5600K)

**Simplification**:
- Reduce visible decor on counters/tables → 1-2 strong pieces max
- Simplify art (lower contrast, quieter)
- Avoid architectural competition

**Depth & Framing**:
- Increase micro-contrast in view to dining/exterior (+0.15)
- Eye flow: kitchen → seating → terrace
- Optional: gentle DoF falloff toward far dining area

**Technical Parameters**:
```python
settings = {
    'exposure': 0.10,
    'contrast': 1.18,
    'clarity': 0.32,
    'white_balance_temp': 5600,  # Neutral-cool
    'vibrance': 0.18,
    'saturation': 0.88,  # Wood desaturation
    'shadow_lift': 0.10,
    'local_adjustments': {
        'cabinetry_island': {'contrast': 0.18},
        'view_to_exterior': {'clarity': 0.15}
    }
}
```

---

## Image 4: Pool / Rear Elevation at Dusk (PRIMARY CAMPAIGN VISUAL)
**File**: `V2_750Picacho_Pool.tiff` (116 MB)  
**Priority**: HIGHEST - Already processed, needs upgrade to Tier 3

### Refinements Required:

**Sky & Ambient Light**:
- Deepen blue hour sky (-0.10 stops)
- Add gradient (darker top, lighter horizon)
- Warm interior/exterior vs. cool sky contrast (2800K vs. 6500K+)

**Pool & Landscaping**:
- Enhance water clarity and reflection (+0.30 clarity selective)
- Increase tile pattern contrast (subtle)
- Cool pool water slightly (5400K selective)
- Fresh, crisp against warm house

**Color Balance**:
- Reduce purple tree saturation (-12%) → Focal without artificial feel
- Harmonize green tones → Avoid neon greens
- Natural landscape palette

**Composition & Focus**:
- Apply vignette centered on covered terrace/living (-0.12)
- Crop small amount from left/right edges
- Tighten composition around pool/façade

**Technical Parameters**:
```python
settings = {
    'exposure': 0.08,
    'highlights': -0.15,
    'contrast': 1.20,
    'clarity': 0.35,
    'white_balance_temp': 5200,
    'vibrance': 0.22,
    'saturation': 0.95,
    'vignette': -0.12,
    'local_adjustments': {
        'sky': {'exposure': -0.10, 'gradient': 'top_to_horizon'},
        'pool_water': {'clarity': 0.30, 'temp': 5400},
        'purple_tree': {'saturation': -0.12},
        'interior_lights': {'temp': 2800, 'glow': 0.08}
    }
}
```

---

## Image 5: Primary Bath + Outdoor Shower
**File**: `V2_750Picacho_PrimaryBathroom.tiff` (275 MB)  
**Goal**: Emphasize indoor-outdoor spa experience

### Refinements Required:

**Composition**:
- Crop from far right to minimize long empty wall
- OR add landscape beyond frosted glass (if re-rendering)
- Keep doorway transition on third for balance

**Lighting**:
- Increase directional light on vanity textures (+0.20 selective)
- Showcase wood fluting, stone countertop richness
- Add gradient on outdoor corridor floor → Avoid flat plane

**Color & Warmth**:
- Cool outdoor tones (5200K selective)
- Keep indoor vanity warm (5600K)
- Contrast cozy interior vs. fresh exterior
- Tame yellow cast on vanity wall tiles

**Decluttering**:
- Remove or reduce vanity objects → 1-2 spa-style items max
- Refine wood tub/shower accessories → Custom, not generic

**Technical Parameters**:
```python
settings = {
    'exposure': 0.12,
    'contrast': 1.18,
    'clarity': 0.35,
    'white_balance_temp': 5600,  # Indoor
    'vibrance': 0.18,
    'local_adjustments': {
        'outdoor_area': {'temp': 5200, 'clarity': 0.25},
        'vanity_textures': {'exposure': 0.20, 'clarity': 0.40},
        'corridor_floor': {'gradient': 'depth'}
    }
}
```

---

## Image 6: Primary Bedroom Opening to Pool
**File**: `V2_750Picacho_PrimaryBedroom.tiff` (137 MB)  
**Goal**: Romantic, calm, ocean/pool oriented

### Refinements Required:

**Tone & Warmth**:
- Reduce overall warmth on stone wall/bedding (-200K)
- Keep sconces warm (2800K selective)
- Neutralize rest of scene
- Add shadow depth behind/under bed and nightstand

**View & Connection**:
- Increase clarity/contrast in pool/ocean view (+0.30)
- Unmistakable indoor-outdoor connection
- Cool exterior tones (water/sky 5400K selective)
- Contrast with warm interior lighting

**Tree & Foliage**:
- Reduce purple tree saturation (-10-15%)
- Natural feel, not overpowering
- Harmonize greens → Avoid neon/artificial

**Composition & Staging**:
- If re-rendering: pull camera back to show more bed base/nightstand
- Don't crowd frame
- Smooth bedding but not "CG perfect" (one intentional fold/throw)

**Technical Parameters**:
```python
settings = {
    'exposure': 0.10,
    'contrast': 1.16,
    'clarity': 0.32,
    'white_balance_temp': 5400,
    'vibrance': 0.18,
    'shadow_lift': 0.10,
    'local_adjustments': {
        'pool_ocean_view': {'clarity': 0.30, 'contrast': 0.20, 'temp': 5400},
        'sconces': {'temp': 2800, 'glow': 0.08},
        'purple_tree': {'saturation': -0.12},
        'bed_furniture': {'shadow_depth': 0.15}
    }
}
```

---

## Processing Strategy

### TIER 3 - Museum Quality (Recommended)
Apply to all 6 images for campaign consistency:

**Full Pipeline**:
1. Material Response (PBR reconstruction)
2. Multi-zone lighting stratification (2700-3200K interior, 5200-6500K exterior)
3. Atmospheric integration (depth-aware)
4. Room-specific color grading (per plan above)
5. Selective local adjustments
6. Chromatic aberration (subtle, peripheral)
7. Luxury final polish (glow, vignette)

**Processing Order**:
1. Image 4 (Pool) - Already good, upgrade to Tier 3
2. Image 1 (Aerial) - Hero shot priority
3. Image 2 (Great Room) - High visibility
4. Image 6 (Primary Bedroom) - Key lifestyle shot
5. Image 5 (Primary Bath) - Spa luxury
6. Image 3 (Kitchen) - Supporting shot

**Estimated Time**: 2-3 minutes per image (12-18 minutes total)

---

## Next Steps

1. Process with custom settings per room (not generic preset)
2. Apply selective/local adjustments using masks
3. Review for consistency across campaign
4. Final polish and export in multiple formats
5. Deliver TIFF masters + web JPEGs

---

**Ready to execute this professional refinement plan?**
