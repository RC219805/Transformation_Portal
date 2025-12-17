# Holdout Image Acquisition Specification
## Water Detection Phase C Validation - Architectural Glass Negative Set

**Version**: v1  
**Date**: 2025-12-17  
**Purpose**: Phase C completion gate - validate multi-scale glass suppressor  
**Target**: Prove detector doesn't false-trigger on architectural glass, reflective surfaces, blue materials  
**Feature Under Test**: Multi-scale grid detection distinguishing pool tiles from glass facades  

---

## Hard Requirements
- ✅ **Zero actual water visible** (no pools, oceans, puddles, even small amounts)
- ✅ **Real photographs** (not renders/synthetic/AI-generated)
- ✅ **Resolution**: ≥1024px short edge (minimum requirement for detection pipeline)
- ✅ **Clean licensing**: Own photo, licensed stock, or explicit permission
- ✅ **No heavy HDR/AI artifacts**: Natural captures preferred

---

## Detector Behavior Being Tested

### Multi-Scale Grid Detection
- **Grid Frequency Analysis**: Distinguish high-frequency pool tile grids (10-30cm spacing) from low-frequency glass mullions (1-2m spacing)
- **Separate architectural grids from water surface patterns**

### Blue Hue Discrimination
- **Color Saturation Tests**: Separate water blue from sky reflection blue or painted blue
- **Ambient blue light vs actual water color**

### Reflectivity Analysis
- **Specular Reflections**: Differentiate water surface from glass/polished stone reflections
- **Mirror-like surfaces that aren't water**

### Edge Detection
- **Boundary Analysis**: Detect water boundaries vs glass frame boundaries
- **Panel seams vs pool coping edges**

---

9. **skylight_reflection_001.jpg** - Glass skylight with sky reflection
10. **skylight_reflection_002.jpg** - Interior skylight with diffused blue light
11. **pool_tiles_closeup_001.jpg** - Pool tiles with grid pattern, no water visible
12. **pool_tiles_closeup_002.jpg** - Blue ceramic pool tiles without water
13. **concrete_blue_tint_001.jpg** - Concrete surface with blue color cast

**Difficulty**: Hard-Medium  
**Why**: Tests grid alignment, blue hue confusers, and pool context without actual water

---

## Image #1: glass_facade_001.jpg
**Category**: Modern office building, reflective glass façade

### Visual Description
Photograph a modern office tower or commercial building with a large, continuous reflective glass façade. Frame the shot to capture at least 3-5 stories of uninterrupted glass panels. Shoot from ground level at a slight upward angle (10-20°) to emphasize the vertical expanse. Include sky reflections in the glass and capture the geometric grid of mullions/window frames.

### Confuser Characteristics
- **Blue Reflections**: Glass reflects blue sky, creating water-like color saturation
- **Geometric Grid**: Window mullions create regular grid patterns (but lower frequency than pool tiles)
- **Reflectivity**: Specular highlights similar to water surface glare
- **Edge Alignment**: Vertical/horizontal glass panel edges may resemble pool boundaries

**Why It Challenges the Detector**: Tests whether grid detection distinguishes large-scale architectural grids (1-2m spacing) from fine pool tile grids (10-30cm spacing).

### Capture Requirements
- **Lighting**: Midday or early afternoon with clear/partly cloudy sky (strong sky reflections)
- **Camera**: Auto mode acceptable; disable HDR/AI processing if possible
- **Resolution**: Minimum 2048×1536px (3MP+) - capture full building facade detail
- **Focus**: Focus on mid-height glass panels; ensure mullions are sharp

### Quality Checkpoints
- ✅ Sky reflections clearly visible in glass (blue tint present)
- ✅ Window mullions/frames form visible grid pattern
- ✅ No actual water visible (pools, fountains, puddles)
- ✅ Minimal lens distortion (avoid ultra-wide angles)
- ⚠️ Avoid: Heavy shadows obscuring glass, shooting directly into sun (lens flare)

### Where to Find It
- **Primary**: Downtown business district (any major city) - office towers built post-1990
- **Alternative**: Modern university buildings, corporate campuses, airport terminals
- **Access**: Public sidewalk photography; no trespassing required (shoot from street level)

---

## Image #2: glass_facade_002.jpg
**Category**: Glass curtain wall, sky reflections

### Visual Description
Capture a glass curtain wall system - a facade where glass extends continuously across multiple floors with minimal visible framing. Frame the shot to show the seamless glass surface with strong sky reflections. Shoot on a day with scattered clouds to create varied blue tones in the reflection. Include at least 20-30 individual glass panels in the frame.

### Confuser Characteristics
- **Continuous Blue Surface**: Minimal visual breaks create water-like continuous blue plane
- **Varied Blue Tones**: Cloud reflections create saturation gradients like rippling water
- **Minimal Texture**: Smooth glass lacks the high-frequency texture of pool tiles
- **Specular Reflections**: Bright cloud highlights mimic sunlight on water

**Why It Challenges the Detector**: Tests blue saturation thresholds and whether detector requires texture/grid to confirm water.

### Capture Requirements
- **Lighting**: Partly cloudy day, late morning (9-11am) or mid-afternoon (2-4pm)
- **Camera**: Smartphone auto mode; enable "vibrant" color mode if available
- **Resolution**: Minimum 1920×1080px (2MP+) - emphasize glass surface area
- **Focus**: Tap to focus on glass mid-frame; ensure reflections are sharp

### Quality Checkpoints
- ✅ Varied blue tones from cloud reflections (gradient, not flat color)
- ✅ Minimal visible frame structure (curtain wall aesthetic)
- ✅ Sharp focus on glass surface (not distant buildings reflected)
- ⚠️ Avoid: Overcast/gray sky (want blue reflections), dirty/streaked glass

### Where to Find It
- **Primary**: Modern hotels, hospitals, corporate headquarters (1990s-present architecture)
- **Alternative**: Museum facades (Getty Center, SFMOMA-style), convention centers
- **Access**: Public areas; photograph from plaza/street without entering building

---

## Image #3: glass_facade_003.jpg
**Category**: Commercial building, tinted glass panels

### Visual Description
Photograph a commercial building with tinted/colored glass (bronze, green, or blue-tinted). Frame to show individual glass panels with visible edges/seams between panels. Shoot at a 30-45° angle to the facade to emphasize panel boundaries and edge reflections. Capture at least 4×4 grid of panels.

### Confuser Characteristics
- **Blue/Cyan Tint**: Some architectural glass has inherent blue/green color cast
- **Panel Edges**: Regular panel seams create linear patterns (horizontal/vertical alignment)
- **Dual Reflections**: Edge framing + glass surface reflections create layered geometry
- **Rectangular Modules**: Repeated rectangular shapes mimic pool/water body geometry

**Why It Challenges the Detector**: Tests edge detection logic - can detector distinguish glass panel seams from pool coping/tile boundaries?

### Capture Requirements
- **Lighting**: Avoid direct sun (creates blown highlights); overcast or shaded side of building
- **Camera**: Manual exposure lock (tap and hold); expose for glass mid-tones
- **Resolution**: Minimum 2048×1536px - capture panel edge detail
- **Focus**: Focus on panel edges/seams (not distant reflections)

### Quality Checkpoints
- ✅ Panel seams/edges clearly visible (not lost in reflections)
- ✅ Tinted glass color visible (bronze/blue/green cast)
- ✅ No motion blur (hold phone steady or use timer)
- ✅ Rectangular panel grid pattern evident
- ⚠️ Avoid: Shooting into direct sunlight, extreme barrel distortion

### Where to Find It
- **Primary**: 1970s-1990s office buildings (common bronze/green tinted glass era)
- **Alternative**: Shopping mall exteriors, parking structure facades
- **Access**: Public sidewalk; shoot during business hours (better lighting)

---

## Image #4: blue_wall_painted_001.jpg
**Category**: Flat blue painted exterior

### Visual Description
Photograph an exterior wall painted solid blue (any blue - cobalt, sky blue, navy, teal). Frame to fill 60-80% of frame with the blue wall surface. Include some architectural context (edge of wall, adjacent structure, ground) to show it's a building, not sky. Shoot perpendicular to wall (straight-on, not angled).

### Confuser Characteristics
- **Flat Blue Saturation**: Solid blue color matches water HSV ranges
- **Minimal Texture**: Smooth painted surface lacks water's reflection/ripples
- **No Grid Pattern**: Tests detector behavior when grid detection returns null
- **Uniform Color**: No brightness variation (unlike water with reflections)

**Why It Challenges the Detector**: Tests pure color-based detection - can detector reject blue surfaces lacking water's physical characteristics (reflections, waves, depth cues)?

### Capture Requirements
- **Lighting**: Diffuse light (overcast or shaded) to minimize glare/hot spots
- **Camera**: Auto white balance; standard color mode (not "vivid")
- **Resolution**: Minimum 1600×1200px (2MP+)
- **Focus**: Ensure wall texture visible (paint surface, not out-of-focus blur)

### Quality Checkpoints
- ✅ Wall fills most of frame (dominant blue surface)
- ✅ Blue color clearly visible (not washed out or too dark)
- ✅ Minimal glare/specular highlights (flat lighting preferred)
- ✅ Some context visible (edge of wall, ground, adjacent structure)
- ⚠️ Avoid: Heavy shadows, uneven lighting, dirty/peeling paint

### Where to Find It
- **Primary**: Modern residential buildings, beach/coastal structures, commercial signage
- **Alternative**: Sports facilities (tennis court walls), school exteriors, shipping containers
- **Access**: Public areas; photograph from sidewalk/parking lot

---

## Image #5: blue_wall_painted_002.jpg
**Category**: Interior blue wall, soft lighting

### Visual Description
Photograph an interior wall painted blue (living room, office, hallway). Frame to show 1-2 meters of wall surface with soft, diffused interior lighting. Include partial view of floor or ceiling edge to provide spatial context. Shoot in ambient room lighting (no flash).

### Confuser Characteristics
- **Softer Blue Tones**: Interior lighting creates muted blue (less saturated than outdoor)
- **Even Illumination**: Soft shadows/highlights unlike water's dynamic reflections
- **Flat Surface**: Tests detector on low-contrast blue surfaces
- **Warm Color Cast**: Incandescent/LED lighting adds yellow tint to blue

**Why It Challenges the Detector**: Tests detection under non-daylight conditions and with color temperature shifts that might alter blue channel values.

### Capture Requirements
- **Lighting**: Ambient room lighting only (lamps, windows); NO flash
- **Camera**: Auto white balance; ISO 400-800 acceptable (some grain OK)
- **Resolution**: Minimum 1920×1080px (2MP+)
- **Focus**: Wall surface; slight depth of field acceptable

### Quality Checkpoints
- ✅ Blue color visible despite warm interior lighting
- ✅ Wall texture discernible (paint finish, not blurry)
- ✅ No flash artifacts (hot spots, harsh shadows)
- ✅ Natural indoor lighting (daylight from windows OK)
- ⚠️ Avoid: Mixed color temperature (fluorescent + incandescent), extreme underexposure

### Where to Find It
- **Primary**: Residential homes (bedrooms, living rooms), offices with colored accent walls
- **Alternative**: Hotels, Airbnb listings, furniture showrooms, design studios
- **Access**: Personal residence or with permission (friend's home, public showroom)

---

## Image #6: reflective_stone_001.jpg
**Category**: Polished granite/marble

### Visual Description
Photograph polished granite or marble (countertop, floor, wall cladding). Frame to show 0.5-1m² of stone surface with visible reflections (overhead lights, windows, objects). Shoot at 30-45° angle to surface to capture specular reflections. Include natural stone pattern/veining.

### Confuser Characteristics
- **Specular Reflections**: Polished stone reflects light like water surface
- **Color Variations**: Some granite has blue/gray tones; marble can be white-blue
- **Natural Patterns**: Veining/crystals create irregular texture (unlike grid tiles)
- **Gloss Finish**: High reflectivity mimics water's mirror-like properties

**Why It Challenges the Detector**: Tests reflectivity-based detection - can detector distinguish stone gloss from water surface reflections?

### Capture Requirements
- **Lighting**: Bright ambient light (office/store lighting) to create strong reflections
- **Camera**: Auto exposure; focus on stone surface (not reflections)
- **Resolution**: Minimum 2048×1536px - capture stone pattern detail
- **Focus**: Stone surface at medium distance (30-100cm from camera)

### Quality Checkpoints
- ✅ Clear reflections visible in polished surface
- ✅ Stone natural pattern/color evident (veining, crystals, color variation)
- ✅ Specular highlights present (bright spots from light sources)
- ✅ No actual water visible (dry surface)
- ⚠️ Avoid: Overly dark stone (black granite with minimal reflections), smudges/dirt obscuring surface

### Where to Find It
- **Primary**: Grocery stores (checkout counters), hotel lobbies (floors), office building lobbies
- **Alternative**: Kitchen showrooms, monument/memorial polished stone, upscale retail stores
- **Access**: Public areas; photograph during open hours without blocking traffic

---

## Image #7: reflective_concrete_001.jpg
**Category**: Wet concrete pavement

### Visual Description
Photograph wet concrete (recently rained, hosed down, or early morning dew). Frame to show 1-2m² of wet concrete surface with visible reflections (sky, buildings, lights). Shoot from standing height looking down at 45-60° angle. Capture the transition from wet to dry concrete if possible.

### Confuser Characteristics
- **Water-Like Reflections**: Wet concrete mirrors sky/surroundings like water
- **Blue Reflections**: Reflects blue sky creating water color similarity
- **Puddles**: May have small puddles (but not a continuous water body)
- **Concrete Texture**: Fine aggregate texture visible through water sheen

**Why It Challenges the Detector**: Tests whether detector can distinguish thin water film (not a pool) from actual water bodies. Wet concrete has water's reflectivity but concrete's texture.

### Capture Requirements
- **Lighting**: Shortly after rain or early morning (6-9am) with dew; clear sky for reflections
- **Camera**: Auto mode; expose for pavement (not sky reflection)
- **Resolution**: Minimum 1920×1080px (2MP+)
- **Focus**: Pavement surface 1-3m from camera

### Quality Checkpoints
- ✅ Wet sheen visible (reflections in concrete surface)
- ✅ Concrete texture still visible (not obscured by thick water layer)
- ✅ No continuous puddles covering entire frame (transition wet/dry preferred)
- ✅ Sky/building reflections evident
- ⚠️ Avoid: Deep puddles (actual water bodies), muddy water (obscures concrete), dry concrete

### Where to Find It
- **Primary**: Parking lots, sidewalks, driveways after rain
- **Alternative**: Building courtyards, tennis courts (hard courts), plazas
- **Access**: Public areas; photograph shortly after rainfall (timing critical)

---

## Image #8: reflective_stone_002.jpg
**Category**: Polished stone floor, specular highlights

### Visual Description
Photograph polished stone flooring (marble, terrazzo, polished concrete) in a commercial/public space. Frame to show 2-3m² of floor with strong overhead lighting creating specular highlights. Include floor pattern (tiles, panels, or continuous surface). Shoot from standing height at 60° downward angle.

### Confuser Characteristics
- **Mirror-Like Reflections**: Highly polished floors reflect lights/ceiling like water
- **Bright Highlights**: Specular reflections create bright spots (like sun on water)
- **Tile Grout Lines**: May have grid pattern from tile installation
- **Glossy Finish**: High reflectivity mimics water surface properties

**Why It Challenges the Detector**: Tests combined grid + reflectivity - polished tile floors have both characteristics but are not water.

### Capture Requirements
- **Lighting**: Indoor commercial lighting (fluorescent/LED); bright, even illumination
- **Camera**: Auto exposure; avoid flash (creates single bright spot)
- **Resolution**: Minimum 2048×1536px - capture floor detail and reflections
- **Focus**: Floor surface 1-2m from camera

### Quality Checkpoints
- ✅ Multiple specular highlights visible (from overhead lights)
- ✅ Stone/tile pattern evident (color, veining, or grout lines)
- ✅ High gloss finish apparent (mirror-like quality)
- ✅ No actual water, spills, or wet cleaning marks
- ⚠️ Avoid: Motion blur (from walking), uneven lighting, dirty/scuffed floors

### Where to Find It
- **Primary**: Shopping mall corridors, hotel lobbies, museum galleries, airport terminals
- **Alternative**: Upscale office building lobbies, convention centers, transit stations
- **Access**: Public areas; photograph during off-peak hours (less foot traffic)

---

## Image #9: skylight_reflection_001.jpg
**Category**: Glass skylight, sky reflection

### Visual Description
Photograph a skylight from interior space looking up. Frame to show glass skylight panels with visible sky through glass and reflections on glass surface. Include skylight frame structure (mullions/supports). Shoot from directly below or slight angle, capturing at least 4-6 skylight panels.

### Confuser Characteristics
- **Blue Sky Visible**: Sky visible through glass creates large blue surface area
- **Dual Blue Layers**: Sky transmission + sky reflection in glass creates intense blue
- **Grid Structure**: Skylight framing creates geometric grid pattern
- **Brightness**: Overexposed sky areas create bright blue-white zones (like water highlights)

**Why It Challenges the Detector**: Tests overhead blue surfaces with grid structure - detector must distinguish upward-facing glass from horizontal water surfaces.

### Capture Requirements
- **Lighting**: Clear or partly cloudy day (blue sky visible through skylight)
- **Camera**: Auto exposure with exposure compensation -0.3 to -0.7 EV (prevent sky blowout)
- **Resolution**: Minimum 1920×1080px (2MP+)
- **Focus**: Skylight frame/glass (not distant clouds)

### Quality Checkpoints
- ✅ Blue sky visible through skylight (not overexposed white)
- ✅ Skylight frame structure clear (mullions, supports)
- ✅ Some glass surface reflections visible (interior reflections in glass)
- ✅ Vertical orientation captures full skylight section
- ⚠️ Avoid: Complete sky blowout (pure white), dirty/frosted glass obscuring sky

### Where to Find It
- **Primary**: Shopping mall atriums, office building lobbies, museums, libraries
- **Alternative**: Residential skylights, transit stations, indoor sports facilities
- **Access**: Public interior spaces; photograph during daytime hours

---

## Image #10: skylight_reflection_002.jpg
**Category**: Interior skylight, diffused blue light

### Visual Description
Photograph a frosted/diffused skylight or skylight with translucent panels. Frame to show soft blue light filtering through diffuser. Capture the even blue glow without visible sky details. Include some interior context (ceiling, walls) to show it's an interior space.

### Confuser Characteristics
- **Soft Blue Glow**: Diffused daylight creates uniform blue wash (like underwater view)
- **No Texture**: Frosted glass eliminates texture cues (smooth, featureless blue)
- **Even Illumination**: No reflections or highlights (unlike water with glare)
- **Recessed Frame**: Skylight well creates depth (like looking down into pool)

**Why It Challenges the Detector**: Tests uniform blue surfaces without texture or grid - extreme case of featureless blue plane.

### Capture Requirements
- **Lighting**: Daytime with clear sky (creates blue light diffusion)
- **Camera**: Auto white balance; auto exposure (skylight should be bright but not blown)
- **Resolution**: Minimum 1600×1200px (2MP+)
- **Focus**: Skylight surface or frame edge

### Quality Checkpoints
- ✅ Soft blue glow visible (not gray or overexposed)
- ✅ No visible sky details through diffuser (smooth surface)
- ✅ Interior context visible (ceiling, walls, frame)
- ⚠️ Avoid: Nighttime shots (no blue light), dirty diffuser (uneven discoloration)

### Where to Find It
- **Primary**: Modern residential homes (bathrooms, hallways), commercial buildings with light wells
- **Alternative**: Spas, indoor pools (skylight over dry areas), office break rooms
- **Access**: Residential (personal home or with permission) or public commercial spaces

---

## Image #11: pool_tiles_closeup_001.jpg
**Category**: Pool tiles, grid pattern, NO water

### Visual Description
Photograph empty/dry pool tiles showing the characteristic grid pattern. Frame closeup (0.5-1m²) to emphasize tile grid spacing and grout lines. Shoot perpendicular to tile surface. Capture classic pool tile colors (blue, white, turquoise ceramic). Ensure tiles are completely dry.

### Confuser Characteristics
- **High-Frequency Grid**: Tile grout creates regular grid (10-30cm spacing) - PRIMARY DETECTOR TARGET
- **Pool Blue Color**: Ceramic pool tiles are typically blue/turquoise
- **Rectangular Modules**: Uniform tile shape creates geometric repetition
- **Pool Context**: Even without water, tiles suggest pool presence

**Why It Challenges the Detector**: CRITICAL TEST - Multi-scale grid detection must distinguish these tiles (pool architectural element) from water surface. Tests whether detector requires water's physical presence or falsely triggers on empty pool tiles.

### Capture Requirements
- **Lighting**: Bright daylight (direct sun acceptable) to show tile color and grout lines clearly
- **Camera**: Auto mode; macro mode if available (close focus)
- **Resolution**: Minimum 2048×1536px - must resolve individual grout lines
- **Focus**: Tile surface; ensure sharp focus on grout lines (critical detail)

### Quality Checkpoints
- ✅ Individual tiles clearly delineated (grout lines sharp)
- ✅ Tiles are COMPLETELY DRY (no water, no wet spots)
- ✅ Grid pattern fills 60-80% of frame
- ✅ Pool tile color evident (blue/turquoise/white ceramic)
- ⚠️ Avoid: Any water visible, blurry grout lines, extreme angle (shows tile perspective distortion)

### Where to Find It
- **Primary**: Empty swimming pools (winter/maintenance), pool under construction/renovation
- **Alternative**: Tile showrooms (pool tile samples), pool supply stores (display tiles)
- **Access**: Private pool with permission, public pool during off-season/maintenance, commercial showrooms

**CRITICAL NOTE**: This is a **hard negative** - detector MUST NOT trigger. If detector fails this, multi-scale grid suppressor is broken.

---

## Image #12: pool_tiles_closeup_002.jpg
**Category**: Blue ceramic pool tiles, NO water

### Visual Description
Photograph a different style of pool tiles - larger format tiles (20-30cm), mosaic tiles (5-10cm), or decorative border tiles. Frame to show 0.3-0.5m² area with clear grid pattern. Shoot at 15-30° angle to show slight depth/dimension. Tiles must be completely dry with visible grout.

### Confuser Characteristics
- **Variable Grid Scale**: Different tile size tests multi-scale grid detection across frequency ranges
- **Saturated Blue**: Glazed ceramic creates intense blue saturation
- **3D Surface**: Slight angle reveals tile edges/bevels (shadow lines mimic water ripples)
- **Decorative Patterns**: Border tiles or mosaics add complexity beyond simple grid

**Why It Challenges the Detector**: Tests grid detection across multiple tile scales - ensures detector doesn't just filter one grid frequency but handles range of architectural grid patterns.

### Capture Requirements
- **Lighting**: Bright diffused light (cloudy day or shaded area) to show tile edges without harsh shadows
- **Camera**: Auto mode; close focus (0.5-1m distance)
- **Resolution**: Minimum 2048×1536px - capture grout and tile edge detail
- **Focus**: Tile surface; slight depth of field acceptable (far tiles slightly soft OK)

### Quality Checkpoints
- ✅ Different tile size/pattern than Image #11 (variety in grid scale)
- ✅ Grout lines clearly visible (grid structure evident)
- ✅ Tiles bone dry (no moisture, condensation, or water droplets)
- ✅ Slight 3D quality visible (tile edges, surface texture)
- ⚠️ Avoid: Water presence, excessive shadows obscuring grout, motion blur

### Where to Find It
- **Primary**: Pool tile showrooms, home improvement stores (tile section), pool under renovation
- **Alternative**: Fountain surrounds (dry), decorative tile installations, architectural salvage yards
- **Access**: Commercial showrooms (public), residential pool with permission

**CRITICAL NOTE**: Another **hard negative** - tests grid suppressor with different tile frequencies.

---

## Image #13: concrete_blue_tint_001.jpg
**Category**: Concrete, blue color cast from sky

### Visual Description
Photograph light-colored concrete (sidewalk, wall, pavement) with blue color cast from sky reflection or ambient blue light. Shoot on clear day when concrete reflects blue sky. Frame to show 1-2m² of concrete surface with visible blue tint but clear concrete texture (aggregate, surface finish).

### Confuser Characteristics
- **Blue Color Cast**: Ambient sky light tints concrete blue-gray
- **Matte Reflectance**: Concrete reflects some sky light (not specular like water)
- **Subtle Blue Saturation**: Less intense than water but measurable in blue channel
- **Texture**: Concrete aggregate visible but may be subtle in bright light

**Why It Challenges the Detector**: Tests low-saturation blue detection - concrete with sky reflection has weak blue channel response. Ensures detector doesn't overcorrect and trigger on ambient blue color casts.

### Capture Requirements
- **Lighting**: Clear blue sky day, midday (10am-2pm) for strong sky reflection
- **Camera**: Auto white balance "daylight" preset if available (preserves blue cast)
- **Resolution**: Minimum 1920×1080px (2MP+)
- **Focus**: Concrete surface 1-3m distance

### Quality Checkpoints
- ✅ Subtle blue tint visible in concrete (not neutral gray)
- ✅ Concrete texture discernible (aggregate, surface finish, imperfections)
- ✅ Clear blue sky context (frame edge or shadow indicates sky source)
- ✅ No actual water on concrete (dry surface)
- ⚠️ Avoid: Overcast/gray sky (want blue ambient light), wet concrete (water confounds test)

### Where to Find It
- **Primary**: Modern concrete sidewalks, building walls, parking structures (light gray concrete)
- **Alternative**: Concrete planters, skate parks, architectural concrete features
- **Access**: Public areas; photograph on clear weather days

---

## General Acquisition Workflow

### Phase 1: Planning (Day 1)
1. Review all 13 specifications
2. Identify accessible locations near you for each category
3. Check weather forecast (clear/partly cloudy days optimal for most shots)
4. Confirm camera/smartphone resolution meets minimums

### Phase 2: Capture (Days 2-5)
1. **Priority Captures** (hardest to find):
   - #11, #12: Pool tiles (requires empty pool access)
   - #1, #2: Modern glass facades (requires downtown access)
   
2. **Opportunistic Captures** (as you encounter):
   - #4, #5: Blue walls (residential/commercial)
   - #6, #7, #8: Reflective surfaces (public spaces)
   
3. **Controlled Captures** (easiest):
   - #9, #10: Skylights (shopping malls, public buildings)
   - #13: Concrete (anywhere, weather-dependent)

### Phase 3: Quality Review (Day 6)
1. Transfer images to computer
2. Verify resolution: `identify -format "%wx%h" image.jpg` (needs ≥1024px short edge)
3. Visual inspection checklist:
   - No actual water visible
   - Key confuser characteristics present
   - Sharp focus (not blurry)
   - Adequate resolution
4. Rename files to match manifest filenames exactly

### Phase 4: Integration (Day 7)
1. Place images in `/Users/rc/Transformation_Portal/data/water_v0/holdout_images/`
2. Generate SHA256 hashes: `shasum -a 256 *.jpg`
3. Update `holdout_manifest.json` with real SHA256 values
4. Run validation: `python scripts/validate_holdout.py` (if exists)
5. Document any substitutions/variations in `HOLDOUT.md`

---

## Troubleshooting & Substitutions

### If You Can't Find Empty Pool Tiles (#11, #12)
**Option A**: Photograph pool tile samples at home improvement store (Lowe's, Home Depot tile section)  
**Option B**: Photograph pool waterline tiles above water level (must be dry)  
**Option C**: Use architectural salvage yard samples  
**Minimum Requirement**: Must show clear tile grid, pool-appropriate blue color, completely dry

### If Glass Facades Not Accessible (#1, #2, #3)
**Option A**: Photograph glass bus shelters, phone booths, or large storefront windows  
**Option B**: Use modern residential glass railings (balconies, staircases)  
**Option C**: Photograph greenhouse or conservatory glass walls  
**Minimum Requirement**: Must show reflective glass surface, blue sky reflections, architectural framing

### If Weather Doesn't Cooperate
- **Need blue sky**: Postpone shots #1, #2, #7, #13 until clear day
- **Overcast OK**: Shots #3, #4, #5, #6, #8, #11, #12 work without sun
- **Indoor-only**: Shots #5, #6, #8, #9, #10 don't depend on weather

### Resolution Issues
- Modern smartphones (2018+) exceed minimum requirements in default mode
- Avoid digital zoom (degrades quality)
- Shoot in native camera app (not Instagram, Snapchat, etc.)
- Check image properties before leaving location

---

## Success Criteria

**Holdout set is COMPLETE when**:
✅ All 13 images captured and named correctly  
✅ All images ≥1024px short edge  
✅ Zero images contain actual water  
✅ Each image clearly represents its confuser category  
✅ Images are real photographs (not AI-generated, not renders)  
✅ SHA256 hashes updated in manifest  

**Holdout set PASSES validation when**:
✅ Water detector triggers ≤1 false positive (max 1 of 13 images)  
✅ If 2+ images trigger detection, multi-scale grid suppressor requires adjustment  

This holdout set provides the **Phase C completion gate** for water detection. Acquiring these images with precision ensures robust validation of the glass suppressor feature.

---

## Storage & Manifest

### Directory Structure
```
data/water_v0/
├── holdout_manifest.json        # Tracked in git (metadata only)
├── holdout_images/               # GITIGNORED (large binaries)
│   ├── glass_facade_001.jpg
│   ├── glass_facade_002.jpg
│   └── ...
└── HOLDOUT_ACQUISITION_SPEC.md  # This file (tracked)
```

### Acquisition Checklist
1. ✅ Collect 13 images following detailed specifications above
2. ✅ Place in `data/water_v0/holdout_images/` (create directory if doesn't exist)
3. ✅ Generate SHA256 hashes: `shasum -a 256 holdout_images/*.jpg`
4. ✅ Update `holdout_manifest.json` with actual filenames + hashes
5. ✅ Run validation: `./scripts/validate_holdout.sh` (if available)
6. ✅ Archive results with timestamp

### Acceptance Gates
- **Max False Triggers**: ≤1 (absolute cap, avoids percentage rounding on small datasets)
  - Effective rate: 1/13 = 7.7% (acceptable for real-world architectural confusers)
- **Telemetry Required**: For any false triggers, review `grid_score_coarse`, `grid_persistence_ratio`, `tile_exempted`
- **Reproducibility**: SHA256 manifest ensures deterministic validation

---

## Legal & Licensing

- **Self-captured photos**: Cleanest option (own copyright)
- **Licensed stock**: Verify commercial/public repo usage rights
- **Client photos**: Get explicit written permission
- **Public domain**: Verify provenance (Wikimedia Commons, Unsplash, etc.)

If repo is public, commit only manifest (metadata). Never commit third-party images without unambiguous license.
