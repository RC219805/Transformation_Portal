# 750 Picacho Lane - Pool Aerial Enhancement Report

## Processing Parameters

- **Input**: `RC_002RC-office750Picacho_Pool 2.tiff`
- **Output**: `750_Picacho_Pool_MBAR_Enhanced.jpg`
- **Analysis Resolution**: 1280px max dimension
- **Clusters**: 8
- **Target Width**: 4096px (4K)
- **Random Seed**: 22

## Material Assignments

| Material | Coverage | Mean RGB | Mean HSV |
|----------|----------|----------|----------|
| **Equitone** | 17.8% | (0.27, 0.25, 0.21) | (0.26, 0.27, 0.28) |
| **Plaster** | 8.9% | (0.79, 0.79, 0.82) | (0.64, 0.05, 0.83) |
| **Stone** | 8.3% | (0.56, 0.51, 0.43) | (0.16, 0.24, 0.57) |
| **Screens** | 14.1% | (0.43, 0.38, 0.32) | (0.18, 0.25, 0.43) |
| **Roof** | 22.3% | (0.67, 0.68, 0.75) | (0.63, 0.11, 0.75) |
| *Unassigned* | 28.6% | - | - |

## MBAR Material Specifications

### Applied Materials

- **Plaster**: Marmorino Palladino Plaster - Westwood Beige
  - Blend: 60%
- **Stone**: Eco Outdoor Bokara Stone - Coastal
  - Blend: 65%
- **Screens**: Grey Gum Screens - Natural
  - Blend: 55%
- **Equitone**: Equitone LT85 Panels - Anthracite
  - Blend: 55%
- **Roof**: Bison Weathered Ipe Pavers
  - Blend: 60%

## Pool Area Specific Notes

This enhancement focuses on the pool and surrounding hardscape:

- **Pool Deck**: Likely identified as stone or roof material (Bokara/Ipe pavers)
- **Pool Water**: May be assigned to equitone or screens (blue-grey tones)
- **Landscaping**: Vegetation typically unassigned or low-confidence clusters
- **Structures**: Plaster walls, bronze details, shade elements

## Recommendations

- For more pool-specific material detection, consider increasing `k` to 10-12 clusters
- Water reflections may benefit from custom water material rule
- Deck materials could use higher blend strength (0.7-0.8) for stronger effect
- Consider masking vegetation areas before processing for cleaner results
