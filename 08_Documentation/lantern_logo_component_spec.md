# Lantern Logo — Component Specification (v1.3)

This living specification aligns the Lantern identity across product, marketing, and engineering deliverables. It mirrors the shared Figma component structure so geometry, motion, and token nomenclature remain synchronized as assets move between teams.

## 1. Component Architecture

- **Variants**
  - `detailed` — 256–512 px use, includes full lattice structure with 2:1.2 stroke ratio.
  - `simplified` — 96–256 px use, removes sub-lattice ribs while preserving silhouette.
  - `glyph` — ≤96 px use, single-stroke glyph tuned for favicons and watermark overlays.
- **Canvas** — Square aspect, default 256 × 256 unit viewBox aligned to the shared 4 pt micro-grid.
- **Clearspace** — Minimum padding equals the internal flame height (`32` units at base scale). Clamp responsive padding via `clamp(12px, 2vw, 20px)` when embedding in adaptive UI shells.

## 2. Geometry & Proportions

- Vessel exterior path maintains proportional ratios of `44 : 22 : 39` for shoulder, waist, and base radii to ensure optical balance.
- Flame geometry is vertically centered on the `128` unit axis with smooth Bezier continuity between primary and secondary flame paths to model natural turbulence.
- Horizontal braces sit at `y=56` and `y=212` to maintain the 3:5 vertical rhythm across all variants.
- Stroke joins are rounded to avoid aliasing when rasterized at small scales.

## 3. Token System

Primitive brand tokens are defined in [`lantern_tokens.json`](../09_Client_Deliverables/Lantern_Logo_Implementation_Kit/lantern_tokens.json) and compiled with Style Dictionary into platform layers.

| Token | Value | Notes |
| --- | --- | --- |
| `color.brand.navy` | `#0B1220` | Base background and glyph stroke for light contexts. |
| `color.brand.cyan` | `#2AC7FF` | Primary stroke and accent hue. |
| `color.brand.azure` | `#56CCF2` | Gradient highlight stop. |
| `color.brand.ice` | `#E5F6FF` | Glyph background and elevated surfaces. |
| `color.brand.white` | `#FFFFFF` | Neutral content surface. |
| `gradient.brand.primary` | Linear, 160°, stops reference `color.brand.azure` → `color.brand.cyan` | Applied on interaction hover/focus states and motion trails; update primitive stops first to cascade changes. |

Semantic variables (e.g., `--lantern-stroke`, `--lantern-hover-glow`) must be derived from these primitives to maintain cross-platform parity.

## 4. Interaction & Motion

- Default hover/focus treatment promotes the gradient stroke (`url(#lantern-gradient)`) and applies a `drop-shadow` derived from `--lantern-hover-glow` only when `(hover: hover) and (pointer: fine)` evaluate to true.
- Reduced-motion contexts (`prefers-reduced-motion: reduce`) must disable timeline loops and freeze the gradient stroke at the initial stop.
- Recommended loop cadence for animated marks: `6 s` total cycle with `1.5 s` dwell at keyframes to preserve calm brand tone.

## 5. Accessibility

- SVG assets include `<title>` elements and `aria-labelledby` relationships for assistive technologies.
- Contrast ratios for glyph variant on `brand-ice` background exceed `WCAG 2.1 AA` for graphic objects.
- Motion fallbacks are required for users who opt out of animation, and gradient transitions should respect system-level reduced-motion preferences.

## 6. Governance & Delivery

- Source tokens and reference assets live in [`09_Client_Deliverables/Lantern_Logo_Implementation_Kit`](../09_Client_Deliverables/Lantern_Logo_Implementation_Kit/).
- Updates follow semantic versioning; bump the `version` key in the token file and note changes in `08_Documentation/Version_History/changelog.md`.
- Automated linting ensures SVG IDs (`lantern-gradient`) remain stable and token names map 1:1 with Figma variables.
- Distribution happens through the `#brand-ops` channel with checksum hashes for each asset to detect drift during vendor handoff.

## 7. Testing Checklist

- ✅ Visual regression across detailed/simplified/glyph variants at 1x and 2x pixel density.
- ✅ Style Dictionary build outputs validated for CSS, iOS, and Android consumers.
- ✅ Accessibility scan confirming ARIA and reduced-motion behavior.
- ✅ Performance budget: hover gradient activation within 150 ms without dropped frames on reference hardware.

Refer to the implementation kit README for integration tips and the latest automation notes.
