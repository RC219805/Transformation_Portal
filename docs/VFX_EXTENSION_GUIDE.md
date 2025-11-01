404: # VFX Extension Guide

This guide explains how to use, configure, and extend the VFX (Visual Effects) extension system in the Transformation Portal. The VFX extension system enables advanced, modular post-processing effects for both image and video pipelines, supporting custom enhancements, compositing, and integration with third-party VFX tools.

---

## Table of Contents

- [Overview](#overview)
- [Installation & Setup](#installation--setup)
- [Enabling VFX Extensions](#enabling-vfx-extensions)
- [Usage Examples](#usage-examples)
- [API Reference](#api-reference)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Related Documentation](#related-documentation)

---

## Overview

The VFX extension system allows users to apply advanced, customizable visual effects to images and videos as part of the Transformation Portal's processing pipelines. VFX extensions can be used for:

- Atmospheric effects (fog, haze, bloom)
- Compositing overlays (logos, watermarks, light leaks)
- Advanced color grading and stylization
- Depth-aware effects (volumetric lighting, depth-of-field)
- Integration with third-party VFX tools (e.g., After Effects, Nuke)

VFX extensions are modular and can be enabled or disabled per-pipeline or per-preset. They are designed to be composable and order-dependent, allowing for complex, production-grade post-processing workflows.

---

## Installation & Setup

VFX extension support is included in the core Transformation Portal package. To use or develop custom VFX extensions:

1. Ensure you have installed the core dependencies:

    ```bash
    pip install -r requirements.txt
    ```

2. (Optional) For advanced effects, install extras:

    ```bash
    pip install -e ".[vfx]"
    ```

3. Place custom VFX extension modules in the `src/transformation_portal/vfx_extensions/` directory or reference them via the configuration.

---

## Enabling VFX Extensions

VFX extensions are enabled via pipeline configuration (YAML or Python dict) or CLI options. Each extension can be configured with parameters such as strength, order, and effect-specific settings.

**Example YAML configuration:**

```yaml
vfx_extensions:
  - name: "atmospheric_fog"
    strength: 0.4
    color: "#e0e6ef"
    depth_aware: true
  - name: "brand_overlay"
    logo_path: "09_Client_Deliverables/Lantern_Logo_Implementation_Kit/lantern_logo.png"
    position: "bottom-right"
    opacity: 0.85