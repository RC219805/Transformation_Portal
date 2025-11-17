"""FLUX diffusion model integration for architectural enhancement.

FLUX.1 represents the state-of-the-art for architectural enhancement:

- **12-billion parameters** with flow matching architecture
- **1-4 step generation** vs 20-50 steps for traditional diffusion (8x speedup)
- **Photorealistic output** indistinguishable from professional photography
- **ControlNet conditioning** for structural preservation
- **Multi-modal prompting** for precise architectural control

Key advantages over Stable Diffusion XL:
1. Faster generation (1-4 steps vs 20-50)
2. Better architectural detail preservation
3. Superior prompt adherence
4. Enhanced realism without artifacts

For luxury real estate:
- Near-real-time client iteration (2-3 min vs 10-15 min)
- Photorealistic enhancement maintaining structure
- Architectural-grade precision with ControlNet
- Professional photography quality output
"""

from transformation_portal.diffusion.flux_pipeline import FLUXPipeline
from transformation_portal.diffusion.flux_controlnet import FLUXControlNet
from transformation_portal.diffusion.architectural_prompts import ArchitecturalPromptBuilder

__all__ = [
    'FLUXPipeline',
    'FLUXControlNet',
    'ArchitecturalPromptBuilder',
]
