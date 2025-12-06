"""
SwinIR Architecture - Placeholder for Official Implementation
==============================================================

To use SwinIR models, download the official architecture from:
https://github.com/JingyunLiang/SwinIR

Installation:
1. Clone SwinIR repository:
   git clone https://github.com/JingyunLiang/SwinIR.git /tmp/swinir
   
2. Copy architecture file:
   cp /tmp/swinir/models/network_swinir.py utils/swinir_arch.py
   
3. Download pre-trained weights:
   wget https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth \
     -O weights/upscaling/swinir_real_4x.pth

License: Apache-2.0
Source: https://github.com/JingyunLiang/SwinIR
Paper: https://arxiv.org/abs/2108.10257

Note: SwinIR provides superior texture preservation compared to Real-ESRGAN,
especially for photographic images. The official implementation is required
for production use.
"""

import sys

# Placeholder error message
class SwinIR:
    """Placeholder class - download official implementation."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "SwinIR architecture not available.\n\n"
            "To use SwinIR models:\n"
            "1. Download from: https://github.com/JingyunLiang/SwinIR\n"
            "2. Copy models/network_swinir.py to utils/swinir_arch.py\n"
            "3. Download weights: https://github.com/JingyunLiang/SwinIR/releases\n\n"
            "Alternative: Use Real-ESRGAN models (already available in basicsr_tp/)\n"
            "  from utils.upscaling_engine import UpscalingModel\n"
            "  model = UpscalingModel.REALESRGAN_4X\n"
        )


def download_swinir():
    """Helper to download SwinIR architecture."""
    print("""
    SwinIR Setup Instructions
    =========================
    
    1. Clone official repository:
       git clone https://github.com/JingyunLiang/SwinIR.git /tmp/swinir
    
    2. Copy architecture (this file):
       cp /tmp/swinir/models/network_swinir.py utils/swinir_arch.py
    
    3. Create weights directory:
       mkdir -p weights/upscaling
    
    4. Download pre-trained models:
       
       # Real-world SR (recommended for photos)
       wget https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth \\
         -O weights/upscaling/swinir_real_4x.pth
       
       # Classical SR (for clean images)
       wget https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth \\
         -O weights/upscaling/swinir_classical_4x.pth
    
    5. Verify installation:
       python -c "from utils.swinir_arch import SwinIR; print('SwinIR ready!')"
    
    For automatic installation, run:
       make setup-swinir
    """)


if __name__ == "__main__":
    download_swinir()
