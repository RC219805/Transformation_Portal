#!/usr/bin/env python3
"""Verification script for APEX Feature Gaps analysis.

Confirms the current state of each gap before implementation.
Run this script to validate the architectural analysis is accurate.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple


def check_gap1_16bit() -> Tuple[bool, List[str]]:
    """Check Gap 1: 16-bit output path."""
    findings = []

    # Check config flags exist
    config_path = Path("src/transformation_portal/lux_depth_v3/config.py")
    if config_path.exists():
        config_content = config_path.read_text()
        if "emit_master16: bool = False" in config_content:
            findings.append("✅ Config: emit_master16 flag exists")
        else:
            findings.append("❌ Config: emit_master16 flag missing")

        if "emit_upscaled16: bool = False" in config_content:
            findings.append("✅ Config: emit_upscaled16 flag exists")
        else:
            findings.append("❌ Config: emit_upscaled16 flag missing")

    # Check CLI parsing
    main_path = Path("src/transformation_portal/lux_depth_v3/__main__.py")
    if main_path.exists():
        main_content = main_path.read_text()
        if "--emit-master16" in main_content:
            findings.append("✅ CLI: --emit-master16 flag exposed")
        else:
            findings.append("❌ CLI: --emit-master16 flag missing")

        if "--emit-upscaled16" in main_content:
            findings.append("✅ CLI: --emit-upscaled16 flag exposed")
        else:
            findings.append("❌ CLI: --emit-upscaled16 flag missing")

    # Check orchestrator Materials V3 handoff (8-bit bug)
    orchestrator_path = Path("src/transformation_portal/lux_depth_v3/orchestrator.py")
    if orchestrator_path.exists():
        orchestrator_content = orchestrator_path.read_text()
        if "enhanced_uint8 = (np.clip(working_image, 0, 1) * 255).astype(np.uint8)" in orchestrator_content:
            findings.append("❌ CONFIRMED BUG: Materials V3 outputs 8-bit PNG handoff (line ~846)")
        else:
            findings.append("⚠️  Materials V3 handoff code may have changed")

        if "enhanced_uint16" in orchestrator_content and "65535" in orchestrator_content:
            findings.append("⚠️  16-bit handoff may already be implemented")

    # Check V2 has 16-bit support (existing capability)
    v2_enhance_path = Path("src/transformation_portal/lux_depth_v3/v2_enhance.py")
    if v2_enhance_path.exists():
        v2_content = v2_enhance_path.read_text()
        if "load_image_preserve_bit_depth" in v2_content:
            findings.append("✅ V2: Has 16-bit TIFF support (existing code)")
        else:
            findings.append("❌ V2: 16-bit support missing")

    passed = all("✅" in f for f in findings[:4])  # Core checks
    return passed, findings


def check_gap2_mps() -> Tuple[bool, List[str]]:
    """Check Gap 2: V2 MPS acceleration."""
    findings = []

    # Check config field exists
    config_path = Path("src/transformation_portal/lux_depth_v3/config.py")
    if config_path.exists():
        config_content = config_path.read_text()
        if 'v2_device: str = "cpu"' in config_content:
            findings.append("✅ Config: v2_device field exists")
        else:
            findings.append("❌ Config: v2_device field missing")

    # Check orchestrator passes it to V2Runner
    orchestrator_path = Path("src/transformation_portal/lux_depth_v3/orchestrator.py")
    if orchestrator_path.exists():
        orchestrator_content = orchestrator_path.read_text()
        if "device=self.config.v2_device" in orchestrator_content:
            findings.append("✅ Orchestrator: Passes v2_device to V2Runner")
        else:
            findings.append("❌ Orchestrator: v2_device not passed")

    # Check V2Runner accepts device parameter
    v2_runner_path = Path("src/transformation_portal/lux_depth_v3/v2_runner.py")
    if v2_runner_path.exists():
        v2_runner_content = v2_runner_path.read_text()
        if "device: str" in v2_runner_content:
            findings.append("✅ V2Runner: Accepts device parameter")
        else:
            findings.append("❌ V2Runner: device parameter missing")

    # Check CLI does NOT expose --v2-device (the gap)
    main_path = Path("src/transformation_portal/lux_depth_v3/__main__.py")
    if main_path.exists():
        main_content = main_path.read_text()
        if "--v2-device" in main_content:
            findings.append("⚠️  CLI: --v2-device already exists (gap may be fixed)")
        else:
            findings.append("❌ CONFIRMED GAP: CLI does not expose --v2-device")

    passed = "CONFIRMED GAP" in findings[-1]
    return passed, findings


def check_gap3_upscaling() -> Tuple[bool, List[str]]:
    """Check Gap 3: ML super-resolution upscaling."""
    findings = []

    # Check upscaling stage exists
    upscaling_stage_path = Path("src/transformation_portal/stage_graph/stages/upscaling.py")
    if upscaling_stage_path.exists():
        findings.append("✅ UpscalingStage exists")

        stage_content = upscaling_stage_path.read_text()
        if 'backend: str = "torch"' in stage_content or "self.backend = backend" in stage_content:
            findings.append("✅ UpscalingStage: Has backend parameter")

        if 'self._upscaler = "bicubic"' in stage_content:
            findings.append("❌ CONFIRMED BUG: Hardcoded to bicubic (line ~141)")

        if "UpscalerRegistry" in stage_content:
            findings.append("⚠️  UpscalerRegistry may already be integrated")
    else:
        findings.append("❌ UpscalingStage missing")

    # Check for Real-ESRGAN references (should not exist yet)
    if Path("src/transformation_portal/upscaling").exists():
        findings.append("⚠️  Upscaling module already exists")
    else:
        findings.append("✅ Upscaling module does not exist (expected)")

    # Check requirements for Real-ESRGAN
    ml_req_path = Path("requirements/ml.txt")
    if ml_req_path.exists():
        ml_content = ml_req_path.read_text()
        if "realesrgan" in ml_content.lower():
            findings.append("⚠️  Real-ESRGAN already in requirements")
        else:
            findings.append("✅ Real-ESRGAN not in requirements (expected)")

    passed = "CONFIRMED BUG" in str(findings)
    return passed, findings


def check_golden_path_preservation() -> Tuple[bool, List[str]]:
    """Verify Golden Path configs are safe defaults."""
    findings = []

    config_path = Path("src/transformation_portal/lux_depth_v3/config.py")
    if config_path.exists():
        config_content = config_path.read_text()

        # Check defaults
        if "emit_master16: bool = False" in config_content:
            findings.append("✅ Golden Path: emit_master16 defaults to False")

        if "emit_upscaled16: bool = False" in config_content:
            findings.append("✅ Golden Path: emit_upscaled16 defaults to False")

        if 'v2_device: str = "cpu"' in config_content:
            findings.append("✅ Golden Path: v2_device defaults to cpu")

        if "v2_upscaler_backend: Optional[str] = None" in config_content or "v2_upscaler" not in config_content:
            findings.append("✅ Golden Path: v2_upscaler defaults to None/bicubic")

    passed = len(findings) == 4 and all("✅" in f for f in findings)
    return passed, findings


def main():
    """Run all verification checks."""
    print("=" * 70)
    print("APEX Feature Gaps Verification")
    print("=" * 70)
    print()

    all_passed = True

    # Gap 1: 16-bit Output
    print("Gap 1: 16-Bit Output Path")
    print("-" * 70)
    passed, findings = check_gap1_16bit()
    for finding in findings:
        print(f"  {finding}")
    print()
    all_passed = all_passed and "CONFIRMED BUG" in str(findings)

    # Gap 2: V2 MPS
    print("Gap 2: V2 MPS Acceleration")
    print("-" * 70)
    passed, findings = check_gap2_mps()
    for finding in findings:
        print(f"  {finding}")
    print()
    all_passed = all_passed and passed

    # Gap 3: ML Upscaling
    print("Gap 3: ML Super-Resolution Upscaling")
    print("-" * 70)
    passed, findings = check_gap3_upscaling()
    for finding in findings:
        print(f"  {finding}")
    print()
    all_passed = all_passed and passed

    # Golden Path Preservation
    print("Golden Path Preservation Check")
    print("-" * 70)
    passed, findings = check_golden_path_preservation()
    for finding in findings:
        print(f"  {finding}")
    print()
    all_passed = all_passed and passed

    # Summary
    print("=" * 70)
    if all_passed:
        print("✅ VERIFICATION PASSED: All three gaps confirmed")
        print("   Implementation plan is accurate and ready to proceed.")
    else:
        print("⚠️  VERIFICATION WARNING: Some gaps may already be fixed")
        print("   Review findings above before implementation.")
    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
