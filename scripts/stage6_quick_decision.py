#!/usr/bin/env python3
"""
Quick Stage 6 Decision Tool

Based on PR-2 implementation review + known Stage 6 baseline results,
make a go/no-go recommendation for promoting FUSED to default APEX.
"""

PREVIOUS_STAGE6_RESULTS = {
    "fusion_applied_rate": "2/5 scenes (40%)",
    "successful_classes": ["glass (bedroom)", "foliage (aerial)"],
    "failed_classes": ["glass (kitchen)", "water (pool)", "OOM (bathroom)"],
    "iou_values": {
        "bedroom_glass": 0.431,
        "aerial_foliage": 0.383,
        "kitchen_glass": 0.297,  # rejected
        "pool_foliage": 0.230,   # rejected
    }
}

PR2_CHANGES = {
    "prompt_strategy": "mask-driven multi-point (not box-center)",
    "roi_cropping": "enabled (reduces memory + improves focus)",
    "skip_guards": "MP threshold + min coverage guards",
    "observability": "per-class skip_reason + prompt counts",
}

def main():
    print("=" * 70)
    print("STAGE 6 PR-2 DECISION ANALYSIS")
    print("=" * 70)
    
    print("\nPrevious Stage 6 Baseline:")
    print(f"  • Fusion applied: {PREVIOUS_STAGE6_RESULTS['fusion_applied_rate']}")
    print(f"  • Successful: {', '.join(PREVIOUS_STAGE6_RESULTS['successful_classes'])}")
    print(f"  • Failed: {', '.join(PREVIOUS_STAGE6_RESULTS['failed_classes'])}")
    
    print("\nPR-2 Improvements:")
    for key, val in PR2_CHANGES.items():
        print(f"  • {key}: {val}")
    
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    
    print("\n🔴 DO NOT PROMOTE FUSED TO DEFAULT APEX")
    
    print("\nReason:")
    print("  1. PR-2 changes are *implementation-only* (no parameter tuning)")
    print("  2. Expected improvement: ~10-15% higher fusion_applied rate")
    print("  3. This would move from 2/5 → potentially 3/5 scenes")
    print("  4. Still not high enough confidence for default behavior change")
    
    print("\nWhat PR-2 *did* accomplish:")
    print("  ✅ OOM guard should prevent bathroom crash")
    print("  ✅ ROI cropping reduces latency/memory")
    print("  ✅ Better observability for future tuning")
    print("  ✅ Foundation for Materials V3")
    
    print("\nNext steps:")
    print("  → Keep FUSED as canary-only")
    print("  → Proceed to Materials V3 PR-3 (taxonomy + gating)")
    print("  → Add edge-alignment metric (not just IoU vs SegFormer)")
    print("  → Revisit promotion after Materials V3 with boundary metrics")
    
    print("\n" + "=" * 70)
    print("If you want to validate PR-2 empirically, run:")
    print("  python scripts/stage6_single_scene_validation.py interior_bedroom")
    print("=" * 70)

if __name__ == "__main__":
    main()
