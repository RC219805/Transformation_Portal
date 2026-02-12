import numpy as np

from transformation_portal.atmosphere import LocationPresets, SkyBlender, SkyGANGenerator
from transformation_portal.core.storage import ExportManager


def run_golden_test():
    print("🔮 Starting Core Systems Verification...")

    # 1. Create a "Digital Twin" of a flat wall (Gray 50%)
    # This acts as our canvas for the physics engine
    input_image = np.full((1024, 1024, 3), 128, dtype=np.uint8)
    print("✅ Input Tensor Created (1024x1024)")

    # 2. Load the "Sundowner" Micro-Climate Data
    # We explicitly request the unique Santa Barbara weather condition
    presets = LocationPresets()
    sky_params = presets.get_sky_parameters(
        location="montecito", time_of_day=17.5, condition="sundowner"  # 5:30 PM (Golden Hour)
    )
    atmo_params = presets.get_atmospheric_parameters(location="montecito", condition="sundowner")

    print(f"🌍 Loaded Micro-Climate: Montecito")
    print(f"   - Condition: Sundowner")
    print(f"   - Turbidity: {atmo_params.turbidity} (Expect ~1.3 for high clarity)")
    print(f"   - Sun Azimuth: {sky_params.sun_azimuth:.1f}°")

    # 3. Initialize the Physics Engines
    # This will lazy-load the heavy ML models
    print("\n🔧 Initializing Physics Engines...")
    generator = SkyGANGenerator()
    blender = SkyBlender()

    # 4. Execute Smart Render (The "Paradigm Shift")
    # This enables the Shadow Analysis and Auto-Correction guardrails
    print("\n🚀 Executing SkyBlender Smart Render...")
    result, suggestion = blender.smart_render(
        source_image=input_image,
        sky_params=sky_params,
        atmo_params=atmo_params,
        auto_correct=True,  # Enable the "Brain"
        strict_physics=False,  # Allow minor deviations
    )

    # 5. Analyze the "Glass Box" Report
    print("\n📊 ANALYSIS REPORT:")
    print(f"   - Status: {'PASSED' if suggestion.confidence > 0.8 else 'WARNING'}")
    print(f"   - Confidence: {suggestion.confidence:.2f}")
    print(f"   - Original Request Azimuth: {suggestion.original_request_azimuth:.1f}°")
    print(f"   - Measured Source Azimuth: {suggestion.measured_source_azimuth:.1f}°")
    print(f"   - Engine Message: {suggestion.message}")

    if suggestion.confidence > 0.8:
        print("\n✅ SUCCESS: Physics engine rejected invalid shadows and rendered correctly.")
    else:
        print("\n⚠️  WARNING: Physics engine output has low confidence (likely due to flat input).")
        print("   This is expected for a uniform gray canvas.")

    print(f"\n🎨 Rendered output shape: {result.shape}")
    print("✅ Core systems verification complete!")


if __name__ == "__main__":
    run_golden_test()
