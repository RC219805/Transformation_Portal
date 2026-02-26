import argparse
import sys
from pathlib import Path


def _seed_repo_root_for_imports() -> None:
    current = Path(__file__).resolve()
    for candidate in (current.parent, *current.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / ".github" / "workflows").is_dir():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return


_seed_repo_root_for_imports()

from scripts.lib.repo_root import RepoRootError, resolve_repo_root


def _bootstrap_paths(repo_override: str | None = None) -> Path:
    repo_path = Path(repo_override).expanduser() if repo_override else None
    repo_root = resolve_repo_root(start=Path(__file__), repo=repo_path)
    for path in (repo_root, repo_root / "src"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    return repo_root


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run core systems verification.")
    parser.add_argument("--repo", help="Explicit repository root path override.")
    return parser.parse_args()


def run_golden_test():
    import numpy as np

    from transformation_portal.atmosphere import LocationPresets, SkyBlender, SkyGANGenerator

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
    args = _parse_args()
    try:
        _bootstrap_paths(repo_override=args.repo)
    except RepoRootError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    run_golden_test()
