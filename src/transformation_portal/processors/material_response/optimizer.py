"""Blueprint generation for enhancing Material Response renderings."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple


@dataclass(frozen=True)
class MetricSnapshot:
    """Bundle of quantitative scores for a single rendering version."""

    luminance: float
    awe: float
    comfort: float
    texture_dimension: float
    future_alignment: float
    luxury_index: float

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "MetricSnapshot":
        """Create a snapshot from a JSON-compatible mapping."""

        return cls(
            luminance=float(data["luminance"]),
            awe=float(data["awe"]),
            comfort=float(data["comfort"]),
            texture_dimension=float(data["texture_dimension"]),
            future_alignment=float(data["future_alignment"]),
            luxury_index=float(data["luxury_index"]),
        )


@dataclass(frozen=True)
class SceneReport:
    """Collection of scores for each deliverable version."""

    name: str
    versions: Mapping[str, MetricSnapshot]

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "SceneReport":
        versions = {
            version_name: MetricSnapshot.from_mapping(version_metrics)
            for version_name, version_metrics in data.get("versions", {}).items()
        }
        return cls(name=str(data["name"]), versions=versions)

    def metric(self, version: str, metric_name: str) -> float:
        """Return ``metric_name`` for ``version`` raising ``KeyError`` on failure."""

        snapshot = self.versions[version]
        return getattr(snapshot, metric_name)


@dataclass(frozen=True)
class MaterialResponseReport:
    """Structured representation of ``material_response_report.json``."""

    generated: str
    analysis_version: str
    scenes: Mapping[str, SceneReport]

    @classmethod
    def load(cls, path: str | Path) -> "MaterialResponseReport":
        with Path(path).open("r", encoding="utf-8") as fp:
            raw: dict[str, Any] = json.load(fp)

        scenes = {
            scene_data["name"]: SceneReport.from_mapping(scene_data)
            for scene_data in raw.get("scenes", [])
        }

        return cls(
            generated=str(raw.get("generated", "")),
            analysis_version=str(raw.get("analysis_version", "")),
            scenes=scenes,
        )

    def iter_scenes(self) -> Iterable[SceneReport]:
        """Yield scenes in the report."""

        return self.scenes.values()

    def top_scene(self, metric_name: str, *, version: str = "regular") -> SceneReport:
        """Return the scene with the highest ``metric_name`` for ``version``."""

        return max(self.iter_scenes(), key=lambda scene: scene.metric(version, metric_name))


@dataclass(frozen=True)
class MaterialDefinition:
    """Physical and rendering properties for a specific material."""

    name: str
    ior: float
    roughness_range: Tuple[float, float]
    displacement_mm: float
    mapping_type: str
    texture_layers: List[str]

    @classmethod
    def herringbone_wood(cls) -> "MaterialDefinition":
        return cls(
            name="herringbone_oak",
            ior=1.53,
            roughness_range=(0.35, 0.65),
            displacement_mm=1.5,
            mapping_type="uv",
            texture_layers=["diffuse", "normal", "displacement", "specularity_inverse"],
        )

    @classmethod
    def stone_pavers(cls) -> "MaterialDefinition":
        return cls(
            name="algorithmic_stone",
            ior=1.57,
            roughness_range=(0.4, 0.7),
            displacement_mm=3.0,
            mapping_type="procedural",
            texture_layers=["diffuse_variants", "roughness_noise", "joint_geometry"],
        )

    @classmethod
    def textured_plaster(cls) -> "MaterialDefinition":
        return cls(
            name="granular_plaster",
            ior=1.52,
            roughness_range=(0.6, 0.8),
            displacement_mm=8.0,
            mapping_type="triplanar",
            texture_layers=["base_normal", "macro_undulation", "roughness_variation"],
        )


class RenderEnhancementPlanner:
    """Translate report metrics into an actionable upgrade blueprint."""

    def __init__(self, report: MaterialResponseReport):
        self.report = report

    @classmethod
    def from_json(cls, path: str | Path) -> "RenderEnhancementPlanner":
        return cls(MaterialResponseReport.load(path))

    def build_blueprint(self) -> dict[str, Any]:
        """Return a nested dictionary describing rendering upgrades."""

        luminance_strategy = self._derive_luminance_strategy()
        awe_alignment = self._derive_awe_alignment()
        comfort_plan = self._derive_comfort_plan()
        texture_dimensions = self._derive_texture_strategy()
        future_alignment = self._derive_future_alignment_strategy()
        lux_plan = self._derive_lux_strategy()
        scene_specific = self._derive_scene_specific_upgrades()

        return {
            "generated": self.report.generated,
            "analysis_version": self.report.analysis_version,
            "luminance_strategy": luminance_strategy,
            "awe_alignment": awe_alignment,
            "comfort_realignment": comfort_plan,
            "texture_dimension_strategy": texture_dimensions,
            "future_alignment": future_alignment,
            "lux_version_strategy": lux_plan,
            "scene_specific_enhancements": scene_specific,
            "narrative": (
                "transcends conventional luxury through orchestrated tension "
                "between photonic drama, tactile richness, and future-forward quiet tech."
            ),
            "technical_workflow": {
                "layer_stack": [
                    "base_render",
                    "targeted_luminance_curves",
                    "hero_surface_texture_pass",
                    "atmospheric_depth_layers",
                    "floating_future_elements",
                    "golden_hour_color_grade",
                ]
            },
            "variation_sets": [
                {
                    "name": "morning_precision",
                    "focus": "heightened focus, tempered comfort",
                    "metrics_bias": {"comfort": -0.05, "awe": +0.03},
                },
                {
                    "name": "twilight_awe",
                    "focus": "maximum awe and luxury allure",
                    "metrics_bias": {"awe": +0.1, "luxury_index": +0.08},
                },
                {
                    "name": "night_future_alignment",
                    "focus": "visionary tech-forward ambience",
                    "metrics_bias": {"future_alignment": +0.12},
                },
            ],
            "ab_testing_framework": {
                "variants": [
                    {"name": "baseline", "delta": 0.0},
                    {"name": "elevated", "delta": +0.15},
                    {"name": "targeted_room_bias", "delta": "per-metric"},
                ],
                "goal": "orchestrate an emotional crescendo instead of uniform gains",
            },
        }

    def _derive_luminance_strategy(self) -> dict[str, Any]:
        hierarchy_targets = []
        for scene in self.report.iter_scenes():
            base_luminance = scene.metric("regular", "luminance")
            if scene.name in {"pool", "aerial"} and base_luminance < 0.3:
                target = round(min(0.32, base_luminance * 1.18), 4)
                hierarchy_targets.append(
                    {
                        "scene": scene.name,
                        "current": base_luminance,
                        "target": target,
                        "focus_areas": [
                            "specular_pool_reflections" if scene.name == "pool" else "roofline_glow",
                            "interior_window_bloom" if scene.name == "aerial" else "architectural_whites",
                        ],
                        "approach": "sculpted masks and dodge layers to avoid uniform brightening",
                    }
                )

        return {
            "reference_luminance": 0.31,
            "notes": "0.30-0.32 scores correlate with top luxury perception. Maintain hierarchy.",
            "targets": hierarchy_targets,
        }

    def _derive_awe_alignment(self) -> Dict[str, Any]:
        kitchen_awe = self.report.scenes["kitchen"].metric("regular", "awe")
        actions = []

        great_room = self.report.scenes.get("great_room")
        if great_room is not None:
            actions.append(
                {
                    "scene": "great_room",
                    "current": great_room.metric("regular", "awe"),
                    "target": 0.85,
                    "moves": [
                        "introduce volumetric sunset shaft through skylight",
                        "amplify contrast on stone wall relief",
                    ],
                }
            )

        pool = self.report.scenes.get("pool")
        if pool is not None:
            actions.append(
                {
                    "scene": "pool",
                    "current": pool.metric("regular", "awe"),
                    "target": 0.75,
                    "moves": [
                        "activate underwater lighting with geometric caustics",
                        "layer fire feature reflections across water surface",
                    ],
                }
            )

        return {
            "benchmark_scene": "kitchen",
            "benchmark_awe": kitchen_awe,
            "actions": actions,
        }

    def _derive_comfort_plan(self) -> dict[str, Any]:
        bedroom = self.report.scenes.get("primary_bedroom")
        if bedroom is None:
            return {}

        comfort = bedroom.metric("regular", "comfort")
        return {
            "scene": "primary_bedroom",
            "current": comfort,
            "target": 0.85,
            "moves": [
                "fold in subtle corner shadows to reintroduce tension",
                "cool ambient color temperature by ~250K",
                "animate sheer curtains for micro-motion",
            ],
        }

    def _derive_texture_strategy(self) -> Dict[str, Any]:
        hero_surfaces = {
            "kitchen": "island_waterfall_edge",
            "great_room": "stone_feature_wall",
            "primary_bedroom": "headboard_textile_panel",
        }

        hero_targets = []
        for scene_name, surface in hero_surfaces.items():
            scene = self.report.scenes.get(scene_name)
            if scene is None:
                continue
            current = scene.metric("regular", "texture_dimension")
            hero_targets.append(
                {
                    "scene": scene_name,
                    "surface": surface,
                    "current": current,
                    "target": 2.25,
                    "method": "microcontrast maps + procedural detail passes",
                }
            )

        return {
            "baseline": 1.9,
            "hero_targets": hero_targets,
            "guardrails": "Maintain supporting surfaces at 1.9 to avoid noise accumulation.",
        }

    def _derive_future_alignment_strategy(self) -> Dict[str, Any]:
        adjustments = []
        for scene in self.report.iter_scenes():
            current = scene.metric("regular", "future_alignment")
            if current < 0.7:
                adjustments.append(
                    {
                        "scene": scene.name,
                        "current": current,
                        "target": 0.72,
                        "interventions": [
                            "float linear LED reveals detached from architecture",
                            "introduce high-polish reflections for spatial ambiguity",
                            "embed discreet sensor-like pin lights",
                        ],
                    }
                )

        return {
            "summary": "Current readings imply contemporary comfort. Layer visionary cues to exceed 0.70.",
            "adjustments": adjustments,
        }

    def _derive_lux_strategy(self) -> Dict[str, Any]:
        entries = []
        for scene in self.report.iter_scenes():
            regular = scene.metric("regular", "luxury_index")
            lux = scene.metric("lux", "luxury_index")
            delta = lux - regular
            if delta < 0.05:
                entries.append(
                    {
                        "scene": scene.name,
                        "delta": round(delta, 4),
                        "actions": [
                            "global golden hour regrade",
                            "prismatic highlights in glass and water",
                            "boost natural material saturation by 20%",
                            "layer atmospheric haze for multi-plane depth",
                        ],
                    }
                )

        return {
            "observation": "Lux variants only marginally outperform baseline.",
            "remedy": entries,
        }

    def _derive_scene_specific_upgrades(self) -> Dict[str, Any]:
        return {
            "aerial": {
                "current_luxury": self.report.scenes["aerial"].metric("regular", "luxury_index"),
                "target": 0.7,
                "moves": [
                    "paint champagne sunset across pool",
                    "project architectural light patterns onto landscaping",
                    "hint at distant coastline haze",
                ],
            },
            "pool": {
                "current_luxury": self.report.scenes["pool"].metric("regular", "luxury_index"),
                "target": 0.72,
                "moves": [
                    "introduce spa steam plumes",
                    "cast caustic light dances on retaining walls",
                    "stage floating floral candles",
                ],
            },
            "great_room": {
                "current_luxury": self.report.scenes["great_room"].metric("regular", "luxury_index"),
                "target": 0.75,
                "moves": [
                    "intensify fire feature for kinetic shadow play",
                    "suspend dust motes inside skylight beam",
                    "animate curtain sway for breathable movement",
                ],
            },
        }


class MaterialAwareEnhancementPlanner(RenderEnhancementPlanner):
    """Extended planner that incorporates material-specific rendering strategies."""

    def __init__(self, report: MaterialResponseReport):
        super().__init__(report)
        self.materials = self._initialize_materials()

    @classmethod
    def from_json(cls, path: str | Path) -> "MaterialAwareEnhancementPlanner":
        return cls(MaterialResponseReport.load(path))

    def _initialize_materials(self) -> dict[str, MaterialDefinition]:
        """Map scenes to their primary materials."""

        return {
            "great_room": MaterialDefinition.herringbone_wood(),
            "aerial": MaterialDefinition.stone_pavers(),
            "kitchen": MaterialDefinition.textured_plaster(),
        }

    def build_blueprint(self) -> dict[str, Any]:
        """Enhanced blueprint with material-aware strategies."""

        base_blueprint = super().build_blueprint()
        base_blueprint["material_integration"] = self._derive_material_strategy()
        base_blueprint["exposure_zones"] = self._derive_exposure_zones()
        base_blueprint["shader_settings"] = self._derive_shader_settings()
        return base_blueprint

    def _derive_material_strategy(self) -> dict[str, Any]:
        """Generate material-specific rendering instructions."""

        strategies: dict[str, Any] = {}
        for scene_name, material in self.materials.items():
            scene = self.report.scenes.get(scene_name)
            if scene is None:
                continue

            current_texture = scene.metric("regular", "texture_dimension")
            target_texture = 2.25 if scene_name == "great_room" else 1.95

            strategies[scene_name] = {
                "material": material.name,
                "current_texture_dimension": current_texture,
                "target_texture_dimension": target_texture,
                "rendering_params": {
                    "ior": material.ior,
                    "roughness_min": material.roughness_range[0],
                    "roughness_max": material.roughness_range[1],
                    "displacement": {
                        "strength_mm": material.displacement_mm,
                        "subdivision_level": 4 if material.displacement_mm > 5 else 3,
                    },
                    "mapping": {
                        "type": material.mapping_type,
                        "layers": material.texture_layers,
                    },
                },
                "optimization_notes": self._get_material_optimization(material, current_texture),
            }

        return strategies

    def _derive_exposure_zones(self) -> dict[str, Any]:
        """Create exposure zones based on luminance hierarchy."""

        zones: List[dict[str, Any]] = []
        for scene in self.report.iter_scenes():
            base_luminance = scene.metric("regular", "luminance")
            ev_adjustment = self._calculate_ev_adjustment(base_luminance, 0.31)
            zones.append(
                {
                    "scene": scene.name,
                    "base_luminance": base_luminance,
                    "ev_adjustment": ev_adjustment,
                    "tone_mapping": "filmic" if base_luminance < 0.3 else "aces",
                    "local_adjustments": self._get_local_adjustments(scene.name),
                }
            )

        return {"zones": zones, "global_reference": 0.31}

    def _derive_shader_settings(self) -> dict[str, Any]:
        """Generate shader-specific settings for each material type."""

        return {
            "wood": {
                "anisotropy": 0.5,
                "anisotropy_rotation": 0.0,
                "clearcoat": 0.3,
                "clearcoat_roughness": 0.1,
                "subsurface": 0.0,
                "specular_workflow": "inverse_grain_correlation",
            },
            "stone": {
                "procedural_variation": {
                    "count": 12,
                    "seed": 42,
                    "distribution": "weighted_random",
                    "mortar_depth": 3.0,
                    "mortar_roughness": 0.85,
                }
            },
            "plaster": {
                "multi_layer_noise": {
                    "fine_scale": 0.2,
                    "fine_strength": 0.3,
                    "macro_scale": 8.0,
                    "macro_strength": 0.7,
                    "roughness_correlation": "follow_macro",
                }
            },
        }

    def _get_material_optimization(self, material: MaterialDefinition, current: float) -> str:
        """Return optimization notes for specific material."""

        if material.name == "herringbone_oak":
            return f"Increase grain contrast by {round((2.25 - current) * 100)}% through displacement"
        if material.name == "algorithmic_stone":
            return "Add 8-12 unique variants with procedural distribution"
        return "Layer dual-frequency noise for organic complexity"

    def _calculate_ev_adjustment(self, current: float, target: float) -> float:
        """Calculate exposure value adjustment in stops."""

        import math

        if current <= 0:
            return 0.0
        return round(math.log2(target / current), 2)

    def _get_local_adjustments(self, scene_name: str) -> List[dict[str, Any]]:
        """Return scene-specific local exposure adjustments."""

        adjustments = {
            "pool": [
                {"area": "water_surface", "ev_delta": +0.5},
                {"area": "underwater", "ev_delta": -0.3},
            ],
            "great_room": [
                {"area": "skylight_shaft", "ev_delta": +1.0},
                {"area": "corner_shadows", "ev_delta": -0.5},
            ],
            "kitchen": [
                {"area": "island_surface", "ev_delta": +0.3},
                {"area": "backsplash", "ev_delta": 0.0},
            ],
        }
        return adjustments.get(scene_name, [])


__all__ = [
    "MaterialResponseReport",
    "MetricSnapshot",
    "MaterialAwareEnhancementPlanner",
    "MaterialDefinition",
    "RenderEnhancementPlanner",
    "SceneReport",
]
