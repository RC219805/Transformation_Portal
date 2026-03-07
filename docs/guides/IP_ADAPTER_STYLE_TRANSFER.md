# IP-Adapter Style Transfer

**Reference-based style transfer for architectural photography**

## Overview

IP-Adapter (Image Prompt Adapter) enables powerful style transfer by learning from reference images. Unlike text prompts alone, IP-Adapter allows you to show the AI exactly what style you want by providing example images.

### Key Benefits

- **Learn from Examples**: Show the AI professional photography you admire
- **Consistent Portfolio Style**: Maintain visual consistency across properties
- **Client-Specific Aesthetics**: Match client's preferred photography style
- **Magazine-Quality Results**: Apply editorial photography aesthetics
- **Fine-Grained Control**: Precise style strength and blending

---

## How IP-Adapter Works

```
Reference Image → CLIP Encoder → Style Features → FLUX + IP-Adapter → Styled Output
Content Image   →                                ↗
```

1. **Reference Encoding**: CLIP vision model extracts style features from reference
2. **Style Injection**: Features are injected into FLUX cross-attention layers
3. **Content Preservation**: ControlNet maintains architectural structure
4. **Style Transfer**: Output matches reference style while preserving content

---

## Quick Start

### Basic Style Transfer

```python
from transformation_portal.style_transfer import IPAdapterStyleTransfer

# Initialize
style_transfer = IPAdapterStyleTransfer()

# Transfer style from reference
result = style_transfer.transfer_style(
    content_image="my_estate.jpg",
    style_reference="architectural_digest_photo.jpg",
    style_strength=0.7
)

# Save result
result.save("styled_estate.jpg")
```

### Use Preset Styles

```python
# Apply pre-configured professional style
result = style_transfer.apply_preset_style(
    content_image="my_estate.jpg",
    preset="architectural_digest",
    strength=0.7
)
```

### Multi-Reference Blending

```python
# Blend multiple style references
result = style_transfer.transfer_multi_style(
    content_image="my_estate.jpg",
    style_references=[
        ("warm_lighting.jpg", 0.5),    # 50% warm lighting
        ("dramatic_sky.jpg", 0.3),     # 30% dramatic sky
        ("editorial_comp.jpg", 0.2)    # 20% editorial composition
    ]
)
```

---

## Style Presets

Pre-configured professional photography styles optimized for architectural photography.

### Editorial Styles

#### Architectural Digest
**Description:** Editorial luxury with warm, sophisticated tones. Perfect balance of aspiration and livability.

```python
result = style_transfer.apply_preset_style(
    content_image="estate.jpg",
    preset="architectural_digest",
    strength=0.75
)
```

**Best for:**
- High-end residential properties
- Luxury estate marketing
- Portfolio showcase pieces

**Characteristics:**
- Color: Warm, rich tones
- Lighting: Sophisticated, balanced
- Mood: Aspirational, inviting

---

#### Dwell Modern
**Description:** Clean, modern aesthetic with natural light emphasis. Scandinavian-influenced minimalism.

```python
result = style_transfer.apply_preset_style(
    content_image="estate.jpg",
    preset="dwell_modern",
    strength=0.70
)
```

**Best for:**
- Modern architecture
- Minimalist interiors
- Eco-friendly properties

**Characteristics:**
- Color: Cool, neutral tones
- Lighting: Natural, diffuse
- Mood: Clean, serene

---

#### Elle Decor Glamorous
**Description:** High-style glamorous aesthetic with dramatic lighting and bold colors.

```python
result = style_transfer.apply_preset_style(
    content_image="estate.jpg",
    preset="elle_decor_glamorous",
    strength=0.80
)
```

**Best for:**
- Fashion-forward properties
- High-style interiors
- Dramatic presentations

**Characteristics:**
- Color: Bold, saturated
- Lighting: Dramatic, high contrast
- Mood: Glamorous, bold

---

### Luxury Real Estate Styles

#### Luxury Real Estate
**Description:** Premium property marketing optimized for high-end listings. Bright, inviting, aspirational.

```python
result = style_transfer.apply_preset_style(
    content_image="estate.jpg",
    preset="luxury_real_estate",
    strength=0.65
)
```

**Best for:**
- MLS listings
- Property marketing materials
- Virtual staging

---

#### Coastal Luxury
**Description:** Sophisticated coastal aesthetic with bright, airy feel and ocean connection.

```python
result = style_transfer.apply_preset_style(
    content_image="ocean_view.jpg",
    preset="coastal_luxury",
    strength=0.70
)
```

**Best for:**
- Montecito/Santa Barbara properties
- Beachfront estates
- Ocean-view properties

---

### Minimalist Styles

#### Minimalist Zen
**Description:** Japanese-influenced minimalism with emphasis on negative space and natural materials.

```python
result = style_transfer.apply_preset_style(
    content_image="estate.jpg",
    preset="minimalist_zen",
    strength=0.75
)
```

---

#### Scandinavian Hygge
**Description:** Warm, inviting Scandinavian aesthetic with cozy textures.

```python
result = style_transfer.apply_preset_style(
    content_image="estate.jpg",
    preset="scandinavian_hygge",
    strength=0.70
)
```

---

### Dramatic Styles

#### Dramatic Moody
**Description:** High-contrast, cinematic aesthetic with rich shadows.

```python
result = style_transfer.apply_preset_style(
    content_image="estate.jpg",
    preset="dramatic_moody",
    strength=0.80
)
```

---

#### Golden Hour Glow
**Description:** Warm golden hour lighting with sun-washed interiors.

```python
result = style_transfer.apply_preset_style(
    content_image="estate.jpg",
    preset="golden_hour_glow",
    strength=0.75
)
```

---

#### Twilight Blue Hour
**Description:** Blue hour twilight with interior lights glowing against dusky sky.

```python
result = style_transfer.apply_preset_style(
    content_image="estate.jpg",
    preset="twilight_blue_hour",
    strength=0.75
)
```

---

## Advanced Features

### Multi-Reference Blending

Combine multiple style references for custom aesthetics:

```python
from transformation_portal.style_transfer import (
    IPAdapterStyleTransfer,
    MultiReferenceBlender,
    ReferenceImageEncoder
)

# Initialize components
style_transfer = IPAdapterStyleTransfer()
encoder = ReferenceImageEncoder()
blender = MultiReferenceBlender()

# Encode references
lighting_style = encoder.encode("warm_lighting.jpg")
color_style = encoder.encode("rich_colors.jpg")
comp_style = encoder.encode("editorial_composition.jpg")

# Hierarchical blend: 50% lighting, 30% color, 20% composition
blended_style = blender.hierarchical_blend(
    feature_groups=[
        [(lighting_style, 1.0)],
        [(color_style, 1.0)],
        [(comp_style, 1.0)]
    ],
    group_weights=[0.5, 0.3, 0.2]
)

# Apply blended style
# (requires custom pipeline integration)
```

---

### Style Interpolation

Create smooth transitions between two styles:

```python
# Generate interpolation from style A to style B
interpolated_images = style_transfer.create_style_interpolation(
    content_image="estate.jpg",
    style1="minimalist.jpg",
    style2="luxurious.jpg",
    num_steps=5
)

# Save interpolation sequence
for i, img in enumerate(interpolated_images):
    img.save(f"interpolation_{i}.jpg")
```

---

### Style Similarity Analysis

Compare styles quantitatively:

```python
# Compute similarity between two reference images
similarity = style_transfer.analyze_style_similarity(
    image1="my_photo_1.jpg",
    image2="my_photo_2.jpg"
)

print(f"Style similarity: {similarity:.2f}")  # 0.0-1.0
# 0.9+ = very similar
# 0.7-0.9 = similar
# 0.5-0.7 = somewhat similar
# <0.5 = different
```

---

### Create Style Library

Build a searchable library from your reference collection:

```python
encoder = ReferenceImageEncoder()

# Create library from directory
encoder.create_style_library(
    reference_dir="references/architectural_digest",
    output_path="libraries/ad_style_library.pkl",
    pattern="*.jpg"
)

# Load and use library
features, metadata = encoder.load_features("libraries/ad_style_library.pkl")
print(f"Library contains {metadata['num_images']} reference images")
```

---

### Extract Portfolio Style

Learn "house style" from photographer's portfolio:

```python
# Collect photographer's best work
portfolio_images = [
    "photographer/best_1.jpg",
    "photographer/best_2.jpg",
    "photographer/best_3.jpg",
    "photographer/best_4.jpg",
    "photographer/best_5.jpg"
]

# Extract averaged style
portfolio_style = style_transfer.extract_style_from_collection(
    reference_images=portfolio_images,
    weights=[0.25, 0.25, 0.20, 0.15, 0.15]  # Weight best examples higher
)

# Save for reuse
encoder = ReferenceImageEncoder()
encoder.save_features(
    portfolio_style,
    "styles/photographer_house_style.pkl",
    metadata={"name": "Photographer House Style", "num_refs": 5}
)
```

---

## Integration with Other Components

### FLUX + IP-Adapter + ControlNet

Maximum quality with structure preservation:

```python
from transformation_portal.diffusion import FLUXControlNet
from transformation_portal.style_transfer import IPAdapterStyleTransfer

# Generate depth map for ControlNet
controlnet = FLUXControlNet(control_types=["depth", "canny"])
control_images = {
    "depth": controlnet.generate_control_image("estate.jpg", "depth"),
    "canny": controlnet.generate_control_image("estate.jpg", "canny")
}

# Transfer style with structure preservation
style_transfer = IPAdapterStyleTransfer()
result = style_transfer.transfer_style(
    content_image="estate.jpg",
    style_reference="AD_reference.jpg",
    style_strength=0.70,
    preserve_structure=True  # Enables ControlNet
)
```

---

### SkyGAN + IP-Adapter

Atmospheric rendering with style transfer:

```python
from transformation_portal.atmosphere import SkyGANGenerator, SkyBlender
from transformation_portal.style_transfer import IPAdapterStyleTransfer

# First: Replace sky
sky_generator = SkyGANGenerator()
blender = SkyBlender()

sky = sky_generator.generate_sky(
    sun_azimuth=270,
    sun_elevation=15,
    time_of_day="golden_hour"
)

sky_replaced = blender.blend_sky("estate.jpg", sky)

# Second: Apply style transfer
style_transfer = IPAdapterStyleTransfer()
final = style_transfer.transfer_style(
    content_image=sky_replaced,
    style_reference="coastal_luxury_ref.jpg",
    style_strength=0.65
)
```

---

### ComfyUI Workflow Integration

Add IP-Adapter to workflows:

```python
from transformation_portal.comfyui import WorkflowBuilder

workflow = (WorkflowBuilder("IP-Adapter Workflow")
    .add_input("estate.jpg")
    .add_scene_analysis()

    # FLUX enhancement
    .add_flux_enhancement(strength=0.45, variant="dev")

    # IP-Adapter style transfer (custom node)
    .add_ip_adapter_style_transfer(
        style_reference="AD_reference.jpg",
        strength=0.70
    )

    # Quality validation
    .add_quality_validation(pass_threshold=7.5)
    .add_output("styled_estate.jpg")
    .build()
)
```

---

## Blending Strategies

The `MultiReferenceBlender` supports multiple blending strategies:

### Weighted Blend
Standard weighted average of references:

```python
blender = MultiReferenceBlender()

result = blender.weighted_blend([
    (style1, 0.5),
    (style2, 0.3),
    (style3, 0.2)
])
```

---

### Max Blend
Takes element-wise maximum - emphasizes strongest features:

```python
result = blender.max_blend([style1, style2, style3])
```

**Use for:** Emphasizing dramatic elements, highlights

---

### Min Blend
Takes element-wise minimum - emphasizes common features:

```python
result = blender.min_blend([style1, style2, style3])
```

**Use for:** Finding consensus style, conservative blending

---

### Adaptive Blend
Automatically weights based on similarity to target:

```python
result = blender.adaptive_blend(
    features_list=[style1, style2, style3],
    target_features=desired_style,
    temperature=1.0  # Lower = more selective
)
```

**Use for:** Matching specific target style, automatic optimization

---

### Hierarchical Blend
Grouped blending for aspect separation:

```python
result = blender.hierarchical_blend(
    feature_groups=[
        # Lighting group (50%)
        [(warm_light, 0.6), (dramatic_light, 0.4)],
        # Color group (30%)
        [(rich_colors, 0.5), (neutral_colors, 0.5)],
        # Composition group (20%)
        [(editorial_comp, 1.0)]
    ],
    group_weights=[0.5, 0.3, 0.2]
)
```

**Use for:** Precise control over different style aspects

---

## Parameter Tuning

### Style Strength

Controls how much style is transferred:

```python
# Subtle (preserves original more)
strength=0.5

# Balanced (recommended)
strength=0.7

# Strong (maximum style transfer)
strength=0.9
```

**Guidelines:**
- 0.4-0.6: Subtle enhancement, keeps original character
- 0.6-0.8: Balanced transfer, clear style influence
- 0.8-1.0: Strong transfer, reference style dominates

---

### Reference Image Quality

**Best practices for reference images:**
- **High resolution**: 1024x1024 or larger
- **Professional quality**: Well-lit, sharp, properly exposed
- **Similar subject**: Architectural photography, not unrelated subjects
- **Clear style**: Strong, identifiable aesthetic
- **Avoid:** Overly processed, filtered, or low-quality images

---

### Multiple References

When blending multiple references:
- **2-3 references**: Ideal for most cases
- **4-5 references**: Can work but may dilute individual styles
- **6+ references**: Risk losing distinctive characteristics

**Weight distribution:**
- Primary style: 40-60%
- Secondary styles: 20-30% each
- Accent styles: 10-20% each

---

## Use Cases

### 1. Match Client's Favorite Photography

Client provides example of style they love:

```python
# Client shows you an AD magazine photo they love
result = style_transfer.transfer_style(
    content_image="client_property.jpg",
    style_reference="client_favorite_example.jpg",
    style_strength=0.75
)
```

---

### 2. Maintain Portfolio Consistency

Apply consistent style across all property photos:

```python
# Extract style from your best work
portfolio_style = style_transfer.extract_style_from_collection([
    "portfolio/best_1.jpg",
    "portfolio/best_2.jpg",
    "portfolio/best_3.jpg"
])

# Apply to all new properties
for property_image in new_properties:
    styled = style_transfer.transfer_style(
        content_image=property_image,
        style_reference=portfolio_style,
        style_strength=0.70
    )
```

---

### 3. Seasonal Style Variations

Create seasonal variations of same property:

```python
# Summer: bright, airy
summer = style_transfer.apply_preset_style(
    "estate.jpg", preset="bright_airy_residential", strength=0.65
)

# Fall: warm, cozy
fall = style_transfer.apply_preset_style(
    "estate.jpg", preset="scandinavian_hygge", strength=0.70
)

# Winter: dramatic, moody
winter = style_transfer.apply_preset_style(
    "estate.jpg", preset="dramatic_moody", strength=0.75
)
```

---

### 4. A/B Testing Styles

Generate multiple style variants for client selection:

```python
styles_to_test = [
    ("architectural_digest", 0.75),
    ("dwell_modern", 0.70),
    ("luxury_real_estate", 0.65),
    ("coastal_luxury", 0.70)
]

variants = []
for preset, strength in styles_to_test:
    result = style_transfer.apply_preset_style(
        content_image="estate.jpg",
        preset=preset,
        strength=strength
    )
    variants.append((preset, result))

# Save for client review
for name, image in variants:
    image.save(f"client_review/{name}.jpg")
```

---

## Performance Optimization

### Model Caching

Cache style features for reuse:

```python
encoder = ReferenceImageEncoder()

# Encode reference once
style_features = encoder.encode("reference.jpg")

# Save for reuse
encoder.save_features(style_features, "cached_styles/my_style.pkl")

# Load and reuse (much faster)
cached_style, metadata = encoder.load_features("cached_styles/my_style.pkl")
```

---

### Batch Processing

Process multiple images with same style:

```python
# Encode reference once
encoder = ReferenceImageEncoder()
style_features = encoder.encode("reference_style.jpg")

# Apply to all images
for image_path in property_images:
    result = style_transfer.transfer_style(
        content_image=image_path,
        style_reference=style_features,  # Reuse encoded features
        style_strength=0.70
    )
    result.save(f"styled/{image_path.name}")
```

---

## Troubleshooting

### Style Transfer Too Subtle

```python
# Increase style strength
strength=0.85  # Instead of 0.70

# Use stronger reference image
# Choose reference with very distinctive style

# Try preset designed for that aesthetic
preset="elle_decor_glamorous"  # Instead of subtle preset
```

---

### Style Transfer Too Strong

```python
# Decrease style strength
strength=0.55  # Instead of 0.70

# Blend with neutral reference
blended = blender.weighted_blend([
    (strong_style, 0.6),
    (neutral_reference, 0.4)
])
```

---

### Loss of Architectural Detail

```python
# Enable ControlNet for structure preservation
result = style_transfer.transfer_style(
    content_image="estate.jpg",
    style_reference="reference.jpg",
    style_strength=0.70,
    preserve_structure=True  # Critical for maintaining architecture
)

# Or reduce style strength
strength=0.60  # More conservative
```

---

### Inconsistent Results

```python
# Use fixed seed for reproducibility
result = style_transfer.transfer_style(
    content_image="estate.jpg",
    style_reference="reference.jpg",
    style_strength=0.70,
    seed=42  # Consistent results
)

# Analyze and cache style features
encoder = ReferenceImageEncoder()
style = encoder.encode("reference.jpg")
encoder.save_features(style, "consistent_style.pkl")
```

---

## Best Practices

### 1. Reference Selection

**Good reference images:**
- Professional architectural photography
- Similar lighting conditions
- Clear, identifiable style
- High resolution and quality
- Appropriate for property type

**Avoid:**
- Amateur snapshots
- Heavily filtered images
- Unrelated subjects
- Low resolution
- Extreme processing

---

### 2. Style Strength Guidelines

| Property Type | Recommended Strength | Notes |
|---------------|---------------------|-------|
| Luxury Estates | 0.70-0.80 | Strong style acceptable |
| Family Homes | 0.60-0.70 | Balanced, inviting |
| Commercial | 0.50-0.65 | Conservative, professional |
| Architectural Showcase | 0.75-0.85 | Artistic freedom |

---

### 3. Workflow Integration

Recommended processing order:
1. **Depth processing** (if needed for ControlNet)
2. **FLUX enhancement** (base quality improvement)
3. **SkyGAN** (if sky replacement needed)
4. **IP-Adapter style transfer** (final aesthetic)
5. **Quality validation**

---

### 4. Client Communication

When presenting style options:
- Show 2-4 distinct styles, not 10+
- Present with reference images so client understands
- Start conservative, can always increase strength
- Save client preferences for future properties

---

## API Reference

### IPAdapterStyleTransfer

```python
class IPAdapterStyleTransfer:
    def __init__(
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        enable_cpu_offload: bool = False
    )

    def transfer_style(
        content_image: Union[str, Path, Image.Image],
        style_reference: Union[str, Path, Image.Image],
        style_strength: float = 0.7,
        prompt: Optional[str] = None,
        num_steps: int = 4,
        guidance_scale: float = 3.5,
        preserve_structure: bool = True,
        seed: Optional[int] = None
    ) -> Image.Image

    def transfer_multi_style(
        content_image: Union[str, Path, Image.Image],
        style_references: List[Tuple[Union[str, Path, Image.Image], float]],
        **kwargs
    ) -> Image.Image

    def apply_preset_style(
        content_image: Union[str, Path, Image.Image],
        preset: str,
        strength: float = 0.7,
        **kwargs
    ) -> Image.Image

    def analyze_style_similarity(
        image1: Union[str, Path, Image.Image],
        image2: Union[str, Path, Image.Image]
    ) -> float

    def create_style_interpolation(
        content_image: Union[str, Path, Image.Image],
        style1: Union[str, Path, Image.Image],
        style2: Union[str, Path, Image.Image],
        num_steps: int = 5
    ) -> List[Image.Image]
```

---

## Conclusion

IP-Adapter style transfer provides unprecedented control over the aesthetic quality of architectural photography enhancement. By learning from professional reference images, you can achieve magazine-quality results that match client expectations and maintain consistent portfolio aesthetics.

**Key Takeaways:**
- Use high-quality reference images from professional photography
- Start with preset styles, then customize as needed
- Blend multiple references for unique aesthetics
- Preserve structure with ControlNet for architectural accuracy
- Build style libraries for consistent portfolio work

**Next Steps:**
1. Explore preset styles to find your aesthetic
2. Collect reference images from your favorite sources
3. Create custom style blends for your properties
4. Integrate with FLUX and SkyGAN for complete enhancement

For questions or advanced use cases, consult the main AI Enhancement Guide or individual component documentation.
