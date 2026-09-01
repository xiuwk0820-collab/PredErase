"""Prompt templates aligned with the paper (Full / short / negative)."""

from __future__ import annotations

# Paper Full prompt (positive)
PROMPT_FULL = (
    "Clean empty background, seamless inpainting, natural lighting, "
    "no object, no person, no cast shadow, no contact shading, no text, "
    "photorealistic."
)

PROMPT_NEGATIVE_FULL = (
    "object, person, animal, text, watermark, logo, blurry, low quality, "
    "extra limbs, distorted, silhouette, floating debris, shadow residual."
)

# w/o Shadow Prompt ablation: short positive; drop "shadow residual" from negative
PROMPT_SHORT = "remove the object and fill the background"

PROMPT_NEGATIVE_SHORT = (
    "object, person, animal, text, watermark, logo, blurry, low quality, "
    "extra limbs, distorted, silhouette, floating debris."
)

# Longer shadow-aware prompt used in some internal FLUX configs (optional)
PROMPT_SHADOW_AWARE = (
    "completely remove the object together with its cast shadow, contact shadow, "
    "and any dark shadow residue on the ground or nearby surfaces; fill the masked "
    "region by extending the surrounding background texture, colors, and lighting; "
    "restore uniform natural illumination with no remaining shadow patches; seamless "
    "inpainting that matches adjacent context, photorealistic, no ghost border, no "
    "blur, no artifacts"
)


def get_prompts(variant: str = "full") -> tuple[str, str]:
    """Return (positive, negative) for ablation variant."""
    v = variant.lower().replace("-", "_").replace(" ", "_")
    if v in ("short", "wo_shadow_prompt", "w_o_shadow_prompt", "no_shadow_prompt"):
        return PROMPT_SHORT, PROMPT_NEGATIVE_SHORT
    if v in ("shadow_aware", "legacy"):
        return PROMPT_SHADOW_AWARE, PROMPT_NEGATIVE_FULL
    return PROMPT_FULL, PROMPT_NEGATIVE_FULL
