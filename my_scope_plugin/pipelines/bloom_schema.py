"""Bloom pipeline configuration (video mode, runtime UI params)."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from scope.core.pipelines.base_schema import (
    BasePipelineConfig,
    ModeDefaults,
    UsageType,
    ui_field_config,
)


class BloomConfig(BasePipelineConfig):
    """Configuration for the Bloom pipeline."""

    pipeline_id: ClassVar[str] = "bloom"
    pipeline_name: ClassVar[str] = "Bloom"
    pipeline_description: ClassVar[str] = "Bloom/glow effect that adds soft light around bright areas"
    supports_prompts: ClassVar[bool] = False
    usage: ClassVar[list] = [UsageType.POSTPROCESSOR]
    modes: ClassVar[dict] = {"video": ModeDefaults(default=True)}

    threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Brightness threshold for bloom extraction",
        json_schema_extra=ui_field_config(order=1, label="Threshold"),
    )
    soft_knee: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Softness of the threshold transition",
        json_schema_extra=ui_field_config(order=2, label="Soft Knee"),
    )
    intensity: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Bloom intensity multiplier",
        json_schema_extra=ui_field_config(order=3, label="Intensity"),
    )
    radius: int = Field(
        default=8,
        ge=1,
        le=48,
        description="Blur radius for the bloom effect",
        json_schema_extra=ui_field_config(order=4, label="Radius"),
    )
    downsample: int = Field(
        default=1,
        ge=1,
        le=4,
        description="Downsample factor for performance (higher = faster, lower quality)",
        json_schema_extra=ui_field_config(order=5, label="Downsample"),
    )
    debug: bool = Field(
        default=False,
        description="Enable debug logging",
        json_schema_extra=ui_field_config(order=6, label="Debug"),
    )
