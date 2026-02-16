"""Bloom pipeline — adds a soft glow around bright areas of video frames."""

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from scope.core.pipelines.interface import Pipeline, Requirements

from .bloom_schema import BloomConfig

if TYPE_CHECKING:
    from scope.core.pipelines.base_schema import BasePipelineConfig

logger = logging.getLogger(__name__)


class BloomPipeline(Pipeline):
    """Applies a bloom/glow post-processing effect to video frames."""

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return BloomConfig

    def __init__(self, device: torch.device | None = None, **kwargs):
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._cached_kernel: torch.Tensor | None = None
        self._cached_radius: int | None = None

    def prepare(self, **kwargs) -> Requirements:
        """Declare that we need 1 input frame."""
        return Requirements(input_size=1)

    def __call__(self, **kwargs) -> dict:
        """Apply bloom effect to input video frames.

        Args:
            video: List of input frame tensors, each (1, H, W, C) in [0, 255].

        Returns:
            Dict with "video" key containing tensor (1, H, W, C) in [0, 1].
        """
        video = kwargs.get("video")
        if video is None:
            raise ValueError("Input video cannot be None for BloomPipeline")

        threshold = float(kwargs.get("threshold", 0.8))
        soft_knee = float(kwargs.get("soft_knee", 0.5))
        intensity = float(kwargs.get("intensity", 1.0))
        radius = max(1, min(48, int(kwargs.get("radius", 8))))
        downsample = max(1, min(4, int(kwargs.get("downsample", 1))))
        debug = bool(kwargs.get("debug", False))

        # Stack frames and normalize [0, 255] -> [0, 1]
        frames = torch.stack([f.squeeze(0) for f in video], dim=0)
        img = frames.to(device=self.device, dtype=torch.float32) / 255.0

        if debug:
            logger.info(
                "[Bloom] in: %s range [%.2f, %.2f] | params: threshold=%.2f "
                "soft_knee=%.2f intensity=%.2f radius=%d downsample=%d device=%s",
                tuple(frames.shape),
                frames.min().item(),
                frames.max().item(),
                threshold,
                soft_knee,
                intensity,
                radius,
                downsample,
                self.device,
            )

        # Short-circuit: no bloom when intensity is zero
        if intensity == 0.0:
            return {"video": img.clamp(0, 1)}

        # BHWC -> BCHW
        x = img.permute(0, 3, 1, 2)

        # Luma (Rec. 709)
        luma = 0.2126 * x[:, 0:1] + 0.7152 * x[:, 1:2] + 0.0722 * x[:, 2:3]

        # Soft threshold to extract highlight mask
        mask = _soft_threshold(luma, threshold, soft_knee)

        if debug:
            logger.info("[Bloom] mask mean: %.6f", mask.mean().item())

        highlights = x * mask

        # Downsample for cheaper blur
        if downsample > 1:
            highlights = F.avg_pool2d(
                highlights, kernel_size=downsample, stride=downsample
            )

        # Separable Gaussian blur
        effective_radius = max(1, radius // downsample) if downsample > 1 else radius
        kernel = self._get_kernel(effective_radius)
        blurred = _separable_blur(highlights, kernel, effective_radius)

        if debug:
            logger.info("[Bloom] blurred mean: %.6f", blurred.mean().item())

        # Upsample back to original resolution
        if downsample > 1:
            blurred = F.interpolate(
                blurred,
                size=(x.shape[2], x.shape[3]),
                mode="bilinear",
                align_corners=False,
            )

        # Combine: additive bloom
        out = x + blurred * intensity
        out = out.clamp(0, 1)

        # BCHW -> BHWC
        out = out.permute(0, 2, 3, 1)

        if debug:
            logger.info(
                "[Bloom] out: [%.4f, %.4f]", out.min().item(), out.max().item()
            )

        return {"video": out}

    def _get_kernel(self, radius: int) -> torch.Tensor:
        """Return a 1-D Gaussian kernel, caching when radius is unchanged."""
        if self._cached_kernel is not None and self._cached_radius == radius:
            return self._cached_kernel.to(self.device)
        kernel = _gaussian_kernel_1d(radius).to(self.device)
        self._cached_kernel = kernel
        self._cached_radius = radius
        return kernel


# ---------------------------------------------------------------------------
# Pure functions (no state)
# ---------------------------------------------------------------------------


def _soft_threshold(luma: torch.Tensor, threshold: float, soft_knee: float) -> torch.Tensor:
    """Compute a soft highlight mask from luma values."""
    knee = threshold * soft_knee
    if knee > 1e-8:
        t = luma - threshold + knee
        mask = torch.clamp(t / (2.0 * knee), 0.0, 1.0)
    else:
        mask = (luma >= threshold).float()
    return mask


def _gaussian_kernel_1d(radius: int) -> torch.Tensor:
    """Create a normalized 1-D Gaussian kernel of size 2*radius+1."""
    size = 2 * radius + 1
    sigma = max(radius / 3.0, 1e-8)
    x = torch.arange(size, dtype=torch.float32) - radius
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    return kernel


def _separable_blur(x: torch.Tensor, kernel: torch.Tensor, radius: int) -> torch.Tensor:
    """Apply separable Gaussian blur (horizontal then vertical) via conv2d."""
    if radius < 1:
        return x

    C = x.shape[1]
    # Shape kernels for grouped depthwise conv
    h_kernel = kernel.view(1, 1, 1, -1).expand(C, -1, -1, -1)
    v_kernel = kernel.view(1, 1, -1, 1).expand(C, -1, -1, -1)

    blurred = F.conv2d(x, h_kernel, padding=(0, radius), groups=C)
    blurred = F.conv2d(blurred, v_kernel, padding=(radius, 0), groups=C)
    return blurred
