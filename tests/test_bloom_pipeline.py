"""Tests for the Bloom pipeline."""

import pytest
import torch

from my_scope_plugin.pipelines.bloom_pipeline import BloomPipeline


@pytest.fixture
def pipeline():
    return BloomPipeline()


class TestBloomPipeline:
    """Core acceptance tests for the Bloom pipeline."""

    def test_correct_shape_and_range(self, pipeline):
        """Output must be (1, H, W, 3) float in [0, 1]."""
        frame = torch.randint(0, 256, (1, 8, 8, 3), dtype=torch.float32)
        result = pipeline(
            video=[frame],
            threshold=0.8,
            soft_knee=0.5,
            intensity=1.0,
            radius=4,
            downsample=1,
        )
        out = result["video"]
        assert out.shape == (1, 8, 8, 3)
        assert out.dtype == torch.float32
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_intensity_zero_returns_normalized(self, pipeline):
        """intensity=0 should be a no-op aside from normalization to [0, 1]."""
        frame = torch.randint(0, 256, (1, 8, 8, 3), dtype=torch.float32)
        result = pipeline(
            video=[frame],
            threshold=0.8,
            soft_knee=0.5,
            intensity=0.0,
            radius=4,
            downsample=1,
        )
        out = result["video"]
        expected = frame / 255.0
        assert torch.allclose(out, expected, atol=1e-5)

    def test_no_highlights_no_bloom(self, pipeline):
        """A dark frame (all values below threshold) should pass through unchanged."""
        frame = torch.full((1, 8, 8, 3), 10.0)
        result = pipeline(
            video=[frame],
            threshold=0.8,
            soft_knee=0.0,
            intensity=1.0,
            radius=4,
            downsample=1,
        )
        out = result["video"]
        expected = frame / 255.0
        assert torch.allclose(out, expected, atol=1e-5)

    def test_bloom_increases_brightness(self, pipeline):
        """A bright pixel should spread light to its neighbors via bloom."""
        frame = torch.full((1, 8, 8, 3), 10.0)
        frame[0, 4, 4, :] = 255.0

        result = pipeline(
            video=[frame],
            threshold=0.5,
            soft_knee=0.5,
            intensity=1.0,
            radius=4,
            downsample=1,
        )
        out = result["video"]
        normalized = frame / 255.0

        # Neighboring pixel (3,4) should be brighter than its original value
        neighbor_out = out[0, 3, 4].max().item()
        neighbor_orig = normalized[0, 3, 4].max().item()
        assert neighbor_out > neighbor_orig, (
            f"Bloom should brighten neighbors: got {neighbor_out}, expected > {neighbor_orig}"
        )

    def test_radius_and_downsample_dont_change_shape(self, pipeline):
        """Output shape must be stable across various radius/downsample combos."""
        frame = torch.randint(0, 256, (1, 16, 16, 3), dtype=torch.float32)
        for radius in [1, 8, 24]:
            for downsample in [1, 2, 4]:
                result = pipeline(
                    video=[frame],
                    threshold=0.8,
                    soft_knee=0.5,
                    intensity=1.0,
                    radius=radius,
                    downsample=downsample,
                )
                out = result["video"]
                assert out.shape == (1, 16, 16, 3), (
                    f"Shape changed for radius={radius}, downsample={downsample}: {out.shape}"
                )
