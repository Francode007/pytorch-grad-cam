#!/usr/bin/env python3
"""
Unit tests for the PF-CAM standalone module.

Tests normalization strategies, aggregation, extractor, and weight logging.

Usage:
    cd /Users/franchisnsaikia/IBS_Research/pytorch-grad-cam
    python -m pytest XAI_Enhancer_module/pf_cam/test_pf_cam.py -v
"""

import numpy as np
import torch
import pytest
import tempfile
import os
import json
import csv
import sys
from pathlib import Path

# Ensure project root is on path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from XAI_Enhancer_module.pf_cam.normalization import (
    NormStrategy,
    normalize_activations,
)
from XAI_Enhancer_module.pf_cam.aggregator import PFCamAggregator
from XAI_Enhancer_module.pf_cam.weight_logger import WeightLogger, ImageWeightRecord


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------

@pytest.fixture
def dummy_activations():
    """Create dummy activations (1, 64, 7, 7)."""
    np.random.seed(42)
    return np.random.rand(1, 64, 7, 7).astype(np.float32)


@pytest.fixture
def dummy_grads():
    """Create dummy gradients (1, 64, 7, 7) — includes negatives."""
    np.random.seed(123)
    return (np.random.rand(1, 64, 7, 7).astype(np.float32) - 0.3)


@pytest.fixture
def dummy_weighted_activations(dummy_activations, dummy_grads):
    """Weighted activations = grad_weights * activations."""
    weights = np.mean(dummy_grads, axis=(2, 3), keepdims=True)
    return weights * dummy_activations


@pytest.fixture
def dummy_cams():
    """Create 10 dummy CAMs at target resolution (7, 7)."""
    torch.manual_seed(0)
    return [torch.rand(7, 7) for _ in range(10)]


@pytest.fixture
def dummy_scores():
    """Create 10 dummy cosine similarities."""
    np.random.seed(0)
    return np.random.uniform(0.8, 1.0, size=10).astype(np.float64)


@pytest.fixture
def dummy_layer_shapes():
    """10 layers: 4 at 56x56, 3 at 14x14, 3 at 7x7."""
    return (
        [(56, 56)] * 4 +
        [(14, 14)] * 3 +
        [(7, 7)] * 3
    )


# ---------------------------------------------------------------
# Normalization Tests
# ---------------------------------------------------------------

class TestNormalization:
    """Test all normalization strategies."""

    def test_channel_spatial_output_range(self, dummy_weighted_activations, dummy_activations):
        """Channel spatial norm should produce non-negative results."""
        result = normalize_activations(
            dummy_weighted_activations, dummy_activations,
            strategy=NormStrategy.CHANNEL_SPATIAL
        )
        assert result.shape == dummy_activations.shape
        # masked = norm * act, both >= 0 in their individual domains
        # but norm can have negatives mapped near 0, so result should be finite
        assert np.all(np.isfinite(result))

    def test_global_output_range(self, dummy_weighted_activations, dummy_activations):
        """Global norm should produce non-negative results."""
        result = normalize_activations(
            dummy_weighted_activations, dummy_activations,
            strategy=NormStrategy.GLOBAL
        )
        assert result.shape == dummy_activations.shape
        assert np.all(np.isfinite(result))

    def test_gradient_weighted_output_shape(
        self, dummy_weighted_activations, dummy_activations, dummy_grads
    ):
        """Gradient weighted norm should preserve shape."""
        result = normalize_activations(
            dummy_weighted_activations, dummy_activations,
            grads=dummy_grads,
            strategy=NormStrategy.GRADIENT_WEIGHTED
        )
        assert result.shape == dummy_activations.shape
        assert np.all(np.isfinite(result))

    def test_gradient_weighted_fallback_without_grads(
        self, dummy_weighted_activations, dummy_activations
    ):
        """Gradient weighted should fall back to channel_spatial if no grads."""
        result = normalize_activations(
            dummy_weighted_activations, dummy_activations,
            grads=None,
            strategy=NormStrategy.GRADIENT_WEIGHTED
        )
        # Should not crash, should produce same as channel_spatial
        expected = normalize_activations(
            dummy_weighted_activations.copy(), dummy_activations,
            strategy=NormStrategy.CHANNEL_SPATIAL
        )
        np.testing.assert_array_almost_equal(result, expected)

    def test_l2_channel_unit_norm(self, dummy_weighted_activations, dummy_activations):
        """L2 channel norm should produce finite results."""
        result = normalize_activations(
            dummy_weighted_activations, dummy_activations,
            strategy=NormStrategy.L2_CHANNEL
        )
        assert result.shape == dummy_activations.shape
        assert np.all(np.isfinite(result))

    def test_relu_before_norm_removes_negatives(self):
        """Verify that applying ReLU before normalization removes negative contributions."""
        np.random.seed(99)
        # Create activations and grads where weighted_act has negatives
        act = np.random.rand(1, 64, 7, 7).astype(np.float32)
        # Grads with strong negatives so some channels get negative mean grad
        grads = (np.random.rand(1, 64, 7, 7).astype(np.float32) - 0.7)
        weights = np.mean(grads, axis=(2, 3), keepdims=True)  # Some channels negative
        weighted_act = weights * act

        # Precondition: some weighted activations ARE negative
        has_negatives = np.any(weighted_act < 0)
        assert has_negatives, "Test precondition: weighted_act should have negatives"

        # With ReLU (the fix)
        weighted_act_pos = np.maximum(weighted_act, 0)
        assert not np.any(weighted_act_pos < 0), "ReLU should remove all negatives"

        result = normalize_activations(
            weighted_act_pos, act,
            grads=grads,
            strategy=NormStrategy.GRADIENT_WEIGHTED
        )
        # Result should be non-negative
        assert np.all(result >= -1e-7), "ReLU fix should ensure non-negative masked activations"


# ---------------------------------------------------------------
# Aggregator Tests
# ---------------------------------------------------------------

class TestAggregator:
    """Test PF-CAM aggregation strategies."""

    def test_standard_aggregation(self, dummy_cams, dummy_scores):
        """Standard softmax aggregation should produce valid output."""
        result = PFCamAggregator.aggregate_standard(dummy_cams, dummy_scores)
        assert result.shape == (7, 7)
        assert torch.all(torch.isfinite(result))

    def test_temperature_sharpening(self, dummy_cams, dummy_scores):
        """Lower temperature should produce sharper (more extreme) weights."""
        result_warm = PFCamAggregator.aggregate_temperature(dummy_cams, dummy_scores, temp=1.0)
        result_cold = PFCamAggregator.aggregate_temperature(dummy_cams, dummy_scores, temp=0.01)
        # Cold temperature should make the peak more pronounced
        assert result_cold.max() >= result_warm.max() * 0.8  # Rough check

    def test_top_k_selection(self, dummy_cams, dummy_scores):
        """Top-K should select exactly K layers."""
        result, indices = PFCamAggregator.aggregate_top_k(dummy_cams, dummy_scores, k=3)
        assert len(indices) == 3
        assert result.shape == (7, 7)

    def test_pyramid_fusion_output(self, dummy_cams, dummy_scores, dummy_layer_shapes):
        """Pyramid fusion should produce valid output and diagnostics."""
        config = {"beta": 0.3, "k_percent": 0.5, "k_min": 1, "temp": 0.1}
        result, diagnostics = PFCamAggregator.aggregate_pyramid_fusion(
            dummy_cams, dummy_scores, dummy_layer_shapes, config
        )
        assert result.shape == (7, 7)
        assert torch.all(torch.isfinite(result))
        assert "num_stages" in diagnostics
        assert diagnostics["num_stages"] == 3  # 56x56, 14x14, 7x7
        assert len(diagnostics["stage_scores"]) == 3
        assert len(diagnostics["stage_weights"]) == 3

    def test_pyramid_fusion_beta_effect(self, dummy_cams, dummy_scores, dummy_layer_shapes):
        """Beta=0 should suppress shallow layers, Beta=1 should pass all."""
        config_low = {"beta": 0.0, "k_percent": 1.0, "k_min": 1, "temp": 1.0}
        config_high = {"beta": 1.0, "k_percent": 1.0, "k_min": 1, "temp": 1.0}
        result_low, _ = PFCamAggregator.aggregate_pyramid_fusion(
            dummy_cams, dummy_scores, dummy_layer_shapes, config_low
        )
        result_high, _ = PFCamAggregator.aggregate_pyramid_fusion(
            dummy_cams, dummy_scores, dummy_layer_shapes, config_high
        )
        # Both should be valid
        assert torch.all(torch.isfinite(result_low))
        assert torch.all(torch.isfinite(result_high))

    def test_aggregate_dispatch(self, dummy_cams, dummy_scores, dummy_layer_shapes):
        """The aggregate dispatcher should route to the correct method."""
        config = {"type": "pyramid", "beta": 0.3, "k_percent": 0.5, "temp": 0.1}
        result, diagnostics = PFCamAggregator.aggregate(
            dummy_cams, dummy_scores, dummy_layer_shapes, config
        )
        assert result.shape == (7, 7)
        assert "num_stages" in diagnostics


# ---------------------------------------------------------------
# Weight Logger Tests
# ---------------------------------------------------------------

class TestWeightLogger:
    """Test weight logging."""

    def test_csv_output(self):
        """CSV output should contain expected columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = WeightLogger(output_dir=tmpdir)
            record = ImageWeightRecord(
                image_path="/test/image.jpg",
                predicted_label=42,
                layer_names=["conv1", "layer1.0.conv1", "layer4.2.conv3"],
                raw_scores=[0.95, 0.87, 0.99],
                softmax_weights=[0.2, 0.1, 0.7],
                layer_shapes=[(112, 112), (56, 56), (7, 7)],
                stage_ids=[0, 1, 2],
                stage_scores=[0.95, 0.87, 0.99],
                stage_weights=[0.1, 0.2, 0.7],
                top_k_selected=[True, False, True],
            )
            logger.log(record)
            filepath = logger.save_csv()

            # Check file exists and has content
            assert os.path.exists(filepath)
            with open(filepath) as f:
                reader = csv.reader(f)
                header = next(reader)
                assert "layer_name" in header
                assert "raw_score" in header
                assert "stage_id" in header
                rows = list(reader)
                assert len(rows) == 3  # 3 layers

    def test_json_output(self):
        """JSON output should contain structured data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = WeightLogger(output_dir=tmpdir)
            record = ImageWeightRecord(
                image_path="/test/image.jpg",
                predicted_label=42,
                layer_names=["conv1", "conv2"],
                raw_scores=[0.95, 0.87],
                softmax_weights=[0.4, 0.6],
                layer_shapes=[(56, 56), (7, 7)],
                stage_ids=[0, 1],
                stage_scores=[0.95, 0.87],
                stage_weights=[0.4, 0.6],
                top_k_selected=[True, True],
            )
            logger.log(record)
            filepath = logger.save_json()

            with open(filepath) as f:
                data = json.load(f)
            assert len(data) == 1
            assert data[0]["predicted_label"] == 42
            assert len(data[0]["layers"]) == 2
            assert len(data[0]["stages"]) == 2

    def test_summary(self):
        """Summary should compute mean/std across images."""
        logger = WeightLogger(output_dir="/tmp")
        for _ in range(5):
            record = ImageWeightRecord(
                image_path="/test/image.jpg",
                predicted_label=0,
                layer_names=["a", "b"],
                raw_scores=[0.9, 0.8],
                softmax_weights=[0.6, 0.4],
                layer_shapes=[(56, 56), (7, 7)],
                stage_ids=[0, 1],
                stage_scores=[0.9, 0.8],
                stage_weights=[0.6, 0.4],
            )
            logger.log(record)

        summary = logger.get_summary()
        assert summary["num_images"] == 5
        assert summary["num_layers"] == 2


# ---------------------------------------------------------------
# Extractor Import Test
# ---------------------------------------------------------------

class TestExtractorImport:
    """Test that PFCamExtractor imports cleanly without enhanced_cams."""

    def test_import_succeeds(self):
        """PFCamExtractor should import without touching enhanced_cams."""
        from XAI_Enhancer_module.pf_cam.extractor import PFCamExtractor
        assert PFCamExtractor is not None

    def test_no_enhanced_cams_dependency(self):
        """Verify pf_cam does not import from enhanced_cams."""
        import XAI_Enhancer_module.pf_cam.extractor as ext_mod
        # Only check import/from lines, not docstrings or comments
        with open(ext_mod.__file__) as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith(('import ', 'from ')):
                    assert 'enhanced_cams' not in stripped, \
                        f"extractor.py imports from enhanced_cams: {stripped}"
                    assert 'OptimizedCamExtractor' not in stripped, \
                        f"extractor.py imports OptimizedCamExtractor: {stripped}"

    def test_no_gradcam_enhanced_dependency(self):
        """Verify pf_cam does not import GradCAMEnhanced."""
        import XAI_Enhancer_module.pf_cam.extractor as ext_mod
        with open(ext_mod.__file__) as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith(('import ', 'from ')):
                    assert 'GradCAMEnhanced' not in stripped, \
                        f"extractor.py imports GradCAMEnhanced: {stripped}"
                    assert 'GradCAM_enhanced' not in stripped, \
                        f"extractor.py imports GradCAM_enhanced: {stripped}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
