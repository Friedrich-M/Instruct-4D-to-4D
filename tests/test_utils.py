"""Metrics, losses and grid sizing."""

import numpy as np
import torch

from instruct4d.utils import (
    TVLoss,
    cal_n_samples,
    colorize_depth,
    mse_to_psnr,
    n_to_reso,
    rgb_ssim,
)


def test_ssim_is_one_for_identical_images():
    image = np.random.rand(32, 32, 3)
    assert abs(rgb_ssim(image, image, 1.0) - 1.0) < 1e-6
    assert rgb_ssim(image, np.random.rand(32, 32, 3), 1.0) < 0.5


def test_psnr_matches_the_definition():
    assert abs(mse_to_psnr(0.01) - 20.0) < 1e-9
    assert abs(mse_to_psnr(1.0)) < 1e-9


def test_total_variation_is_zero_on_a_flat_map():
    assert TVLoss()(torch.ones(2, 3, 8, 8)).item() == 0.0
    assert TVLoss()(torch.randn(2, 3, 8, 8)).item() > 0


def test_total_variation_handles_a_degenerate_axis():
    """TensoRF's line factors are (1, C, N, 1): the width axis has no pairs."""
    line = torch.randn(1, 8, 32, 1)
    value = TVLoss()(line).item()
    assert np.isfinite(value) and value > 0
    # A single row and column has no variation at all.
    assert TVLoss()(torch.randn(1, 4, 1, 1)).item() == 0.0


def test_voxel_budget_is_split_by_extent():
    bbox = torch.tensor([[0.0, 0.0, 0.0], [4.0, 2.0, 1.0]])
    reso = n_to_reso(1000, bbox)
    assert reso[0] > reso[1] > reso[2]
    assert abs(np.prod(reso) - 1000) / 1000 < 0.2


def test_sample_count_scales_with_resolution():
    assert cal_n_samples([100, 100, 100], 0.5) > cal_n_samples([50, 50, 50], 0.5)


def test_depth_colourisation_ignores_empty_background():
    depth = np.ones((8, 8)) * 2.0
    depth[0, 0] = 0.0                      # background, must not set the scale
    image, (low, high) = colorize_depth(depth)
    assert image.shape == (8, 8, 3) and image.dtype == np.uint8
    assert low == 2.0 and high == 2.0
