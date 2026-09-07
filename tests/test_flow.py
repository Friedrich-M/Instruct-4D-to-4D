"""Optical-flow warping and the forward-backward consistency check."""

import numpy as np

from instruct4d.flow.utils import blend_with_mask, consistency_mask, warp_by_flow

HEIGHT, WIDTH = 40, 48


def constant_flow(dx, dy=0.0):
    flow = np.zeros((HEIGHT, WIDTH, 2), np.float32)
    flow[..., 0] = dx
    flow[..., 1] = dy
    return flow


def test_zero_flow_leaves_an_image_alone():
    image = (np.random.rand(HEIGHT, WIDTH, 3) * 255).astype(np.uint8)
    assert np.array_equal(warp_by_flow(image, constant_flow(0.0)), image)


def test_constant_flow_shifts_an_image():
    image = (np.random.rand(HEIGHT, WIDTH, 3) * 255).astype(np.uint8)
    shifted = warp_by_flow(image, constant_flow(3.0))
    # Sampling 3px to the right means output column c holds input column c + 3.
    assert np.allclose(shifted[5:-5, 5:-8], image[5:-5, 8:-5], atol=1)


def test_a_consistent_flow_pair_is_accepted():
    mask = consistency_mask(constant_flow(-3.0), constant_flow(3.0))
    assert mask[:, 6:-6].all()


def test_an_inconsistent_flow_pair_is_rejected():
    """A round trip that does not return to the starting pixel is an occlusion.

    Both flows pointing the same way is exactly that: following one and then the
    other lands twice as far away instead of back where it started.
    """
    mask = consistency_mask(constant_flow(-3.0), constant_flow(-3.0))
    assert not mask[:, 6:-6].any()


def test_the_tolerance_grows_with_flow_magnitude():
    """The threshold is relative, so the same absolute error means different
    things for slow and fast motion. Both pairs below round-trip 3px out."""
    slow = consistency_mask(constant_flow(1.0), constant_flow(2.0))
    fast = consistency_mask(constant_flow(10.0), constant_flow(-7.0))
    assert not slow[:, 6:-6].any(), "3px is a lot when the flow is only 1px"
    assert fast[:, 18:-18].all(), "3px is little when the flow is 10px"


def test_blend_picks_per_pixel():
    warped = np.ones((HEIGHT, WIDTH, 3), np.float32)
    fallback = np.zeros((HEIGHT, WIDTH, 3), np.float32)
    mask = np.zeros((HEIGHT, WIDTH), bool)
    mask[:, : WIDTH // 2] = True
    blended = blend_with_mask(warped, fallback, mask)
    assert (blended[:, : WIDTH // 2] == 1).all()
    assert (blended[:, WIDTH // 2 :] == 0).all()
