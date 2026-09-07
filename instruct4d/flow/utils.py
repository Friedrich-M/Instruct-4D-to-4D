"""Optical-flow helpers used by the temporal half of the editing pipeline.

Section 3.3 of the paper propagates an edit along time by warping an already
edited frame onto its neighbours with RAFT flow, then letting anchor-aware
InstructPix2Pix repaint whatever the warp could not explain.  This module holds
the pieces that supports: loading RAFT, estimating a forward/backward flow pair,
turning that pair into an occlusion mask, and applying a flow to an image.
"""

from typing import Tuple

import argparse

import cv2
import numpy as np
import torch

from .raft import RAFT
from .raft.utils import InputPadder

#: Forward-backward consistency thresholds. A pixel survives when its round-trip
#: error is below ``ALPHA_REL`` times the total flow magnitude plus ``ALPHA_ABS``
#: pixels, which tolerates large motion while still rejecting occlusions.
ALPHA_REL = 0.5
ALPHA_ABS = 0.5

#: Refinement iterations for a RAFT forward pass; the value used throughout the
#: paper's experiments.
RAFT_ITERS = 20


def load_raft(checkpoint: str, device: torch.device | str) -> RAFT:
    """Load a frozen RAFT model in evaluation mode.

    The public RAFT checkpoints were saved from a :class:`~torch.nn.DataParallel`
    wrapper, so every key carries a ``module.`` prefix.  Wrapping and immediately
    unwrapping is the simplest way to load them without rewriting the keys.

    Args:
        checkpoint: Path to a RAFT ``.pth`` checkpoint, e.g. ``raft-things.pth``.
        device: Device to place the model on.
    """
    model = torch.nn.DataParallel(RAFT(args=argparse.Namespace(small=False, mixed_precision=False)))
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model = model.module.to(device)
    model.requires_grad_(False)
    model.eval()
    return model


@torch.no_grad()
def estimate_flow_pair(
    raft: RAFT,
    source: torch.Tensor,
    target: torch.Tensor,
    iters: int = RAFT_ITERS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate the forward and backward flow between two frames.

    Args:
        raft: A model from :func:`load_raft`.
        source: Source frame as ``(1, 3, H, W)`` in ``[0, 255]``.
        target: Target frame as ``(1, 3, H, W)`` in ``[0, 255]``.
        iters: RAFT refinement iterations.

    Returns:
        ``(forward, backward)`` flows as ``(H, W, 2)`` arrays, where ``forward``
        maps source pixels to target pixels and ``backward`` the reverse.
    """
    # RAFT needs both sides divisible by 8; the padding is stripped again below.
    padder = InputPadder(source.shape)
    source, target = padder.pad(source, target)

    _, forward = raft(source, target, iters=iters, test_mode=True)
    _, backward = raft(target, source, iters=iters, test_mode=True)

    forward = padder.unpad(forward[0]).cpu().numpy().transpose(1, 2, 0)
    backward = padder.unpad(backward[0]).cpu().numpy().transpose(1, 2, 0)
    return forward, backward


def warp_by_flow(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Resample ``image`` through ``flow``.

    Args:
        image: ``(H, W, C)`` image.
        flow: ``(H, W, 2)`` flow whose vectors point from each output pixel back
            into ``image``.

    Returns:
        The warped image, same shape as ``image``.
    """
    height, width = flow.shape[:2]
    # cv2.remap wants absolute sample coordinates, so add the pixel grid.
    coords = flow.copy()
    coords[:, :, 0] += np.arange(width)
    coords[:, :, 1] += np.arange(height)[:, np.newaxis]
    return cv2.remap(image, coords, None, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)


def consistency_mask(flow: np.ndarray, opposite_flow: np.ndarray) -> np.ndarray:
    """Mark pixels whose flow survives a forward-backward round trip.

    Following the flow into the other frame and back should land on the starting
    pixel.  Where it does not, the correspondence is unreliable -- typically an
    occlusion or a disocclusion -- and the warped colour must not be trusted.

    Args:
        flow: The flow being validated, ``(H, W, 2)``.
        opposite_flow: The flow in the opposite direction, ``(H, W, 2)``.

    Returns:
        A boolean ``(H, W)`` mask that is ``True`` where the warp is reliable.
    """
    round_trip = warp_by_flow(opposite_flow, flow)
    error = np.linalg.norm(flow + round_trip, axis=-1)
    magnitude = np.linalg.norm(flow, axis=-1) + np.linalg.norm(round_trip, axis=-1)
    return error < ALPHA_REL * magnitude + ALPHA_ABS


def blend_with_mask(warped: np.ndarray, fallback: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Take ``warped`` where ``mask`` holds and ``fallback`` everywhere else."""
    mask = mask[..., None]
    return warped * mask + fallback * (1 - mask)
