"""The editing pipelines, driven by deterministic stand-ins for the heavy models.

Diffusion and optical flow are replaced by stubs, so these exercise the parts
this repository actually implements: which frames go into a batch, which slot
holds the anchor, and where each result is written back.
"""

import argparse

import numpy as np
import torch

from instruct4d.editing import FrameBuffer, KeyFrameEditor, TemporalPropagator

NUM_FRAMES, NUM_CAMERAS, HEIGHT, WIDTH = 6, 5, 8, 10
SEQUENCE = 3


class StubIP2P:
    """Deterministic stand-in for the diffusion model.

    Mixes every frame with the batch's anchor, so a caller that puts the anchor
    in the wrong slot, or orders the batch differently, changes the output.
    """

    def __init__(self):
        self.batches = []

    def edit_sequence(self, images, images_cond, **kwargs):
        images = images.float()
        self.batches.append(images.clone())
        anchor = images[0:1]
        offsets = torch.arange(images.shape[0], dtype=torch.float32).view(-1, 1, 1, 1)
        return (0.5 * images + 0.3 * anchor + 0.01 * offsets).clamp(0, 1)


class StubRAFT:
    """Returns a fixed flow, so warping is deterministic and content-free."""

    def __call__(self, a, b, iters=20, test_mode=True):
        flow = torch.zeros(a.shape[0], 2, a.shape[2], a.shape[3])
        flow[:, 0] = 1.0
        return None, flow


def make_args():
    return argparse.Namespace(
        sequence_length=SEQUENCE, prompt="make it snow", guidance_scale=7.5,
        image_guidance_scale=1.5, diffusion_steps=8, refine_diffusion_steps=3,
        refine_num_steps=600, restview_refine_diffusion_steps=3,
        restview_refine_num_steps=700, ip2p_device="cpu",
    )


def make_scene(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    flat = torch.rand(NUM_FRAMES * NUM_CAMERAS * HEIGHT * WIDTH, 3)
    frames = FrameBuffer(flat, NUM_FRAMES, NUM_CAMERAS, HEIGHT, WIDTH)

    focal = 9.0
    ys, xs = torch.meshgrid(
        torch.arange(HEIGHT).float(), torch.arange(WIDTH).float(), indexing="ij"
    )
    points = []
    for camera in range(NUM_CAMERAS):
        z = torch.full((HEIGHT, WIDTH), -3.0 - 0.05 * camera)
        points.append(torch.stack(
            [-(xs + 0.5 - WIDTH / 2) / focal * z, (ys + 0.5 - HEIGHT / 2) / focal * z, z],
            dim=-1,
        ))
    points = torch.stack(points)

    intrinsics = torch.tensor(
        [[focal, 0, WIDTH / 2], [0, focal, HEIGHT / 2], [0, 0, 1.0]]
    ).expand(NUM_CAMERAS, 3, 3).clone()
    extrinsics = torch.eye(4).expand(NUM_CAMERAS, 4, 4).clone()
    for camera in range(NUM_CAMERAS):
        extrinsics[camera, 0, 3] = 0.02 * camera
    return frames, points, intrinsics, extrinsics


# ------------------------------------------------------------------ buffer
def test_get_returns_a_copy_not_a_view():
    """A view would keep tracking later writes, and whether it did would depend
    on whether the target device forced a transfer."""
    flat = torch.zeros(2 * 3 * 4 * 5, 3)
    buffer = FrameBuffer(flat, 2, 3, 4, 5)

    single = buffer.get(0, 1)
    several = buffer.get(0, [1, 2])
    buffer.set_many([0, 1], 1, torch.ones(2, 3, 4, 5))

    assert single.eq(0).all(), "a single-camera read must not track later writes"
    assert several.eq(0).all(), "a multi-camera read must not track later writes"


# ------------------------------------------------------------ key frame
def test_key_frame_editing_touches_only_the_key_frame():
    frames, points, intrinsics, extrinsics = make_scene()
    before = frames.snapshot()

    KeyFrameEditor(
        StubIP2P(), frames, before.clone(), points, intrinsics, extrinsics,
        make_args(), torch.device("cpu"),
    ).run(key_frame=0, warp_ratio=0.5, warm_up_steps=2)

    after = frames.snapshot()
    assert not torch.equal(after[0], before[0]), "the key frame should have changed"
    assert torch.equal(after[1:], before[1:]), "no other frame may be touched"
    assert torch.isfinite(after).all()


def test_key_frame_editing_covers_every_camera():
    frames, points, intrinsics, extrinsics = make_scene()
    before = frames.snapshot()

    KeyFrameEditor(
        StubIP2P(), frames, before.clone(), points, intrinsics, extrinsics,
        make_args(), torch.device("cpu"),
    ).run(key_frame=0, warp_ratio=1.0, warm_up_steps=2)

    after = frames.snapshot()
    for camera in range(NUM_CAMERAS):
        assert not torch.equal(after[0, camera], before[0, camera]), f"camera {camera} untouched"


def test_key_frame_batches_are_the_configured_length():
    frames, points, intrinsics, extrinsics = make_scene()
    ip2p = StubIP2P()
    KeyFrameEditor(
        ip2p, frames, frames.snapshot(), points, intrinsics, extrinsics,
        make_args(), torch.device("cpu"),
    ).run(key_frame=0, warp_ratio=0.5, warm_up_steps=1)

    # The first call edits the sampled cameras together; the rest repaint the
    # warped remainder, each anchored on its own first view.
    assert ip2p.batches[0].shape[0] == SEQUENCE
    for batch in ip2p.batches[1:]:
        assert batch.shape[0] <= SEQUENCE + 1
        assert torch.equal(batch[0], batch[1]), "the anchor slot duplicates the window's first view"


# ------------------------------------------------------------ propagation
def make_propagator(frames, ip2p=None):
    return TemporalPropagator(
        ip2p or StubIP2P(), StubRAFT(), frames, frames.snapshot(), make_args()
    )


def test_propagation_reaches_every_frame_and_camera():
    frames, *_ = make_scene()
    before = frames.snapshot()
    make_propagator(frames).run(key_frame=0)

    after = frames.snapshot()
    assert torch.isfinite(after).all()
    for frame in range(NUM_FRAMES):
        for camera in range(NUM_CAMERAS):
            assert not torch.equal(after[frame, camera], before[frame, camera]), (
                f"frame {frame}, camera {camera} was never written"
            )


def test_every_window_is_anchored_on_the_current_key_frame():
    """The anchor must be re-read each window: the first window rewrites the key
    frame, and later windows have to see that update."""
    frames, *_ = make_scene()
    ip2p = StubIP2P()
    propagator = TemporalPropagator(ip2p, StubRAFT(), frames, frames.snapshot(), make_args())
    propagator._propagate_camera(camera=0, key_frame=0)

    windows = len(range(0, NUM_FRAMES, SEQUENCE))
    assert len(ip2p.batches) == windows

    anchors = [batch[0] for batch in ip2p.batches]
    assert not torch.equal(anchors[0], anchors[1]), (
        "the anchor was hoisted out of the loop, so it missed the key frame's update"
    )
    # The last window saw whatever the buffer held once every earlier window
    # had been written back.
    assert torch.allclose(anchors[-1], frames.get(0, 0)[0], atol=1e-5)


def test_the_key_frame_is_not_flow_warped_into_itself():
    """Frame 0 is the source of the edit; the first window must leave it alone
    before repainting."""
    frames, *_ = make_scene()
    propagator = make_propagator(frames)

    images = frames.images[[0, 1, 2], 0].clone()
    original = images.clone()
    warped = propagator._warp_window_forward(images, frames.snapshot()[[0, 1, 2], 0], 0, 0, 3)

    assert torch.equal(warped[0], original[0]), "the key frame slot must be untouched"
    assert not torch.equal(warped[1], original[1]), "later slots must be warped"


def test_propagation_leaves_the_conditioning_images_alone():
    frames, *_ = make_scene()
    originals = frames.snapshot()
    reference = originals.clone()
    TemporalPropagator(StubIP2P(), StubRAFT(), frames, originals, make_args()).run(key_frame=0)
    assert torch.equal(originals, reference), "the condition must never drift"


# ------------------------------------------------------------ full pipeline
def test_full_pipeline_is_deterministic():
    results = []
    for _ in range(2):
        frames, points, intrinsics, extrinsics = make_scene(seed=3)
        originals = frames.snapshot()
        args = make_args()
        KeyFrameEditor(
            StubIP2P(), frames, originals, points, intrinsics, extrinsics,
            args, torch.device("cpu"),
        ).run(key_frame=0, warp_ratio=0.5, warm_up_steps=2)
        TemporalPropagator(StubIP2P(), StubRAFT(), frames, originals, args).run(key_frame=0)
        results.append(frames.snapshot())
    assert torch.equal(results[0], results[1])
