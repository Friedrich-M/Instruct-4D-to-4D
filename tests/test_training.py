"""Ray sampling, the editable frame buffer, and configuration parsing."""

import glob
import os

import pytest
import torch

from instruct4d.config import build_field, parse_args
from instruct4d.editing.buffer import FrameBuffer
from instruct4d.training import MotionSampler, UniformSampler

FRAMES, PER_FRAME, BATCH = 6, 400, 64


def frame_of(indices):
    return set((indices // PER_FRAME).tolist())


@pytest.mark.parametrize("kind", ["uniform", "motion"])
def test_a_batch_never_spans_two_frames(kind):
    """The field gathers its feature planes once per timestamp, so it cannot
    render a batch drawn from several frames."""
    rgbs = torch.rand(FRAMES * PER_FRAME, 3)
    sampler = (
        UniformSampler(FRAMES * PER_FRAME, FRAMES, BATCH)
        if kind == "uniform"
        else MotionSampler(rgbs, FRAMES, BATCH)
    )
    for _ in range(30):
        indices = sampler.nextids()
        assert indices.min() >= 0 and indices.max() < FRAMES * PER_FRAME
        assert len(frame_of(indices)) == 1


def test_motion_sampler_finds_the_moving_pixels():
    rgbs = torch.zeros(FRAMES * PER_FRAME, 3)
    moving = set(range(50, 90))
    for frame in range(FRAMES):
        for pixel in moving:
            rgbs[frame * PER_FRAME + pixel] = frame / FRAMES

    sampler = MotionSampler(rgbs, FRAMES, BATCH)
    for frame in range(FRAMES):
        assert set(sampler.motion_indices[frame].tolist()) <= moving

    hits = total = 0
    for _ in range(50):
        indices = sampler.nextids() % PER_FRAME
        hits += sum(int(i) in moving for i in indices.tolist())
        total += len(indices)
    # Moving pixels are 10% of the frame but should take far more of the budget.
    assert hits / total > 0.1


def test_a_completely_static_scene_still_samples():
    sampler = MotionSampler(torch.zeros(FRAMES * PER_FRAME, 3), FRAMES, BATCH)
    assert all(len(v) == 0 for v in sampler.motion_indices.values())
    assert len(sampler.nextids()) > 0


def test_rays_must_divide_evenly_into_frames():
    with pytest.raises(ValueError, match="frame-major"):
        UniformSampler(101, 5, 4)


def test_frame_buffer_writes_through_to_the_flat_colours():
    num_frames, num_cameras, height, width = 3, 4, 5, 6
    flat = torch.zeros(num_frames * num_cameras * height * width, 3)
    buffer = FrameBuffer(flat, num_frames, num_cameras, height, width)
    snapshot = buffer.snapshot()

    buffer.set(1, [0, 2], torch.ones(2, 3, height, width))
    images = flat.view(num_frames, num_cameras, height, width, 3)
    assert images[1, 0].eq(1).all() and images[1, 2].eq(1).all()
    assert images[1, 1].eq(0).all(), "an untouched camera must not change"
    assert images[0].eq(0).all(), "an untouched frame must not change"
    assert snapshot.eq(0).all(), "snapshot must be a copy, not a view"

    buffer.set_many([0, 2], 1, torch.full((2, 3, height, width), 0.5))
    assert images[0, 1].eq(0.5).all() and images[2, 1].eq(0.5).all()

    assert buffer.get(1, [0, 2]).shape == (2, 3, height, width)
    assert buffer.get(1, 0).shape == (1, 3, height, width)


def test_frame_buffer_rejects_a_mismatched_size():
    with pytest.raises(ValueError, match="expected"):
        FrameBuffer(torch.zeros(7, 3), 3, 4, 5, 6)


MINIMAL_ARGS = [
    "--expname", "test", "--num_frames", "4",
    "--n_lamb_sigma", "4", "--n_lamb_sigma", "4", "--n_lamb_sigma", "4",
    "--n_lamb_sh", "8", "--n_lamb_sh", "8", "--n_lamb_sh", "8",
    "--data_dim_color", "9", "--shadingMode", "MLP_Fea", "--fea2denseAct", "relu",
    "--view_pe", "0", "--fea_pe", "0", "--ld_per_frame", "0.5", "--ndc_ray", "1",
]


def test_config_parses_and_builds_a_field():
    args = parse_args(editing=True, cmd=MINIMAL_ARGS + [
        "--prompt", "make it snow", "--frame_list", "[0, 1, 2, 3]",
    ])
    assert args.frame_list == [0, 1, 2, 3]
    assert args.prompt == "make it snow"

    field = build_field(args, torch.tensor([[-1.0] * 3, [1.0] * 3]), [16, 16, 16], "cpu", [0.0, 1.0])
    assert field.num_frames == 4


def test_missing_required_options_are_reported():
    with pytest.raises(SystemExit, match="num_frames"):
        parse_args(cmd=["--expname", "test"])


def test_editing_options_are_only_registered_for_editing():
    with pytest.raises(SystemExit):
        parse_args(editing=False, cmd=MINIMAL_ARGS + ["--prompt", "make it snow"])


@pytest.mark.parametrize(
    "path", sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "configs", "n3dv", "*.txt")))
)
def test_every_shipped_config_parses(path):
    """A config key the parser does not know is a hard error, so keep them in sync."""
    editing = os.path.basename(path).startswith("edit_")
    args = parse_args(editing=editing, cmd=["--config", path])
    assert args.dataset_name == "n3dv_dynamic"
    assert args.datadir.startswith("./data/")


def test_background_task_surfaces_a_failure():
    """A bare Thread swallows exceptions, which would let a broken editing run
    finish quietly and look like a successful one."""
    from instruct4d.utils import BackgroundTask

    ran = []
    BackgroundTask(lambda: ran.append(1), name="ok").start().join()
    assert ran == [1]

    def boom():
        raise ValueError("editing blew up")

    task = BackgroundTask(boom, name="temporal-propagation").start()
    with pytest.raises(RuntimeError, match="temporal-propagation"):
        task.join()
    assert task.failed


def test_background_task_forwards_keyword_arguments():
    from instruct4d.utils import BackgroundTask

    seen = {}
    BackgroundTask(lambda **kw: seen.update(kw), name="kw", key_frame=3).start().join()
    assert seen == {"key_frame": 3}
