"""The streaming 4D field: rendering, checkpointing and regularisers."""

import pytest
import torch

from instruct4d.fields import FIELDS, StreamTensorCP, StreamTensorVMSplit
from instruct4d.rendering import render_rays

FIELD_KWARGS = dict(
    density_n_comp=[8, 4, 4],
    appearance_n_comp=[16, 8, 8],
    app_dim=9,
    shadingMode="MLP_Fea",
    near_far=[0.0, 1.0],
    step_ratio=0.5,
    fea2denseAct="relu",
    num_frames=6,
    ld_per_frame=0.5,
    view_pe=0,
    fea_pe=0,
)
AABB = torch.tensor([[-1.5, -1.67, -1.0], [1.5, 1.67, 1.0]])


def make_field(cls=StreamTensorVMSplit, **overrides):
    kwargs = {**FIELD_KWARGS, **overrides}
    return cls(AABB, [24, 26, 16], "cpu", **kwargs)


def make_rays(n=64, time=0.25):
    rays = torch.rand(n, 7) * 2 - 1
    rays[:, 6] = time
    return rays


@pytest.mark.parametrize("cls", [StreamTensorVMSplit, StreamTensorCP])
def test_forward_returns_finite_colour_and_depth(cls):
    field = make_field(cls)
    rgb, depth = field(make_rays(), white_bg=False, ndc_ray=True, N_samples=16)
    assert rgb.shape == (64, 3) and depth.shape == (64,)
    assert torch.isfinite(rgb).all() and torch.isfinite(depth).all()
    assert (rgb >= 0).all() and (rgb <= 1).all()


def test_a_chunk_must_hold_one_timestamp():
    field = make_field()
    rays = make_rays(8)
    rays[4:, 6] = 0.75
    with pytest.raises(ValueError, match="single timestamp"):
        field(rays, ndc_ray=True, N_samples=8)


def test_features_change_over_time():
    """The streaming buffer must actually produce different features per frame."""
    field = make_field()
    points = torch.rand(32, 3) * 2 - 1
    early = field.compute_densityfeature(points, torch.tensor(0.0))
    late = field.compute_densityfeature(points, torch.tensor(0.9))
    assert not torch.allclose(early, late)


def test_checkpoint_round_trip(tmp_path):
    field = make_field()
    path = tmp_path / "ckpt.th"
    field.save(str(path))

    ckpt = torch.load(str(path), weights_only=False)
    assert ckpt["kwargs"]["num_frames"] == 6
    assert ckpt["kwargs"]["ld_per_frame"] == 0.5

    rebuilt = FIELDS["StreamTensorVMSplit"](**{**ckpt["kwargs"], "device": "cpu"})
    rebuilt.load(ckpt)
    for key, value in field.state_dict().items():
        assert torch.equal(value, rebuilt.state_dict()[key])


def test_load_ignores_buffers_from_older_checkpoints():
    """Released checkpoints carry a buffer from a since-removed feature."""
    field = make_field()
    ckpt = {"kwargs": field.get_kwargs(), "state_dict": dict(field.state_dict())}
    ckpt["state_dict"]["target_portion"] = torch.tensor([0, 0, 1])
    make_field().load(ckpt)


def test_load_still_rejects_a_genuinely_wrong_checkpoint():
    field = make_field()
    ckpt = {"kwargs": field.get_kwargs(), "state_dict": dict(field.state_dict())}
    del ckpt["state_dict"]["basis_mat.weight"]
    with pytest.raises(RuntimeError):
        make_field().load(ckpt)


def test_upsampling_grows_the_grid_and_keeps_it_finite():
    field = make_field()
    before = [tuple(p.shape) for p in field.density_plane]
    field.upsample_volume_grid([32, 34, 22])
    after = [tuple(p.shape) for p in field.density_plane]
    assert all(a[-1] > b[-1] for a, b in zip(after, before))
    assert all(torch.isfinite(p).all() for p in field.density_plane)


def test_regularisers_are_finite():
    """Total variation must not divide by zero on the (1, C, N, 1) line factors."""
    from instruct4d.utils import TVLoss

    field = make_field()
    reg = TVLoss()
    for value in (
        field.density_L1(),
        field.vector_comp_diffs(),
        field.TV_loss_density(reg),
        field.TV_loss_app(reg),
        field.feat_diff_loss(0.4),
    ):
        assert torch.isfinite(value), value


def test_ray_filtering_drops_rays_outside_the_box():
    field = make_field()
    torch.manual_seed(0)
    rays = torch.randn(400, 7) * 1.5
    rgbs = torch.rand(400, 3)
    kept_rays, kept_rgbs = field.filtering_rays(rays, rgbs, bbox_only=True)
    assert len(kept_rays) == len(kept_rgbs)
    assert 0 < len(kept_rays) < 400


def test_chunking_does_not_change_the_render():
    field = make_field()
    field.eval()
    rays = make_rays(48, time=0.0)
    torch.manual_seed(1)
    whole, _ = render_rays(rays, field, chunk=48, N_samples=8, ndc_ray=True, device="cpu")
    torch.manual_seed(1)
    chunked, _ = render_rays(rays, field, chunk=16, N_samples=8, ndc_ray=True, device="cpu")
    assert torch.allclose(whole, chunked, atol=1e-6)
