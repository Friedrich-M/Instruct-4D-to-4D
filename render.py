"""Render a trained or edited 4D radiance field from a checkpoint.

    python render.py --config configs/n3dv/train_coffee_50_2.txt \\
        --datadir data/neural_3d/coffee_martini \\
        --ckpt log/neural_3d/train_coffee_50_2/ckpt-99999.th \\
        --render_test 1 --render_path 1

Outputs land beside the checkpoint, under a directory named after ``--expname``.
The field's geometry is recovered from the ``config.py`` that ``train.py`` wrote
next to the weights, with the current arguments layered on top.
"""

import os

import numpy as np
import torch
import torchvision
from tqdm.auto import tqdm

from instruct4d.config import build_field, load_checkpoint_config, parse_args
from instruct4d.data import DATASETS
from instruct4d.rendering import evaluate, render_path, render_rays
from instruct4d.training import set_seed
from instruct4d.utils import convert_sdf_samples_to_ply, n_to_reso

#: Iso-level at which the density field is turned into a surface.
MESH_LEVEL = 0.005


def load_field(args, dataset, device):
    """Rebuild the field described by ``args`` and load the checkpoint into it."""
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"checkpoint not found: {args.ckpt}")

    aabb = dataset.scene_bbox.to(device)
    reso = n_to_reso(args.N_voxel_final, aabb)
    field = build_field(args, aabb, reso, device, dataset.near_far)
    field.load(torch.load(args.ckpt, map_location=device))
    print(f"loaded checkpoint {args.ckpt}")
    return field


@torch.no_grad()
def export_mesh(args, field, frame: float = 0.0) -> None:
    """Marching-cubes the density field at one timestamp and write a PLY.

    The scene moves, so a mesh only means anything for a single ``frame``.

    Note that the field is rebuilt from ``args`` rather than from the
    checkpoint's own ``kwargs``: checkpoints written before this refactor record
    a placeholder ``num_frames`` of 1, which would produce a field whose tensors
    cannot hold the saved weights.
    """
    alpha, _ = field.getDenseAlpha(frame=frame)
    path = f"{os.path.splitext(args.ckpt)[0]}.ply"
    convert_sdf_samples_to_ply(alpha.cpu(), path, bbox=field.aabb.cpu(), level=MESH_LEVEL)


@torch.no_grad()
def render_all_views(args, field, dataset, device, output_dir: str) -> None:
    """Dump every (frame, camera) render as an individual image.

    Useful for computing metrics outside this repository, or for assembling
    per-camera videos.  Existing files are skipped so an interrupted run can be
    resumed.
    """
    os.makedirs(output_dir, exist_ok=True)
    width, height = dataset.img_wh
    num_frames = len(dataset.frame_list)
    num_cameras = len(dataset.poses)
    rays = dataset.all_rays.view(num_frames, num_cameras, height * width, -1)

    for frame in tqdm(range(num_frames), desc="frame"):
        for camera in range(num_cameras):
            path = os.path.join(output_dir, f"{frame:04d}_{camera:02d}.png")
            if os.path.exists(path):
                continue
            rgb, _ = render_rays(
                rays[frame, camera], field, chunk=4096, N_samples=-1,
                ndc_ray=args.ndc_ray, white_bg=dataset.white_bg, device=device,
            )
            torchvision.utils.save_image(rgb.view(height, width, 3).permute(2, 0, 1), path)
    print(f"wrote per-view renders to {output_dir}")


def main() -> None:
    set_seed()
    args = parse_args()
    if args.ckpt is None:
        raise SystemExit("--ckpt is required")
    args = load_checkpoint_config(args)
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_cls = DATASETS[args.dataset_name]
    test_dataset = dataset_cls(
        args.datadir, split="test", downsample=args.downsample_train,
        is_stack=True, num_frames=args.num_frames, frame_list=args.frame_list,
    )
    field = load_field(args, test_dataset, device)
    logfolder = os.path.join(os.path.dirname(args.ckpt), args.expname)

    if args.export_mesh:
        export_mesh(args, field)
        return

    if args.render_test:
        psnrs = evaluate(
            test_dataset, field, f"{logfolder}/imgs_test_all/", N_vis=-1, N_samples=-1,
            white_bg=test_dataset.white_bg, ndc_ray=args.ndc_ray, device=device,
        )
        if psnrs:
            print(f"{args.expname} test psnr: {np.mean(psnrs):.3f}")

    if args.render_path:
        render_path(
            test_dataset, field, test_dataset.render_path, f"{logfolder}/imgs_path_all/",
            N_samples=-1, white_bg=test_dataset.white_bg, ndc_ray=args.ndc_ray, device=device,
        )

    if args.render_train or args.render_all_views:
        train_dataset = dataset_cls(
            args.datadir, split="train", downsample=args.downsample_train,
            is_stack=True, num_frames=args.num_frames, frame_list=args.frame_list,
        )
        if args.render_train:
            psnrs = evaluate(
                train_dataset, field, f"{logfolder}/imgs_train_all/", N_vis=-1, N_samples=-1,
                white_bg=train_dataset.white_bg, ndc_ray=args.ndc_ray, device=device,
            )
            if psnrs:
                print(f"{args.expname} train psnr: {np.mean(psnrs):.3f}")
        if args.render_all_views:
            render_all_views(args, field, train_dataset, device, f"{logfolder}/views/")


if __name__ == "__main__":
    main()
