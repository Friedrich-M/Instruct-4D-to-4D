"""Reconstruct a 4D radiance field from a multi-view video dataset.

This is the prerequisite for editing: ``edit.py`` starts from a field that
already reproduces the scene.  Nothing here is specific to Instruct 4D-to-4D --
it is ordinary streaming-TensoRF reconstruction.

    python train.py --config configs/n3dv/train_coffee_50_2.txt \\
        --render_test 1 --render_path 1
"""

import sys

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from instruct4d.config import build_field, parse_args
from instruct4d.data import DATASETS
from instruct4d.rendering import evaluate
from instruct4d.training import (
    CHECKPOINTS_PER_RUN,
    FieldOptimizer,
    MotionSampler,
    make_log_folder,
    render_outputs,
    save_checkpoint,
    set_seed,
    upsample_schedule,
)
from instruct4d.utils import cal_n_samples, n_to_reso


def reconstruct(args, device) -> None:
    """Fit a field to the dataset and render the requested outputs."""
    logfolder = make_log_folder(args)
    summary_writer = SummaryWriter(logfolder)

    dataset_cls = DATASETS[args.dataset_name]
    train_dataset = dataset_cls(
        args.datadir, split="train", downsample=args.downsample_train,
        is_stack=False, num_frames=args.num_frames, frame_list=args.frame_list,
    )
    test_dataset = dataset_cls(
        args.datadir, split="test", downsample=args.downsample_train,
        is_stack=True, num_frames=args.num_frames, frame_list=args.frame_list,
    )
    white_bg = train_dataset.white_bg

    # Reconstruction starts coarse and upsamples, so the grid begins small.
    aabb = train_dataset.scene_bbox.to(device)
    reso = n_to_reso(args.N_voxel_init, aabb)
    n_samples = min(args.nSamples, cal_n_samples(reso, args.step_ratio))
    field = build_field(args, aabb, reso, device, train_dataset.near_far)

    all_rays, all_rgbs = train_dataset.all_rays, train_dataset.all_rgbs
    if not args.ndc_ray:
        all_rays, all_rgbs = field.filtering_rays(all_rays, all_rgbs, bbox_only=True)
    sampler = MotionSampler(all_rgbs, args.num_frames, args.batch_size)

    optimizer = FieldOptimizer(field, args, device, white_bg, n_samples)
    voxel_budgets = upsample_schedule(args)
    save_every = max(args.n_iters // CHECKPOINTS_PER_RUN, 1)

    torch.cuda.empty_cache()
    recent_psnrs, test_psnrs = [], [0]

    progress = tqdm(range(args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout)
    for iteration in progress:
        ray_idx = sampler.nextids()
        mse, psnr = optimizer.step(
            all_rays[ray_idx], all_rgbs[ray_idx].to(device), summary_writer, iteration
        )
        recent_psnrs.append(psnr)

        if iteration % args.progress_refresh_rate == 0:
            progress.set_description(
                f"iter {iteration:05d} "
                f"train_psnr={float(np.mean(recent_psnrs)):.2f} "
                f"test_psnr={float(np.mean(test_psnrs)):.2f} mse={mse:.6f}"
            )
            recent_psnrs = []

        if args.N_vis != 0 and iteration % args.vis_every == args.vis_every - 1:
            test_psnrs = evaluate(
                test_dataset, field, f"{logfolder}/imgs_vis/", N_vis=args.N_vis,
                prefix=f"{iteration:06d}_", N_samples=optimizer.n_samples,
                white_bg=white_bg, ndc_ray=args.ndc_ray, compute_extra_metrics=False,
                device=device,
            )
            summary_writer.add_scalar("test/psnr", np.mean(test_psnrs), global_step=iteration)

        if iteration in args.upsamp_list:
            reso = n_to_reso(voxel_budgets.pop(0), field.aabb)
            optimizer.n_samples = min(args.nSamples, cal_n_samples(reso, args.step_ratio))
            field.upsample_volume_grid(reso)
            # Upsampling replaces the grid tensors, so Adam's state is stale.
            lr_scale = (
                1.0
                if args.lr_upsample_reset
                else args.lr_decay_target_ratio ** (iteration / args.n_iters)
            )
            optimizer.reset_optimizer(lr_scale)

        if iteration % save_every == 0:
            # Written on the first iteration too, so a full disk fails fast.
            save_checkpoint(field, logfolder, iteration)

    save_checkpoint(field, logfolder, args.n_iters)
    render_outputs(args, field, train_dataset, test_dataset, logfolder, white_bg, device)


def main() -> None:
    set_seed()
    args = parse_args()
    print(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reconstruct(args, device)


if __name__ == "__main__":
    main()
