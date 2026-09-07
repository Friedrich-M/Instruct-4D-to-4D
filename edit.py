"""Edit a trained 4D radiance field with a text instruction.

    python edit.py --config configs/n3dv/edit_coffee_50_2.txt \\
        --datadir data/neural_3d/coffee_martini \\
        --ckpt log/neural_3d/train_coffee_50_2/ckpt-99999.th \\
        --prompt 'What if it was painted by Van Gogh?'

The pipeline has two stages.  First one timestamp -- the key frame -- is edited
across all of its cameras and made spatially consistent (:mod:`instruct4d.editing.key_view`).
Then that edit is carried along time by a flow-guided sliding window
(:mod:`instruct4d.editing.propagation`).

The second stage runs on a background thread while the radiance field is being
re-optimised on the main one, so the field keeps absorbing edited frames as they
are produced instead of waiting for the whole sequence.  That is also why the
diffusion model is placed on a separate GPU by default: the two stages are
running at the same time.
"""

import sys

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from instruct4d.config import build_field, parse_args
from instruct4d.data import DATASETS
from instruct4d.editing import (
    DebugImageWriter,
    FrameBuffer,
    KeyFrameEditor,
    TemporalPropagator,
    compute_view_points,
)
from instruct4d.flow import load_raft
from instruct4d.ip2p import SequenceInstructPix2Pix
from instruct4d.rendering import evaluate, render_path
from instruct4d.training import (
    CHECKPOINTS_PER_RUN,
    FieldOptimizer,
    MotionSampler,
    UniformSampler,
    make_log_folder,
    render_outputs,
    save_checkpoint,
    set_seed,
)
from instruct4d.utils import BackgroundTask, cal_n_samples, n_to_reso

#: Fraction of the run spent oversampling moving pixels.  Motion needs the most
#: supervision early; once it has converged, uniform sampling stops starving the
#: background.
MOTION_SAMPLING_FRACTION = 0.5

#: The timestamp the edit starts from.
KEY_FRAME = 0


def edit(args, device) -> None:
    """Run the full editing pipeline and re-optimise the field."""
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

    # The checkpoint is fully trained, so the grid starts at its final size.
    aabb = test_dataset.scene_bbox.to(device)
    reso = n_to_reso(args.N_voxel_final, aabb)
    n_samples = min(args.nSamples, cal_n_samples(reso, args.step_ratio))
    field = build_field(args, aabb, reso, device, test_dataset.near_far)
    field.load(torch.load(args.ckpt, map_location=device))
    print(f"loaded checkpoint {args.ckpt}")

    ip2p = SequenceInstructPix2Pix(
        device=args.ip2p_device, use_full_precision=args.ip2p_use_full_precision
    )
    raft = load_raft(args.raft_ckpt, args.ip2p_device)
    print("RAFT loaded")

    all_rays, all_rgbs = train_dataset.all_rays, train_dataset.all_rgbs
    width, height = train_dataset.img_wh
    num_frames = len(train_dataset.frame_list)
    num_cameras = len(train_dataset.poses)

    if not args.ndc_ray:
        all_rays, all_rgbs = field.filtering_rays(all_rays, all_rgbs, bbox_only=True)

    frames = FrameBuffer(all_rgbs, num_frames, num_cameras, height, width)
    # The conditioning images must stay the *original* scene: InstructPix2Pix
    # anchors its edit to them, so letting them drift would compound the edit.
    originals = frames.snapshot()

    debug = DebugImageWriter(f"{logfolder}/edit_debug" if args.save_debug_images else None)
    cache_dir = f"{args.cache}/{args.expname}"

    view_points = compute_view_points(
        field, all_rays, train_dataset, KEY_FRAME, num_frames, num_cameras, args, device, cache_dir
    )

    KeyFrameEditor(
        ip2p, frames, originals, view_points,
        train_dataset.intrinsics, train_dataset.extrinsics, args, device, debug,
    ).run(key_frame=KEY_FRAME, warp_ratio=args.warp_ratio, warm_up_steps=args.warm_up_steps)

    propagator = TemporalPropagator(ip2p, raft, frames, originals, args, debug)
    # Propagation runs alongside optimisation so the field absorbs edited
    # frames as they appear. BackgroundTask re-raises on join, so a failure
    # there cannot pass for a finished run.
    propagation = BackgroundTask(
        propagator.run, name="temporal-propagation", key_frame=KEY_FRAME
    ).start()
    print("temporal propagation running in the background")

    optimise(args, field, frames, all_rays, all_rgbs, test_dataset, logfolder,
             summary_writer, white_bg, n_samples, device)

    propagation.join()
    save_checkpoint(field, logfolder, args.n_iters)
    render_outputs(args, field, train_dataset, test_dataset, logfolder, white_bg, device)


def optimise(args, field, frames, all_rays, all_rgbs, test_dataset, logfolder,
             summary_writer, white_bg, n_samples, device) -> None:
    """Re-fit the field to the training colours as they are being edited."""
    motion_sampler = MotionSampler(all_rgbs, args.num_frames, args.batch_size)
    uniform_sampler = UniformSampler(all_rgbs.shape[0], args.num_frames, args.batch_size)
    switch_at = MOTION_SAMPLING_FRACTION * args.n_iters

    optimizer = FieldOptimizer(field, args, device, white_bg, n_samples)
    save_every = max(args.n_iters // (CHECKPOINTS_PER_RUN // 2), 1)

    torch.cuda.empty_cache()
    recent_psnrs, test_psnrs = [], [0]

    progress = tqdm(range(args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout)
    for iteration in progress:
        sampler = motion_sampler if iteration < switch_at else uniform_sampler
        ray_idx = sampler.nextids()

        # The propagation thread rewrites these colours as it goes, so the read
        # has to be atomic with respect to its writes.
        with frames.lock:
            rays, rgbs = all_rays[ray_idx], all_rgbs[ray_idx].to(device)

        mse, psnr = optimizer.step(rays, rgbs, summary_writer, iteration)
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
            render_path(
                test_dataset, field, test_dataset.render_path, f"{logfolder}/imgs_path_all/",
                N_samples=-1, white_bg=white_bg, ndc_ray=args.ndc_ray, device=device,
            )

        if iteration % save_every == 0:
            save_checkpoint(field, logfolder, iteration)


def main() -> None:
    set_seed()
    args = parse_args(editing=True)
    if args.ckpt is None:
        raise SystemExit("--ckpt is required: editing starts from a trained field")
    print(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    edit(args, device)


if __name__ == "__main__":
    main()
