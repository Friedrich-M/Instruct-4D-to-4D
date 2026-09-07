"""Run bookkeeping shared by reconstruction and editing.

Both entry points create a log directory, dump the resolved configuration into
it, write checkpoints periodically and render the same set of outputs at the
end.  None of that is specific to either, so it lives here.
"""

import datetime
import os
from typing import List

import numpy as np
import torch

from ..rendering import evaluate, render_path

#: Fixed so runs are comparable; the value itself has no significance.
SEED = 20211202

#: Checkpoints are written this many times over a run, so a crash never costs
#: more than a fraction of the training.
CHECKPOINTS_PER_RUN = 10


def set_seed(seed: int = SEED) -> None:
    """Seed torch and numpy, and pin the default dtype."""
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(seed)
    np.random.seed(seed)


def make_log_folder(args) -> str:
    """Create the run's output directory and dump the resolved configuration.

    The dump matters: ``render.py`` reads it back to rebuild a field with the
    same geometry the checkpoint was trained with.

    Args:
        args: Parsed configuration.

    Returns:
        Path to the created directory.
    """
    suffix = datetime.datetime.now().strftime("-%Y%m%d-%H%M%S") if args.add_timestamp else ""
    logfolder = f"{args.basedir}/{args.expname}{suffix}"
    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(f"{logfolder}/imgs_vis", exist_ok=True)
    args.dump(os.path.join(logfolder, "config.py"))
    print(f"logging to {logfolder}")
    return logfolder


def save_checkpoint(field, logfolder: str, iteration: int) -> None:
    """Write a checkpoint, replacing any earlier one in the same folder.

    Only the latest is kept: these grids are large, and an intermediate state of
    a run is rarely worth the disk.
    """
    for name in os.listdir(logfolder):
        if name.endswith(".th"):
            os.remove(os.path.join(logfolder, name))
    path = f"{logfolder}/ckpt-{iteration}.th"
    field.save(path)
    print(f"checkpoint saved: {path}")


def upsample_schedule(args) -> List[int]:
    """Voxel budgets for each upsampling step.

    The budget grows geometrically rather than linearly, so every step
    multiplies the resolution by a constant factor.
    """
    steps = len(args.upsamp_list) + 1
    budgets = torch.round(
        torch.exp(torch.linspace(np.log(args.N_voxel_init), np.log(args.N_voxel_final), steps))
    ).long()
    return budgets.tolist()[1:]


def render_outputs(args, field, train_dataset, test_dataset, logfolder, white_bg, device) -> None:
    """Render whichever of the train/test/path outputs the config asked for."""
    if args.render_train:
        psnrs = evaluate(
            train_dataset, field, f"{logfolder}/imgs_train_all/", N_vis=-1, N_samples=-1,
            white_bg=white_bg, ndc_ray=args.ndc_ray, device=device,
        )
        if psnrs:
            print(f"{args.expname} train psnr: {np.mean(psnrs):.3f}")

    if args.render_test:
        psnrs = evaluate(
            test_dataset, field, f"{logfolder}/imgs_test_all/", N_vis=-1, N_samples=-1,
            white_bg=white_bg, ndc_ray=args.ndc_ray, device=device,
        )
        if psnrs:
            print(f"{args.expname} test psnr: {np.mean(psnrs):.3f}")

    if args.render_path:
        render_path(
            test_dataset, field, test_dataset.render_path, f"{logfolder}/imgs_path_all/",
            N_samples=-1, white_bg=white_bg, ndc_ray=args.ndc_ray, device=device,
        )
