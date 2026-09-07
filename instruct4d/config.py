"""Command-line and file configuration for the multi-view entry points.

``train.py``, ``edit.py`` and ``render.py`` share one option set; editing adds a
second group on top.  Options can be given on the command line or in a config
file under ``configs/`` -- ConfigArgParse treats the two identically, with the
command line winning.

Note that the config file's key set must match the parser exactly: an unknown
key in a config file is an error, not a warning.
"""

import ast
import os
from typing import Optional, Sequence

import configargparse
import mmengine
import torch

from .data import DATASETS
from .fields import FIELDS


def build_parser(editing: bool = False) -> configargparse.ArgumentParser:
    """Build the argument parser.

    Args:
        editing: Also register the instruction-editing options.
    """
    parser = configargparse.ArgumentParser()
    _add_experiment_args(parser)
    _add_data_args(parser)
    _add_field_args(parser)
    _add_optimisation_args(parser)
    _add_rendering_args(parser)
    if editing:
        _add_editing_args(parser)
    return parser


def parse_args(editing: bool = False, cmd: Optional[Sequence[str]] = None) -> mmengine.Config:
    """Parse configuration into an attribute-accessible config object.

    Args:
        editing: Register the editing options.
        cmd: Argument list to parse; defaults to ``sys.argv``.

    Returns:
        An :class:`mmengine.Config` holding every option.
    """
    parser = build_parser(editing=editing)
    args = parser.parse_args(cmd) if cmd is not None else parser.parse_args()

    args.datadir = os.path.expanduser(args.datadir)
    # `frame_list` arrives as the literal text "[0, 5, 10]" from the config file.
    args.frame_list = ast.literal_eval(args.frame_list) if args.frame_list else []
    # `action="append"` leaves these as None when nothing sets them.
    args.upsamp_list = args.upsamp_list or []

    config = mmengine.Config(vars(args))
    if config.cfg_options is not None:
        config.merge_from_dict(config.cfg_options)

    for required in ("num_frames", "n_lamb_sigma", "n_lamb_sh"):
        if config.get(required) is None:
            raise SystemExit(
                f"--{required} is required; pass a config file from configs/ with --config"
            )
    return config


def load_checkpoint_config(args: mmengine.Config) -> mmengine.Config:
    """Recover the training run's configuration, overridden by the current args.

    A checkpoint is only meaningful alongside the field geometry it was trained
    with, and ``train.py`` dumps that next to the weights.  Rendering therefore
    starts from the dumped config and layers the current invocation on top.
    """
    dumped = f"{os.path.dirname(args.ckpt)}/config.py"
    if not os.path.exists(dumped):
        raise FileNotFoundError(
            f"no config.py beside the checkpoint at {dumped}; it is written next to "
            "the weights by train.py and is needed to rebuild the field"
        )
    merged = mmengine.Config.fromfile(dumped)
    merged.merge_from_dict(args)
    return merged


def build_field(args: mmengine.Config, aabb: torch.Tensor, reso, device, near_far):
    """Construct the radiance field described by ``args``.

    Args:
        args: Parsed configuration.
        aabb: ``(2, 3)`` scene bounding box.
        reso: ``[nx, ny, nz]`` initial grid resolution.
        device: Device to build on.
        near_far: Ray bounds from the dataset.
    """
    return FIELDS[args.model_name](
        aabb,
        reso,
        device,
        density_n_comp=args.n_lamb_sigma,
        appearance_n_comp=args.n_lamb_sh,
        app_dim=args.data_dim_color,
        near_far=near_far,
        shadingMode=args.shadingMode,
        alphaMask_thres=args.alpha_mask_thre,
        density_shift=args.density_shift,
        distance_scale=args.distance_scale,
        pos_pe=args.pos_pe,
        view_pe=args.view_pe,
        fea_pe=args.fea_pe,
        featureC=args.featureC,
        step_ratio=args.step_ratio,
        fea2denseAct=args.fea2denseAct,
        num_frames=args.num_frames,
        ld_per_frame=args.ld_per_frame,
    )


# ----------------------------------------------------------------------
# Option groups
# ----------------------------------------------------------------------
def _add_experiment_args(parser) -> None:
    group = parser.add_argument_group("experiment")
    group.add_argument("--config", is_config_file=True, help="path to a config file")
    group.add_argument("--expname", type=str, help="experiment name; names the output directory")
    group.add_argument("--basedir", type=str, default="./log", help="where checkpoints and logs go")
    group.add_argument(
        "--add_timestamp", type=int, default=0, help="append a timestamp to the output directory"
    )
    group.add_argument(
        "--progress_refresh_rate", type=int, default=10, help="iterations between progress updates"
    )
    group.add_argument("--N_vis", type=int, default=5, help="views to render when visualising; -1 for all")
    group.add_argument("--vis_every", type=int, default=10000, help="iterations between visualisations")
    group.add_argument(
        "--cfg_options",
        nargs="+",
        action=mmengine.DictAction,
        help="override any option as key=value",
    )


def _add_data_args(parser) -> None:
    group = parser.add_argument_group("data")
    group.add_argument(
        "--datadir", type=str, default="./data/neural_3d/coffee_martini", help="scene directory"
    )
    group.add_argument(
        "--dataset_name", type=str, default="n3dv_dynamic", choices=sorted(DATASETS), help="dataset loader"
    )
    group.add_argument("--downsample_train", type=float, default=1.0, help="image downsampling factor")
    group.add_argument("--batch_size", type=int, default=4096, help="rays per optimisation step")
    group.add_argument("--num_frames", type=int, help="length of the time axis")
    group.add_argument(
        "--frame_list",
        type=str,
        default="[]",
        help="explicit frame indices as a Python list literal; empty means the first num_frames",
    )


def _add_field_args(parser) -> None:
    group = parser.add_argument_group("field")
    group.add_argument(
        "--model_name", type=str, default="StreamTensorVMSplit", choices=sorted(FIELDS),
        help="tensor decomposition to use",
    )
    group.add_argument("--n_lamb_sigma", type=int, action="append", help="density components per axis")
    group.add_argument("--n_lamb_sh", type=int, action="append", help="appearance components per axis")
    group.add_argument("--data_dim_color", type=int, default=27, help="appearance feature width")
    group.add_argument("--ld_per_frame", type=float, default=1, help="new feature channels per frame")
    group.add_argument(
        "--alpha_mask_thre", type=float, default=0.0001, help="opacity threshold for the alpha mask"
    )
    group.add_argument(
        "--distance_scale", type=float, default=25, help="scales sample spacing into density units"
    )
    group.add_argument(
        "--density_shift", type=float, default=-10,
        help="softplus bias so a zero feature gives zero density",
    )
    group.add_argument("--shadingMode", type=str, default="MLP_PE", help="appearance decoder")
    group.add_argument("--pos_pe", type=int, default=6, help="encoding octaves for position")
    group.add_argument("--view_pe", type=int, default=6, help="encoding octaves for view direction")
    group.add_argument("--fea_pe", type=int, default=6, help="encoding octaves for the feature")
    group.add_argument("--featureC", type=int, default=128, help="hidden width of the shading MLP")
    group.add_argument("--fea2denseAct", type=str, default="softplus", choices=["softplus", "relu"],
                       help="activation mapping features to density")
    group.add_argument("--N_voxel_init", type=int, default=100**3, help="initial voxel budget")
    group.add_argument("--N_voxel_final", type=int, default=300**3, help="final voxel budget")
    group.add_argument(
        "--upsamp_list", type=int, action="append", help="iterations at which to upsample the grid"
    )
    group.add_argument("--ckpt", type=str, default=None, help="checkpoint to load")


def _add_optimisation_args(parser) -> None:
    group = parser.add_argument_group("optimisation")
    group.add_argument("--n_iters", type=int, default=30000, help="total optimisation steps")
    group.add_argument("--lr_init", type=float, default=0.02, help="learning rate for the grids")
    group.add_argument("--lr_basis", type=float, default=1e-3, help="learning rate for the MLPs")
    group.add_argument(
        "--lr_decay_iters", type=int, default=-1,
        help="iterations over which the learning rate decays; -1 means n_iters",
    )
    group.add_argument(
        "--lr_decay_target_ratio", type=float, default=0.1,
        help="fraction of the initial learning rate reached at the end of decay",
    )
    group.add_argument(
        "--lr_upsample_reset", type=int, default=1, help="reset the learning rate after upsampling"
    )
    group.add_argument("--L1_weight_initial", type=float, default=0.0, help="density L1 weight")
    group.add_argument("--Ortho_weight", type=float, default=0.0, help="component orthogonality weight")
    group.add_argument("--TV_weight_density", type=float, default=0.0, help="density total-variation weight")
    group.add_argument("--TV_weight_app", type=float, default=0.0, help="appearance total-variation weight")
    group.add_argument(
        "--feat_diff_weight", type=float, default=0.0, help="temporal feature-difference weight"
    )


def _add_rendering_args(parser) -> None:
    group = parser.add_argument_group("rendering")
    group.add_argument("--ndc_ray", type=int, default=0, help="sample in normalised device coordinates")
    group.add_argument(
        "--nSamples", type=int, default=int(1e6),
        help="samples per ray; the resolution-derived estimate is used when smaller",
    )
    group.add_argument("--step_ratio", type=float, default=0.5, help="samples per voxel along a ray")
    group.add_argument("--render_test", type=int, default=0, help="render the held-out views")
    group.add_argument("--render_train", type=int, default=0, help="render the training views")
    group.add_argument("--render_path", type=int, default=0, help="render the spiral camera path")
    group.add_argument(
        "--render_all_views", type=int, default=0,
        help="dump every (frame, camera) render as an individual image",
    )
    group.add_argument("--export_mesh", type=int, default=0, help="export a mesh and exit")


def _add_editing_args(parser) -> None:
    group = parser.add_argument_group("editing")
    group.add_argument(
        "--prompt", type=str, default="don't change the image", help="the editing instruction"
    )
    group.add_argument(
        "--guidance_scale", type=float, default=7.5, help="text guidance scale for InstructPix2Pix"
    )
    group.add_argument(
        "--image_guidance_scale", type=float, default=1.5,
        help="image guidance scale; higher keeps the result closer to the original",
    )
    group.add_argument(
        "--diffusion_steps", type=int, default=20, help="denoising steps for the key-frame edit"
    )
    group.add_argument(
        "--refine_diffusion_steps", type=int, default=6,
        help="denoising steps when repainting a flow-warped window",
    )
    group.add_argument(
        "--refine_num_steps", type=int, default=700,
        help="noise level when repainting a flow-warped window; lower preserves more of the warp",
    )
    group.add_argument(
        "--restview_refine_diffusion_steps", type=int, default=10,
        help="denoising steps when repainting depth-warped views",
    )
    group.add_argument(
        "--restview_refine_num_steps", type=int, default=800,
        help="noise level when repainting depth-warped views",
    )
    group.add_argument(
        "--sequence_length", type=int, default=5,
        help="frames or views edited together in one anchor-aware batch; reduce this first if you hit CUDA OOM",
    )
    group.add_argument(
        "--warm_up_steps", type=int, default=12, help="annealed rounds of key-frame editing"
    )
    group.add_argument(
        "--warp_ratio", type=float, default=0.5,
        help="how strongly a depth-warped edit replaces a non-edited view",
    )
    group.add_argument(
        "--ip2p_device", type=str, default="cuda:1",
        help="device for InstructPix2Pix; editing and optimisation run concurrently, so this "
             "should normally be a second GPU",
    )
    group.add_argument(
        "--ip2p_use_full_precision", action="store_true",
        help="run the diffusion model in fp32 instead of fp16",
    )
    group.add_argument(
        "--raft_ckpt", type=str, default="./weights/raft-things.pth", help="RAFT optical-flow checkpoint"
    )
    group.add_argument("--cache", type=str, default="./cache", help="where to cache rendered depth")
    group.add_argument(
        "--save_debug_images", action="store_true",
        help="write intermediate editing visualisations under the log directory",
    )
