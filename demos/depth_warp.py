"""Propagate an edit between two camera views using rendered depth.

Component (3).  The source view is edited in 2D, then reprojected into the
target view through the depth the 4D NeRF renders.  Where the two views see the
same surface the transfer is exact; where they do not, the 3D consistency check
rejects the correspondence and those pixels are left black.

The point cloud and per-view pixel map come from the example bundle referenced
in the README, since they are outputs of a trained field.

    python demos/depth_warp.py --source_img examples/coffee_cam_2x/0.png \\
        --target_img examples/coffee_cam_2x/1.png \\
        --prompt 'What if it was painted by Van Gogh?' \\
        --pts_path examples/pts_0.pt --warp_path examples/warp_0.pt
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image

from diffusers import StableDiffusionInstructPix2PixPipeline

from _common import prepare_output_dir  # also puts the repository root on sys.path
from instruct4d.editing import apply_warp

IP2P_SOURCE = "timbrooks/instruct-pix2pix"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_img", type=str, required=True, help="view to edit")
    parser.add_argument("--target_img", type=str, required=True, help="view to warp onto")
    parser.add_argument("--prompt", type=str, required=True, help="editing instruction")
    parser.add_argument("--pts_path", type=str, required=True,
                        help="(num_views, H, W, 3) world points, from a trained field")
    parser.add_argument("--warp_path", type=str, required=True,
                        help="(num_views, num_views, H, W, 2) per-view pixel maps")
    parser.add_argument("--steps", type=int, default=20, help="denoising steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="text guidance")
    parser.add_argument("--image_guidance_scale", type=float, default=1.5, help="image guidance")
    parser.add_argument("--diff_thres", type=float, default=0.2,
                        help="3D disagreement above which a correspondence is rejected")
    parser.add_argument("--output_dir", type=str, default="./demo_output", help="where to write")
    parser.add_argument("--seed", type=int, default=7070)
    return parser.parse_args()


def view_index(path: str) -> int:
    """Recover a view index from a filename like ``.../3.png`` or ``cam_3.png``."""
    return int(os.path.splitext(os.path.basename(path))[0].split("_")[-1])


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    output_dir = prepare_output_dir(args.output_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    points = torch.load(args.pts_path).to(device)
    pixel_maps = torch.load(args.warp_path).to(device)
    source_id, target_id = view_index(args.source_img), view_index(args.target_img)

    source = Image.open(args.source_img).convert("RGB")
    target = Image.open(args.target_img).convert("RGB")
    width, height = source.size

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        IP2P_SOURCE, torch_dtype=torch.float16, safety_checker=None
    ).to(device)
    pipe.enable_attention_slicing()

    with torch.no_grad():
        edited = pipe(
            args.prompt, image=source, num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale, image_guidance_scale=args.image_guidance_scale,
            output_type="pt",
        ).images
    if edited.shape[-2:] != (height, width):
        edited = F.interpolate(edited, (height, width), mode="bilinear", align_corners=False)
    edited = edited.squeeze(0).permute(1, 2, 0).float()

    source_t = torch.from_numpy(np.asarray(source) / 255).to(device).float()
    target_t = torch.from_numpy(np.asarray(target) / 255).to(device).float()

    warped, mask, _ = apply_warp(
        pixel_maps[source_id][target_id], target_t, edited,
        points[target_id], points[source_id], diff_thres=args.diff_thres,
    )
    print(f"warp covered {mask.float().mean() * 100:.1f}% of the target view")

    # Top row: the two original views. Bottom row: the edited source and the
    # edit carried into the target view.
    sheet = torch.cat(
        [torch.cat([source_t, target_t], dim=1), torch.cat([edited, warped], dim=1)], dim=0
    )
    path = os.path.join(output_dir, "depth_warp.png")
    torchvision.utils.save_image(sheet.permute(2, 0, 1), path)
    print(f"wrote {path}: originals on top, edited source and its warp below")


if __name__ == "__main__":
    main()
