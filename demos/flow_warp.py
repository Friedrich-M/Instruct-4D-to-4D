"""Warp an edited frame onto a later one using RAFT optical flow.

Half of component (2).  The edit is applied to the source frame only; flow then
carries it to the target.  Two images are written: the raw warp, and the warp
after the forward-backward consistency check has masked out the pixels the flow
cannot explain.  Those masked regions are what the sliding-window demo repaints.

    python demos/flow_warp.py --source_img examples/coffee_frame_2x/3.png \\
        --target_img examples/coffee_frame_2x/6.png \\
        --prompt 'What if it was painted by Van Gogh?'
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image

from diffusers import StableDiffusionInstructPix2PixPipeline

from _common import prepare_output_dir  # also puts the repository root on sys.path
from instruct4d.flow import (
    consistency_mask,
    estimate_flow_pair,
    flow_to_image,
    load_raft,
    warp_by_flow,
)

IP2P_SOURCE = "timbrooks/instruct-pix2pix"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_img", type=str, required=True, help="frame to edit")
    parser.add_argument("--target_img", type=str, required=True, help="frame to warp onto")
    parser.add_argument("--prompt", type=str, required=True, help="editing instruction")
    parser.add_argument("--raft_ckpt", type=str, default="./weights/raft-things.pth")
    parser.add_argument("--steps", type=int, default=20, help="denoising steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="text guidance")
    parser.add_argument("--image_guidance_scale", type=float, default=1.5, help="image guidance")
    parser.add_argument("--output_dir", type=str, default="./demo_output/flow_warp")
    return parser.parse_args()


def as_raft_input(path: str, device) -> torch.Tensor:
    array = np.array(Image.open(path).convert("RGB")).astype(np.uint8)
    return torch.from_numpy(array).permute(2, 0, 1).float()[None].to(device)


def main():
    args = parse_args()
    output_dir = prepare_output_dir(args.output_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    raft = load_raft(args.raft_ckpt, device)
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        IP2P_SOURCE, torch_dtype=torch.float16, safety_checker=None
    ).to(device)
    pipe.enable_attention_slicing()

    source = as_raft_input(args.source_img, device)
    target = as_raft_input(args.target_img, device)
    height, width = source.shape[-2:]

    edited = pipe(
        args.prompt,
        image=Image.open(args.source_img).convert("RGB"),
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
        output_type="pt",
    ).images
    edited = torch.nn.functional.interpolate(
        edited, size=(height, width), mode="bilinear", align_corners=False
    )
    edited_np = (edited[0].permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)

    # Flow is estimated between the *unedited* frames: the edit changes colours
    # everywhere, which would confuse a photometric flow estimator.
    forward, backward = estimate_flow_pair(raft, source, target)
    reliable = consistency_mask(backward, forward)

    # The backward flow maps target pixels back into the source, which is the
    # direction needed to pull the edited source onto the target.
    warped = warp_by_flow(edited_np, backward)
    masked = (warped * reliable[..., None]).astype(np.uint8)

    as_uint8 = lambda t: t[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    outputs = {
        "1_source.png": as_uint8(source),
        "2_target.png": as_uint8(target),
        "3_source_edited.png": edited_np,
        "4_flow_forward.png": flow_to_image(forward),
        "5_flow_backward.png": flow_to_image(backward),
        "6_consistency_mask.png": (reliable * 255).astype(np.uint8),
        "7_warp_unmasked.png": warped.astype(np.uint8),
        "8_warp_masked.png": masked,
    }
    for name, array in outputs.items():
        Image.fromarray(array).save(os.path.join(output_dir, name))
    print(f"wrote {len(outputs)} images to {output_dir}")
    print(f"the mask keeps {reliable.mean() * 100:.1f}% of pixels; "
          "the rest is what the sliding window has to repaint")


if __name__ == "__main__":
    main()
