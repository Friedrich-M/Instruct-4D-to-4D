"""Propagate an edit along a sequence with a flow-guided sliding window.

Component (2) end to end.  The first window is edited outright.  Every window
after it is initialised by warping the previous window's overlapping frame
forward with RAFT flow, and then repainted by anchor-aware InstructPix2Pix with
a short, low-noise schedule -- just enough to fill what the warp could not
explain, and not enough to invent a new appearance.

    python demos/sliding_window.py --image_dir examples/coffee_frame_2x/ \\
        --prompt 'What if it was painted by Van Gogh?' \\
        --sequence_length 6 --resize 1024 \\
        --guidance_scale 10.5 --image_guidance_scale 1.5 \\
        --painting_diffusion_steps 5 --painting_num_train_timesteps 600
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from _common import (  # also puts the repository root on sys.path
    load_frames,
    prepare_output_dir,
    sorted_image_paths,
)
from instruct4d.flow import (
    blend_with_mask,
    consistency_mask,
    estimate_flow_pair,
    load_raft,
    warp_by_flow,
)
from instruct4d.ip2p import SequenceInstructPix2Pix


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image_dir", type=str, required=True, help="directory of frames")
    parser.add_argument("--prompt", type=str, required=True, help="editing instruction")
    parser.add_argument("--sequence_length", type=int, default=5, help="frames per window")
    parser.add_argument("--overlap_length", type=int, default=1, help="frames shared between windows")
    parser.add_argument("--resize", type=int, default=1024, help="length of the longer side")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="text guidance")
    parser.add_argument("--image_guidance_scale", type=float, default=1.5, help="image guidance")
    parser.add_argument("--steps", type=int, default=20, help="denoising steps for the first window")
    parser.add_argument(
        "--painting_diffusion_steps", type=int, default=5, help="denoising steps when repainting"
    )
    parser.add_argument(
        "--painting_num_train_timesteps", type=int, default=600,
        help="noise level when repainting; lower preserves more of the warp",
    )
    parser.add_argument("--raft_ckpt", type=str, default="./weights/raft-things.pth")
    parser.add_argument("--output_dir", type=str, default="./demo_output/sliding_window")
    parser.add_argument("--seed", type=int, default=7070)
    return parser.parse_args()


def warp_window(raft, window, conditions, overlap, device):
    """Warp the window's first frame onto every later frame in the window."""
    warped = window.clone()
    reference = (window[:1] * 255.0).float()
    reference_cond = (conditions[:1] * 255.0).float()

    for i in range(overlap, len(window)):
        current = (window[i : i + 1] * 255.0).float()
        current_cond = (conditions[i : i + 1] * 255.0).float()

        forward, backward = estimate_flow_pair(raft, reference_cond, current_cond)
        reliable = consistency_mask(backward, forward)

        ref_np = reference[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        cur_np = current[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        blended = blend_with_mask(warp_by_flow(ref_np, backward), cur_np, reliable)

        frame = torch.from_numpy(blended / 255.0).permute(2, 0, 1).to(window)
        if frame.shape[-2:] != window.shape[-2:]:
            frame = F.interpolate(
                frame[None], size=window.shape[-2:], mode="bilinear", align_corners=False
            )[0]
        warped[i] = frame
    return warped


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    output_dir = prepare_output_dir(args.output_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    paths = sorted_image_paths(args.image_dir)
    print(f"loaded {len(paths)} frames from {args.image_dir}")
    frames = load_frames(paths, args.resize, device)
    conditions = frames.clone()

    ip2p = SequenceInstructPix2Pix(device=device)
    raft = load_raft(args.raft_ckpt, device)

    stride = args.sequence_length - args.overlap_length
    for batch_index, start in enumerate(range(0, len(frames), stride)):
        end = start + args.sequence_length
        if end > len(frames):
            break
        window = frames[start:end]
        window_cond = conditions[start:end]

        if batch_index > 0:
            window = warp_window(raft, window, window_cond, args.overlap_length, device)
            torchvision.utils.save_image(
                window.float(), f"{output_dir}/window{batch_index:02d}_warped.png",
                nrow=args.sequence_length, padding=0,
            )

        # The first window has nothing to inherit, so it gets the full schedule;
        # later windows only need to repair the warp.
        first = batch_index == 0
        edited = ip2p.edit_sequence(
            images=window,
            images_cond=window_cond,
            prompt=args.prompt,
            guidance_scale=args.guidance_scale,
            image_guidance_scale=args.image_guidance_scale,
            diffusion_steps=args.steps if first else args.painting_diffusion_steps,
            noisy_latent_type="noisy_latent",
            T=1000 if first else args.painting_num_train_timesteps,
        )
        stage = "edited" if first else "painted"
        torchvision.utils.save_image(
            edited.float(), f"{output_dir}/window{batch_index:02d}_{stage}.png",
            nrow=args.sequence_length, padding=0,
        )

        # Carry the overlapping tail into the next window.
        frames[end - args.overlap_length : end] = edited[-args.overlap_length :].to(frames)

    print(f"wrote per-window images to {output_dir}")


if __name__ == "__main__":
    main()
