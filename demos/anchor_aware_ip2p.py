"""Edit a batch of frames consistently with anchor-aware InstructPix2Pix.

Component (1) of the framework.  Running the single-image editor on each frame
independently gives each one its own interpretation of the prompt; the
anchor-aware variant edits the whole batch in one pass so they share an
appearance.  Compare this against demos/single_view_ip2p.py on the same frames.

    python demos/anchor_aware_ip2p.py --image_dir examples/coffee_frame_2x/ \\
        --prompt 'What if it was painted by Van Gogh?' \\
        --sequence_length 6 --resize 1024 --steps 20 \\
        --guidance_scale 10.5 --image_guidance_scale 1.5
"""

import argparse
import os

import torch

from _common import (  # also puts the repository root on sys.path
    load_frames,
    output_name,
    prepare_output_dir,
    save_sheet,
    sorted_image_paths,
)
from instruct4d.ip2p import SequenceInstructPix2Pix


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image_dir", type=str, required=True, help="directory of frames")
    parser.add_argument("--prompt", type=str, required=True, help="editing instruction")
    parser.add_argument("--sequence_length", type=int, default=6, help="frames edited together")
    parser.add_argument("--resize", type=int, default=None, help="length of the longer side")
    parser.add_argument("--steps", type=int, default=20, help="denoising steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="text guidance")
    parser.add_argument("--image_guidance_scale", type=float, default=1.5, help="image guidance")
    parser.add_argument("--output_dir", type=str, default="./demo_output", help="where to write")
    parser.add_argument("--seed", type=int, default=17070, help="random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    output_dir = prepare_output_dir(args.output_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    paths = sorted_image_paths(args.image_dir)[: args.sequence_length]
    print(f"loaded {len(paths)} frames from {args.image_dir}")
    frames = load_frames(paths, args.resize, device)

    ip2p = SequenceInstructPix2Pix(device=device)
    edited = ip2p.edit_sequence(
        images=frames,
        # With nothing already edited to condition on, the frames condition on
        # themselves, exactly as stock InstructPix2Pix does.
        images_cond=frames,
        prompt=args.prompt,
        guidance_scale=args.guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
        diffusion_steps=args.steps,
        noisy_latent_type="noisy_latent",
    )

    path = save_sheet(edited, os.path.join(output_dir, output_name(args.prompt, "anchor_aware_ip2p")))
    print(f"wrote {path}")
    print("compare it against single_view_ip2p.py --image_dir on the same directory")


if __name__ == "__main__":
    main()
