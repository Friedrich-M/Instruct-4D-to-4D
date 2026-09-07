"""Edit images one at a time with stock InstructPix2Pix.

This is the 2D editor the paper builds on, with none of the 4D machinery. It has
two uses.

Given ``--image_path`` it edits one frame, which is the first thing to try when
an edit is not coming out as expected: if InstructPix2Pix cannot produce the
result on a single frame, no amount of 4D consistency will recover it.

Given ``--image_dir`` it edits every frame *independently* and writes them as one
contact sheet. Run ``anchor_aware_ip2p.py`` on the same directory and compare the
two sheets: this one gives each frame its own interpretation of the prompt, which
is the inconsistency the anchor-aware editor removes.

    python demos/single_view_ip2p.py --image_dir examples/coffee_frame_2x/ \\
        --prompt 'What if it was painted by Van Gogh?' --resize 1024
"""

import argparse
import os

import torch
from PIL import Image

from diffusers import StableDiffusionInstructPix2PixPipeline

from _common import (
    fit_to_multiple,
    load_frames,
    output_name,
    prepare_output_dir,
    save_sheet,
    sorted_image_paths,
)

IP2P_SOURCE = "timbrooks/instruct-pix2pix"


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image_path", type=str, help="a single image to edit")
    source.add_argument(
        "--image_dir", type=str, help="a directory of frames, each edited independently"
    )
    parser.add_argument("--prompt", type=str, required=True, help="editing instruction")
    parser.add_argument("--sequence_length", type=int, default=6, help="frames to take from --image_dir")
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

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        IP2P_SOURCE, torch_dtype=torch.float16, safety_checker=None
    ).to(device)
    pipe.enable_attention_slicing()

    def edit(image: Image.Image) -> Image.Image:
        return pipe(
            args.prompt,
            image=fit_to_multiple(image, args.resize),
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            image_guidance_scale=args.image_guidance_scale,
        ).images[0]

    if args.image_path:
        original = Image.open(args.image_path).convert("RGB")
        # The model works at its own resolution; put the result back.
        edited = edit(original).resize(original.size, Image.Resampling.LANCZOS)
        path = os.path.join(output_dir, output_name(args.prompt, "ip2p"))
        edited.save(path)
        print(f"wrote {path}")
        return

    paths = sorted_image_paths(args.image_dir)[: args.sequence_length]
    print(f"editing {len(paths)} frames from {args.image_dir}, one at a time")

    # Every frame is a separate diffusion run with a separate noise draw, which
    # is exactly why their appearances disagree.
    edited = [edit(Image.open(p).convert("RGB")) for p in paths]

    sheet_dir = prepare_output_dir(os.path.join(output_dir, "frames"))
    for path, image in zip(paths, edited):
        image.save(os.path.join(sheet_dir, os.path.basename(path)))

    sheet = load_frames(sorted_image_paths(sheet_dir), args.resize, "cpu")
    path = save_sheet(sheet, os.path.join(output_dir, output_name(args.prompt, "ip2p_independent")))
    print(f"wrote {path}")
    print("compare it against anchor_aware_ip2p.py on the same directory")


if __name__ == "__main__":
    main()
