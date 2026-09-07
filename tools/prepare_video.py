"""Explode a DyNeRF scene's per-camera videos into image frames.

The dataset ships one ``.mp4`` per camera; the loader wants individual frames.
Run this once per scene, before training:

    python tools/prepare_video.py data/neural_3d/coffee_martini

Given a scene directory of::

    coffee_martini/
        cam00.mp4
        cam01.mp4
        poses_bounds.npy

it produces::

    coffee_martini/
        frames/
            cam00/000001.jpg ...
            cam01/000001.jpg ...

Requires ``ffmpeg`` on the PATH.  Cameras whose frames already exist are
skipped, so an interrupted run can simply be repeated.
"""

import argparse
import os
import shutil
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir", type=str, help="scene directory containing the .mp4 files")
    parser.add_argument(
        "--overwrite", action="store_true", help="re-extract cameras that already have frames"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg was not found on the PATH; install it and try again")

    cameras = sorted(f[: -len(".mp4")] for f in os.listdir(args.data_dir) if f.endswith(".mp4"))
    if not cameras:
        sys.exit(f"no .mp4 files found in {args.data_dir}")

    frames_dir = os.path.join(args.data_dir, "frames")
    for camera in cameras:
        camera_dir = os.path.join(frames_dir, camera)
        if os.path.isdir(camera_dir) and os.listdir(camera_dir) and not args.overwrite:
            print(f"{camera}: already extracted, skipping")
            continue

        os.makedirs(camera_dir, exist_ok=True)
        print(f"{camera}: extracting frames")
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", os.path.join(args.data_dir, f"{camera}.mp4"),
                os.path.join(camera_dir, "%06d.jpg"),
            ],
            check=True,
        )

    print(f"extracted {len(cameras)} cameras into {frames_dir}")


if __name__ == "__main__":
    main()
