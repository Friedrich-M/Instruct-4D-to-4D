"""Loader for the Neural 3D Video (DyNeRF) multi-view dataset.

Each scene is a set of synchronised, forward-facing videos plus an LLFF-style
``poses_bounds.npy``.  Run ``tools/prepare_video.py`` first to explode the
``.mp4`` files into ``frames/<camera>/<index>.jpg``.

Every ray is stored as a 7-vector ``[origin(3), direction(3), time]``, where
``time`` is the frame index normalised to ``[0, 1]``.  The radiance field reads
that last channel to index its per-frame feature planes, which is what makes the
representation 4D.

Rays and colours are laid out frame-major::

    index = frame * (num_cameras * H * W) + camera * (H * W) + pixel

The editing pipeline relies on that ordering when it reshapes the flat buffers
back into ``(frame, camera, H, W, 3)`` images.
"""

import os
from typing import Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from tqdm import tqdm

from .pose_utils import center_poses, get_spiral
from .ray_utils import get_ray_directions, get_rays, ndc_rays_blender

#: In DyNeRF's ``coffee_martini`` one camera is out of sync with the rest, so
#: the official evaluation protocol drops it.  Keyed by scene directory name.
UNSYNCHRONISED_CAMERAS = {"coffee_martini": 12}

#: Held-out camera for the test split, matching the DyNeRF protocol.
TEST_CAMERA = 0

#: Scene bounds in NDC.  The NDC cube is ``[-1, 1]^3`` by construction; the
#: ``x``/``y`` extents are widened to the 16:9-ish aspect of these scenes.
SCENE_BBOX = torch.tensor([[-1.5, -1.67, -1.0], [1.5, 1.67, 1.0]])

#: Near/far in NDC, where the far plane maps to exactly 1.
NEAR_FAR = [0.0, 1.0]


class N3DVDynamicDataset(Dataset):
    """A DyNeRF scene as a flat bundle of space-time rays.

    Attributes:
        all_rays: ``(N, 7)`` rays, or ``(num_images, H * W, 7)`` when stacked.
        all_rgbs: ``(N, 3)`` colours, or ``(num_images, H, W, 3)`` when stacked.
        poses: ``(num_cameras, 3, 4)`` recentred camera-to-world matrices.
        intrinsics: ``(num_cameras, 3, 3)`` pinhole intrinsics.
        extrinsics: ``(num_cameras, 4, 4)`` world-to-camera matrices.  Used by
            the pseudo-view propagation step to reproject between views.
        render_path: ``(num_frames, 4, 4)`` spiral path for fly-through videos.
        directions: ``(H, W, 3)`` camera-frame ray directions, shared by all
            cameras since they share intrinsics.
    """

    def __init__(
        self,
        datadir: str,
        split: str = "train",
        downsample: float = 4,
        is_stack: bool = False,
        num_frames: int = 30,
        frame_list: Optional[Sequence[int]] = None,
    ):
        """
        Args:
            datadir: Scene directory containing the ``.mp4`` files, the exploded
                ``frames/`` directory and ``poses_bounds.npy``.
            split: ``"train"`` loads every camera; anything else loads only
                :data:`TEST_CAMERA`.
            downsample: Integer factor by which to shrink the images.  Resized
                copies are cached under ``frames<factor>x/`` inside ``datadir``.
            is_stack: Keep one entry per image instead of flattening to pixels.
                Training wants the flat layout, evaluation the stacked one.
            num_frames: Length of the time axis; sets the ``[0, 1]`` time scale.
            frame_list: Explicit frame indices to load.  Defaults to the first
                ``num_frames`` frames.
        """
        self.root_dir = datadir
        self.split = split
        self.num_frames = num_frames
        self.frame_list = list(frame_list) if frame_list else list(range(num_frames))
        self.is_stack = is_stack
        self.downsample = downsample
        self.transform = T.ToTensor()

        self.white_bg = False
        self.near_far = list(NEAR_FAR)
        self.scene_bbox = SCENE_BBOX.clone()

        self.read_meta()

    def read_meta(self) -> None:
        """Load poses and images, and turn them into space-time rays."""
        cameras, poses, near_fars = self._read_poses()
        self._setup_intrinsics(poses)
        self._build_rays(cameras, near_fars)

    def _read_poses(self):
        """Read ``poses_bounds.npy`` and bring the poses into the NDC frame."""
        cameras = sorted(f[: -len(".mp4")] for f in os.listdir(self.root_dir) if f.endswith(".mp4"))
        poses_bounds = np.load(os.path.join(self.root_dir, "poses_bounds.npy"))

        scene = os.path.basename(os.path.normpath(self.root_dir))
        for known_scene, drop in UNSYNCHRONISED_CAMERAS.items():
            if known_scene in scene:
                print(f"[{scene}] dropping unsynchronised camera {drop}")
                poses_bounds = np.delete(poses_bounds, drop, axis=0)
                cameras.pop(drop)
                break

        if poses_bounds.shape[0] != len(cameras):
            raise ValueError(
                f"{self.root_dir}: poses_bounds.npy has {poses_bounds.shape[0]} entries "
                f"but {len(cameras)} videos were found"
            )

        # Columns 0..14 are a (3, 5) matrix per camera: a 3x4 pose plus a column
        # holding (height, width, focal).  The last two columns are near/far.
        poses = poses_bounds[:, :15].reshape(-1, 3, 5)
        near_fars = poses_bounds[:, -2:]

        self._original_hw_focal = poses[0, :, -1]

        # LLFF stores rotations as "down right back"; NeRF expects "right up
        # back".  See https://github.com/bmild/nerf/issues/34.
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], axis=-1)
        poses, _ = center_poses(poses)

        # Rescale so the nearest surface sits slightly beyond depth 1, which is
        # where the NDC projection is best conditioned.
        scale_factor = near_fars.min() * 1.8 * 0.6
        near_fars = near_fars / scale_factor
        poses[..., 3] /= scale_factor

        self.poses = poses
        return cameras, poses, near_fars

    def _setup_intrinsics(self, poses: np.ndarray) -> None:
        """Derive resolution, intrinsics, extrinsics and the render path."""
        height, width, focal = self._original_hw_focal
        self.img_wh = np.array([int(width / self.downsample), int(height / self.downsample)])
        self.focal = [focal * self.img_wh[0] / width, focal * self.img_wh[1] / height]

        num_cameras = len(poses)
        intrinsics = torch.zeros((num_cameras, 3, 3), dtype=torch.float32)
        intrinsics[:, 0, 0] = float(self.focal[0])
        intrinsics[:, 1, 1] = float(self.focal[1])
        intrinsics[:, 0, 2] = self.img_wh[0] * 0.5
        intrinsics[:, 1, 2] = self.img_wh[1] * 0.5
        intrinsics[:, 2, 2] = 1.0
        self.intrinsics = intrinsics

        c2w = torch.from_numpy(poses).float()
        bottom = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).expand(num_cameras, 1, 4)
        self.extrinsics = torch.inverse(torch.cat([c2w, bottom], dim=1))

        width_px, height_px = self.img_wh
        self.directions = get_ray_directions(height_px, width_px, self.focal)

    def _build_rays(self, cameras: Sequence[str], near_fars: np.ndarray) -> None:
        """Load every (frame, camera) image and emit its rays."""
        self.render_path = get_spiral(
            self.poses, near_fars, rads_scale=0.4, zrate=-0.5, n_views=self.num_frames
        )

        width, height = self.img_wh
        camera_ids = list(range(len(self.poses))) if self.split == "train" else [TEST_CAMERA]

        all_rays, all_rgbs = [], []
        for time_index, frame in enumerate(tqdm(self.frame_list, desc="frame")):
            for camera_id in tqdm(camera_ids, desc="camera", leave=False):
                image = self._load_image(cameras[camera_id], frame)
                all_rgbs.append(image.view(3, -1).permute(1, 0))

                c2w = torch.FloatTensor(self.poses[camera_id])
                rays_o, rays_d = get_rays(self.directions, c2w)
                rays_o, rays_d = ndc_rays_blender(height, width, self.focal[0], 1.0, rays_o, rays_d)

                # Normalised timestamp; a single-frame scene is pinned at 0.
                t = time_index / (self.num_frames - 1) if self.num_frames > 1 else 0.0
                time_channel = torch.full([*rays_o.shape[:-1], 1], t)
                all_rays.append(torch.cat([rays_o, rays_d, time_channel], dim=1))

        if self.is_stack:
            self.all_rays = torch.stack(all_rays, 0)
            self.all_rgbs = torch.stack(all_rgbs, 0).reshape(-1, height, width, 3)
        else:
            self.all_rays = torch.cat(all_rays, 0)
            self.all_rgbs = torch.cat(all_rgbs, 0)

    def _load_image(self, camera: str, frame: int) -> torch.Tensor:
        """Load one frame, caching a downsampled copy next to the originals."""
        name = f"{camera}/{frame + 1:06d}.jpg"
        cache_dir = f"{self.root_dir}/frames{int(self.downsample):d}x"
        cached = f"{cache_dir}/{name}"

        if os.path.isfile(cached):
            image = Image.open(cached).convert("RGB")
        else:
            image = Image.open(f"{self.root_dir}/frames/{name}").convert("RGB")
            image = image.resize(self.img_wh, Image.LANCZOS)
            os.makedirs(f"{cache_dir}/{camera}", exist_ok=True)
            image.save(cached)
        return self.transform(image)

    def __len__(self) -> int:
        return len(self.all_rgbs)

    def __getitem__(self, idx: int) -> dict:
        return {"rays": self.all_rays[idx], "rgbs": self.all_rgbs[idx]}
