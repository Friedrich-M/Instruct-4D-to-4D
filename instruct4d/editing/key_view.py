"""Key pseudo-view editing (Sections 3.2 and 3.4).

Editing starts at a single timestamp, the *key frame*.  Treating that timestamp
as a static 3D scene, the problem reduces to editing a set of views of one scene
consistently -- what the paper calls the pseudo-3D view.

Each warm-up round does three things:

1. Sample a handful of cameras and edit them together with anchor-aware
   InstructPix2Pix, so they already share an appearance.
2. Cross-warp those edited views into one another using the rendered depth and
   average the results, which reconciles whatever disagreement is left.
3. Warp the reconciled edit onto every remaining camera, blending rather than
   replacing so the field is nudged toward the edit instead of being snapped to
   it.

Rounds are annealed: early rounds inject a lot of noise and change the scene
boldly, later rounds inject less and mostly consolidate.  On the final round the
warped remaining views are refined once more by the diffusion model, which
repairs the regions no source view could explain.
"""

import math
import os
from typing import Optional, Sequence

import numpy as np
import torch

from ..ip2p import SequenceInstructPix2Pix
from ..rendering import render_rays
from .buffer import FrameBuffer
from .debug import DebugImageWriter
from .warping import apply_warp, project_points_to_view, unproject_depth_ndc

#: Rejection threshold when reconciling the jointly edited views.  They already
#: agree closely, so a tight threshold keeps only high-confidence matches.
SAMPLED_DIFF_THRESHOLD = 0.05

#: Rejection threshold when warping onto views that were not edited.  Tighter
#: still, because a wrong colour here is propagated into the field.
REMAINING_DIFF_THRESHOLD = 0.02

#: Noise level is annealed between these fractions of the full schedule across
#: the warm-up rounds.
ANNEAL_MIN = 0.8
ANNEAL_MAX = 1.0


class KeyFrameEditor:
    """Edits every camera of one timestamp into a mutually consistent state."""

    def __init__(
        self,
        ip2p: SequenceInstructPix2Pix,
        frames: FrameBuffer,
        originals: torch.Tensor,
        view_points: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        args,
        device,
        debug: Optional[DebugImageWriter] = None,
    ):
        """
        Args:
            ip2p: The anchor-aware editor.
            frames: The training colours, edited in place.
            originals: ``(frame, camera, H, W, 3)`` unedited images, used as the
                conditioning signal.
            view_points: ``(camera, H, W, 3)`` world-space points for the key
                frame, from :func:`compute_view_points`.
            intrinsics: ``(camera, 3, 3)`` intrinsics.
            extrinsics: ``(camera, 4, 4)`` world-to-camera matrices.
            args: Parsed configuration.
            device: Device the warping runs on.
            debug: Optional writer for intermediate visualisations.
        """
        self.ip2p = ip2p
        self.frames = frames
        self.originals = originals
        self.args = args
        self.device = device
        self.debug = debug or DebugImageWriter(None)

        # Every warp reads this geometry, and it never changes. Uploading it
        # once keeps the projection on the GPU and takes the per-pair host
        # transfers out of the inner loop entirely.
        self.view_points = view_points.to(device)
        self.intrinsics = intrinsics.to(device)
        self.extrinsics = extrinsics.to(device)

    def run(self, key_frame: int = 0, warp_ratio: float = 0.5, warm_up_steps: int = 12) -> None:
        """Edit the key frame in place.

        Args:
            key_frame: Which timestamp to edit.
            warp_ratio: How strongly the warped edit replaces a non-sampled
                view.  ``1.0`` overwrites it outright; the default blends half
                and half so the field is not jerked to the edit in one round.
            warm_up_steps: Number of annealed rounds.
        """
        print(f"editing key frame {key_frame}")
        camera_ids = list(range(self.frames.num_cameras))

        for step in range(warm_up_steps):
            is_final = step == warm_up_steps - 1
            sampled = sorted(
                np.random.choice(camera_ids, self.args.sequence_length, replace=False).tolist()
            )
            remaining = sorted(set(camera_ids) - set(sampled))

            sampled_images = self.frames.get(key_frame, sampled, self.device)
            sampled_cond = self._condition(key_frame, sampled)
            remaining_images = self.frames.get(key_frame, remaining, self.device)
            remaining_cond = self._condition(key_frame, remaining)

            self.debug.save(step, "1_sampled_input", sampled_images)

            edited = self._edit_sampled_views(sampled_images, sampled_cond, step, warm_up_steps)
            self._reconcile_sampled_views(edited, sampled)
            self.debug.save(step, "2_sampled_edited", edited)

            warped = self._warp_onto_remaining(edited, sampled, remaining_images, remaining, warp_ratio)
            self.debug.save(step, "3_remaining_warped", warped)

            if is_final:
                warped = self._refine_remaining(warped, remaining_cond)
                self.debug.save(step, "4_remaining_refined", warped)

            self.frames.set(key_frame, sampled, edited)
            self.frames.set(key_frame, remaining, warped)

        print("key frame editing done")

    # ------------------------------------------------------------------
    def _condition(self, key_frame: int, cameras: Sequence[int]) -> torch.Tensor:
        """The unedited images for ``cameras``, as ``(N, 3, H, W)`` on device."""
        images = self.originals[key_frame, cameras].to(self.device)
        return images.permute(0, 3, 1, 2)

    def _edit_sampled_views(
        self,
        images: torch.Tensor,
        images_cond: torch.Tensor,
        step: int,
        total_steps: int,
    ) -> torch.Tensor:
        """Edit the sampled cameras jointly, with a cosine-annealed noise level.

        The first round runs the full schedule and is free to change the scene
        substantially; later rounds run a shorter one and mostly reinforce what
        is already there.
        """
        cosine = 0.5 * (1 + math.cos(math.pi * step / total_steps))
        scale = ANNEAL_MIN + cosine * (ANNEAL_MAX - ANNEAL_MIN)

        edited = self.ip2p.edit_sequence(
            images=images,
            images_cond=images_cond,
            prompt=self.args.prompt,
            guidance_scale=self.args.guidance_scale,
            image_guidance_scale=self.args.image_guidance_scale,
            diffusion_steps=int(self.args.diffusion_steps * scale),
            noisy_latent_type="noisy_latent",
            T=int(1000 * scale),
        )
        return edited.to(self.device, dtype=torch.float32)

    def _reconcile_sampled_views(self, edited: torch.Tensor, sampled: Sequence[int]) -> None:
        """Average each sampled view with every other one warped into it.

        Editing a batch jointly gets the views close, but not identical.
        Averaging the mutually warped versions pulls them the rest of the way
        onto a single consistent appearance.  Modifies ``edited`` in place.
        """
        for target_pos, target_cam in enumerate(sampled):
            blended, weights = self._warp_average(edited, sampled, target_cam, SAMPLED_DIFF_THRESHOLD)
            covered = weights != 0
            blended[covered] /= weights[covered].unsqueeze(-1)
            edited[target_pos].permute(1, 2, 0)[covered] = blended[covered]

    def _warp_onto_remaining(
        self,
        edited: torch.Tensor,
        sampled: Sequence[int],
        remaining_images: torch.Tensor,
        remaining: Sequence[int],
        warp_ratio: float,
    ) -> torch.Tensor:
        """Blend the edit into the cameras that were not edited directly."""
        warped_all = remaining_images.clone()
        for target_pos, target_cam in enumerate(remaining):
            blended, weights = self._warp_average(
                edited, sampled, target_cam, REMAINING_DIFF_THRESHOLD
            )
            covered = weights != 0
            blended[covered] /= weights[covered].unsqueeze(-1)

            target = warped_all[target_pos].permute(1, 2, 0)
            current = remaining_images[target_pos].permute(1, 2, 0)
            target[covered] = blended[covered] * warp_ratio + current[covered] * (1 - warp_ratio)
        return warped_all

    def _warp_average(
        self,
        source_images: torch.Tensor,
        source_cameras: Sequence[int],
        target_camera: int,
        diff_thres: float,
    ):
        """Warp every source view into ``target_camera`` and accumulate.

        Each source contributes in proportion to how much of the target it can
        explain, so a view that mostly misses the target barely counts.  Returns
        the unnormalised sum and the accumulated weights; the caller divides.
        """
        height, width = self.frames.height, self.frames.width
        accumulator = torch.zeros((height, width, 3), dtype=torch.float32, device=self.device)
        weights = torch.zeros((height, width), dtype=torch.float32, device=self.device)

        target_points = self.view_points[target_camera]
        for source_pos, source_camera in enumerate(source_cameras):
            pixel_map = project_points_to_view(
                target_points,
                self.intrinsics[source_camera],
                self.extrinsics[source_camera],
            )
            source_points = self.view_points[source_camera]
            source_image = source_images[source_pos].permute(1, 2, 0).float()

            warped, mask, _ = apply_warp(
                pixel_map,
                torch.zeros_like(source_image),
                source_image,
                target_points,
                source_points,
                diff_thres=diff_thres,
            )
            coverage = (mask != 0).sum() / mask.numel()
            accumulator[mask] += warped[mask] * coverage
            weights[mask] += coverage

        return accumulator, weights

    def _refine_remaining(self, warped: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Repaint the warped views in windows, each anchored on its first view.

        Warping leaves holes wherever no source camera could see the surface.
        A short, low-noise diffusion pass fills them while the anchor keeps the
        repaint consistent with the rest of the window.
        """
        refined = warped.clone()
        window = self.args.sequence_length
        for start in range(0, refined.shape[0], window):
            end = min(start + window, refined.shape[0])
            # The window's own first view doubles as its anchor, so index 0 of
            # the batch is the anchor and its output is discarded.
            batch = torch.cat([refined[start : start + 1], refined[start:end]], dim=0)
            batch_cond = torch.cat([cond[start : start + 1], cond[start:end]], dim=0)

            edited = self.ip2p.edit_sequence(
                images=batch,
                images_cond=batch_cond,
                prompt=self.args.prompt,
                guidance_scale=self.args.guidance_scale,
                image_guidance_scale=self.args.image_guidance_scale,
                diffusion_steps=self.args.restview_refine_diffusion_steps,
                noisy_latent_type="noisy_latent",
                T=self.args.restview_refine_num_steps,
            ).to(self.device, dtype=torch.float32)
            refined[start:end] = edited[1:]
        return refined


@torch.no_grad()
def compute_view_points(
    field,
    all_rays: torch.Tensor,
    dataset,
    key_frame: int,
    num_frames: int,
    num_cameras: int,
    args,
    device,
    cache_dir: Optional[str] = None,
) -> torch.Tensor:
    """Render the key frame's depth and unproject it to world-space points.

    Every warp in :class:`KeyFrameEditor` needs to know where each pixel of each
    camera sits in 3D.  That only depends on the trained geometry, so it is
    computed once and cached: re-rendering depth for every camera costs minutes.

    Args:
        field: The trained radiance field.
        all_rays: ``(N, 7)`` training rays, laid out frame-major.
        dataset: Supplies ``img_wh`` and ``focal``.
        key_frame: Which timestamp to render.
        num_frames: Length of the time axis.
        num_cameras: Number of cameras.
        args: Parsed configuration; supplies ``ndc_ray``.
        device: Device to render on.
        cache_dir: If given, points are written here and reused next run.

    Returns:
        ``(num_cameras, H, W, 3)`` world-space points, on the CPU.
    """
    width, height = dataset.img_wh
    cache_path = os.path.join(cache_dir, "view_points.pt") if cache_dir else None

    if cache_path and os.path.exists(cache_path):
        print(f"loaded cached view points from {cache_path}")
        return torch.load(cache_path).cpu()

    rays_by_view = all_rays.view(num_frames, num_cameras, height * width, -1)[key_frame]
    points = []
    for camera in range(num_cameras):
        rays = rays_by_view[camera]
        _, depth = render_rays(
            rays.to(device),
            field,
            chunk=2048,
            N_samples=-1,
            ndc_ray=args.ndc_ray,
            white_bg=dataset.white_bg,
            device=device,
        )
        points.append(
            unproject_depth_ndc(
                rays.cpu(), depth.view(-1, 1).cpu(), height, width, dataset.focal
            )
        )
        torch.cuda.empty_cache()

    points = torch.stack(points, dim=0)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(points, cache_path)
        print(f"cached view points to {cache_path}")
    return points
