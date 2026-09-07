"""Base class for the streaming TensoRF 4D radiance field.

The scene is a tensor-decomposed voxel grid, as in TensoRF, extended along time
following the streaming formulation of NeRFPlayer: the feature channels of each
plane/line are treated as a rolling buffer, and a frame's features are read by
selecting and blending a slice of that buffer.  ``ld_per_frame`` controls how
many new channels each frame introduces, which trades memory for the amount of
change a scene can express over time.

The pieces that vary between decompositions -- how the volume is stored, how a
feature is looked up, how the grid is upsampled -- are left abstract here and
implemented in :mod:`instruct4d.fields.tensorf`.

Method and attribute names follow the original TensoRF implementation so the two
can be read side by side.
"""

import time
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .sh import eval_sh_bases


def positional_encoding(positions: torch.Tensor, freqs: int) -> torch.Tensor:
    """Standard NeRF sinusoidal encoding.

    Args:
        positions: ``(..., D)`` values to encode.
        freqs: Number of octaves.

    Returns:
        ``(..., 2 * freqs * D)`` encoding.
    """
    freq_bands = (2 ** torch.arange(freqs).float()).to(positions.device)
    pts = (positions[..., None] * freq_bands).reshape(
        positions.shape[:-1] + (freqs * positions.shape[-1],)
    )
    return torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)


def raw2alpha(
    sigma: torch.Tensor, dist: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Turn densities into per-sample alphas and compositing weights.

    Args:
        sigma: ``(N_rays, N_samples)`` densities.
        dist: ``(N_rays, N_samples)`` distances between consecutive samples.

    Returns:
        ``(alpha, weights, transmittance_to_background)``.
    """
    alpha = 1.0 - torch.exp(-sigma * dist)
    # Transmittance is the running product of "not yet absorbed"; the leading
    # 1 makes the first sample fully visible.
    transmittance = torch.cumprod(
        torch.cat([torch.ones(alpha.shape[0], 1).to(alpha.device), 1.0 - alpha + 1e-10], dim=-1),
        dim=-1,
    )
    weights = alpha * transmittance[:, :-1]
    return alpha, weights, transmittance[:, -1:]


def SHRender(xyz_sampled, viewdirs, features):
    """Decode appearance features as spherical-harmonic RGB coefficients."""
    sh_mult = eval_sh_bases(2, viewdirs)[:, None]
    rgb_sh = features.view(-1, 3, sh_mult.shape[-1])
    return torch.relu(torch.sum(sh_mult * rgb_sh, dim=-1) + 0.5)


def RGBRender(xyz_sampled, viewdirs, features):
    """Use the appearance features directly as RGB."""
    return features


class AlphaGridMask(nn.Module):
    """A binary occupancy grid used to skip empty space during ray marching."""

    def __init__(self, device, aabb: torch.Tensor, alpha_volume: torch.Tensor):
        super().__init__()
        self.device = device
        self.aabb = aabb.to(device)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invgridSize = 1.0 / self.aabbSize * 2
        self.alpha_volume = alpha_volume.view(1, 1, *alpha_volume.shape[-3:])
        self.gridSize = torch.LongTensor(
            [alpha_volume.shape[-1], alpha_volume.shape[-2], alpha_volume.shape[-3]]
        ).to(device)

    def sample_alpha(self, xyz_sampled: torch.Tensor) -> torch.Tensor:
        """Trilinearly sample the occupancy grid at world-space points."""
        xyz_sampled = self.normalize_coord(xyz_sampled)
        return F.grid_sample(
            self.alpha_volume, xyz_sampled.view(1, -1, 1, 1, 3), align_corners=True
        ).view(-1)

    def normalize_coord(self, xyz_sampled: torch.Tensor) -> torch.Tensor:
        """Map world-space points into the grid's ``[-1, 1]`` sampling space."""
        return (xyz_sampled - self.aabb[0]) * self.invgridSize - 1


class MLPRender_Fea(nn.Module):
    """Shading MLP conditioned on the appearance feature and the view direction.

    The feature itself is positionally encoded, which lets a low-dimensional
    feature express high-frequency appearance.
    """

    def __init__(self, in_channels: int, viewpe: int = 6, feape: int = 6, featureC: int = 128):
        super().__init__()
        self.viewpe = viewpe
        self.feape = feape
        self.in_mlpC = 2 * viewpe * 3 + 2 * feape * in_channels + 3 + in_channels
        self.mlp = nn.Sequential(
            nn.Linear(self.in_mlpC, featureC),
            nn.ReLU(inplace=True),
            nn.Linear(featureC, featureC),
            nn.ReLU(inplace=True),
            nn.Linear(featureC, 3),
        )
        nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.feape > 0:
            indata.append(positional_encoding(features, self.feape))
        if self.viewpe > 0:
            indata.append(positional_encoding(viewdirs, self.viewpe))
        return torch.sigmoid(self.mlp(torch.cat(indata, dim=-1)))


class MLPRender_PE(nn.Module):
    """Shading MLP conditioned on the encoded *position* and view direction."""

    def __init__(self, in_channels: int, viewpe: int = 6, pospe: int = 6, featureC: int = 128):
        super().__init__()
        self.viewpe = viewpe
        self.pospe = pospe
        self.in_mlpC = (3 + 2 * viewpe * 3) + (3 + 2 * pospe * 3) + in_channels
        self.mlp = nn.Sequential(
            nn.Linear(self.in_mlpC, featureC),
            nn.ReLU(inplace=True),
            nn.Linear(featureC, featureC),
            nn.ReLU(inplace=True),
            nn.Linear(featureC, 3),
        )
        nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.pospe > 0:
            indata.append(positional_encoding(pts, self.pospe))
        if self.viewpe > 0:
            indata.append(positional_encoding(viewdirs, self.viewpe))
        return torch.sigmoid(self.mlp(torch.cat(indata, dim=-1)))


class MLPRender(nn.Module):
    """Shading MLP conditioned on the view direction only."""

    def __init__(self, in_channels: int, viewpe: int = 6, featureC: int = 128):
        super().__init__()
        self.viewpe = viewpe
        self.in_mlpC = (3 + 2 * viewpe * 3) + in_channels
        self.mlp = nn.Sequential(
            nn.Linear(self.in_mlpC, featureC),
            nn.ReLU(inplace=True),
            nn.Linear(featureC, featureC),
            nn.ReLU(inplace=True),
            nn.Linear(featureC, 3),
        )
        nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.viewpe > 0:
            indata.append(positional_encoding(viewdirs, self.viewpe))
        return torch.sigmoid(self.mlp(torch.cat(indata, dim=-1)))


#: Selectable appearance decoders, keyed by the ``--shadingMode`` flag.
SHADING_MODES = ("MLP_PE", "MLP_Fea", "MLP", "SH", "RGB")


class StreamTensorBase(nn.Module):
    """Shared machinery for the streaming tensor-decomposed radiance fields.

    Subclasses supply the actual factorisation by implementing
    :meth:`init_svd_volume`, :meth:`compute_densityfeature`,
    :meth:`compute_appfeature`, :meth:`get_optparam_groups`,
    :meth:`upsample_volume_grid` and :meth:`shrink`.
    """

    def __init__(
        self,
        aabb: torch.Tensor,
        gridSize: Sequence[int],
        device,
        density_n_comp: int = 8,
        appearance_n_comp: int = 24,
        app_dim: int = 27,
        shadingMode: str = "MLP_PE",
        alphaMask: Optional[AlphaGridMask] = None,
        near_far: Sequence[float] = (2.0, 6.0),
        density_shift: float = -10,
        alphaMask_thres: float = 0.001,
        distance_scale: float = 25,
        rayMarch_weight_thres: float = 0.0001,
        pos_pe: int = 6,
        view_pe: int = 6,
        fea_pe: int = 6,
        featureC: int = 128,
        step_ratio: float = 2.0,
        fea2denseAct: str = "softplus",
        num_frames: int = 1,
        ld_per_frame: float = 1,
    ):
        """
        Args:
            aabb: ``(2, 3)`` scene bounding box.
            gridSize: ``[nx, ny, nz]`` voxel resolution.
            device: Device the field lives on.
            density_n_comp: Per-axis component counts for the density tensor.
            appearance_n_comp: Per-axis component counts for the appearance tensor.
            app_dim: Width of the appearance feature handed to the shading MLP.
            shadingMode: One of :data:`SHADING_MODES`.
            alphaMask: Optional pre-computed occupancy grid.
            near_far: Ray bounds, in NDC for forward-facing scenes.
            density_shift: Bias added before the softplus, so a zero feature maps
                to zero density.
            alphaMask_thres: Occupancy threshold when building the alpha mask.
            distance_scale: Multiplier on sample spacing, absorbing the scene's
                arbitrary world scale into the density units.
            rayMarch_weight_thres: Samples with less compositing weight than this
                skip the (expensive) appearance branch entirely.
            pos_pe: Positional-encoding octaves for position.
            view_pe: Positional-encoding octaves for view direction.
            fea_pe: Positional-encoding octaves for the appearance feature.
            featureC: Hidden width of the shading MLP.
            step_ratio: Samples per voxel along a ray.
            fea2denseAct: ``"softplus"`` or ``"relu"``.
            num_frames: Length of the time axis.
            ld_per_frame: New feature channels introduced per frame.  Values
                above 1 must be integral and are folded into ``level_dim_multi``.
        """
        super().__init__()

        self.density_n_comp = density_n_comp
        self.app_n_comp = appearance_n_comp
        self.app_dim = app_dim
        self.aabb = aabb
        self.alphaMask = alphaMask
        self.device = device
        self.num_frames = num_frames

        self.density_shift = density_shift
        self.alphaMask_thres = alphaMask_thres
        self.distance_scale = distance_scale
        self.rayMarch_weight_thres = rayMarch_weight_thres
        self.fea2denseAct = fea2denseAct

        self.near_far = near_far
        self.step_ratio = step_ratio

        self.update_stepSize(gridSize)

        # Which pair of axes each of the three planes spans, and which axis the
        # matching line runs along.
        self.matMode = [[0, 1], [0, 2], [1, 2]]
        self.vecMode = [2, 1, 0]
        self.comp_w = [1, 1, 1]

        self.ld_per_frame = ld_per_frame
        self.level_dim_multi = 1
        if self.ld_per_frame > 1:
            if self.ld_per_frame % 1 != 0:
                raise ValueError(f"ld_per_frame above 1 must be integral, got {ld_per_frame}")
            self.level_dim_multi = int(self.ld_per_frame * 2)
            for k in [*self.density_n_comp, *self.app_n_comp]:
                if k % self.level_dim_multi != 0:
                    raise ValueError(
                        f"component count {k} must be divisible by level_dim_multi "
                        f"{self.level_dim_multi}"
                    )
            self.ld_per_frame = 0.5
        # How many frames share a channel before a new one is introduced.
        self.new_dim_interval = int(1 / self.ld_per_frame)
        self.density_list = [self.register_frame_permute_index(k) for k in self.density_n_comp]
        self.app_list = [self.register_frame_permute_index(k) for k in self.app_n_comp]

        self.init_svd_volume(gridSize[0], device)

        self.shadingMode = shadingMode
        self.pos_pe, self.view_pe, self.fea_pe, self.featureC = pos_pe, view_pe, fea_pe, featureC
        self.init_render_func(shadingMode, pos_pe, view_pe, fea_pe, featureC, device)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def init_render_func(self, shadingMode, pos_pe, view_pe, fea_pe, featureC, device):
        """Build the appearance decoder selected by ``shadingMode``."""
        if shadingMode == "MLP_PE":
            self.renderModule = MLPRender_PE(self.app_dim, view_pe, pos_pe, featureC).to(device)
        elif shadingMode == "MLP_Fea":
            self.renderModule = MLPRender_Fea(self.app_dim, view_pe, fea_pe, featureC).to(device)
        elif shadingMode == "MLP":
            self.renderModule = MLPRender(self.app_dim, view_pe, featureC).to(device)
        elif shadingMode == "SH":
            self.renderModule = SHRender
        elif shadingMode == "RGB":
            if self.app_dim != 3:
                raise ValueError("shadingMode 'RGB' requires app_dim == 3")
            self.renderModule = RGBRender
        else:
            raise ValueError(f"shadingMode must be one of {SHADING_MODES}, got {shadingMode!r}")

    def update_stepSize(self, gridSize: Sequence[int]) -> None:
        """Recompute ray-marching step size after a resolution change."""
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invaabbSize = 2.0 / self.aabbSize
        self.gridSize = torch.LongTensor(gridSize).to(self.device)
        self.units = self.aabbSize / (self.gridSize - 1)
        self.stepSize = torch.mean(self.units) * self.step_ratio
        self.aabbDiag = torch.sqrt(torch.sum(torch.square(self.aabbSize)))
        # Enough steps to cross the volume along its longest diagonal.
        self.nSamples = int((self.aabbDiag / self.stepSize).item()) + 1

    def normalize_coord(self, xyz_sampled: torch.Tensor) -> torch.Tensor:
        """Map world-space points into the grid's ``[-1, 1]`` sampling space."""
        return (xyz_sampled - self.aabb[0]) * self.invaabbSize - 1

    # ------------------------------------------------------------------
    # Implemented by subclasses
    # ------------------------------------------------------------------
    def init_svd_volume(self, res, device):
        raise NotImplementedError

    def compute_densityfeature(self, xyz_sampled, frame):
        raise NotImplementedError

    def compute_appfeature(self, xyz_sampled, frame):
        raise NotImplementedError

    def get_optparam_groups(self, lr_init_spatial=0.02, lr_init_network=0.001):
        raise NotImplementedError

    def upsample_volume_grid(self, res_target):
        raise NotImplementedError

    def shrink(self, new_aabb):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def get_kwargs(self) -> dict:
        """Constructor arguments needed to rebuild this field."""
        return {
            "aabb": self.aabb,
            "gridSize": self.gridSize.tolist(),
            "density_n_comp": self.density_n_comp,
            "appearance_n_comp": self.app_n_comp,
            "app_dim": self.app_dim,
            "density_shift": self.density_shift,
            "alphaMask_thres": self.alphaMask_thres,
            "distance_scale": self.distance_scale,
            "rayMarch_weight_thres": self.rayMarch_weight_thres,
            "fea2denseAct": self.fea2denseAct,
            "near_far": self.near_far,
            "step_ratio": self.step_ratio,
            "shadingMode": self.shadingMode,
            "pos_pe": self.pos_pe,
            "view_pe": self.view_pe,
            "fea_pe": self.fea_pe,
            "featureC": self.featureC,
            "num_frames": self.num_frames,
            "ld_per_frame": self.ld_per_frame,
        }

    def save(self, path: str) -> None:
        """Write weights plus the kwargs needed to rebuild the field.

        The alpha mask is stored bit-packed rather than as a float volume, which
        is roughly a 32x saving on what is otherwise the largest entry.
        """
        ckpt = {"kwargs": self.get_kwargs(), "state_dict": self.state_dict()}
        if self.alphaMask is not None:
            alpha_volume = self.alphaMask.alpha_volume.bool().cpu().numpy()
            ckpt["alphaMask.shape"] = alpha_volume.shape
            ckpt["alphaMask.mask"] = np.packbits(alpha_volume.reshape(-1))
            ckpt["alphaMask.aabb"] = self.alphaMask.aabb.cpu()
        torch.save(ckpt, path)

    def load(self, ckpt: dict) -> None:
        """Restore weights (and the alpha mask, if present) from a checkpoint."""
        if "alphaMask.aabb" in ckpt:
            length = np.prod(ckpt["alphaMask.shape"])
            alpha_volume = torch.from_numpy(
                np.unpackbits(ckpt["alphaMask.mask"])[:length].reshape(ckpt["alphaMask.shape"])
            )
            self.alphaMask = AlphaGridMask(
                self.device,
                ckpt["alphaMask.aabb"].to(self.device),
                alpha_volume.float().to(self.device),
            )

        state_dict = ckpt["state_dict"]
        # Earlier versions of this field carried an unused deformation branch
        # whose buffers are still present in the released checkpoints. Drop them
        # rather than refuse to load. Missing keys are still an error.
        stale = set(state_dict) - set(self.state_dict())
        if stale:
            print(f"ignoring {len(stale)} entry/entries from an older checkpoint: {sorted(stale)}")
            state_dict = {k: v for k, v in state_dict.items() if k not in stale}
        self.load_state_dict(state_dict)

    # ------------------------------------------------------------------
    # Ray sampling
    # ------------------------------------------------------------------
    def sample_ray_ndc(self, rays_o, rays_d, is_train=True, N_samples=-1):
        """Sample uniformly in NDC depth, where the far plane is finite."""
        N_samples = N_samples if N_samples > 0 else self.nSamples
        near, far = self.near_far
        interpx = torch.linspace(near, far, N_samples).unsqueeze(0).to(rays_o)
        if is_train:
            # Jitter within each interval so training does not lock onto a
            # fixed depth grid.
            interpx = interpx + torch.rand_like(interpx).to(rays_o) * ((far - near) / N_samples)

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = ((self.aabb[0] > rays_pts) | (rays_pts > self.aabb[1])).any(dim=-1)
        return rays_pts, interpx, ~mask_outbbox

    def sample_ray(self, rays_o, rays_d, is_train=True, N_samples=-1):
        """Sample at a fixed world-space step from where the ray enters the box."""
        N_samples = N_samples if N_samples > 0 else self.nSamples
        near, far = self.near_far

        # Slab test against the bounding box; the guard avoids dividing by a
        # zero direction component.
        vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (self.aabb[1] - rays_o) / vec
        rate_b = (self.aabb[0] - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)

        rng = torch.arange(N_samples)[None].float()
        if is_train:
            rng = rng.repeat(rays_d.shape[-2], 1)
            rng = rng + torch.rand_like(rng[:, [0]])
        interpx = t_min[..., None] + self.stepSize * rng.to(rays_o.device)

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = ((self.aabb[0] > rays_pts) | (rays_pts > self.aabb[1])).any(dim=-1)
        return rays_pts, interpx, ~mask_outbbox

    # ------------------------------------------------------------------
    # Occupancy
    # ------------------------------------------------------------------
    @torch.no_grad()
    def getDenseAlpha(self, gridSize=None, frame: float = 0.0):
        """Evaluate opacity on a dense grid at one timestamp.

        Used for mesh export.  The scene moves, so a mesh is only meaningful for
        a single ``frame``.
        """
        gridSize = self.gridSize if gridSize is None else gridSize
        samples = torch.stack(
            torch.meshgrid(
                torch.linspace(0, 1, gridSize[0]),
                torch.linspace(0, 1, gridSize[1]),
                torch.linspace(0, 1, gridSize[2]),
            ),
            dim=-1,
        ).to(self.device)
        dense_xyz = self.aabb[0] * (1 - samples) + self.aabb[1] * samples

        alpha = torch.zeros_like(dense_xyz[..., 0])
        # One x-slice at a time; the full volume does not fit in memory at the
        # resolutions used here.
        for i in range(gridSize[0]):
            alpha[i] = self.compute_alpha(
                dense_xyz[i].view(-1, 3), self.stepSize, frame=frame
            ).view((gridSize[1], gridSize[2]))
        return alpha, dense_xyz

    @torch.no_grad()
    def updateAlphaMask(self, gridSize=(200, 200, 200)):
        """Not supported for the streaming field.

        A single alpha mask cannot be shared across frames: geometry that is
        empty at one timestamp may be occupied at another, so masking would
        erase moving content.  Configs therefore set
        ``update_AlphaMask_list = [-1]``.
        """
        raise NotImplementedError(
            "the streaming field does not support alpha-mask updates; "
            "leave update_AlphaMask_list = [-1]"
        )

    @torch.no_grad()
    def filtering_rays(self, all_rays, all_rgbs, N_samples=256, chunk=10240 * 5, bbox_only=False):
        """Drop rays that never enter the scene bounding box.

        Args:
            all_rays: ``(N, 7)`` rays.
            all_rgbs: ``(N, 3)`` matching colours.
            N_samples: Samples per ray when testing against the alpha mask.
            chunk: Rays processed per batch.
            bbox_only: Use a cheap slab test instead of the alpha mask.

        Returns:
            The surviving ``(rays, rgbs)``.
        """
        print("filtering rays ...")
        started = time.time()

        total = torch.tensor(all_rays.shape[:-1]).prod()
        mask_filtered = []
        for idx_chunk in torch.split(torch.arange(total), chunk):
            rays_chunk = all_rays[idx_chunk].to(self.device)
            rays_o, rays_d = rays_chunk[..., :3], rays_chunk[..., 3:6]
            if bbox_only:
                vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
                rate_a = (self.aabb[1] - rays_o) / vec
                rate_b = (self.aabb[0] - rays_o) / vec
                t_min = torch.minimum(rate_a, rate_b).amax(-1)
                t_max = torch.maximum(rate_a, rate_b).amin(-1)
                mask_inbbox = t_max > t_min
            else:
                xyz_sampled, _, _ = self.sample_ray(
                    rays_o, rays_d, N_samples=N_samples, is_train=False
                )
                mask_inbbox = (
                    self.alphaMask.sample_alpha(xyz_sampled).view(xyz_sampled.shape[:-1]) > 0
                ).any(-1)
            mask_filtered.append(mask_inbbox.cpu())

        mask_filtered = torch.cat(mask_filtered).view(all_rgbs.shape[:-1])
        kept = torch.sum(mask_filtered) / total
        print(f"ray filtering done in {time.time() - started:.1f}s, kept {kept:.3f}")
        return all_rays[mask_filtered], all_rgbs[mask_filtered]

    def feature2density(self, density_features: torch.Tensor) -> torch.Tensor:
        """Map raw density features to non-negative densities."""
        if self.fea2denseAct == "softplus":
            return F.softplus(density_features + self.density_shift)
        if self.fea2denseAct == "relu":
            return F.relu(density_features)
        raise ValueError(f"fea2denseAct must be 'softplus' or 'relu', got {self.fea2denseAct!r}")

    def compute_alpha(
        self, xyz_locs: torch.Tensor, length: float = 1, frame: float = 0.0
    ) -> torch.Tensor:
        """Opacity of a set of points over a step of ``length``.

        Args:
            xyz_locs: ``(N, 3)`` world-space points.
            length: Step length the opacity is integrated over.
            frame: Timestamp in ``[0, 1]`` at which to evaluate the field.
        """
        if self.alphaMask is not None:
            alpha_mask = self.alphaMask.sample_alpha(xyz_locs) > 0
        else:
            alpha_mask = torch.ones_like(xyz_locs[:, 0], dtype=bool)

        sigma = torch.zeros(xyz_locs.shape[:-1], device=xyz_locs.device)
        if alpha_mask.any():
            xyz_sampled = self.normalize_coord(xyz_locs[alpha_mask])
            frame_t = torch.as_tensor(frame, device=xyz_locs.device)
            sigma[alpha_mask] = self.feature2density(
                self.compute_densityfeature(xyz_sampled, frame_t)
            )
        return 1 - torch.exp(-sigma * length).view(xyz_locs.shape[:-1])

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def forward(self, rays_chunk, white_bg=True, is_train=False, ndc_ray=False, N_samples=-1):
        """Volume-render one chunk of rays.

        Args:
            rays_chunk: ``(N, 7)`` rays as ``[origin, direction, time]``.  Every
                ray in the chunk must share the same timestamp, because the
                feature planes are gathered once per frame.
            white_bg: Composite over white instead of black.
            is_train: Enables sample jitter and random background augmentation.
            ndc_ray: Sample in NDC rather than world space.
            N_samples: Samples per ray; ``-1`` uses the resolution-derived count.

        Returns:
            ``(rgb_map, depth_map)``.
        """
        viewdirs = rays_chunk[:, 3:6]
        frame = rays_chunk[:, -1].unique()
        if len(frame) != 1:
            raise ValueError(f"a ray chunk must hold a single timestamp, got {frame.tolist()}")

        if ndc_ray:
            xyz_sampled, z_vals, ray_valid = self.sample_ray_ndc(
                rays_chunk[:, :3], viewdirs, is_train=is_train, N_samples=N_samples
            )
            dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
            # In NDC the directions are not unit length; rescale the step
            # lengths so density stays in world units.
            rays_norm = torch.norm(viewdirs, dim=-1, keepdim=True)
            dists = dists * rays_norm
            viewdirs = viewdirs / rays_norm
        else:
            xyz_sampled, z_vals, ray_valid = self.sample_ray(
                rays_chunk[:, :3], viewdirs, is_train=is_train, N_samples=N_samples
            )
            dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
        viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled.shape)

        if self.alphaMask is not None:
            alpha_mask = self.alphaMask.sample_alpha(xyz_sampled[ray_valid]) > 0
            ray_invalid = ~ray_valid
            ray_invalid[ray_valid] |= ~alpha_mask
            ray_valid = ~ray_invalid

        sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
        rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)

        if ray_valid.any():
            xyz_sampled = self.normalize_coord(xyz_sampled)
            sigma[ray_valid] = self.feature2density(
                self.compute_densityfeature(xyz_sampled[ray_valid], frame)
            )

        alpha, weight, _ = raw2alpha(sigma, dists * self.distance_scale)

        # Shading is the expensive half, so it only runs where a sample can
        # still contribute meaningfully to the pixel.
        app_mask = weight > self.rayMarch_weight_thres
        if app_mask.any():
            app_features = self.compute_appfeature(xyz_sampled[app_mask], frame)
            rgb[app_mask] = self.renderModule(
                xyz_sampled[app_mask], viewdirs[app_mask], app_features
            )

        acc_map = torch.sum(weight, dim=-1)
        rgb_map = torch.sum(weight[..., None] * rgb, dim=-2)

        # Randomly compositing over white during training regularises the
        # density: a surface that is only opaque against black gets penalised.
        if white_bg or (is_train and torch.rand((1,)) < 0.5):
            rgb_map = rgb_map + (1.0 - acc_map[..., None])
        rgb_map = rgb_map.clamp(0, 1)

        with torch.no_grad():
            depth_map = torch.sum(weight * z_vals, dim=-1)
            # Unoccupied rays fall back to the ray's own far bound.
            depth_map = depth_map + (1.0 - acc_map) * rays_chunk[..., -2]

        return rgb_map, depth_map

    # ------------------------------------------------------------------
    # Streaming feature buffer
    # ------------------------------------------------------------------
    def register_frame_permute_index(self, level_dim: int):
        """Precompute which feature channels each frame reads, and in what order.

        The channel buffer behaves like a queue: as time advances, the oldest
        channel is retired and a fresh one takes its place, while the rest are
        rotated.  Doing that bookkeeping once here keeps the per-ray lookup in
        :meth:`get_embeds` down to an index and a blend.

        Args:
            level_dim: Number of components in the plane being indexed.

        Returns:
            ``(index, permute)`` tensors of shape ``(num_rows, ...)``, one row
            per distinct channel configuration along time.
        """
        level_dim = int(level_dim / self.level_dim_multi)
        if self.ld_per_frame > 1:
            raise NotImplementedError("ld_per_frame > 1 is normalised in __init__")

        # Row 0 reads channel 0 and the newly added channel `level_dim`, then
        # the remaining channels in order.
        index_init = [0, level_dim] + list(range(1, level_dim))
        permute_base = list(range(1, level_dim))
        slot = 0  # position at which the fresh channel is spliced in
        permute_init = permute_base[:slot] + [0] + permute_base[slot:]

        index_list = [torch.as_tensor(index_init, dtype=torch.long)]
        permute_list = [torch.as_tensor(permute_init, dtype=torch.long)]

        for frame_i in range(1, self.num_frames - 1):
            if self.new_dim_interval == 0 or frame_i % self.new_dim_interval != 0:
                continue
            slot = slot + 1
            if slot >= level_dim:
                slot = 0
            last_index_max = index_list[-1].max().item()
            last_index_min = index_list[-1].min().item()
            previous = index_list[-1][1:][permute_list[-1]].tolist()
            previous.pop(slot)
            index_list.append(
                torch.as_tensor([last_index_min + 1, last_index_max + 1] + previous, dtype=torch.long)
            )
            permute_list.append(
                torch.as_tensor(permute_base[:slot] + [0] + permute_base[slot:], dtype=torch.long)
            )
        return torch.stack(index_list, 0), torch.stack(permute_list, 0)

    def get_embeds(self, frame, embedding, frame_index, frame_index_permute):
        """Gather the feature channels active at ``frame``.

        A frame generally falls between two channel configurations, so the two
        boundary channels are linearly blended.  That blend is what makes the
        representation continuous in time rather than piecewise constant.

        Args:
            frame: Timestamp in ``[0, 1]``.
            embedding: ``(1, n_comp, d0, d1)`` plane or line parameters.
            frame_index: Channel indices per row, from
                :meth:`register_frame_permute_index`.
            frame_index_permute: Matching output ordering.

        Returns:
            The gathered embedding, flattened back to ``(1, C, d0, d1)``.
        """
        if self.new_dim_interval == 0:
            return embedding

        _, _, ndim0, ndim1 = embedding.shape
        embedding = embedding.reshape([1, -1, self.level_dim_multi, ndim0, ndim1])

        if frame == 1:
            # The final timestamp sits exactly on the last configuration.
            row_idx = -1
            left, right = 1, 0
        else:
            row_value = frame * len(frame_index)
            row_idx = int(row_value)
            left, right = row_value - row_idx, row_idx + 1 - row_value

        feat_idx = frame_index[row_idx]
        if left == 0:
            feat = embedding[:, torch.cat([feat_idx[[0]], feat_idx[2:]], 0)]
        elif right == 0:
            feat = embedding[:, feat_idx[1:]]
        else:
            feat = embedding[:, feat_idx]
            feat = torch.cat([feat[:, [0]] * right + feat[:, [1]] * left, feat[:, 2:]], dim=1)
        return feat[:, frame_index_permute[row_idx]].flatten(1, 2)

    def feat_diff(self, frame, embedding, frame_index) -> torch.Tensor:
        """Mean absolute difference between the two channels being blended.

        Penalising this keeps consecutive frames' features close, which damps
        temporal flicker.
        """
        _, _, ndim0, ndim1 = embedding.shape
        embedding = embedding.reshape([1, -1, self.level_dim_multi, ndim0, ndim1])
        feat_idx = frame_index[int(frame * len(frame_index))]
        feat = embedding[:, feat_idx]
        return (feat[:, [0]] - feat[:, [1]]).abs().mean()

    def feat_diff_loss(self, frame) -> torch.Tensor:
        raise NotImplementedError
