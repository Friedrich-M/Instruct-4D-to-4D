"""Tensor factorisations of the streaming 4D radiance field.

Two decompositions from TensoRF, each extended along time by
:class:`~instruct4d.fields.tensor_base.StreamTensorBase`:

:class:`StreamTensorVMSplit`
    Vector-Matrix.  Each component is a plane times a line, giving a compact but
    expressive grid.  This is the default and what every shipped config uses.
:class:`StreamTensorCP`
    CANDECOMP/PARAFAC.  Each component is a product of three lines: far smaller,
    correspondingly less expressive.

The two differ only in how a feature is stored and looked up; everything about
ray marching, shading and time is inherited.
"""

from math import ceil
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F

from .tensor_base import StreamTensorBase


def _line_plane_coordinates(
    xyz_sampled: torch.Tensor, mat_mode: Sequence[Sequence[int]], vec_mode: Sequence[int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project sample points onto the plane and line grids.

    ``grid_sample`` wants ``(N, H, W, 2)`` coordinates, so each projection is
    reshaped into a degenerate 1-pixel-tall image.  Lines only need one
    coordinate, and the other is filled with zeros.

    Args:
        xyz_sampled: ``(N, 3)`` points already in ``[-1, 1]``.
        mat_mode: Axis pairs spanned by each plane.
        vec_mode: Axis along which each line runs.

    Returns:
        ``(plane_coords, line_coords)``, each ``(3, N, 1, 2)``.
    """
    plane = torch.stack(
        (
            xyz_sampled[..., mat_mode[0]],
            xyz_sampled[..., mat_mode[1]],
            xyz_sampled[..., mat_mode[2]],
        )
    ).detach().view(3, -1, 1, 2)

    line = torch.stack(
        (
            xyz_sampled[..., vec_mode[0]],
            xyz_sampled[..., vec_mode[1]],
            xyz_sampled[..., vec_mode[2]],
        )
    )
    line = torch.stack((torch.zeros_like(line), line), dim=-1).detach().view(3, -1, 1, 2)
    return plane, line


class StreamTensorVMSplit(StreamTensorBase):
    """Vector-Matrix decomposition: three plane/line pairs per field."""

    def init_svd_volume(self, res, device):
        self.density_plane, self.density_line = self.init_one_svd(
            self.density_n_comp, self.gridSize, 0.1, device
        )
        self.app_plane, self.app_line = self.init_one_svd(
            self.app_n_comp, self.gridSize, 0.1, device
        )
        # Projects the concatenated per-plane features down to `app_dim`.
        self.basis_mat = torch.nn.Linear(sum(self.app_n_comp), self.app_dim, bias=False).to(device)

    def init_one_svd(self, n_component, gridSize, scale, device):
        """Allocate one plane/line pair per axis, widened for the time axis.

        Beyond the components a static scene would need, extra channels are
        allocated to hold the frames the streaming buffer rotates through.
        """
        widen = lambda c: c + ceil((self.num_frames - 1) * self.ld_per_frame) * self.level_dim_multi

        plane_coef, line_coef = [], []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef.append(
                torch.nn.Parameter(
                    scale
                    * torch.randn((1, widen(n_component[i]), gridSize[mat_id_1], gridSize[mat_id_0]))
                )
            )
            line_coef.append(
                torch.nn.Parameter(
                    scale * torch.randn((1, widen(n_component[i]), gridSize[vec_id], 1))
                )
            )
        return (
            torch.nn.ParameterList(plane_coef).to(device),
            torch.nn.ParameterList(line_coef).to(device),
        )

    def get_optparam_groups(self, lr_init_spatialxyz=0.02, lr_init_network=0.001) -> List[dict]:
        """Parameter groups, with a higher learning rate for the grids than the MLPs."""
        grad_vars = [
            {"params": self.density_line, "lr": lr_init_spatialxyz},
            {"params": self.density_plane, "lr": lr_init_spatialxyz},
            {"params": self.app_line, "lr": lr_init_spatialxyz},
            {"params": self.app_plane, "lr": lr_init_spatialxyz},
            {"params": self.basis_mat.parameters(), "lr": lr_init_network},
        ]
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars.append({"params": self.renderModule.parameters(), "lr": lr_init_network})
        return grad_vars

    # ------------------------------------------------------------------
    # Regularisers
    # ------------------------------------------------------------------
    def vectorDiffs(self, vector_comps) -> torch.Tensor:
        """Mean off-diagonal Gram entry, i.e. how non-orthogonal the lines are."""
        total = 0
        for comp in vector_comps:
            n_comp, n_size = comp.shape[1:-1]
            flat = comp.view(n_comp, n_size)
            dotp = torch.matmul(flat, flat.transpose(-1, -2))
            # Drop the diagonal by reading the Gram matrix off-by-one.
            non_diagonal = dotp.view(-1)[1:].view(n_comp - 1, n_comp + 1)[..., :-1]
            total = total + torch.mean(torch.abs(non_diagonal))
        return total

    def vector_comp_diffs(self) -> torch.Tensor:
        """Orthogonality penalty across all line components."""
        return self.vectorDiffs(self.density_line) + self.vectorDiffs(self.app_line)

    def density_L1(self) -> torch.Tensor:
        """L1 sparsity penalty on the density grids."""
        total = 0
        for idx in range(len(self.density_plane)):
            total = total + torch.mean(torch.abs(self.density_plane[idx]))
            total = total + torch.mean(torch.abs(self.density_line[idx]))
        return total

    def TV_loss_density(self, reg) -> torch.Tensor:
        """Total variation on the density grids.

        Planes are weighted 10x the lines because they hold two spatial axes and
        so carry most of the high-frequency detail.
        """
        total = 0
        for idx in range(len(self.density_plane)):
            total = total + reg(self.density_plane[idx]) * 1e-2 + reg(self.density_line[idx]) * 1e-3
        return total

    def TV_loss_app(self, reg) -> torch.Tensor:
        """Total variation on the appearance grids."""
        total = 0
        for idx in range(len(self.app_plane)):
            total = total + reg(self.app_plane[idx]) * 1e-2 + reg(self.app_line[idx]) * 1e-3
        return total

    def feat_diff_loss(self, frame) -> torch.Tensor:
        """Penalise the gap between the two channels blended at ``frame``.

        Keeps neighbouring timestamps' features close, damping temporal flicker.
        """
        loss = 0
        for idx in range(len(self.density_plane)):
            loss = loss + self.feat_diff(frame, self.density_plane[idx], self.density_list[idx][0])
            loss = loss + self.feat_diff(frame, self.density_line[idx], self.density_list[idx][0])
        for idx in range(len(self.app_plane)):
            loss = loss + self.feat_diff(frame, self.app_plane[idx], self.app_list[idx][0])
            loss = loss + self.feat_diff(frame, self.app_line[idx], self.app_list[idx][0])
        return loss

    # ------------------------------------------------------------------
    # Feature lookup
    # ------------------------------------------------------------------
    def compute_densityfeature(self, xyz_sampled, frame) -> torch.Tensor:
        """Density feature: sum over components of plane value times line value."""
        coordinate_plane, coordinate_line = _line_plane_coordinates(
            xyz_sampled, self.matMode, self.vecMode
        )
        sigma_feature = torch.zeros((xyz_sampled.shape[0],), device=xyz_sampled.device)
        for idx in range(len(self.density_plane)):
            plane_coef = F.grid_sample(
                self.get_embeds(frame.item(), self.density_plane[idx], *self.density_list[idx]),
                coordinate_plane[[idx]],
                align_corners=True,
            ).view(-1, *xyz_sampled.shape[:1])
            line_coef = F.grid_sample(
                self.get_embeds(frame.item(), self.density_line[idx], *self.density_list[idx]),
                coordinate_line[[idx]],
                align_corners=True,
            ).view(-1, *xyz_sampled.shape[:1])
            sigma_feature = sigma_feature + torch.sum(plane_coef * line_coef, dim=0)
        return sigma_feature

    def compute_appfeature(self, xyz_sampled, frame) -> torch.Tensor:
        """Appearance feature: per-component products, then a linear projection."""
        coordinate_plane, coordinate_line = _line_plane_coordinates(
            xyz_sampled, self.matMode, self.vecMode
        )
        plane_coefs, line_coefs = [], []
        for idx in range(len(self.app_plane)):
            plane_coefs.append(
                F.grid_sample(
                    self.get_embeds(frame.item(), self.app_plane[idx], *self.app_list[idx]),
                    coordinate_plane[[idx]],
                    align_corners=True,
                ).view(-1, *xyz_sampled.shape[:1])
            )
            line_coefs.append(
                F.grid_sample(
                    self.get_embeds(frame.item(), self.app_line[idx], *self.app_list[idx]),
                    coordinate_line[[idx]],
                    align_corners=True,
                ).view(-1, *xyz_sampled.shape[:1])
            )
        plane_coefs, line_coefs = torch.cat(plane_coefs), torch.cat(line_coefs)
        return self.basis_mat((plane_coefs * line_coefs).T)

    # ------------------------------------------------------------------
    # Resolution changes
    # ------------------------------------------------------------------
    @torch.no_grad()
    def up_sampling_VM(self, plane_coef, line_coef, res_target):
        """Bilinearly resample every plane and line to a new resolution."""
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef[i] = torch.nn.Parameter(
                F.interpolate(
                    plane_coef[i].data,
                    size=(res_target[mat_id_1], res_target[mat_id_0]),
                    mode="bilinear",
                    align_corners=True,
                )
            )
            line_coef[i] = torch.nn.Parameter(
                F.interpolate(
                    line_coef[i].data, size=(res_target[vec_id], 1), mode="bilinear", align_corners=True
                )
            )
        return plane_coef, line_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        """Grow the grid, coarse-to-fine, part-way through training."""
        self.app_plane, self.app_line = self.up_sampling_VM(self.app_plane, self.app_line, res_target)
        self.density_plane, self.density_line = self.up_sampling_VM(
            self.density_plane, self.density_line, res_target
        )
        self.update_stepSize(res_target)
        print(f"upsampled grid to {res_target}")

    @torch.no_grad()
    def shrink(self, new_aabb):
        """Crop the grids to a tighter bounding box, freeing empty space."""
        xyz_min, xyz_max = new_aabb
        t_l = torch.round((xyz_min - self.aabb[0]) / self.units).long()
        b_r = torch.round((xyz_max - self.aabb[0]) / self.units).long() + 1
        b_r = torch.stack([b_r, self.gridSize]).amin(0)

        for i in range(len(self.vecMode)):
            mode0 = self.vecMode[i]
            self.density_line[i] = torch.nn.Parameter(
                self.density_line[i].data[..., t_l[mode0] : b_r[mode0], :]
            )
            self.app_line[i] = torch.nn.Parameter(
                self.app_line[i].data[..., t_l[mode0] : b_r[mode0], :]
            )
            mode0, mode1 = self.matMode[i]
            self.density_plane[i] = torch.nn.Parameter(
                self.density_plane[i].data[..., t_l[mode1] : b_r[mode1], t_l[mode0] : b_r[mode0]]
            )
            self.app_plane[i] = torch.nn.Parameter(
                self.app_plane[i].data[..., t_l[mode1] : b_r[mode1], t_l[mode0] : b_r[mode0]]
            )

        new_aabb = self._snap_aabb_to_grid(new_aabb, t_l, b_r)
        new_size = b_r - t_l
        self.aabb = new_aabb
        self.update_stepSize((new_size[0], new_size[1], new_size[2]))

    def _snap_aabb_to_grid(self, new_aabb, t_l, b_r):
        """Round a requested bounding box out to the voxel boundaries it hit."""
        if self.alphaMask is not None and torch.all(self.alphaMask.gridSize == self.gridSize):
            return new_aabb
        t_l_r, b_r_r = t_l / (self.gridSize - 1), (b_r - 1) / (self.gridSize - 1)
        corrected = torch.zeros_like(new_aabb)
        corrected[0] = (1 - t_l_r) * self.aabb[0] + t_l_r * self.aabb[1]
        corrected[1] = (1 - b_r_r) * self.aabb[0] + b_r_r * self.aabb[1]
        return corrected


class StreamTensorCP(StreamTensorBase):
    """CP decomposition: every component is a product of three lines."""

    def init_svd_volume(self, res, device):
        self.density_line = self.init_one_svd(self.density_n_comp[0], self.gridSize, 0.2, device)
        self.app_line = self.init_one_svd(self.app_n_comp[0], self.gridSize, 0.2, device)
        self.basis_mat = torch.nn.Linear(self.app_n_comp[0], self.app_dim, bias=False).to(device)

    def init_one_svd(self, n_component, gridSize, scale, device):
        """Allocate one line per axis, widened for the time axis."""
        widen = lambda c: c + ceil((self.num_frames - 1) * self.ld_per_frame) * self.level_dim_multi
        line_coef = [
            torch.nn.Parameter(
                scale * torch.randn((1, widen(n_component), gridSize[self.vecMode[i]], 1))
            )
            for i in range(len(self.vecMode))
        ]
        return torch.nn.ParameterList(line_coef).to(device)

    def get_optparam_groups(self, lr_init_spatialxyz=0.02, lr_init_network=0.001) -> List[dict]:
        grad_vars = [
            {"params": self.density_line, "lr": lr_init_spatialxyz},
            {"params": self.app_line, "lr": lr_init_spatialxyz},
            {"params": self.basis_mat.parameters(), "lr": lr_init_network},
        ]
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars.append({"params": self.renderModule.parameters(), "lr": lr_init_network})
        return grad_vars

    def _sample_lines(self, lines, embed_list, coordinate_line, xyz_sampled, frame) -> torch.Tensor:
        """Multiply the three per-axis line samples together."""
        result = None
        for axis in range(3):
            coef = F.grid_sample(
                self.get_embeds(frame, lines[axis], *embed_list),
                coordinate_line[[axis]],
                align_corners=True,
            ).view(-1, *xyz_sampled.shape[:1])
            result = coef if result is None else result * coef
        return result

    def compute_densityfeature(self, xyz_sampled, frame) -> torch.Tensor:
        _, coordinate_line = _line_plane_coordinates(xyz_sampled, self.matMode, self.vecMode)
        line_coef = self._sample_lines(
            self.density_line, self.density_list[0], coordinate_line, xyz_sampled, frame.item()
        )
        return torch.sum(line_coef, dim=0)

    def compute_appfeature(self, xyz_sampled, frame) -> torch.Tensor:
        _, coordinate_line = _line_plane_coordinates(xyz_sampled, self.matMode, self.vecMode)
        line_coef = self._sample_lines(
            self.app_line, self.app_list[0], coordinate_line, xyz_sampled, frame.item()
        )
        return self.basis_mat(line_coef.T)

    def density_L1(self) -> torch.Tensor:
        return sum(torch.mean(torch.abs(line)) for line in self.density_line)

    def TV_loss_density(self, reg) -> torch.Tensor:
        return sum(reg(line) * 1e-3 for line in self.density_line)

    def TV_loss_app(self, reg) -> torch.Tensor:
        return sum(reg(line) * 1e-3 for line in self.app_line)

    @torch.no_grad()
    def up_sampling_Vector(self, density_line_coef, app_line_coef, res_target):
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            density_line_coef[i] = torch.nn.Parameter(
                F.interpolate(
                    density_line_coef[i].data,
                    size=(res_target[vec_id], 1),
                    mode="bilinear",
                    align_corners=True,
                )
            )
            app_line_coef[i] = torch.nn.Parameter(
                F.interpolate(
                    app_line_coef[i].data,
                    size=(res_target[vec_id], 1),
                    mode="bilinear",
                    align_corners=True,
                )
            )
        return density_line_coef, app_line_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        self.density_line, self.app_line = self.up_sampling_Vector(
            self.density_line, self.app_line, res_target
        )
        self.update_stepSize(res_target)
        print(f"upsampled grid to {res_target}")

    @torch.no_grad()
    def shrink(self, new_aabb):
        xyz_min, xyz_max = new_aabb
        t_l = torch.round((xyz_min - self.aabb[0]) / self.units).long()
        b_r = torch.round((xyz_max - self.aabb[0]) / self.units).long() + 1
        b_r = torch.stack([b_r, self.gridSize]).amin(0)

        for i in range(len(self.vecMode)):
            mode0 = self.vecMode[i]
            self.density_line[i] = torch.nn.Parameter(
                self.density_line[i].data[..., t_l[mode0] : b_r[mode0], :]
            )
            self.app_line[i] = torch.nn.Parameter(
                self.app_line[i].data[..., t_l[mode0] : b_r[mode0], :]
            )

        if self.alphaMask is None or not torch.all(self.alphaMask.gridSize == self.gridSize):
            t_l_r, b_r_r = t_l / (self.gridSize - 1), (b_r - 1) / (self.gridSize - 1)
            corrected = torch.zeros_like(new_aabb)
            corrected[0] = (1 - t_l_r) * self.aabb[0] + t_l_r * self.aabb[1]
            corrected[1] = (1 - b_r_r) * self.aabb[0] + b_r_r * self.aabb[1]
            new_aabb = corrected

        new_size = b_r - t_l
        self.aabb = new_aabb
        self.update_stepSize((new_size[0], new_size[1], new_size[2]))
