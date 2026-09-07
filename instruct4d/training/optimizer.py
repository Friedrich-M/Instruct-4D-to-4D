"""The optimisation step shared by reconstruction and editing.

Both entry points run the same inner loop -- render a batch of rays, compare
against the target colours, add the grid regularisers, step Adam -- and differ
only in where the target colours come from and in what happens between steps.
:class:`FieldOptimizer` owns that inner loop so the two schedules above it stay
short enough to read.
"""

import torch

from ..rendering import render_rays
from ..utils.losses import TVLoss
from ..utils.metrics import mse_to_psnr


class FieldOptimizer:
    """Adam plus the exponential learning-rate decay and grid regularisers.

    Attributes:
        lr_factor: Per-iteration multiplier that decays the learning rate from
            its initial value to ``lr_decay_target_ratio`` of it.
    """

    def __init__(self, field, args, device, white_bg: bool, n_samples: int):
        """
        Args:
            field: The radiance field to optimise.
            args: Parsed configuration; see :mod:`instruct4d.config`.
            device: Device to render on.
            white_bg: Composite over a white background.
            n_samples: Samples per ray.
        """
        self.field = field
        self.args = args
        self.device = device
        self.white_bg = white_bg
        self.n_samples = n_samples

        if args.lr_decay_iters > 0:
            decay_iters = args.lr_decay_iters
        else:
            decay_iters = args.n_iters
        self.lr_factor = args.lr_decay_target_ratio ** (1 / decay_iters)
        print(f"lr decay: target ratio {args.lr_decay_target_ratio} over {decay_iters} iters")

        self.optimizer = torch.optim.Adam(
            field.get_optparam_groups(args.lr_init, args.lr_basis), betas=(0.9, 0.99)
        )

        self.tv_reg = TVLoss()
        self.ortho_weight = args.Ortho_weight
        self.l1_weight = args.L1_weight_initial
        self.tv_weight_density = args.TV_weight_density
        self.tv_weight_app = args.TV_weight_app

    def reset_optimizer(self, lr_scale: float = 1.0) -> None:
        """Rebuild Adam after the grid resolution changed.

        Upsampling replaces the grid parameters with new tensors, so the old
        optimiser state refers to tensors that no longer exist.
        """
        self.optimizer = torch.optim.Adam(
            self.field.get_optparam_groups(
                self.args.lr_init * lr_scale, self.args.lr_basis * lr_scale
            ),
            betas=(0.9, 0.99),
        )

    def step(self, rays: torch.Tensor, rgbs: torch.Tensor, summary_writer=None, iteration: int = 0):
        """Run one optimisation step.

        Args:
            rays: ``(B, 7)`` training rays.
            rgbs: ``(B, 3)`` target colours, already on ``device``.
            summary_writer: Optional TensorBoard writer for the loss breakdown.
            iteration: Global step, used as the TensorBoard x-axis.

        Returns:
            ``(mse, psnr)`` for this batch.
        """
        rgb_map, _ = render_rays(
            rays,
            self.field,
            chunk=self.args.batch_size,
            N_samples=self.n_samples,
            white_bg=self.white_bg,
            ndc_ray=self.args.ndc_ray,
            device=self.device,
            is_train=True,
        )

        photometric = torch.mean((rgb_map - rgbs) ** 2)
        total_loss = photometric

        if self.ortho_weight > 0:
            ortho = self.field.vector_comp_diffs()
            total_loss = total_loss + self.ortho_weight * ortho
            _log(summary_writer, "train/reg_ortho", ortho, iteration)

        if self.l1_weight > 0:
            l1 = self.field.density_L1()
            total_loss = total_loss + self.l1_weight * l1
            _log(summary_writer, "train/reg_l1", l1, iteration)

        # The TV weights decay alongside the learning rate, so the grids are
        # smoothed hard early on and left alone once detail starts to appear.
        tv_loss = None
        if self.tv_weight_density > 0:
            self.tv_weight_density *= self.lr_factor
            tv_loss = self.field.TV_loss_density(self.tv_reg) * self.tv_weight_density
            total_loss = total_loss + tv_loss
            _log(summary_writer, "train/reg_tv_density", tv_loss, iteration)

        if self.tv_weight_app > 0:
            self.tv_weight_app *= self.lr_factor
            app_tv = self.field.TV_loss_app(self.tv_reg) * self.tv_weight_app
            # NOTE: `tv_loss` still holds the density term, so when both TV
            # weights are active the density term is counted twice. This matches
            # the configuration the published results were produced with; set
            # `tv_loss = 0` here to weight the two terms independently.
            tv_loss = (tv_loss if tv_loss is not None else 0) + app_tv
            total_loss = total_loss + tv_loss
            _log(summary_writer, "train/reg_tv_app", tv_loss, iteration)

        if self.args.feat_diff_weight > 0:
            # Evaluated at one random timestamp per step; averaging over all of
            # them would cost as much as the render itself.
            frame = torch.randint(0, self.field.num_frames, [1]).item() / self.field.num_frames
            total_loss = total_loss + self.field.feat_diff_loss(frame) * self.args.feat_diff_weight

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        for param_group in self.optimizer.param_groups:
            param_group["lr"] *= self.lr_factor

        mse = photometric.detach().item()
        psnr = mse_to_psnr(mse)
        if summary_writer is not None:
            summary_writer.add_scalar("train/mse", mse, global_step=iteration)
            summary_writer.add_scalar("train/PSNR", psnr, global_step=iteration)
        return mse, psnr


def _log(writer, tag: str, value: torch.Tensor, iteration: int) -> None:
    if writer is not None:
        writer.add_scalar(tag, value.detach().item(), global_step=iteration)
