"""Regularisers applied to the TensoRF feature planes."""

import torch
from torch import nn


class TVLoss(nn.Module):
    """Isotropic total-variation penalty on a 4D feature map.

    Encourages neighbouring voxels to hold similar features, which suppresses
    the high-frequency noise that a sparse tensor decomposition is prone to.
    """

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(B, C, H, W)`` feature map.  TensoRF's line factors have a
                width of 1, so the horizontal term is skipped for them; see
                :meth:`_axis_variation`.

        Returns:
            Scalar loss, averaged over the batch and normalised by element count
            so the value does not depend on the grid resolution.
        """
        batch_size = x.shape[0]
        tv_h = self._axis_variation(x, axis=2)
        tv_w = self._axis_variation(x, axis=3)
        return self.weight * 2 * (tv_h + tv_w) / batch_size

    @staticmethod
    def _axis_variation(x: torch.Tensor, axis: int) -> torch.Tensor:
        """Mean squared difference between neighbours along one axis.

        An axis of extent 1 has no neighbouring pairs and contributes nothing.
        Guarding that is not optional: TensoRF's line factors are ``(1, C, N, 1)``,
        so without it the normalising count is zero and the loss is NaN.
        """
        if x.shape[axis] < 2:
            return x.new_zeros(())
        leading = x.narrow(axis, 1, x.shape[axis] - 1)
        trailing = x.narrow(axis, 0, x.shape[axis] - 1)
        diff = torch.pow(leading - trailing, 2)
        return diff.sum() / (diff.numel() // x.shape[0])
