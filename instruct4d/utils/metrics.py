"""Image-quality metrics used to score renders against ground truth."""

from typing import Dict

import numpy as np
import scipy.signal
import torch

#: LPIPS networks are heavy, so they are built once and reused.
_LPIPS_CACHE: Dict[str, "torch.nn.Module"] = {}


def mse_to_psnr(mse: float) -> float:
    """Convert a mean squared error on ``[0, 1]`` images to PSNR in dB."""
    return -10.0 * np.log(mse) / np.log(10.0)


def rgb_lpips(gt: np.ndarray, pred: np.ndarray, net_name: str, device) -> float:
    """Learned Perceptual Image Patch Similarity; lower is better.

    Args:
        gt: ``(H, W, 3)`` ground truth in ``[0, 1]``.
        pred: ``(H, W, 3)`` prediction in ``[0, 1]``.
        net_name: Backbone, either ``"alex"`` or ``"vgg"``.
        device: Device to run the backbone on.
    """
    if net_name not in _LPIPS_CACHE:
        import lpips

        if net_name not in ("alex", "vgg"):
            raise ValueError(f"net_name must be 'alex' or 'vgg', got {net_name!r}")
        print(f"initialising LPIPS ({net_name})")
        _LPIPS_CACHE[net_name] = lpips.LPIPS(net=net_name, version="0.1").eval().to(device)

    gt_t = torch.from_numpy(gt).permute(2, 0, 1).contiguous().to(device)
    pred_t = torch.from_numpy(pred).permute(2, 0, 1).contiguous().to(device)
    return _LPIPS_CACHE[net_name](gt_t, pred_t, normalize=True).item()


def rgb_ssim(
    img0,
    img1,
    max_val: float,
    filter_size: int = 11,
    filter_sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    return_map: bool = False,
):
    """Structural similarity between two images; higher is better.

    Follows the reference mip-NeRF implementation, which applies a separable
    Gaussian window and evaluates only on fully-covered pixels (``mode="valid"``).

    Args:
        img0: ``(H, W, 3)`` image.
        img1: ``(H, W, 3)`` image of the same shape.
        max_val: Dynamic range of the inputs, i.e. ``1.0`` for ``[0, 1]`` images.
        filter_size: Side length of the Gaussian window.
        filter_sigma: Standard deviation of the Gaussian window.
        k1: Stabilising constant for the luminance term.
        k2: Stabilising constant for the contrast term.
        return_map: Return the per-pixel SSIM map instead of its mean.
    """
    img0, img1 = np.asarray(img0), np.asarray(img1)
    if img0.ndim != 3 or img0.shape[-1] != 3:
        raise ValueError(f"expected an (H, W, 3) image, got {img0.shape}")
    if img0.shape != img1.shape:
        raise ValueError(f"shape mismatch: {img0.shape} vs {img1.shape}")

    # Separable 1D Gaussian: two 1D passes are much cheaper than one 2D pass.
    half = filter_size // 2
    shift = (2 * half - filter_size + 1) / 2
    weights = np.exp(-0.5 * ((np.arange(filter_size) - half + shift) / filter_sigma) ** 2)
    weights /= np.sum(weights)

    def blur(z):
        return np.stack(
            [
                scipy.signal.convolve2d(
                    scipy.signal.convolve2d(z[..., i], weights[:, None], mode="valid"),
                    weights[None, :],
                    mode="valid",
                )
                for i in range(z.shape[-1])
            ],
            axis=-1,
        )

    mu0, mu1 = blur(img0), blur(img1)
    mu00, mu11, mu01 = mu0 * mu0, mu1 * mu1, mu0 * mu1
    sigma00 = np.maximum(0.0, blur(img0**2) - mu00)
    sigma11 = np.maximum(0.0, blur(img1**2) - mu11)
    sigma01 = blur(img0 * img1) - mu01
    # Clamp the covariance to the Cauchy-Schwarz bound; numerical error in the
    # blur can otherwise push it outside the valid range.
    sigma01 = np.sign(sigma01) * np.minimum(np.sqrt(sigma00 * sigma11), np.abs(sigma01))

    c1, c2 = (k1 * max_val) ** 2, (k2 * max_val) ** 2
    numerator = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denominator = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numerator / denominator
    return ssim_map if return_map else np.mean(ssim_map)
