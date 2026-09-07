"""Real spherical-harmonic bases, for the ``SH`` shading mode.

Selected with ``--shadingMode SH``, which decodes the appearance feature as SH
coefficients instead of running it through an MLP. Cheaper and view-dependent by
construction, but less expressive; every shipped config uses ``MLP_Fea``.

Coefficients follow the standard real SH normalisation, hard-coded up to degree
four rather than evaluated from the general formula.
"""

import torch

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]

def eval_sh_bases(deg, dirs):
    """Evaluate the SH basis functions at a set of unit directions.

    The bases are returned without being combined, so a caller multiplies them
    by its own coefficients.

    Args:
        deg: Maximum SH degree, 0 to 4.
        dirs: ``(..., 3)`` unit directions.

    Returns:
        ``(..., (deg + 1) ** 2)`` basis values.
    """
    if not 0 <= deg <= 4:
        raise ValueError(f"only SH degrees 0-4 are supported, got {deg}")
    result = torch.empty((*dirs.shape[:-1], (deg + 1) ** 2), dtype=dirs.dtype, device=dirs.device)
    result[..., 0] = C0
    if deg > 0:
        x, y, z = dirs.unbind(-1)
        result[..., 1] = -C1 * y
        result[..., 2] = C1 * z
        result[..., 3] = -C1 * x
        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result[..., 4] = C2[0] * xy
            result[..., 5] = C2[1] * yz
            result[..., 6] = C2[2] * (2.0 * zz - xx - yy)
            result[..., 7] = C2[3] * xz
            result[..., 8] = C2[4] * (xx - yy)
            if deg > 2:
                result[..., 9] = C3[0] * y * (3 * xx - yy)
                result[..., 10] = C3[1] * xy * z
                result[..., 11] = C3[2] * y * (4 * zz - xx - yy)
                result[..., 12] = C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
                result[..., 13] = C3[4] * x * (4 * zz - xx - yy)
                result[..., 14] = C3[5] * z * (xx - yy)
                result[..., 15] = C3[6] * x * (xx - 3 * yy)
                if deg > 3:
                    result[..., 16] = C4[0] * xy * (xx - yy)
                    result[..., 17] = C4[1] * yz * (3 * xx - yy)
                    result[..., 18] = C4[2] * xy * (7 * zz - 1)
                    result[..., 19] = C4[3] * yz * (7 * zz - 3)
                    result[..., 20] = C4[4] * (zz * (35 * zz - 30) + 3)
                    result[..., 21] = C4[5] * xz * (7 * zz - 3)
                    result[..., 22] = C4[6] * (xx - yy) * (7 * zz - 1)
                    result[..., 23] = C4[7] * xz * (xx - 3 * yy)
                    result[..., 24] = C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))
    return result
