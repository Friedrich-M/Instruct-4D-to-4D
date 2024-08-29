import torch
import torch.nn.functional as F
import numpy as np
import cv2

enhomo0 = lambda x: torch.cat([x, torch.ones([1, *x.shape[1:]], device=x.device, dtype=x.dtype)], dim=0)
dehomo0 = lambda x: x[:-1] / x[-1:]

def warp_pts_AfromB(ptsA, intriB, w2cB):
    """
    Args:
        pts: (H, W, 3)
        intrinsics: (3, 3)
        w2c: (4, 4)
    """
    H, W, _ = ptsA.shape
    x = ptsA.reshape(-1, 3).T # (3, H*W)
    x = dehomo0(w2cB @ enhomo0(x)) # (3, H*W)
    x = intriB @ x # (3, H*W)
    pixB = dehomo0(x).T # (H*W, 3)
    pixB = pixB.reshape(H, W, 2)
    
    return pixB

def apply_warp(
    AfromB: torch.Tensor, # (H, W, 2) 
    imgA: torch.Tensor, # (H, W, 3) the ground truth image
    imgB: torch.Tensor, # (H, W, 3) the image to be warped
    ptsA: torch.Tensor, # (H, W, 3) the 3D points in imgA
    ptsB: torch.Tensor, # (H, W, 3) the 3D points in imgB
    u=None, d=None, l=None, r=None, # the region of imgA to be warped
    diff_thres = 0.2, # the threshold of the distance between the 3D points in imgA and imgB
    default = torch.tensor([0, 0, 0]) # the default color of the pixels in imgA that are not warped from imgB
):
    """Warp imgB to imgA
    """
    u, d, l, r = u or 0, d or imgA.shape[0], l or 0, r or imgA.shape[1]
    default = default.to(dtype=imgA.dtype, device=imgA.device)
    
    Y, X = (AfromB - 0.5).unbind(-1) # (H, W)
    mask = (u <= X) & (X <= d - 1) & (l <= Y) & (Y <= r - 1) 
    X = ((X - u) / (d - 1 - u) * 2 - 1) * mask 
    Y = ((Y - l) / (r - 1 - l) * 2 - 1) * mask 
    pix = torch.stack([-Y, X], dim=-1).unsqueeze(0) 
    imgA = F.grid_sample(imgB[None, u:d, l:r].permute(0, 3, 1, 2), pix, mode='bilinear', align_corners=True).squeeze(0).permute(1, 2, 0) 
    pts_AfromB = F.grid_sample(ptsB[None, u:d, l:r].permute(0, 3, 1, 2), pix, mode='bilinear', align_corners=True).squeeze(0).permute(1, 2, 0) 
    diff = (ptsA - pts_AfromB).norm(dim=-1) 
    
    mask &= (diff < diff_thres)
    imgA[~mask] = default 
    
    return imgA, mask, diff

def apply_warp_latents(
    AfromB: torch.Tensor, # (H, W, 2) Each pixel's (x,y) position in B for A
    latentA: torch.Tensor, # (H, W, 4) the ground truth features
    latentB: torch.Tensor, # (H, W, 4) the features to be warped
    ptsA: torch.Tensor, # (H, W, 3) the 3D points in latentA
    ptsB: torch.Tensor, # (H, W, 3) the 3D points in latentB
    u=None, d=None, l=None, r=None, # the region of latentA to be warped
    diff_thres = 0.2, # the threshold of the distance between the 3D points in latentA and latentB
    default = torch.tensor([0, 0, 0, 0]) # the default feature for the pixels in latentA that are not warped from latentB
):
    """Warp latentB to latentA"""
    u, d, l, r = u or 0, d or latentA.shape[0], l or 0, r or latentA.shape[1]
    default = default.to(dtype=latentA.dtype, device=latentA.device)
    
    Y, X = (AfromB - 0.5).unbind(-1) # (H, W)
    mask = (u <= X) & (X <= d - 1) & (l <= Y) & (Y <= r - 1)
    X = ((X - u) / (d - 1 - u) * 2 - 1) * mask
    Y = ((Y - l) / (r - 1 - l) * 2 - 1) * mask
    pix = torch.stack([-Y, X], dim=-1).unsqueeze(0)
    
    # The grid_sample function is called with a 4-dimensional input instead of 3
    latentA_warped = F.grid_sample(latentB[None, u:d, l:r].permute(0, 3, 1, 2), pix, mode='bilinear', align_corners=True).squeeze(0).permute(1, 2, 0)
    pts_AfromB = F.grid_sample(ptsB[None, u:d, l:r].permute(0, 3, 1, 2), pix, mode='bilinear', align_corners=True).squeeze(0).permute(1, 2, 0)
    diff = (ptsA - pts_AfromB).norm(dim=-1)
    
    mask &= (diff < diff_thres)
    latentA_warped[~mask] = default
    
    return latentA_warped, mask


def warp_flow(img, flow): 
    # warp image according to flow
    h, w = flow.shape[:2]
    flow_new = flow.copy() 
    flow_new[:, :, 0] += np.arange(w) 
    flow_new[:, :, 1] += np.arange(h)[:, np.newaxis] 

    res = cv2.remap(
        img, flow_new, None, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT
    )
    return res

def compute_bwd_mask(fwd_flow, bwd_flow):
    # compute the backward mask
    alpha_1 = 0.5 
    alpha_2 = 0.5

    fwd2bwd_flow = warp_flow(fwd_flow, bwd_flow)
    bwd_lr_error = np.linalg.norm(bwd_flow + fwd2bwd_flow, axis=-1)

    bwd_mask = (
        bwd_lr_error
        < alpha_1
        * (np.linalg.norm(bwd_flow, axis=-1) + np.linalg.norm(fwd2bwd_flow, axis=-1))
        + alpha_2
    )

    return bwd_mask
