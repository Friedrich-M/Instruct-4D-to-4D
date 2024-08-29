# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from RAFT.raft import RAFT
from RAFT.utils import flow_viz
from RAFT.utils.utils import InputPadder
import torch.nn.functional as F
import torchvision

from flow_utils import *

DEVICE = "cuda"


def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    return img_flo


def load_image(imfile):
    long_dim = 768
    img = np.array(Image.open(imfile).convert("RGB")).astype(np.uint8)

    # Portrait Orientation
    if img.shape[0] > img.shape[1]:
        input_h = long_dim
        input_w = int(round(float(input_h) / img.shape[0] * img.shape[1]))
    # Landscape Orientation
    else:
        input_w = long_dim
        input_h = int(round(float(input_w) / img.shape[1] * img.shape[0]))

    print("flow input w %d h %d" % (input_w, input_h))
    img = cv2.resize(img, (input_w, input_h), cv2.INTER_LINEAR)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def resize_flow(flow, img_h, img_w):
    flow_h, flow_w = flow.shape[0], flow.shape[1]
    flow[:, :, 0] *= float(img_w) / float(flow_w)
    flow[:, :, 1] *= float(img_h) / float(flow_h)
    flow = cv2.resize(flow, (img_w, img_h), cv2.INTER_LINEAR)

    return flow


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


def compute_fwdbwd_mask(fwd_flow, bwd_flow):
    alpha_1 = 0.5 
    alpha_2 = 0.5

    bwd2fwd_flow = warp_flow(bwd_flow, fwd_flow) 
    fwd_lr_error = np.linalg.norm(fwd_flow + bwd2fwd_flow, axis=-1) 
    fwd_mask = (
        fwd_lr_error
        < alpha_1
        * (np.linalg.norm(fwd_flow, axis=-1) + np.linalg.norm(bwd2fwd_flow, axis=-1))
        + alpha_2
    ) 

    fwd2bwd_flow = warp_flow(fwd_flow, bwd_flow)
    bwd_lr_error = np.linalg.norm(bwd_flow + fwd2bwd_flow, axis=-1)

    bwd_mask = (
        bwd_lr_error
        < alpha_1
        * (np.linalg.norm(bwd_flow, axis=-1) + np.linalg.norm(fwd2bwd_flow, axis=-1))
        + alpha_2
    )

    return fwd_mask, bwd_mask


def run(args, input_path, output_path, output_img_path):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(input_path, "*.png")) + glob.glob(
            os.path.join(input_path, "*.jpg")
        )

        images = sorted(images)
        img_train = cv2.imread(images[0])
        for i in range(len(images) - 1):
            print(i)
            image1 = load_image(images[i]) 
            image2 = load_image(images[i + 1])

            padder = InputPadder(image1.shape) 
            image1, image2 = padder.pad(image1, image2) 
            
            _, flow_fwd = model(image1, image2, iters=20, test_mode=True) 
            _, flow_bwd = model(image2, image1, iters=20, test_mode=True)

            flow_fwd = padder.unpad(flow_fwd[0]).cpu().numpy().transpose(1, 2, 0) 
            flow_bwd = padder.unpad(flow_bwd[0]).cpu().numpy().transpose(1, 2, 0) 
 
            flow_fwd = resize_flow(flow_fwd, img_train.shape[0], img_train.shape[1]) 
            flow_bwd = resize_flow(flow_bwd, img_train.shape[0], img_train.shape[1])

            mask_fwd, mask_bwd = compute_fwdbwd_mask(flow_fwd, flow_bwd) 
            
            import ipdb; ipdb.set_trace()
            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            img1_warp = warp(image1, flow_up)
            save_warp = img1_warp / 255.0
            torchvision.utils.save_image(save_warp, 'warp.png')
            
            np.savez(
                os.path.join(output_path, "%05d_fwd.npz" % i),
                flow=flow_fwd,
                mask=mask_fwd,
            ) 
            np.savez(
                os.path.join(output_path, "%05d_bwd.npz" % (i + 1)),
                flow=flow_bwd,
                mask=mask_bwd,
            ) 

            # Save flow_img
            Image.fromarray(flow_viz.flow_to_image(flow_fwd)).save(
                os.path.join(output_img_path, "%05d_fwd.png" % i)
            ) 
            Image.fromarray(flow_viz.flow_to_image(flow_bwd)).save(
                os.path.join(output_img_path, "%05d_bwd.png" % (i + 1))
            )

            # Save mask_img
            Image.fromarray(mask_fwd).save(
                os.path.join(output_img_path, "%05d_fwd_mask.png" % i)
            ) 
            Image.fromarray(mask_bwd).save(
                os.path.join(output_img_path, "%05d_bwd_mask.png" % (i + 1))
            )
            
def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    
    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flo
    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid)
    mask = torch.ones(x.size()).to(DEVICE)
    mask = F.grid_sample(mask, vgrid)

    mask[mask < 0.999] = 0
    mask[mask > 0] = 1

    return output
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, help="Dataset path")
    parser.add_argument("--model", help="restore RAFT checkpoint")
    parser.add_argument("--small", action="store_true", help="use small model")
    parser.add_argument(
        "--mixed_precision", action="store_true", help="use mixed precision"
    )
    args = parser.parse_args()

    input_path = os.path.join(args.dataset_path)
    output_path = os.path.join(args.dataset_path, "flow")
    output_img_path = os.path.join(args.dataset_path, "flow_png")
    create_dir(output_path)
    create_dir(output_img_path)

    run(args, input_path, output_path, output_img_path)
