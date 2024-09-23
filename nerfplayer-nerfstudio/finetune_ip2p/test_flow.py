import argparse
import numpy as np
import torch
from PIL import Image, ImageOps

from RAFT.raft import RAFT
from RAFT.utils.utils import InputPadder
import flow_viz
import torch.nn.functional as F
from diffusers import (
    StableDiffusionInstructPix2PixPipeline,
)
import torchvision
import math

from flow_utils import *

DEVICE = "cuda"

def load_image(imfile):
    img = np.array(Image.open(imfile).convert("RGB")).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float() # [3, H, W]
    return img[None].to(DEVICE) # [1, 3, H, W]

def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)
    
    return img_flo # [H, 2*W, 3]

def viz_flow(flo):
    flo = flo[0].permute(1,2,0).cpu().numpy()
    flo = flow_viz.flow_to_image(flo)
    
    return flo 

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


def run(args, image1_path, image2_path):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()
    
    ip2p = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix", torch_dtype=torch.float16, safety_checker=None).to(DEVICE)
    
    prompt = "What if it was painted by Edvard Munch?"
    steps=20
    guidance_scale=6.5
    image_guidance_scale=1.5
    
    ip2p.enable_xformers_memory_efficient_attention()
    ip2p.enable_model_cpu_offload()

    with torch.no_grad():
        image1 = load_image(image1_path) 
        image2 = load_image(image2_path) 

        padder = InputPadder(image1.shape) 
        image1, image2 = padder.pad(image1, image2) 
        
        image = Image.open(image1_path).convert("RGB")
        edit_image = ip2p(prompt, image=image, num_inference_steps=steps, guidance_scale=guidance_scale, image_guidance_scale=image_guidance_scale, output_type="pt").images # [1, 3, H, W] [0, 1]
        edit_image = torch.nn.functional.interpolate(edit_image, size=(image1.shape[2], image1.shape[3]), mode='bilinear', align_corners=False) # [1, 3, H, W] [0, 1]
        
        import ipdb; ipdb.set_trace()
        
        _, flow_fwd = model(image1, image2, iters=20, test_mode=True) 
        _, flow_bwd = model(image2, image1, iters=20, test_mode=True)
        
        flow_fwd = padder.unpad(flow_fwd[0]).cpu().numpy().transpose(1, 2, 0) 
        flow_bwd = padder.unpad(flow_bwd[0]).cpu().numpy().transpose(1, 2, 0) 
        
        fwd_viz = flow_viz.flow_to_image(flow_fwd)
        bwd_viz = flow_viz.flow_to_image(flow_bwd)
        
        Image.fromarray(fwd_viz).save('fwd_viz.png')
        Image.fromarray(bwd_viz).save('bwd_viz.png')
        
        mask_fwd, mask_bwd = compute_fwdbwd_mask(flow_fwd, flow_bwd)
         
        Image.fromarray((mask_fwd * 255).astype(np.uint8)).save('mask_fwd.png')
        Image.fromarray((mask_bwd * 255).astype(np.uint8)).save('mask_bwd.png')
        
        image1 = (image1[0].permute(1,2,0).cpu().numpy()).astype(np.uint8)
        image2 = (image2[0].permute(1,2,0).cpu().numpy()).astype(np.uint8)
        edit_image = (edit_image[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
        
        Image.fromarray(image1).save('image1.png')
        Image.fromarray(image2).save('image2.png')
        Image.fromarray(edit_image).save('edit_image.png')
        
        warp_to_image1 = warp_flow(image2, flow_fwd)
        warp_to_image2 = warp_flow(edit_image, flow_bwd)
        
        Image.fromarray(warp_to_image1.astype(np.uint8)).save('warp_to_image1.png')
        Image.fromarray(warp_to_image2.astype(np.uint8)).save('warp_to_image2.png')
        
        warp_to_image1 = warp_to_image1 * mask_fwd[:,:,np.newaxis]
        warp_to_image2 = warp_to_image2 * mask_bwd[:,:,np.newaxis]
        
        Image.fromarray(warp_to_image1.astype(np.uint8)).save('warp_to_image1_masked.png')
        Image.fromarray(warp_to_image2.astype(np.uint8)).save('warp_to_image2_masked.png')
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='weights/raft-things.pth', help="restore RAFT checkpoint")
    parser.add_argument("--small", action="store_true", help="use small model")
    parser.add_argument(
        "--mixed_precision", action="store_true", help="use mixed precision"
    ) 
    args = parser.parse_args()

    image1_path = 'data/coffee/000003.jpg'
    image2_path = 'data/coffee/000006.jpg'
    run(args, image1_path, image2_path)
