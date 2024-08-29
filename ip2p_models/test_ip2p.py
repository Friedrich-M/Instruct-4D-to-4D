from diffusers import (
    StableDiffusionInstructPix2PixPipeline,
)
import configargparse as argparse
import torch
import math
from PIL import Image, ImageOps

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ip2p = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix", torch_dtype=torch.float16, safety_checker=None).to(device)  

ip2p.enable_xformers_memory_efficient_attention()
ip2p.enable_model_cpu_offload()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="./examples/coffee_frame_2x/0.png")
    parser.add_argument("--prompt", type=str, default="What if it was painted by Van Gogh?")
    parser.add_argument("--resize", type=int, default=None)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--image_guidance_scale", type=float, default=1.5)
    return parser.parse_args()

args = parse_args()
image = Image.open(args.image_path).convert("RGB")
original_width, original_height = image.size
if args.resize is None:
    args.resize = max(original_width, original_height)
factor = args.resize / max(original_width, original_height)
factor = math.ceil(min(original_width, original_height) * factor / 64) * 64 / min(original_width, original_height)
width = int((original_width * factor) // 64) * 64
height = int((original_height * factor) // 64) * 64
image = ImageOps.fit(image, (width, height), method=Image.Resampling.LANCZOS)

prompt = args.prompt
steps = args.steps
guidance_scale = args.guidance_scale
image_guidance_scale = args.image_guidance_scale
images = ip2p(prompt, image=image, num_inference_steps=steps, guidance_scale=guidance_scale, image_guidance_scale=image_guidance_scale).images[0]
# resizes the image to the original size
images = ImageOps.fit(images, (original_width, original_height), method=Image.Resampling.LANCZOS)

images.save(f"ip2p_{prompt.split(' ')[-1].replace('?', '')}_{args.image_path.split('/')[-1]}")


