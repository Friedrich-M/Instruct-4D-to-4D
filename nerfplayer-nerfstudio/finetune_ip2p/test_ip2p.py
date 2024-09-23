from diffusers import (
    StableDiffusionInstructPix2PixPipeline,
)
import torch
import math
from PIL import Image, ImageOps

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
ip2p = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix", torch_dtype=torch.float16, safety_checker=None).to(device)  

ip2p.enable_xformers_memory_efficient_attention()
ip2p.enable_model_cpu_offload()

image_path = '/u/linzhan/nerfplayer-nerfstudio/key_frame_condition_image_fox.png'
image = Image.open(image_path).convert("RGB")
original_width, original_height = image.size
# factor = 512 / max(original_width, original_height)
# factor = math.ceil(min(original_width, original_height) * factor / 64) * 64 / min(original_width, original_height)
# width = int((original_width * factor) // 64) * 64
# height = int((original_height * factor) // 64) * 64
# image = ImageOps.fit(image, (width, height), method=Image.Resampling.LANCZOS)

image.save("ip2p_original.png")

# prompt = "What if it was painted by Edvard Munch?"
prompt = "Turn the cat into a fox"
generator = torch.manual_seed(1371)
steps=20
guidance_scale=6.5
image_guidance_scale=1.5
images = ip2p(prompt, image=image, num_inference_steps=steps, guidance_scale=guidance_scale, image_guidance_scale=image_guidance_scale).images[0]
# resizes the image to the original size
images = ImageOps.fit(images, (original_width, original_height), method=Image.Resampling.LANCZOS)

images.save("ip2p_edit.png")


