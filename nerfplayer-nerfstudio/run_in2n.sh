data_dir=data/dycheck/mochi-high-five
load_dir=log/dycheck/mochi-high-five/nerfplayer-nerfacto/2024-09-22_212045/nerfstudio_models
output_dir=logs

max_num_iterations=20000
prompt="turn the cat into a fox"
# prompt="turn the cat into a tiger"
guidance_scale=8.5
image_guidance_scale=1.5
diffusion_steps=10
ip2p_device='cuda:1'
lower_bound=0.02
upper_bound=0.98

ns-train in2n-nerfacto --data ${data_dir} --load-dir ${load_dir} --max-num-iterations ${max_num_iterations} --pipeline.prompt "${prompt}" --pipeline.guidance-scale ${guidance_scale} --pipeline.image-guidance-scale ${image_guidance_scale} --pipeline.diffusion-steps ${diffusion_steps} --output-dir ${output_dir} --pipeline.ip2p-device "${ip2p_device}" --pipeline.lower-bound ${lower_bound} --pipeline.upper-bound ${upper_bound}