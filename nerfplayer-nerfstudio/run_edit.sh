data_dir=data/dycheck/mochi-high-five
load_dir=log/dycheck/mochi-high-five/nerfplayer-nerfacto/2024-09-22_212045/nerfstudio_models
output_dir=log/dycheck

max_num_iterations=20000
prompt="What if it was painted by Edward Hopper?"
guidance_scale=7.5
image_guidance_scale=1.5
diffusion_steps=20
ip2p_device='cuda:1'
resize_512=True
refine_diffusion_steps=5
refine_num_steps=600

ns-train edit-nerfacto --data ${data_dir} --load-dir ${load_dir} --max-num-iterations ${max_num_iterations} --pipeline.prompt "${prompt}" --pipeline.guidance-scale ${guidance_scale} --pipeline.image-guidance-scale ${image_guidance_scale} --pipeline.diffusion-steps ${diffusion_steps} --pipeline.ip2p-device "${ip2p_device}" --pipeline.resize-512 ${resize_512} --pipeline.refine-diffusion-steps ${refine_diffusion_steps} --pipeline.refine-num-steps ${refine_num_steps} --output-dir ${output_dir}