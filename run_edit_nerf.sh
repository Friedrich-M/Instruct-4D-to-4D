export CUDA_VISIBLE_DEVICES=0,1,
export PYTHONPATH=$PYTHONPATH:./

python stream_edit.py --config configs/n3dv/edit_coffee_50_2.txt \
    --datadir data/neural_3d/coffee_martini \
    --basedir log/neural_3d --expname edit_coffee_50_2 \
    --cache cache/neural_3d \
    --ckpt log/neural_3d/train_coffee_50_2/ckpt-99999.th \
    --prompt 'What if it was painted by Van Gogh?' \
    --guidance_scale 9.5 --image_guidance_scale 1.5 \
    --diffusion_steps 20 --refine_num_steps 600 --refine_diffusion_steps 4 \
    --restview_refine_num_steps 700 --restview_refine_diffusion_steps 6
