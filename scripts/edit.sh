#!/usr/bin/env bash
# Edit a trained 4D NeRF with a text instruction.
#
# Two GPUs: the radiance field is optimised on the first while the diffusion
# model edits frames on the second. Point --ip2p_device at a single GPU only if
# it has the memory to hold both.
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

CONFIG=${CONFIG:-configs/n3dv/edit_coffee_50_2.txt}
CKPT=${CKPT:-log/neural_3d/train_coffee_50_2/ckpt-99999.th}
PROMPT=${PROMPT:-"What if it was painted by Van Gogh?"}

python edit.py --config "${CONFIG}" \
    --ckpt "${CKPT}" \
    --prompt "${PROMPT}" \
    --guidance_scale 9.5 --image_guidance_scale 1.5 \
    --diffusion_steps 20 \
    --refine_diffusion_steps 4 --refine_num_steps 600 \
    --restview_refine_diffusion_steps 6 --restview_refine_num_steps 700 \
    --ip2p_device cuda:1
