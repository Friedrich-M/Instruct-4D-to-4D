#!/usr/bin/env bash
# Render held-out views and a fly-through video from a checkpoint.
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

CONFIG=${CONFIG:-configs/n3dv/train_coffee_50_2.txt}
CKPT=${CKPT:-log/neural_3d/train_coffee_50_2/ckpt-99999.th}

python render.py --config "${CONFIG}" --ckpt "${CKPT}" \
    --render_test 1 --render_path 1
