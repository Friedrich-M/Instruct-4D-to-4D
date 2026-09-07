#!/usr/bin/env bash
# Reconstruct a 4D NeRF from a multi-view video scene. Editing starts from this.
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

CONFIG=${CONFIG:-configs/n3dv/train_coffee_50_2.txt}

python train.py --config "${CONFIG}" --render_test 1 --render_path 1
