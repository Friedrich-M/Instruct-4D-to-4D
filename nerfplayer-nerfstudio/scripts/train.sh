#!/usr/bin/env bash
# Reconstruct a single-view 4D NeRF, the starting point for editing.
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

DATA_DIR=${DATA_DIR:-./data/dycheck/mochi-high-five}
OUTPUT_DIR=${OUTPUT_DIR:-./log/dycheck}
MAX_ITERS=${MAX_ITERS:-30000}

ns-train nerfplayer-nerfacto \
    --data "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --max-num-iterations "${MAX_ITERS}"
