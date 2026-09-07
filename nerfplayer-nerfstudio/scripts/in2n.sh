#!/usr/bin/env bash
# Instruct-NeRF2NeRF baseline: edits each frame independently, with no
# cross-frame consistency. Run this to reproduce the comparison in the paper.
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

DATA_DIR=${DATA_DIR:-./data/dycheck/mochi-high-five}
LOAD_DIR=${LOAD_DIR:?set LOAD_DIR to a nerfstudio_models directory from scripts/train.sh}
OUTPUT_DIR=${OUTPUT_DIR:-./log/dycheck}

PROMPT=${PROMPT:-"turn the cat into a fox"}
MAX_ITERS=${MAX_ITERS:-20000}

ns-train in2n-nerfacto \
    --data "${DATA_DIR}" \
    --load-dir "${LOAD_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --max-num-iterations "${MAX_ITERS}" \
    --pipeline.prompt "${PROMPT}" \
    --pipeline.guidance-scale 8.5 \
    --pipeline.image-guidance-scale 1.5 \
    --pipeline.diffusion-steps 10 \
    --pipeline.lower-bound 0.02 \
    --pipeline.upper-bound 0.98 \
    --pipeline.ip2p-device "cuda:1"
