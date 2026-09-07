#!/usr/bin/env bash
# Edit a trained single-view 4D NeRF with a text instruction.
#
# LOAD_DIR must point at the nerfstudio_models directory of a run produced by
# scripts/train.sh; it contains a timestamp, so set it to your own run.
set -euo pipefail

# Editing and optimisation run at the same time, so both GPUs are used.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

DATA_DIR=${DATA_DIR:-./data/dycheck/mochi-high-five}
LOAD_DIR=${LOAD_DIR:?set LOAD_DIR to a nerfstudio_models directory from scripts/train.sh}
OUTPUT_DIR=${OUTPUT_DIR:-./log/dycheck}

PROMPT=${PROMPT:-"What if it was painted by Edward Hopper?"}
MAX_ITERS=${MAX_ITERS:-20000}

ns-train edit-nerfacto \
    --data "${DATA_DIR}" \
    --load-dir "${LOAD_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --max-num-iterations "${MAX_ITERS}" \
    --pipeline.prompt "${PROMPT}" \
    --pipeline.guidance-scale 7.5 \
    --pipeline.image-guidance-scale 1.5 \
    --pipeline.diffusion-steps 20 \
    --pipeline.refine-diffusion-steps 5 \
    --pipeline.refine-num-steps 700 \
    --pipeline.sequence-length 5 \
    --pipeline.resize-512 True \
    --pipeline.ip2p-device "cuda:1"
