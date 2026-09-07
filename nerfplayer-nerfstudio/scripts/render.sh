#!/usr/bin/env bash
# Render a video from an edited single-view 4D NeRF.
#
# LOAD_DIR must point at the nerfstudio_models directory of a run produced by
# scripts/edit.sh. The video is written inside that run's directory.
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

DATA_DIR=${DATA_DIR:-./data/dycheck/mochi-high-five}
LOAD_DIR=${LOAD_DIR:?set LOAD_DIR to a nerfstudio_models directory from scripts/edit.sh}
OUTPUT_DIR=${OUTPUT_DIR:-./log/dycheck}

ns-train edit-nerfacto \
    --data "${DATA_DIR}" \
    --load-dir "${LOAD_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --render-mode True
