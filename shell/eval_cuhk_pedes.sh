#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export SUGARCREPE_CONFIG=configs/sugarcrepe_cuhk_pedes.yaml
export RETRIEVAL_CONFIG=configs/text_to_image_retrieval_cuhk_pedes.yaml
uv run python text-to-image-retrieval.py
uv run python sugarcrepe-pedes.py
