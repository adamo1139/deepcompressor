#!/usr/bin/env bash
set -euo pipefail

# Usage: bash test_qwen_image_edit_nvfp4.sh [OUTPUT_DIRNAME]

OUTDIR=${1:-test-qwen-image-edit-nvfp4}

python -m deepcompressor.app.diffusion.ptq \
  "./examples/diffusion/configs/model/qwen-image-edit.yaml" \
  "./examples/diffusion/configs/svdquant/nvfp4.yaml" \
  --output-dirname "${OUTDIR}"


