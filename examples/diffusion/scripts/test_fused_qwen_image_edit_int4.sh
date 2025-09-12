#!/usr/bin/env bash
set -euo pipefail

# Usage: bash test_fused_qwen_image_edit_int4.sh [OUTPUT_DIRNAME]

OUTDIR=${1:-test-fused-qwen-image-edit-int4}

python -m deepcompressor.app.diffusion.ptq \
  "./examples/diffusion/configs/model/fused-qwen-image-edit.yaml" \
  "./examples/diffusion/configs/svdquant/int4.yaml" \
  --output-dirname "${OUTDIR}"
