#!/usr/bin/env bash
set -euo pipefail

# Usage: bash test_qwen_image_edit_nvfp4.sh [OUTPUT_DIRNAME]

OUTDIR=${1:-test-qwen-image-edit-nvfp4}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

python -m deepcompressor.app.diffusion.ptq \
  "$REPO_ROOT/examples/diffusion/configs/model/qwen-image-edit.yaml" \
  "$REPO_ROOT/examples/diffusion/configs/svdquant/nvfp4.yaml" \
  --output-dirname "${OUTDIR}"


