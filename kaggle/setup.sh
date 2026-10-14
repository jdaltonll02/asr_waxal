#!/usr/bin/env bash
# WAXAL-NET Kaggle setup script
# Run this cell first in any Kaggle notebook to install dependencies.
set -euo pipefail

echo "=== GPU Info ==="
nvidia-smi || echo "No GPU detected"

echo "=== Installing requirements ==="
pip install --quiet \
  torch==2.8.0 \
  torchaudio==2.8.0 \
  accelerate==1.10.1 \
  transformers==4.51.3 \
  huggingface_hub==0.32.6 \
  'datasets[audio]==3.6.0' \
  numpy==1.26.4 \
  jiwer==4.0.0 \
  omnilingual-asr==0.1.0 \
  PyYAML==6.0.3 \
  soundfile==0.13.1 \
  'wandb==0.19.11'

echo "=== Setting up HF cache ==="
mkdir -p /kaggle/working/hf_cache
export HF_HOME=/kaggle/working/hf_cache
export TRANSFORMERS_CACHE=/kaggle/working/hf_cache
export WANDB_DISABLED=true

echo "=== Disk usage ==="
df -h /kaggle/working

echo "=== Setup complete ==="
